#!/usr/bin/env python3
"""
zed_capture.py — record everything the ZED gives us, for later analysis.

Two steps, because writing full dumps live is slower than the camera (each one
costs a dropped frame), while replaying an SVO never drops anything:

    1. Record live -> out/captures/<YYYYmmdd_HHMMSS>/
           session.json        camera model, serial, SDK, calibration, settings
           recording.svo2      every frame, stereo video (H264) + IMU, ~2 MB/s
           frames.jsonl        one line per frame: timestamp, IMU, camera pose, dropped frames
    2. Extract from the SVO -> out/captures/recording_extract_<HHMMSS>/
           session.json, frames.jsonl   as above
           frames/000120_left.png  _right.png  _depth.npy  _confidence.npy  [_xyz.npy]
                               ~10 MB per frame (+11 MB with --pointcloud)

depth.npy is float32 metres (NaN/inf = no depth). confidence.npy is the SDK's
0-100 confidence (lower = more confident). xyz.npy is HxWx3 float32 metres in
RIGHT_HANDED_Z_UP (x right, y forward, z up).

Usage:
    python3 vision/depth/zed_capture.py --seconds 30                       # 1. record
    python3 vision/depth/zed_capture.py --svo out/captures/<run>/recording.svo2   # 2. extract all
    python3 vision/depth/zed_capture.py --svo <file> --dump-every 5 --pointcloud
    python3 vision/depth/zed_capture.py --seconds 10 --dump-every 30       # live dumps (drops frames)

    import numpy as np
    depth = np.load("out/captures/<run>/frames/000120_depth.npy")

Only one process can open the ZED at a time — close ZED_Explorer / ZED_Depth_Viewer first.
"""

import argparse
import json
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zed_depth import DEFAULT_FPS, DEFAULT_MODE, DEFAULT_RES, MODES, OUT_DIR, ZedDepth, _write_json_atomic

# ── Config ─────────────────────────────────────────────────────────────────────
CAPTURE_DIR  = OUT_DIR / "captures"
MIN_FREE_GB  = 3.0                  # stop recording before the shared disk fills up
SVO_CODECS   = ["H264", "H265", "LOSSLESS"]   # first one the encoder accepts wins


# ── Static info ────────────────────────────────────────────────────────────────

def _cam_params(c) -> dict:
    return {"fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy,
            "h_fov_deg": c.h_fov, "v_fov_deg": c.v_fov,
            "distortion": [float(x) for x in c.disto]}


def session_info(cam: ZedDepth, args, svo_codec) -> dict:
    info = cam.cam.get_camera_information()
    cfg  = info.camera_configuration
    cal  = cfg.calibration_parameters
    return {
        "time":              time.strftime("%Y-%m-%d %H:%M:%S"),
        "sdk":               sl.Camera.get_sdk_version(),
        "camera_model":      str(info.camera_model),
        "serial":            info.serial_number,
        "camera_firmware":   cfg.firmware_version,
        "sensors_firmware":  info.sensors_configuration.firmware_version,
        "resolution":        [cam.width, cam.height],
        "fps":               cfg.fps,
        "depth_mode":        cam.mode,
        "coordinate_system": "RIGHT_HANDED_Z_UP",
        "units":             "METER",
        "baseline_m":        cal.get_camera_baseline(),
        "left":              _cam_params(cal.left_cam),
        "right":             _cam_params(cal.right_cam),
        "stereo_translation_m": [float(x) for x in cal.stereo_transform.get_translation().get()],
        "svo_codec":         svo_codec,
        "dump_every":        args.dump_every,
        "pointcloud":        args.pointcloud,
        "positional_tracking": not args.no_tracking,
    }


# ── Per-frame info ─────────────────────────────────────────────────────────────

def frame_record(cam: ZedDepth, idx, sensors, pose) -> dict:
    zed = cam.cam
    r = {
        "frame":        idx,
        "timestamp_ns": zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds(),
        "grab_ms":      round(cam.last_grab_ms, 2),
        "dropped":      zed.get_frame_dropped_count(),
    }

    if zed.get_sensors_data(sensors, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
        imu = sensors.get_imu_data()
        r["imu"] = {
            "accel_mps2":    [round(float(v), 4) for v in imu.get_linear_acceleration()],
            "gyro_dps":      [round(float(v), 4) for v in imu.get_angular_velocity()],
            "orientation_q": [round(float(v), 5) for v in imu.get_pose().get_orientation().get()],
        }

    if pose is not None:
        state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        r["pose"] = {
            "state":         str(state),
            "translation_m": [round(float(v), 4) for v in pose.get_translation().get()],
            "orientation_q": [round(float(v), 5) for v in pose.get_orientation().get()],
            "confidence":    pose.pose_confidence,
        }
    return r


def snapshot_frame(cam: ZedDepth, pointcloud, mats) -> dict:
    """Copy this frame's images + depth/confidence (+ xyz) out of the SDK buffers."""
    zed, arrays = cam.cam, {}
    for view, tag in ((sl.VIEW.LEFT, "left"), (sl.VIEW.RIGHT, "right")):
        zed.retrieve_image(mats["img"], view)
        arrays[tag] = mats["img"].get_data()[:, :, :3].copy()        # BGRA -> BGR
    arrays["depth"] = cam.depth_map().copy()
    zed.retrieve_measure(mats["conf"], sl.MEASURE.CONFIDENCE)
    arrays["confidence"] = mats["conf"].get_data().copy()
    if pointcloud:
        zed.retrieve_measure(mats["xyz"], sl.MEASURE.XYZ)
        arrays["xyz"] = mats["xyz"].get_data()[:, :, :3].copy()
    return arrays


class DumpWriter:
    """
    Writes frame dumps on a background thread so disk I/O never stalls grab().
    If the writer falls behind, new dumps are skipped rather than queued forever.
    """

    def __init__(self, frames_dir: Path, max_pending=4):
        self.frames_dir = frames_dir
        self.q = queue.Queue(maxsize=max_pending)
        self.written = self.skipped = 0
        self.write_ms = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, idx, arrays, block=False) -> bool:
        try:
            self.q.put((idx, arrays), block=block)
            return True
        except queue.Full:
            self.skipped += 1
            return False

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            idx, arrays = item
            t0 = time.perf_counter()
            stem = self.frames_dir / f"{idx:06d}"
            for tag, a in arrays.items():
                if tag in ("left", "right"):
                    cv2.imwrite(f"{stem}_{tag}.png", a, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                else:
                    np.save(f"{stem}_{tag}.npy", a)
            self.write_ms.append((time.perf_counter() - t0) * 1000)
            self.written += 1

    def close(self):
        """Finish everything already queued."""
        self.q.put(None)
        self._thread.join()


# ── Capture ────────────────────────────────────────────────────────────────────

def enable_svo(zed, path: Path):
    """Start SVO recording with the first codec that works. Returns its name or None."""
    for codec in SVO_CODECS:
        params = sl.RecordingParameters(str(path), getattr(sl.SVO_COMPRESSION_MODE, codec))
        err = zed.enable_recording(params)
        if err == sl.ERROR_CODE.SUCCESS:
            return codec
        print(f"[capture] SVO {codec} unavailable: {err}")
    return None


def run_capture(args):
    replay = bool(args.svo)
    if args.dump_every is None:                  # live: SVO only (no drops); replay: every frame
        args.dump_every = 1 if replay else 0

    run_dir = Path(args.out) / (f"{Path(args.svo).stem}_extract_{time.strftime('%H%M%S')}"
                                if replay else time.strftime("%Y%m%d_%H%M%S"))
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage(run_dir).free / 1e9
    print(f"[capture] writing to {run_dir}  ({free_gb:.1f} GB free)")
    if free_gb < MIN_FREE_GB:
        sys.exit(f"[capture] refusing to start: under {MIN_FREE_GB} GB free")

    cam = ZedDepth(args.mode, args.resolution, args.fps, svo=args.svo)
    cam.open()
    writer = None
    try:
        zed = cam.cam
        src = f"SVO {args.svo}" if replay else "live camera"
        print(f"[capture] opened {src}: {args.mode} {cam.width}x{cam.height} in {cam.open_s:.1f}s")

        svo_codec = None if (args.no_svo or replay) else enable_svo(zed, run_dir / "recording.svo2")

        pose = None
        if not args.no_tracking:
            err = zed.enable_positional_tracking(sl.PositionalTrackingParameters())
            if err == sl.ERROR_CODE.SUCCESS:
                pose = sl.Pose()
            else:
                print(f"[capture] positional tracking unavailable: {err}")

        info = session_info(cam, args, svo_codec)
        info["source"] = str(Path(args.svo).resolve()) if replay else "live"
        _write_json_atomic(run_dir / "session.json", info)

        sensors = sl.SensorsData()
        mats = {"img": sl.Mat(), "conf": sl.Mat(), "xyz": sl.Mat()}
        writer = DumpWriter(frames_dir)
        frames = 0
        dropped_start = zed.get_frame_dropped_count()     # drops during open/tracking init
        copy_ms, t_win, win_frames = [], time.time(), 0
        t_start = time.time()

        with open(run_dir / "frames.jsonl", "w") as log:
            while replay or time.time() - t_start < args.seconds:     # replay runs to the end
                if not cam.grab():
                    break
                rec = frame_record(cam, frames, sensors, pose)
                if args.dump_every and frames % args.dump_every == 0:
                    t0 = time.perf_counter()
                    # replay can wait for the disk (no frames to lose); live must not
                    rec["dumped"] = writer.submit(frames, snapshot_frame(cam, args.pointcloud, mats),
                                                  block=replay)
                    copy_ms.append((time.perf_counter() - t0) * 1000)
                log.write(json.dumps(rec) + "\n")
                frames += 1
                win_frames += 1

                now = time.time()
                if now - t_win >= 1.0:
                    w_ms = np.mean(writer.write_ms[-10:]) if writer.write_ms else 0
                    print(f"  {win_frames / (now - t_win):5.1f} FPS | frame {frames:5d} | "
                          f"dumps {writer.written:4d} written, {writer.skipped} skipped "
                          f"(copy {np.mean(copy_ms) if copy_ms else 0:4.0f} ms, write {w_ms:4.0f} ms) | "
                          f"dropped {rec['dropped'] - dropped_start:3d}")
                    t_win, win_frames = now, 0
                    copy_ms.clear()
                    if shutil.disk_usage(run_dir).free / 1e9 < MIN_FREE_GB:
                        print("[capture] disk nearly full — stopping")
                        break
        dropped = 0 if replay else zed.get_frame_dropped_count() - dropped_start
    finally:
        if cam.is_open:
            cam.cam.disable_recording()
        cam.close()
        if writer:
            print("[capture] finishing queued dumps…")
            writer.close()

    size_mb = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file()) / 1e6
    print(f"[capture] {frames} frames ({dropped} dropped while recording), "
          f"{writer.written} dumps written, {writer.skipped} skipped, {size_mb:.0f} MB -> {run_dir}")


# ── Standalone ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--mode", default=DEFAULT_MODE, choices=MODES)
    ap.add_argument("--resolution", default=DEFAULT_RES)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--svo", help="extract from this recording instead of the live camera")
    ap.add_argument("--dump-every", type=int, default=None,
                    help="full per-frame dump every N frames, 0 = none "
                         "(default: 0 live, so nothing is dropped; 1 when extracting from --svo)")
    ap.add_argument("--pointcloud", action="store_true", help="also dump xyz.npy (11 MB/frame)")
    ap.add_argument("--no-svo", action="store_true", help="skip the SVO2 recording")
    ap.add_argument("--no-tracking", action="store_true", help="skip positional tracking (pose)")
    ap.add_argument("--out", default=str(CAPTURE_DIR))
    args = ap.parse_args()

    try:
        run_capture(args)
    except KeyboardInterrupt:
        print("\n[capture] interrupted")             # finally: recording + camera closed
