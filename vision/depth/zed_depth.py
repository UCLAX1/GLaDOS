#!/usr/bin/env python3
"""
zed_depth.py — ZED depth grab loop + depth-mode benchmark.

Importable module. Wraps a ZED camera (live or an SVO recording) and answers
"how far away is this pixel?" and "where is this pixel in 3D?".

Coordinates are metres in RIGHT_HANDED_Z_UP, relative to the left lens:
x right, y forward (out of the lens), z up — the same z-up convention as the
MuJoCo / control frame. depth_at() returns the distance along the lens axis
(i.e. the y of point_at()), not the straight-line range.

Usage:
    from vision.depth.zed_depth import ZedDepth

    with ZedDepth(mode="NEURAL_LIGHT") as cam:
        while cam.grab():
            d = cam.depth_at(640, 360)     # metres, or None where there is no depth
            p = cam.point_at(640, 360)     # (x, y, z) metres, or None

Run standalone to test (no windows; snapshots go to --out):
    python3 vision/depth/zed_depth.py                       # live: FPS + centre depth, 10 s
    python3 vision/depth/zed_depth.py --mode NEURAL --seconds 30
    python3 vision/depth/zed_depth.py --svo clip.svo2       # replay a recording
    python3 vision/depth/zed_depth.py --benchmark           # every depth mode -> table + JSON

Only one process can open the ZED at a time — close ZED_Explorer / ZED_Depth_Viewer first.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyzed.sl as sl

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_MODE  = "NEURAL_LIGHT"
DEFAULT_RES   = "HD720"
DEFAULT_FPS   = 30
MAX_DEPTH_M   = 10.0                # ZED Mini depth is unreliable well before this
PATCH_RADIUS  = 2                   # depth_at() takes the median of a 5x5 patch

# Legacy (classic stereo) modes first, then the neural ones, cheapest to dearest.
MODES = ["PERFORMANCE", "QUALITY", "ULTRA", "NEURAL_LIGHT", "NEURAL", "NEURAL_PLUS"]

OUT_DIR = Path(__file__).resolve().parent / "out"


class ZedDepth:
    """
    One ZED camera (or SVO file) with depth enabled.

    Parameters
    ----------
    mode : str
        sl.DEPTH_MODE name, e.g. "NEURAL_LIGHT", "PERFORMANCE".
    resolution : str
        sl.RESOLUTION name. ZED Mini: "HD2K", "HD1080", "HD720", "VGA".
    fps : int
        Camera frame rate. ZED Mini HD720 supports 15/30/60.
    svo : str | Path | None
        Replay this .svo/.svo2 recording instead of the live camera.
    """

    def __init__(self, mode=DEFAULT_MODE, resolution=DEFAULT_RES, fps=DEFAULT_FPS,
                 svo=None, max_depth_m=MAX_DEPTH_M):
        self.mode        = mode
        self.resolution  = resolution
        self.fps         = fps
        self.svo         = svo
        self.max_depth_m = max_depth_m

        self.cam      = sl.Camera()
        self._runtime = sl.RuntimeParameters()
        self._depth   = sl.Mat()
        self._xyz     = sl.Mat()
        self._image   = sl.Mat()
        self._depth_fresh = False
        self._xyz_fresh   = False

        self.width = self.height = 0
        self.open_s        = 0.0     # how long open() took (neural modes are slow first time)
        self.last_grab_ms  = 0.0
        self.is_open       = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def open(self):
        init = sl.InitParameters()
        init.depth_mode             = getattr(sl.DEPTH_MODE, self.mode)
        init.camera_resolution      = getattr(sl.RESOLUTION, self.resolution)
        init.camera_fps             = self.fps
        init.coordinate_units       = sl.UNIT.METER
        init.coordinate_system      = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
        init.depth_maximum_distance = self.max_depth_m
        if self.svo:
            init.set_from_svo_file(str(self.svo))
            init.svo_real_time_mode = False      # every frame, as fast as we can

        t0 = time.time()
        err = self.cam.open(init)
        self.open_s = time.time() - t0
        if err > sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed ({self.mode}): {err}")

        res = self.cam.get_camera_information().camera_configuration.resolution
        self.width, self.height = res.width, res.height
        self.is_open = True
        return self

    def close(self):
        if self.is_open:
            self.cam.close()
            self.is_open = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ── Per-frame ──────────────────────────────────────────────────────────────

    def grab(self) -> bool:
        """Grab the next frame and compute its depth. False at end of SVO or on error."""
        t0 = time.perf_counter()
        err = self.cam.grab(self._runtime)
        self.last_grab_ms = (time.perf_counter() - t0) * 1000
        if err > sl.ERROR_CODE.SUCCESS:          # negative codes are warnings, still a frame
            if err != sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print(f"[zed] grab failed: {err}")
            return False
        self._depth_fresh = self._xyz_fresh = False
        return True

    def depth_map(self) -> np.ndarray:
        """HxW float32 metres for the current frame. NaN/inf where there is no depth."""
        if not self._depth_fresh:
            self.cam.retrieve_measure(self._depth, sl.MEASURE.DEPTH)
            self._depth_fresh = True
        return self._depth.get_data()

    def depth_at(self, u: int, v: int, radius: int = PATCH_RADIUS):
        """Median depth (m) of the patch around pixel (u, v), or None if it has none."""
        d = self.depth_map()
        patch = d[max(v - radius, 0):v + radius + 1, max(u - radius, 0):u + radius + 1]
        patch = patch[np.isfinite(patch)]
        return float(np.median(patch)) if patch.size else None

    def point_at(self, u: int, v: int):
        """(x, y, z) metres of pixel (u, v) in the camera frame, or None."""
        if not self._xyz_fresh:
            self.cam.retrieve_measure(self._xyz, sl.MEASURE.XYZ)
            self._xyz_fresh = True
        err, p = self._xyz.get_value(u, v)
        if err != sl.ERROR_CODE.SUCCESS or not np.all(np.isfinite(p[:3])):
            return None
        return float(p[0]), float(p[1]), float(p[2])

    def save_snapshot(self, prefix) -> list:
        """Write <prefix>_left.png and <prefix>_depth.png (SDK's grey depth render)."""
        written = []
        for view, tag in ((sl.VIEW.LEFT, "left"), (sl.VIEW.DEPTH, "depth")):
            self.cam.retrieve_image(self._image, view)
            path = f"{prefix}_{tag}.png"
            if self._image.write(path) == sl.ERROR_CODE.SUCCESS:
                written.append(path)
        return written


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_json_atomic(path: Path, data):
    """Temp file + rename, so a reader never sees a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


def _fmt(x, spec):
    return "—" if x is None else format(x, spec)


# ── Live test ──────────────────────────────────────────────────────────────────

def run_live(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with ZedDepth(args.mode, args.resolution, args.fps, args.svo) as cam:
        cu, cv = cam.width // 2, cam.height // 2
        print(f"[zed] opened {args.mode} {cam.width}x{cam.height}@{args.fps} "
              f"in {cam.open_s:.1f}s — centre pixel ({cu}, {cv})")

        t_start = t_win = time.time()
        frames = win_frames = 0
        grab_ms, retr_ms = [], []
        while time.time() - t_start < args.seconds:
            if not cam.grab():
                break
            frames += 1
            win_frames += 1
            grab_ms.append(cam.last_grab_ms)

            t0 = time.perf_counter()
            d = cam.depth_at(cu, cv)
            retr_ms.append((time.perf_counter() - t0) * 1000)

            now = time.time()
            if now - t_win >= 1.0:
                p = cam.point_at(cu, cv)
                p_str = "—" if p is None else f"({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})"
                print(f"  {win_frames / (now - t_win):5.1f} FPS | "
                      f"grab {np.mean(grab_ms):5.1f} ms  retrieve {np.mean(retr_ms):4.1f} ms | "
                      f"centre depth {_fmt(d, '.3f')} m  xyz {p_str}")
                t_win, win_frames = now, 0
                grab_ms.clear()
                retr_ms.clear()

        if frames:
            for path in cam.save_snapshot(out / f"live_{args.mode}"):
                print(f"[zed] wrote {path}")
    print(f"[zed] closed after {frames} frames")


# ── Benchmark ──────────────────────────────────────────────────────────────────

def bench_mode(mode, args, out: Path) -> dict:
    """Open one depth mode, warm up, measure, snapshot, close."""
    r = {"mode": mode}
    cam = ZedDepth(mode, args.resolution, args.fps, args.svo)
    try:
        cam.open()
    except RuntimeError as e:
        print(f"  {mode}: {e}")
        r["error"] = str(e)
        return r

    try:
        r["open_s"] = round(cam.open_s, 1)
        cu, cv = cam.width // 2, cam.height // 2

        for _ in range(args.warmup):             # let exposure + neural nets settle
            cam.grab()

        grab_ms, retr_ms, fill, centre = [], [], [], []
        frames = 0
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            if not cam.grab():
                break
            frames += 1
            grab_ms.append(cam.last_grab_ms)
            if frames % args.sample_every == 0:  # don't let analysis eat into the FPS
                t1 = time.perf_counter()
                d = cam.depth_map()
                retr_ms.append((time.perf_counter() - t1) * 1000)
                fill.append(float(np.isfinite(d).mean()))
                c = cam.depth_at(cu, cv)
                if c is not None:
                    centre.append(c)
        elapsed = time.time() - t0

        r.update({
            "fps":            round(frames / elapsed, 1) if elapsed else 0.0,
            "grab_ms_mean":   round(float(np.mean(grab_ms)), 1) if grab_ms else None,
            "grab_ms_p95":    round(float(np.percentile(grab_ms, 95)), 1) if grab_ms else None,
            "retrieve_ms":    round(float(np.mean(retr_ms)), 1) if retr_ms else None,
            "fill_pct":       round(100 * float(np.mean(fill)), 1) if fill else None,
            "centre_m":       round(float(np.median(centre)), 3) if centre else None,
            "centre_std_mm":  round(1000 * float(np.std(centre)), 1) if len(centre) > 1 else None,
            "snapshots":      cam.save_snapshot(out / f"bench_{mode}"),
        })
    finally:
        cam.close()

    print(f"  {mode:<12} open {r['open_s']:5.1f}s | {r['fps']:5.1f} FPS | "
          f"grab {_fmt(r['grab_ms_mean'], '.1f')} ms (p95 {_fmt(r['grab_ms_p95'], '.1f')}) | "
          f"fill {_fmt(r['fill_pct'], '.1f')}% | centre {_fmt(r['centre_m'], '.3f')} m "
          f"± {_fmt(r['centre_std_mm'], '.1f')} mm")
    return r


def run_benchmark(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modes = args.modes or MODES

    info = {
        "sdk":          sl.Camera.get_sdk_version(),
        "resolution":   args.resolution,
        "camera_fps":   args.fps,
        "seconds":      args.seconds,
        "svo":          str(args.svo) if args.svo else None,
        "loadavg_1m":   round(os.getloadavg()[0], 2),
        "mem_avail_gb": round(_mem_available_gb(), 2),
        "time":         time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"[bench] {len(modes)} modes, {args.seconds}s each at {args.resolution}@{args.fps} "
          f"(load {info['loadavg_1m']}, {info['mem_avail_gb']} GB free)")
    print("[bench] keep the scene still — centre ± is frame-to-frame jitter\n")

    results = []
    for mode in modes:
        results.append(bench_mode(mode, args, out))
        time.sleep(1.0)                          # give the USB device a moment between opens

    path = out / f"depth_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    _write_json_atomic(path, {"info": info, "results": results})

    print("\n| mode | open (s) | FPS | grab ms (mean / p95) | fill % | centre (m) | centre ± (mm) |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            print(f"| {r['mode']} | failed | — | — | — | — | — |")
            continue
        print(f"| {r['mode']} | {r['open_s']} | {r['fps']} | "
              f"{_fmt(r['grab_ms_mean'], '.1f')} / {_fmt(r['grab_ms_p95'], '.1f')} | "
              f"{_fmt(r['fill_pct'], '.1f')} | {_fmt(r['centre_m'], '.3f')} | "
              f"{_fmt(r['centre_std_mm'], '.1f')} |")
    print(f"\n[bench] wrote {path}")


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", default=DEFAULT_MODE, choices=MODES)
    ap.add_argument("--resolution", default=DEFAULT_RES)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--svo", help="replay a .svo/.svo2 file instead of the live camera")
    ap.add_argument("--seconds", type=float, default=10.0, help="run / measure time per mode")
    ap.add_argument("--out", default=str(OUT_DIR), help="where snapshots + results go")
    ap.add_argument("--benchmark", action="store_true", help="measure every depth mode")
    ap.add_argument("--modes", nargs="+", choices=MODES, help="benchmark only these modes")
    ap.add_argument("--warmup", type=int, default=30, help="frames to discard per mode")
    ap.add_argument("--sample-every", type=int, default=5,
                    help="analyse depth every Nth frame during the benchmark")
    args = ap.parse_args()

    try:
        run_benchmark(args) if args.benchmark else run_live(args)
    except KeyboardInterrupt:
        print("\n[zed] interrupted")             # context managers / finally close the camera
