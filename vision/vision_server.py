#!/usr/bin/env python3
"""
vision_server.py — Perception server for the X1 Jetson.

Runs on the Jetson. Fuses:
  - Gesture / arm detection  (gestures_revised.py)
  - Face count               (face_memory.py)
  - Depth to nearest person  (ZED SDK — optional, gracefully absent)

Serves a single HTTP endpoint:
  GET /perception  →  JSON snapshot of the latest frame

Usage (from X1_GLaDOS/ root on the Jetson):
    source vision/.venv/bin/activate   # or whichever venv has mediapipe + cv2
    python3 vision/vision_server.py

By default listens on 0.0.0.0:5555.
Set env vars to override:
    VISION_HOST=0.0.0.0 VISION_PORT=5555 python3 vision/vision_server.py

The Mac-side client (vision/vision_client.py) polls this endpoint at ~5 Hz
and injects the result into the GLaDOS brain's context.
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
HOST = os.environ.get("VISION_HOST", "0.0.0.0")
PORT = int(os.environ.get("VISION_PORT", "5555"))

# Camera index; ZED Mini is usually 0 on the Jetson. Override with VISION_CAM.
CAM_INDEX = int(os.environ.get("VISION_CAM", "0"))

# ── Optional ZED import ───────────────────────────────────────────────────────
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    print("[vision_server] pyzed not found — depth disabled")

# ── MediaPipe / gesture imports ───────────────────────────────────────────────
try:
    import cv2
    import mediapipe as mp
    from vision.gestures_revised import (
        detect_arm_gestures,
        update_wrist_history,
        clear_wrist_history,
        LatestFrame,
        hands,
        hands_pool,
        pose,
    )
    GESTURE_AVAILABLE = True
except Exception as e:
    GESTURE_AVAILABLE = False
    print(f"[vision_server] gesture detection unavailable: {e}")

try:
    from vision.face_memory import FaceMemory
    FACE_AVAILABLE = True
    face_memory = FaceMemory()
except Exception as e:
    FACE_AVAILABLE = False
    print(f"[vision_server] face memory unavailable: {e}")

# ── Shared perception state ───────────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "gestures":        [],      # e.g. ["right: waving", "left: thumbs up"]
    "face_count":      0,       # number of recognised faces in view
    "known_faces":     [],      # names of recognised people (if any)
    "nearest_person_m": None,   # float metres, or null if no depth
    "fps":             0.0,
    "ts":              0.0,     # unix timestamp of last update
}


def set_state(**kwargs):
    with _lock:
        _state.update(kwargs)
        _state["ts"] = time.time()


def get_state() -> dict:
    with _lock:
        return dict(_state)


# ── ZED depth helper ──────────────────────────────────────────────────────────

class ZEDDepth:
    """Thin wrapper — opens the ZED Mini and grabs depth to the nearest object."""

    def __init__(self):
        self.zed  = sl.Camera()
        init      = sl.InitParameters()
        init.depth_mode           = sl.DEPTH_MODE.PERFORMANCE
        init.coordinate_units     = sl.UNIT.METER
        init.camera_resolution    = sl.RESOLUTION.HD720
        init.camera_fps           = 30
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {err}")
        self.depth_map = sl.Mat()
        self.runtime   = sl.RuntimeParameters()

    def nearest_person_m(self) -> float | None:
        """Return depth (m) of the closest non-zero pixel in the centre crop."""
        if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
        arr = self.depth_map.get_data()
        if arr is None:
            return None
        h, w = arr.shape[:2]
        # sample the centre quarter of the frame
        crop = arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        valid = crop[(crop > 0.1) & (crop < 20)]   # 10 cm – 20 m
        if valid.size == 0:
            return None
        return float(valid.min())

    def close(self):
        self.zed.close()


# ── Perception loop ───────────────────────────────────────────────────────────

def perception_loop():
    """Background thread: grab frames, run models, update shared state."""
    zed = None
    if ZED_AVAILABLE:
        try:
            zed = ZEDDepth()
            print("[vision_server] ZED depth active")
        except Exception as e:
            print(f"[vision_server] ZED init failed: {e} — depth disabled")

    cap = None
    frame_obj = None
    if GESTURE_AVAILABLE:
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(CAM_INDEX)
        _cap.set(_cv2.CAP_PROP_FRAME_WIDTH, 2560)
        _cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame_obj = LatestFrame(_cap)
        print("[vision_server] camera open")

    frame_times = []
    MAX_FPS_WINDOW = 30

    while True:
        t0 = time.time()
        gestures   = []
        face_count = 0
        known      = []
        depth      = None

        # ── Gesture + face ──────────────────────────────────────────────────
        if GESTURE_AVAILABLE and frame_obj is not None:
            ok, frame = frame_obj.read()
            if ok and frame is not None:
                left = frame[:, : frame.shape[1] // 2]   # left ZED lens
                rgb  = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

                hands_fut    = hands_pool.submit(hands.process, rgb)
                pose_results = pose.process(rgb)
                hand_results = hands_fut.result()

                if pose_results.pose_landmarks:
                    lm = pose_results.pose_landmarks.landmark
                    update_wrist_history(lm)
                    gestures = detect_arm_gestures(lm, hand_results)
                else:
                    clear_wrist_history()

                if FACE_AVAILABLE:
                    names = face_memory.identify(left)
                    known = [n for n in names if n != "unknown"]
                    face_count = len(names)

        # ── Depth ───────────────────────────────────────────────────────────
        if zed is not None:
            depth = zed.nearest_person_m()

        # ── FPS ─────────────────────────────────────────────────────────────
        frame_times.append(time.time())
        if len(frame_times) > MAX_FPS_WINDOW:
            frame_times.pop(0)
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0

        set_state(
            gestures=gestures,
            face_count=face_count,
            known_faces=known,
            nearest_person_m=depth,
            fps=round(fps, 1),
        )

        # target ~15 Hz; subtract processing time already elapsed
        elapsed = time.time() - t0
        time.sleep(max(0.0, 1 / 15 - elapsed))

    # cleanup (unreachable in normal operation)
    if frame_obj:
        frame_obj.release()
    if zed:
        zed.close()


# ── HTTP server ───────────────────────────────────────────────────────────────

class PerceptionHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/perception", "/perception/"):
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps(get_state()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass   # silence per-request logs


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"[vision_server] starting perception loop...")
    t = threading.Thread(target=perception_loop, daemon=True)
    t.start()
    time.sleep(0.5)   # let the camera open

    server = HTTPServer((HOST, PORT), PerceptionHandler)
    print(f"[vision_server] serving on http://{HOST}:{PORT}/perception")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[vision_server] shutting down")


if __name__ == "__main__":
    main()
