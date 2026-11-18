"""
camera.py — the one place that opens the camera for the demo.

A FrameSource reads frames on a background thread and keeps only the newest
one, so every module sees a fresh frame and nothing queues up. Sources:
    zed        ZED through the ZED SDK (pyzed): left image + depth in metres
    webcam:N   any OpenCV camera; a ZED without the SDK shows up here as one
               side-by-side stereo image, which is cropped to the left lens
    <file>     a video file, looped (backup if the camera fails on the day)
    auto       zed if pyzed and a ZED are present, otherwise webcam:0

Usage (test mode, shows the raw feed and FPS):
    python vision/demo/camera.py --source auto
"""

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

VISION_DIR = Path(__file__).resolve().parents[1]
RECONNECT_S = 2.0          # wait between attempts when the camera drops


@dataclass
class Frame:
    seq: int                         # increases by one per new frame
    t: float                         # time.time() when it was read
    image: np.ndarray                # HxWx3 BGR, left lens only
    depth: Optional[np.ndarray]      # HxW float32 metres (NaN = unknown), ZED SDK only


def zed_available() -> bool:
    try:
        import pyzed.sl as sl
        return len(sl.Camera.get_device_list()) > 0
    except Exception:
        return False


class FrameSource:
    def __init__(self, spec: str = "auto", width: int = 1280, height: int = 720, rotate: int = 0):
        if spec == "auto":
            spec = "zed" if zed_available() else "webcam:0"
        self.spec = spec
        self.kind = ("zed" if spec == "zed" else
                     "webcam" if spec.startswith("webcam") else "video")
        self.width, self.height = width, height
        # camera mounted sideways / upside down (GLaDOS hangs from the ceiling)
        self.rotate = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                       270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotate]
        self.status = "starting"
        self.fps = 0.0
        self._frame: Optional[Frame] = None
        self._lock = threading.Lock()
        self._fresh = threading.Condition(self._lock)
        self._running = False
        self._thread = None

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def has_depth(self) -> bool:
        return self.kind == "zed"

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._frame

    def wait_newer(self, seq: int, timeout: float = 1.0) -> Optional[Frame]:
        """Block until a frame with a higher seq than `seq` exists (or timeout)."""
        with self._fresh:
            self._fresh.wait_for(lambda: self._frame is not None and self._frame.seq > seq,
                                 timeout=timeout)
            return self._frame

    # ── Reader thread ─────────────────────────────────────────────────────────

    def _publish(self, image, depth=None):
        if self.rotate is not None:               # depth turns with the image so pixels still line up
            image = cv2.rotate(image, self.rotate)
            depth = cv2.rotate(depth, self.rotate) if depth is not None else None
        now = time.time()
        with self._fresh:
            prev = self._frame
            if prev is not None:
                dt = max(now - prev.t, 1e-6)
                self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1 / dt
            self._frame = Frame((prev.seq + 1) if prev else 1, now, image, depth)
            self._fresh.notify_all()

    def _run(self):
        # Outer loop = reconnect: any failure closes the camera, waits, reopens.
        while self._running:
            try:
                if self.kind == "zed":
                    self._run_zed()
                else:
                    self._run_opencv()
            except Exception as e:
                self.status = f"error: {e}"
                print(f"[camera] {e}", file=sys.stderr)
            if self._running:
                time.sleep(RECONNECT_S)

    def _run_zed(self):
        sys.path.insert(0, str(VISION_DIR / "depth"))
        from zed_depth import ZedDepth
        res = "HD720" if self.height >= 720 else "VGA"
        # 180 is done by the SDK on the GPU (free); 90/270 still rotate below
        flip = self.rotate == cv2.ROTATE_180
        zed = ZedDepth(mode=os.environ.get("ZED_DEPTH_MODE", "NEURAL_LIGHT"), resolution=res, flip=flip)
        rotate, self.rotate = self.rotate, (None if flip else self.rotate)
        self.status = "opening ZED (first neural-depth start can take a minute)"
        zed.open()
        self.status = f"ZED {zed.width}x{zed.height} {zed.mode}"
        try:
            while self._running:
                if not zed.grab():
                    raise RuntimeError("ZED grab failed")
                self._publish(zed.left_image(), zed.depth_map().copy())
        finally:
            self.rotate = rotate
            zed.close()

    def _run_opencv(self):
        if self.kind == "webcam":
            index = int(self.spec.split(":")[1]) if ":" in self.spec else 0
            cap = cv2.VideoCapture(index)
            # A ZED without the SDK gives both lenses in one image; asking for
            # 2x width gets 1280x720 per lens. Normal webcams pick their nearest mode.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width * 2)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self.spec)
        if not cap.isOpened():
            raise RuntimeError(f"could not open {self.spec}")
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.status = f"{self.spec} open"
        try:
            while self._running:
                t0 = time.time()
                ok, image = cap.read()
                if not ok:
                    if self.kind == "video":            # loop the backup clip
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    raise RuntimeError(f"{self.spec} stopped giving frames")
                h, w = image.shape[:2]
                if w > 3 * h:                           # side-by-side stereo: left lens
                    image = image[:, :w // 2].copy()
                    h, w = image.shape[:2]
                self.status = f"{self.spec} {w}x{h}"
                self._publish(image)
                if self.kind == "video":                # play at real speed
                    time.sleep(max(0.0, 1 / video_fps - (time.time() - t0)))
        finally:
            cap.release()


# ── Standalone ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="auto")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    args = ap.parse_args()
    src = FrameSource(args.source, rotate=args.rotate).start()
    print("Press Q to quit")
    seq = 0
    try:
        while True:
            f = src.wait_newer(seq)
            if f is None or f.seq == seq:
                print(f"[camera] {src.status}")
                continue
            seq = f.seq
            img = f.image.copy()
            cv2.putText(img, f"{src.status}  {src.fps:.1f} fps", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("camera test", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        src.stop()
        cv2.destroyAllWindows()
