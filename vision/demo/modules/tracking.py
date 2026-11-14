"""
tracking.py — person detection + tracking (YOLO26n + ByteTrack).

Same settings as vision/object_tracker.py's find_people(tracking=True):
person class only, NMS-free head, ByteTrack keeps IDs across frames. That
script picks its model from sys.argv at import, so the call is repeated here
instead of importing it.

Result: [{"id": 3, "box": (x1, y1, x2, y2), "conf": 0.87}, ...]
"""

import os
import platform

import numpy as np

from workers import Worker, onnx_threads

JETSON = platform.machine() == "aarch64" and os.path.exists("/etc/nv_tegra_release")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
# On the Jetson the ONNX export (setup_demo.py makes it) runs ~1.5x faster than torch's CPU path.
MODEL = os.environ.get("DEMO_YOLO", "yolo26n.onnx" if JETSON else "yolo26n.pt")
PERSON = 0
# smaller input on the Jetson CPU: ~4x fewer pixels, still finds people at room distance
IMGSZ = int(os.environ.get("DEMO_YOLO_IMGSZ", "320" if JETSON else "640"))


class TrackerWorker(Worker):
    key, label, hotkey = "track", "Person tracking", "1"

    def load(self):
        import torch
        from ultralytics import YOLO
        if JETSON:
            torch.set_num_threads(2)          # leave cores for MediaPipe / InsightFace
        self.device = ("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
        name = MODEL
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path) and name.endswith(".onnx"):   # not exported: use torch
            name = name.replace(".onnx", ".pt")
            path = os.path.join(MODELS_DIR, name)
        self.model = YOLO(path if os.path.exists(path) else name, task="detect")
        # the ONNX session is built on the first call, so warm up inside the thread cap
        with onnx_threads(2 if JETSON else 0):
            self.model.predict(np.zeros((IMGSZ, IMGSZ, 3), np.uint8), imgsz=IMGSZ,
                               device=self.device, verbose=False)
        self.message = f"{name} {IMGSZ}px {self.device}"

    def process(self, frame):
        r = self.model.track(frame.image, persist=True, tracker="bytetrack.yaml",
                             classes=[PERSON], nms=False, iou=0.7, imgsz=IMGSZ,
                             device=self.device, verbose=False)[0]
        if r.boxes.id is None:                  # nothing tracked yet
            return []
        return [{"id": int(i), "box": tuple(int(v) for v in b), "conf": float(c)}
                for i, b, c in zip(r.boxes.id.tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
