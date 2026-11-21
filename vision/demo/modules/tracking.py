"""
tracking.py — person detection + tracking (YOLO26n + ByteTrack).

Same settings as vision/object_tracker.py's find_people(tracking=True):
person class only, NMS-free head, ByteTrack keeps IDs across frames. That
script picks its model from sys.argv at import, so the call is repeated here
instead of importing it.

Two backends, picked at load time (the panel shows which one is running):
  gpu  YOLO26n NMS-free ONNX on onnxruntime-gpu (Jetson): TensorRT FP16 with a
       cached engine (~2.4 ms/frame), or CUDA if the engine isn't built yet.
       ByteTrack is ultralytics' own tracker with its bytetrack.yaml settings.
  cpu  ultralytics YOLO(...).track() on ONNX (Jetson) or torch (laptop / Mac MPS).
       Used whenever the GPU path is missing or fails to load.

setup_demo.py exports models/yolo26n_320_e2e.onnx and builds the TensorRT
engine (models/trt_cache/, ~8 min once). Without that cache the GPU path uses
CUDA, so a missing engine never makes the demo sit for minutes at startup.

Env: DEMO_TRACK_BACKEND=auto|gpu|cpu, DEMO_YOLO, DEMO_YOLO_IMGSZ.

Result: [{"id": 3, "box": (x1, y1, x2, y2), "conf": 0.87}, ...]

Usage (standalone check: time both backends on a video, compare their boxes):
    vision/demo/.venv/bin/python vision/demo/modules/tracking.py <video> [--frames 150]
"""

import glob
import os
import platform
import time

import cv2
import numpy as np

from workers import Worker, onnx_threads

JETSON = platform.machine() == "aarch64" and os.path.exists("/etc/nv_tegra_release")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
# On the Jetson the ONNX export (setup_demo.py makes it) runs ~1.5x faster than torch's CPU path.
MODEL = os.environ.get("DEMO_YOLO", "yolo26n.onnx" if JETSON else "yolo26n.pt")
PERSON = 0
# smaller input on the Jetson CPU: ~4x fewer pixels, still finds people at room distance
IMGSZ = int(os.environ.get("DEMO_YOLO_IMGSZ", "320" if JETSON else "640"))
BACKEND = os.environ.get("DEMO_TRACK_BACKEND", "auto")

GPU_ONNX = os.path.join(MODELS_DIR, f"yolo26n_{IMGSZ}_e2e.onnx")   # NMS-free head: (1, 300, 6) out
TRT_CACHE = os.path.join(MODELS_DIR, "trt_cache")
TRACK_CONF = 0.1          # ByteTrack uses low-score boxes too (track()'s default conf)


# ── GPU detector ───────────────────────────────────────────────────────────────

def trt_engine_built() -> bool:
    return bool(glob.glob(os.path.join(TRT_CACHE, "*.engine")))


def trt_provider_options() -> dict:
    os.makedirs(TRT_CACHE, exist_ok=True)
    return {"trt_fp16_enable": True,
            "trt_engine_cache_enable": True, "trt_engine_cache_path": TRT_CACHE,
            "trt_timing_cache_enable": True, "trt_timing_cache_path": TRT_CACHE,
            "trt_max_workspace_size": 1 << 30}


class GpuDetector:
    """YOLO26n NMS-free ONNX on onnxruntime-gpu. Call with a BGR frame -> (N, 6) person boxes."""

    def __init__(self, onnx_path=GPU_ONNX, imgsz=IMGSZ, allow_trt_build=False):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError("onnxruntime has no GPU support here")
        if not os.path.exists(onnx_path):
            raise RuntimeError(f"{os.path.basename(onnx_path)} not exported (run setup_demo.py)")
        providers = []
        # building the engine takes ~8 min: only do it when asked (setup), else use CUDA
        if "TensorrtExecutionProvider" in avail and (trt_engine_built() or allow_trt_build):
            providers.append(("TensorrtExecutionProvider", trt_provider_options()))
        providers += ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.name = {"TensorrtExecutionProvider": "TensorRT fp16", "CUDAExecutionProvider": "CUDA"}.get(
            self.sess.get_providers()[0], self.sess.get_providers()[0])
        self.input = self.sess.get_inputs()[0].name
        self.imgsz = imgsz

    def __call__(self, bgr, conf=TRACK_CONF):
        # letterbox exactly like ultralytics' LetterBox (centered, grey 114 padding)
        h, w = bgr.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = round(w * r), round(h * r)
        dw, dh = (self.imgsz - nw) / 2, (self.imgsz - nh) / 2
        top, left = round(dh - 0.1), round(dw - 0.1)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0

        out = self.sess.run(None, {self.input: x})[0][0]          # (300, 6): x1 y1 x2 y2 score cls
        d = out[(out[:, 4] >= conf) & (out[:, 5] == PERSON)].copy()
        d[:, [0, 2]] = ((d[:, [0, 2]] - left) / r).clip(0, w)
        d[:, [1, 3]] = ((d[:, [1, 3]] - top) / r).clip(0, h)
        return d


def make_bytetrack():
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    return BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml"))))


# ── Worker ─────────────────────────────────────────────────────────────────────

class TrackerWorker(Worker):
    key, label, hotkey = "track", "Person tracking", "1"
    gpu = None                                # GpuDetector when the GPU backend loaded

    def load(self):
        if BACKEND in ("auto", "gpu"):
            try:
                self._load_gpu()
                return
            except Exception as e:
                if BACKEND == "gpu":
                    raise
                print(f"[track] GPU backend unavailable ({e}); using CPU")
        self._load_cpu()

    def _load_gpu(self):
        from ultralytics.engine.results import Boxes
        self.Boxes = Boxes
        self.gpu = GpuDetector()
        self.gpu(np.zeros((IMGSZ, IMGSZ, 3), np.uint8))           # warm up (loads the engine)
        self.tracker = make_bytetrack()
        note = "" if self.gpu.name.startswith("TensorRT") else " (TensorRT engine not built: setup_demo.py)"
        self.message = f"yolo26n {IMGSZ}px GPU {self.gpu.name}{note}"

    def _load_cpu(self):
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
        if self.gpu is not None:
            d = self.gpu(frame.image)
            t = self.tracker.update(self.Boxes(d, frame.image.shape[:2]), frame.image)
            # rows: x1 y1 x2 y2 track_id score cls idx
            return [{"id": int(r[4]), "box": tuple(int(v) for v in r[:4]), "conf": float(r[5])} for r in t]
        r = self.model.track(frame.image, persist=True, tracker="bytetrack.yaml",
                             classes=[PERSON], nms=False, iou=0.7, imgsz=IMGSZ,
                             device=self.device, verbose=False)[0]
        if r.boxes.id is None:                  # nothing tracked yet
            return []
        return [{"id": int(i), "box": tuple(int(v) for v in b), "conf": float(c)}
                for i, b, c in zip(r.boxes.id.tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]


# ── Standalone ─────────────────────────────────────────────────────────────────

def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-6)


if __name__ == "__main__":
    import argparse
    import sys
    import types
    ap = argparse.ArgumentParser(description="time GPU vs CPU tracking on a video and compare boxes")
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--build-trt", action="store_true", help="build the TensorRT engine if missing (~8 min)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    if not frames:
        sys.exit(f"could not read {args.video}")

    if args.build_trt:
        t = time.time()
        GpuDetector(allow_trt_build=True)(frames[0])
        print(f"[track] TensorRT engine ready in {time.time() - t:.0f}s ({TRT_CACHE})")

    runs = {}
    for name in ("gpu", "cpu"):
        w = TrackerWorker(source=None, hub={})
        try:
            (w._load_gpu if name == "gpu" else w._load_cpu)()
        except Exception as e:
            print(f"[{name}] unavailable: {e}")
            continue
        out, t0 = [], time.perf_counter()
        for f in frames:
            out.append(w.process(types.SimpleNamespace(image=f)))
        ms = 1000 * (time.perf_counter() - t0) / len(frames)
        runs[name] = out
        print(f"[{name}] {w.message}: {ms:.1f} ms/frame, "
              f"{sum(len(o) for o in out) / len(out):.2f} people/frame")

    if len(runs) == 2:
        ious, missing = [], 0
        for g, c in zip(runs["gpu"], runs["cpu"]):
            for cb in c:
                best = max((_iou(cb["box"], gb["box"]) for gb in g), default=0)
                ious.append(best)
                missing += best < 0.5
        print(f"[compare] {len(ious)} CPU boxes: mean best IoU with GPU {np.mean(ious) if ious else 0:.3f}, "
              f"{missing} without a GPU match (IoU<0.5)")
