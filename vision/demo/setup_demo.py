#!/usr/bin/env python3
"""
setup_demo.py — one command to get the vision demo running on a new machine.

    1. creates vision/demo/.venv
    2. installs torch (CPU build on Linux/Windows, MPS-capable on Mac) and
       requirements-demo.txt
    3. on a Jetson with the ZED SDK, makes pyzed importable in the venv
    4. on a Jetson (JetPack 6, Python 3.10), swaps onnxruntime for NVIDIA's GPU
       build, so tracking (TensorRT), faces and expression run on the GPU;
       if that fails the CPU build goes back and the demo runs as before
    5. downloads every model now, so the demo needs no internet on the day:
       yolo26n.pt, InsightFace buffalo_l, HSEmotion, MediaPipe face detector;
       on the Jetson also builds the TensorRT engine for tracking (~8 min, once)
    6. runs the demo headless for a few seconds and prints a go/no-go table

Usage (from the repo root; needs Python 3.9-3.12, MediaPipe has no 3.13 wheels):
    python3 vision/demo/setup_demo.py              # full setup + check
    python3 vision/demo/setup_demo.py --check-only # just the check (webcam/ZED in front of a person)
    python3 vision/demo/setup_demo.py --check-only --source <video>
"""

import argparse
import os
import platform
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path

DEMO = Path(__file__).resolve().parent
VISION = DEMO.parent
VENV = DEMO / ".venv"
MODELS = DEMO / "models"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
# NVIDIA's onnxruntime build for JetPack 6 / CUDA 12.6 (CUDA + TensorRT), Python 3.10 only
JETSON_ORT = "onnxruntime-gpu==1.23.0"
JETSON_INDEX = "https://pypi.jetson-ai-lab.io/jp6/cu126"
FACE_DETECTOR_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                     "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
JETSON = platform.machine() == "aarch64" and Path("/etc/nv_tegra_release").exists()


# a sourced ROS workspace's PYTHONPATH would leak into the venv; keep it out
ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def run(*cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, env=ENV, **kw)


def step(msg):
    print(f"\n== {msg}")


def make_venv():
    step(f"virtual env {VENV}")
    if not PY.exists():
        venv.create(VENV, with_pip=True)
    run(PY, "-m", "pip", "install", "-q", "--upgrade", "pip", "wheel", "setuptools")


def install():
    step("torch")
    if sys.platform == "darwin":
        run(PY, "-m", "pip", "install", "-q", "torch", "torchvision")
    else:   # the CPU build: PyPI's Linux aarch64 build is for servers and won't use the Jetson GPU either
        run(PY, "-m", "pip", "install", "-q", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cpu")
    step("requirements-demo.txt (insightface builds from source: needs a C/C++ compiler)")
    run(PY, "-m", "pip", "install", "-q", "-r", DEMO / "requirements-demo.txt")
    # ultralytics pulls opencv-python and mediapipe pulls opencv-contrib-python; both
    # provide cv2 and the second install can leave it broken. Keep only contrib.
    run(PY, "-m", "pip", "uninstall", "-q", "-y", "opencv-python")
    run(PY, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", "opencv-contrib-python<4.12")


def jetson_gpu():
    """Replace onnxruntime with the Jetson GPU build. Any failure puts the CPU build back."""
    release = Path("/etc/nv_tegra_release").read_text()
    if not release.startswith("# R36") or sys.version_info[:2] != (3, 10):
        print("  GPU onnxruntime needs JetPack 6 + Python 3.10: staying on the CPU build")
        return
    step(f"{JETSON_ORT} (GPU) in place of onnxruntime")
    # both builds install the same `onnxruntime` package: never leave the two together
    run(PY, "-m", "pip", "uninstall", "-q", "-y", "onnxruntime", "onnxruntime-gpu")
    try:
        run(PY, "-m", "pip", "install", "-q", JETSON_ORT, "numpy<2", "protobuf<5", "--extra-index-url", JETSON_INDEX)
        run(PY, "-c", "import onnxruntime as o; p = o.get_available_providers(); print('  providers', p); "
                      "assert 'CUDAExecutionProvider' in p")
    except subprocess.CalledProcessError:
        print("  GPU build failed: putting the CPU onnxruntime back (the demo runs on the CPU)")
        subprocess.run([str(PY), "-m", "pip", "uninstall", "-q", "-y", "onnxruntime-gpu"], env=ENV)
        run(PY, "-m", "pip", "install", "-q", "onnxruntime")


def link_pyzed():
    """The ZED SDK installs pyzed for the system python; expose only that package to the venv."""
    out = subprocess.run(["python3", "-c", "import pyzed, os; print(os.path.dirname(pyzed.__file__))"],
                         capture_output=True, text=True, env=ENV)
    if out.returncode != 0:
        print("  no pyzed on this machine: depth distance will be unavailable")
        return
    step("pyzed (ZED SDK) -> venv")
    link_dir = VENV / "zed_sdk"
    link_dir.mkdir(exist_ok=True)
    link = link_dir / "pyzed"
    if not link.exists():
        link.symlink_to(out.stdout.strip())
    site = subprocess.run([str(PY), "-c", "import site; print(site.getsitepackages()[0])"],
                          capture_output=True, text=True, check=True, env=ENV).stdout.strip()
    (Path(site) / "zed_sdk.pth").write_text(str(link_dir) + "\n")
    run(PY, "-c", "import pyzed.sl as sl; print('  pyzed', sl.Camera().get_sdk_version())")


def download_models():
    step("models (cached so the demo works offline)")
    MODELS.mkdir(exist_ok=True)
    tflite = VISION / "face_detector.tflite"
    if not tflite.exists():
        print(f"  downloading {tflite.name}")
        tmp = tflite.with_suffix(".part")
        urllib.request.urlretrieve(FACE_DETECTOR_URL, tmp)
        os.replace(tmp, tflite)
    code = f"""
import os, sys
sys.path.insert(0, {str(VISION)!r})
os.chdir({str(MODELS)!r})                      # ultralytics downloads into the cwd
from ultralytics import YOLO
YOLO("yolo26n.pt")
print("  yolo26n.pt ok")
if {JETSON!r} and not os.path.exists("yolo26n.onnx"):   # faster than torch on the Jetson CPU
    YOLO("yolo26n.pt").export(format="onnx", imgsz=320, verbose=False)
    print("  yolo26n.onnx exported")
if {JETSON!r} and not os.path.exists("yolo26n_320_e2e.onnx"):   # GPU path: NMS-free (1, 300, 6) output
    out = YOLO("yolo26n.pt").export(format="onnx", imgsz=320, end2end=True, verbose=False)
    os.replace(out, "yolo26n_320_e2e.onnx")
    YOLO("yolo26n.pt").export(format="onnx", imgsz=320, verbose=False)   # the export reused yolo26n.onnx's name
    print("  yolo26n_320_e2e.onnx exported")
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
print("  HSEmotion ok")
import face_memory                              # downloads buffalo_l to ~/.insightface
print("  InsightFace buffalo_l ok")
"""
    run(PY, "-c", code)
    if JETSON:
        build_trt()


def build_trt():
    """Build the TensorRT engine for tracking once; until then tracking uses CUDA, then the CPU."""
    code = f"""
import sys, time, numpy as np
sys.path[:0] = [{str(DEMO)!r}, {str(DEMO / 'modules')!r}]
import tracking as t
if t.trt_engine_built():
    print("  TensorRT engine already built"); sys.exit()
try:
    det = t.GpuDetector(allow_trt_build=True)
except Exception as e:
    print(f"  no GPU tracking ({{e}}): the demo uses the CPU"); sys.exit()
print("  building the TensorRT engine for tracking (~8 min, only once)...", flush=True)
t0 = time.time()
det(np.zeros((320, 320, 3), np.uint8))
print(f"  {{det.name}} ready in {{time.time() - t0:.0f}}s")
"""
    step("TensorRT engine for tracking")
    if subprocess.run([str(PY), "-c", code], env=ENV).returncode != 0:
        print("  engine build failed: tracking uses CUDA / the CPU instead")


def check(source, seconds):
    step(f"check: every module on {source} for {seconds}s")
    return subprocess.run([str(PY), str(DEMO / "run_demo.py"), "--source", source,
                           "--check", str(seconds)], env=ENV).returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--source", default="auto")
    ap.add_argument("--seconds", type=int, default=30)
    args = ap.parse_args()
    if not (3, 9) <= sys.version_info[:2] <= (3, 12):
        sys.exit(f"Python {platform.python_version()} found; MediaPipe needs 3.9-3.12 "
                 "(e.g. `python3.12 vision/demo/setup_demo.py`).")
    if not args.check_only:
        make_venv()
        install()
        if JETSON:
            jetson_gpu()
            link_pyzed()
        download_models()
    rc = check(args.source, args.seconds)
    print("\nGO: run the demo with" if rc == 0 else "\nNOT READY: see the table above. Run with",
          f"\n  {PY} {DEMO / 'run_demo.py'}")
    sys.exit(rc)
