"""
workers.py — run each vision module on its own thread against the shared camera.

A Worker loads its model on its thread (so the window opens at once), then
repeatedly takes the newest frame, processes it and stores the latest result.
Errors never escape: a module that can't load shows as "unavailable", and one
that throws mid-run shows "error" and keeps retrying. Every worker can be
turned on and off while the demo runs, which frees CPU for the others.

Subclasses implement:
    load()                    import + build models (raise to mark unavailable)
    process(frame) -> result  one frame in, any result out
    want(frame) -> bool       optional: skip frames (e.g. only when needed)
"""

import sys
import threading
import time
import traceback
from pathlib import Path

VISION_DIR = Path(__file__).resolve().parents[1]
if str(VISION_DIR) not in sys.path:            # so modules can import the team scripts
    sys.path.insert(0, str(VISION_DIR))

ERROR_RETRY_S = 1.0
JETSON = Path("/etc/nv_tegra_release").exists()


class Worker:
    key = "?"           # short name used on the command line and in the web UI
    label = "?"         # shown in the panel
    hotkey = "?"        # keyboard toggle
    max_hz = None       # cap the rate (None = as fast as the CPU allows)

    def __init__(self, source, hub: dict):
        self.source = source
        self.hub = hub                   # key -> Worker, to read other modules' results
        self.enabled = True
        self.state = "loading"           # loading | running | off | unavailable | error
        self.message = ""                # module notes (model, known faces, ...)
        self.error = ""                  # last error, shown while state == "error"
        self.result = None
        self.result_seq = 0              # frame seq the result came from
        self.result_t = 0.0
        self.ms = 0.0                    # smoothed processing time per call
        self.hz = 0.0                    # smoothed calls per second
        self.load_s = 0.0
        self.errors = 0
        self.paused_for = ""             # set by another module to borrow the CPU (e.g. face registration)
        self.paused_until = 0.0          # safety net: the pause lapses at this time even if never lifted
        self._running = False
        self._thread = None
        self._last_run = 0.0

    # ── Subclass hooks ────────────────────────────────────────────────────────

    def load(self):
        pass

    def process(self, frame):
        raise NotImplementedError

    def want(self, frame) -> bool:
        return True

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, name=self.key, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False

    def toggle(self):
        if self.state in ("unavailable", "loading"):
            return
        self.enabled = not self.enabled
        self.state = "running" if self.enabled else "off"
        if not self.enabled:
            self.result = None

    def fresh_result(self, max_age_s: float = 1.0):
        """The latest result, or None if it is too old to draw (module off/stalled)."""
        if self.result is None or time.time() - self.result_t > max_age_s:
            return None
        return self.result

    def status_line(self) -> str:
        if self.paused_for and self.state == "running":
            return f"paused while {self.paused_for}"
        if self.state == "running":
            return f"{self.hz:4.1f} Hz {self.ms:5.0f} ms  {self.message}"
        if self.state == "loading":
            return "loading..."
        detail = self.error if self.state == "error" else self.message
        return self.state + (f": {detail}" if detail else "")

    # ── Thread ────────────────────────────────────────────────────────────────

    def _run(self):
        t0 = time.time()
        try:
            self.load()
        except Exception as e:
            self.state, self.message = "unavailable", f"{type(e).__name__}: {e}"[:120]
            print(f"[{self.key}] unavailable: {e}", file=sys.stderr)
            return
        self.load_s = time.time() - t0
        self.state = "running" if self.enabled else "off"
        print(f"[{self.key}] loaded in {self.load_s:.1f}s")

        seq = 0
        while self._running:
            if self.paused_for and time.time() > self.paused_until:
                self.paused_for = ""             # whoever paused us never resumed: resume anyway
            if not self.enabled or self.paused_for:
                self.hz = 0.0
                time.sleep(0.1)
                continue
            frame = self.source.wait_newer(seq, timeout=0.5)
            if frame is None or frame.seq == seq:
                continue
            if self.max_hz:
                wait = self._last_run + 1 / self.max_hz - time.time()
                if wait > 0:
                    time.sleep(wait)
                    frame = self.source.latest()
            if not self.want(frame):
                time.sleep(0.02)
                continue
            seq = frame.seq
            start = time.time()
            try:
                result = self.process(frame)
            except Exception as e:
                self.errors += 1
                self.state, self.error = "error", f"{type(e).__name__}: {e}"[:120]
                print(f"[{self.key}] error #{self.errors}: {e}", file=sys.stderr)
                if self.errors <= 3:
                    traceback.print_exc()
                time.sleep(ERROR_RETRY_S)
                continue
            end = time.time()
            if self.enabled:                      # may have been switched off meanwhile
                self.result, self.result_seq, self.result_t = result, frame.seq, end
                self.state, self.error = "running", ""
            dt_ms = (end - start) * 1000
            self.ms = 0.8 * self.ms + 0.2 * dt_ms if self.ms else dt_ms
            if self._last_run:
                period = max(start - self._last_run, 1e-6)
                self.hz = 0.8 * self.hz + 0.2 / period if self.hz else 1 / period
            self._last_run = start


def boxes_contain(box, x, y) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


class onnx_threads:
    """
    with onnx_threads(2): import face_memory
    Caps the CPU threads of every onnxruntime session created inside the block.
    Each session otherwise starts one thread per core, and three libraries doing
    that at once on the Jetson's 6 cores made every module ~10x slower.
    """
    def __init__(self, n):
        self.n = n

    def __enter__(self):
        if not self.n:
            return
        import onnxruntime as ort
        self.ort, self.orig, n = ort, ort.InferenceSession, self.n

        class Capped(self.orig):
            def __init__(s, path, sess_options=None, *a, **kw):
                if sess_options is None:
                    sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = n
                sess_options.inter_op_num_threads = 1
                super().__init__(path, sess_options, *a, **kw)
        ort.InferenceSession = Capped

    def __exit__(self, *exc):
        if self.n:
            self.ort.InferenceSession = self.orig


# ── GPU ───────────────────────────────────────────────────────────────────────

GPU_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def onnx_gpu_available() -> bool:
    """onnxruntime-gpu installed (setup_demo.py does it on the Jetson) and not turned off with DEMO_ONNX_GPU=0."""
    import os
    if os.environ.get("DEMO_ONNX_GPU", "1") == "0":
        return False
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def sessions_to_gpu(sessions, tag) -> str:
    """
    Move already-built onnxruntime sessions (InsightFace, HSEmotion: their code asks for
    CPU only) onto CUDA. Any failure leaves them on the CPU, where they worked before.
    Returns "GPU" or "CPU" for the panel.
    """
    if not onnx_gpu_available():
        return "CPU"
    try:
        for s in sessions:
            s.set_providers(GPU_PROVIDERS)
        if all(s.get_providers()[0] == "CUDAExecutionProvider" for s in sessions):
            return "GPU"
    except Exception as e:
        print(f"[{tag}] GPU unavailable ({e}); staying on CPU", file=sys.stderr)
    for s in sessions:
        try:
            s.set_providers(["CPUExecutionProvider"])
        except Exception:
            pass
    return "CPU"
