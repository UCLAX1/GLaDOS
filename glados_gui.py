#!/usr/bin/env python3
"""
glados_gui.py — GLaDOS launcher: sim view + I/O display.

source listening/venv/bin/activate
JETSON_IP=100.73.92.3 python3 glados_gui.py
"""

import importlib
import os
import pathlib
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

import mujoco

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QFont, QWheelEvent
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QSizePolicy, QPushButton,
)

from central_ai.brain import GladosBrain
from control.mujoco_control import MujocoControl
from actions.sequence import Sequence, TICK_RATE
from speech.speech_queue import enqueue
from listening.listener import SpeechListener
from vision.vision_client import VisionClient

# ── Config ────────────────────────────────────────────────────────────────────
GGUF      = ROOT / "central_ai/benchmarking_tools/Bonsai-8B-Q1_0.gguf"
MODEL_XML = ROOT / "sim/model/glados.xml"
JETSON_IP = os.environ.get("JETSON_IP", "")
FPS       = 30

GESTURE_MAP = {
    "head_tilt":       "curious_peek_horiz",
    "recoil":          "confused_scan",
    "slow_sweep":      "scan",
    "lean_in":         "curious_peek_vert",
    "dismissive_turn": "look_away",
    "nod":             "nod",
    "idle":            None,
}
HOME = {"lower_arm": 20, "main_swivel": 0, "nod": 10}

# ── MuJoCo setup ─────────────────────────────────────────────────────────────
try:
    from sim import params as _p
    _model = _p.load_model("glados.xml")
except Exception:
    _model = mujoco.MjModel.from_xml_path(str(MODEL_XML))

model    = _model
data     = mujoco.MjData(model)
robot    = MujocoControl(model, data)

# Increase offscreen framebuffer so resizing the window never hits the 480px default cap.
model.vis.global_.offheight = 2048
model.vis.global_.offwidth  = 2048

renderer = mujoco.Renderer(model, height=400, width=540)

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.distance  = 2.0
cam.azimuth   = 90.0
cam.elevation = -20.0

# ── Shared state ──────────────────────────────────────────────────────────────
brain        = GladosBrain()
brain_lock   = threading.Lock()
vision       = VisionClient(JETSON_IP or None)
_utt_q: queue.Queue = queue.Queue(maxsize=1)
_gest_q: queue.Queue = queue.Queue(maxsize=1)
procs:   list[subprocess.Popen] = []
listener = None


# ── Pipeline threads ──────────────────────────────────────────────────────────

def dispatch_speech(text: str):
    enqueue(text)


def dispatch_gesture(gesture: str):
    action = GESTURE_MAP.get(gesture)
    if not action:
        return
    try:
        _gest_q.get_nowait()
    except queue.Empty:
        pass
    _gest_q.put_nowait(action)


def on_transcription(text: str, signals):
    if len(text.strip().split()) < 2:
        return
    signals.user_said.emit(text)
    try:
        _utt_q.get_nowait()
    except queue.Empty:
        pass
    _utt_q.put_nowait(text)


def _brain_loop(signals):
    while True:
        text       = _utt_q.get()
        perception = vision.context_string()
        parts      = []
        with brain_lock:
            for sentence in brain.respond_stream(text, perception=perception):
                dispatch_speech(sentence)
                parts.append(sentence)
        if parts:
            signals.glados_said.emit(" ".join(parts))
        response = brain.last_response or {}
        dispatch_gesture(response.get("gesture", "idle"))


def _gesture_loop():
    def tick():
        time.sleep(TICK_RATE)
    while True:
        name = _gest_q.get()
        try:
            mod = importlib.import_module(f"actions.scripts.{name}")
            getattr(mod, name)(Sequence(robot, tick_fn=tick, running_fn=lambda: True)).play()
        except Exception as e:
            print(f"[gesture] {name!r}: {e}")


def _home_pose():
    def tick():
        time.sleep(TICK_RATE)
    Sequence(robot, tick_fn=tick).pose(duration=1.5, lerp=True, **HOME).play()


# ── Qt signals ────────────────────────────────────────────────────────────────

class Signals(QObject):
    user_said   = Signal(str)
    glados_said = Signal(str)


# ── Sim viewport (interactive camera) ────────────────────────────────────────

class SimView(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._last   = None
        self._button = None

    def mousePressEvent(self, ev):
        self._last   = ev.position().toPoint()
        self._button = ev.button()

    def mouseReleaseEvent(self, ev):
        self._last = None

    def mouseMoveEvent(self, ev):
        if self._last is None:
            return
        import math
        cur = ev.position().toPoint()
        dx  = cur.x() - self._last.x()
        dy  = cur.y() - self._last.y()
        self._last = cur
        if self._button == Qt.LeftButton:
            cam.azimuth   = (cam.azimuth - dx * 0.4) % 360
            cam.elevation = max(-89.0, min(89.0, cam.elevation - dy * 0.4))
        elif self._button == Qt.RightButton:
            pan = cam.distance * 0.0015
            az  = math.radians(cam.azimuth)
            cam.lookat[0] += math.sin(az) * dx * pan
            cam.lookat[1] += -math.cos(az) * dx * pan
            cam.lookat[2] -= dy * pan

    def wheelEvent(self, ev: QWheelEvent):
        cam.distance = max(0.3, cam.distance * (0.9 if ev.angleDelta().y() > 0 else 1.1))


# ── Main window ───────────────────────────────────────────────────────────────

BG  = "#111111"
FG  = "#dddddd"
DIM = "#3a3a3a"
GRN = "#4caf50"


class GladosWindow(QWidget):
    def __init__(self, signals: Signals):
        super().__init__()
        self.signals = signals
        self.setWindowTitle("GLaDOS")
        self.setStyleSheet(f"background:{BG}; color:{FG};")
        self.resize(860, 500)
        self._build()
        self._connect_signals()

        # Sim render
        self._render_t = QTimer(self)
        self._render_t.timeout.connect(self._tick)
        self._render_t.start(1000 // FPS)

        # Service dots
        self._status_t = QTimer(self)
        self._status_t.timeout.connect(self._refresh_dots)
        self._status_t.start(2000)
        self._refresh_dots()

        threading.Thread(target=_home_pose, daemon=True).start()

    # ── Layout ─────────────────────────────────────────────────────────────

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sim
        self.sim = SimView()
        self.sim.setStyleSheet("background:#000;")
        self.sim.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sim.setCursor(Qt.CrossCursor)
        root.addWidget(self.sim, stretch=1)

        # Right panel
        panel = QFrame()
        panel.setFixedWidth(230)
        panel.setStyleSheet("background:#161616; border-left:1px solid #1e1e1e;")
        col = QVBoxLayout(panel)
        col.setContentsMargins(18, 20, 18, 20)
        col.setSpacing(0)

        # Title
        t = QLabel("GLaDOS")
        t.setFont(QFont("Helvetica Neue", 17, QFont.Bold))
        t.setStyleSheet("border:none;")
        col.addWidget(t)
        col.addSpacing(16)

        # Service dots
        self._dots: dict[str, QLabel] = {}
        for key, label in [("llm", "LLM"), ("speech", "Speech"), ("jetson", "Jetson")]:
            row = QHBoxLayout()
            row.setSpacing(8)
            dot = QLabel("●")
            dot.setFont(QFont("Helvetica Neue", 10))
            dot.setStyleSheet(f"color:{DIM}; border:none;")
            self._dots[key] = dot
            lbl = QLabel(label)
            lbl.setFont(QFont("Helvetica Neue", 11))
            lbl.setStyleSheet("border:none;")
            row.addWidget(dot); row.addWidget(lbl); row.addStretch()
            col.addLayout(row)
            col.addSpacing(3)

        col.addSpacing(18)

        div = QFrame(); div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("color:#222; background:#222; max-height:1px; border:none;")
        col.addWidget(div)
        col.addSpacing(18)

        # I/O boxes
        def _io(header, color):
            h = QLabel(header)
            h.setFont(QFont("Menlo", 8, QFont.Bold))
            h.setStyleSheet(f"color:{color}; border:none;")
            b = QLabel("—")
            b.setFont(QFont("Helvetica Neue", 11))
            b.setStyleSheet("border:none;")
            b.setWordWrap(True)
            b.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            b.setMinimumHeight(56)
            return h, b

        h_you, self.lbl_you = _io("YOU", "#7ec8e3")
        col.addWidget(h_you); col.addSpacing(4); col.addWidget(self.lbl_you)
        col.addSpacing(16)

        h_gl, self.lbl_gl = _io("GLADOS", "#f5a623")
        col.addWidget(h_gl); col.addSpacing(4); col.addWidget(self.lbl_gl)

        col.addStretch()

        # Quit button
        q = QPushButton("QUIT")
        q.setFont(QFont("Helvetica Neue", 11, QFont.Bold))
        q.setFixedHeight(36)
        q.setCursor(Qt.PointingHandCursor)
        q.setStyleSheet(
            "QPushButton{background:#1e1e1e;color:#666;border:none;border-radius:6px;}"
            "QPushButton:hover{background:#2a2a2a;color:#ccc;}"
        )
        q.clicked.connect(self.close)
        col.addWidget(q)

        root.addWidget(panel)

    def _connect_signals(self):
        self.signals.user_said.connect(self.lbl_you.setText)
        self.signals.glados_said.connect(self.lbl_gl.setText)

    # ── Dots ───────────────────────────────────────────────────────────────

    def _refresh_dots(self):
        def llm_up():
            try:
                urllib.request.urlopen("http://localhost:8080/health", timeout=0.5)
                return True
            except Exception:
                return False

        checks = {
            "llm":    llm_up,
            "speech": lambda: any(p.poll() is None for p in procs),
            "jetson": vision.is_connected,
        }
        for key, fn in checks.items():
            try:
                up = fn()
            except Exception:
                up = False
            self._dots[key].setStyleSheet(
                f"color:{'#4caf50' if up else DIM}; border:none;"
            )

    # ── Sim ────────────────────────────────────────────────────────────────

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        global renderer
        w = max(64, self.sim.width())
        h = max(64, self.sim.height())
        renderer = mujoco.Renderer(model, height=h, width=w)

    def _tick(self):
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=cam)
        px = renderer.render()
        h, w, ch = px.shape
        img = QImage(px.data, w, h, w * ch, QImage.Format_RGB888)
        self.sim.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, ev):
        shutdown()
        ev.accept()


# ── Subprocess helpers ────────────────────────────────────────────────────────

def start_proc(cmd, name):
    print(f"[launch] {name}...")
    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    procs.append(p)
    return p


def wait_for_llama(timeout=90):
    print("[launch] waiting for llama-server", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://localhost:8080/health", timeout=1)
            print(" ✓"); return
        except Exception:
            print(".", end="", flush=True); time.sleep(1)
    print("\n[launch] llama-server not responding — continuing")


def shutdown():
    global listener
    if listener:
        try:
            listener.stop()
        except Exception:
            pass
        listener = None
    vision.stop()
    for p in reversed(procs):
        try:
            p.terminate()
        except Exception:
            pass
    os._exit(0)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global listener

    for d in ["/tmp/glados_queue", "/tmp/glados_sim_queue"]:
        p = pathlib.Path(d); p.mkdir(exist_ok=True)
        for f in p.glob("*"): f.unlink(missing_ok=True)

    start_proc(["llama-server", "-m", str(GGUF), "--port", "8080"], "llama-server")
    wait_for_llama()
    start_proc([sys.executable, "speech/glados_daemon.py"], "glados-daemon")
    time.sleep(1)

    vision.start()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    signals = Signals()
    threading.Thread(target=_brain_loop,  args=(signals,), daemon=True, name="brain").start()
    threading.Thread(target=_gesture_loop, daemon=True, name="gesture").start()

    listener = SpeechListener(
        on_transcription=lambda text: on_transcription(text, signals),
        language="en",
        model_final="distil-small.en",
    )
    threading.Thread(target=listener.start, daemon=True, name="listener").start()

    win = GladosWindow(signals)
    win.show()

    signal.signal(signal.SIGINT, lambda *_: shutdown())
    app.aboutToQuit.connect(shutdown)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
