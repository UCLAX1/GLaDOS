#!/usr/bin/env python3
"""
gui_launch.py — GLaDOS pipeline with embedded MuJoCo GUI.

Run with the listening venv Python (once: pip install PySide6 pillow mujoco):
    source listening/venv/bin/activate
    python3 gui_launch.py

Starts llama-server and glados_daemon as subprocesses, runs the listener
and brain loop as threads, and renders the sim live inside a PySide6 window.
"""

import importlib
import os
# torch/faster-whisper leave a semaphore behind when we os._exit(); harmless,
# macOS reclaims it. Silence the resource_tracker warning (inherited by its subprocess).
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:multiprocessing.resource_tracker")
import pathlib
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import mujoco
import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QPoint
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QTextCursor, QWheelEvent, QPainter, QBrush, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QTextEdit, QFrame, QSizePolicy, QPushButton,
    QSlider, QCheckBox, QLineEdit, QGroupBox, QScrollArea, QTabWidget, QSplitter,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from central_ai.brain import GladosBrain, build_turn
import json
from control.mujoco_control import MujocoControl
from actions.sequence import Sequence, TICK_RATE
from speech.speech_queue import enqueue, clear as clear_tts_queue
from listening.listener import SpeechListener
from vision.vision_client import VisionClient

# ── Config ────────────────────────────────────────────────────────────────────

ROOT      = pathlib.Path(__file__).parent
GGUF      = ROOT / "central_ai/benchmarking_tools/Bonsai-8B-Q1_0.gguf"
MODEL_XML = ROOT / "sim/model/glados.xml"
sys.path.insert(0, str(ROOT / "sim"))
HARDWARE   = bool(os.environ.get("GLADOS_HARDWARE"))
LLAMA_CMD  = "llama-server"
JETSON_IP  = os.environ.get("JETSON_IP", "")

SIM_W, SIM_H = 640, 480
FPS = 30

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

# ── MuJoCo ───────────────────────────────────────────────────────────────────

try:
    from sim import params as _sim_params
    model = _sim_params.load_model("glados.xml")
except (ImportError, Exception):
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
# Bump offscreen framebuffer so the renderer can grow with the window.
model.vis.global_.offheight = 2048
model.vis.global_.offwidth  = 2048
data     = mujoco.MjData(model)
robot    = MujocoControl(model, data)
renderer = mujoco.Renderer(model, height=SIM_H, width=SIM_W)

# Free camera — manipulated by mouse in SimView
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.distance  = 2.0
cam.azimuth   = 90.0
cam.elevation = -20.0

# ── Shared state ──────────────────────────────────────────────────────────────

import collections as _collections

brain        = GladosBrain()
brain_lock   = threading.Lock()
vision       = VisionClient(JETSON_IP or None)   # polls Jetson; no-op if JETSON_IP unset
# Input turns: FIFO (oldest first). A deque + condition instead of
# Queue(maxsize=1), which silently dropped older inputs ("latest wins").
_turn_q   = _collections.deque()
_turn_cv  = threading.Condition()
_cancel_gen = 0          # bumped on interrupt; in-flight replies with an older gen stop
_gesture_q:   queue.Queue = queue.Queue(maxsize=1)
_tts_queue   = _collections.deque()   # local mirror of sentences awaiting TTS
_last_unspoken: list[str] = []        # sentences queued but not yet played at interrupt time
_currently_speaking = threading.Event()  # set while audio is playing
auto_interrupt = threading.Event()    # toggleable: interrupt on barge-in
auto_interrupt.set()                  # on by default
_muted    = threading.Event()   # mute: block TTS output
_deafened = threading.Event()   # deafen: ignore mic input
_admin_console: "AdminConsole | None" = None  # set after window creation
procs: list[subprocess.Popen] = []
listener: "SpeechListener | None" = None   # set in main(); used by shutdown()

# ── Pipeline ──────────────────────────────────────────────────────────────────

def dispatch_speech(text: str):
    if not _muted.is_set():
        enqueue(text)

def dispatch_gesture(gesture: str):
    action = GESTURE_MAP.get(gesture)
    if not action:
        return
    try:
        _gesture_q.get_nowait()
    except queue.Empty:
        pass
    _gesture_q.put_nowait(action)

def _interrupt_speech(signals):
    """Clear the TTS queue and record what was left unspoken."""
    global _last_unspoken, _cancel_gen
    _cancel_gen += 1    # stop the reply that is still being generated
    # Save whatever was still pending (hasn't started playing)
    _last_unspoken = list(_tts_queue)
    _tts_queue.clear()
    clear_tts_queue()   # delete filesystem queue files
    signals.pipe_queue.emit([])
    signals.pipe_speak.emit("—")

# Admin "next-turn inject" — plain Python state, written by the GUI thread,
# read by the brain thread. (Never touch Qt widgets from the brain thread.)
_inject_lock    = threading.Lock()
_inject_text    = ""
_inject_persist = False


def _take_inject() -> str:
    global _inject_text
    with _inject_lock:
        text = _inject_text.strip()
        if text and not _inject_persist:
            _inject_text = ""
            if _admin_console is not None:
                _admin_console.inject_consumed.emit()   # clears the box on the GUI thread
        return text


def submit_turn(signals, heard: str = "", event: str = "speech", from_mic: bool = False):
    """
    Single entry point for every input: mic, admin text, vision events, the
    'input visuals' button. The brain loop merges it with the live vision
    snapshot into ONE JSON message.
    """
    heard = (heard or "").strip()
    if event == "speech":
        if not heard or (from_mic and _deafened.is_set()):
            return
        # Barge-in only applies to real speech
        if (_currently_speaking.is_set() or _tts_queue) and auto_interrupt.is_set():
            _interrupt_speech(signals)

    turn = {"heard": heard, "event": event}
    with _turn_cv:
        if event != "speech":
            # Camera events only matter "now": skip if anything is already waiting
            if _turn_q:
                return
        else:
            # Speech outranks stale camera events that haven't been handled yet
            for t in [t for t in _turn_q if t["event"] != "speech"]:
                _turn_q.remove(t)
        _turn_q.append(turn)            # FIFO: oldest is handled first
        _turn_cv.notify()
    signals.status.emit("thinking")


def on_transcription(text: str, signals):
    """SpeechListener callback."""
    submit_turn(signals, heard=text, event="speech", from_mic=True)


def _brain_loop(signals):
    global _last_unspoken
    while True:
        with _turn_cv:
            while not _turn_q:
                _turn_cv.wait()
            turn = _turn_q.popleft()                     # oldest first
            if turn["event"] == "speech":
                # Speech that piled up while she was busy is one continued
                # thought (VAD splits on pauses) — merge it, in spoken order.
                parts = [turn["heard"]]
                while _turn_q and _turn_q[0]["event"] == "speech":
                    parts.append(_turn_q.popleft()["heard"])
                turn = {"heard": " ".join(parts), "event": "speech"}
        my_gen = _cancel_gen
        t0     = time.time()

        # No "you were interrupted" note: at our latency the unfinished line is
        # stale by the time she replies, and it made her rehash old answers.
        notes = []
        inject = _take_inject()
        if inject:
            notes.append(inject)

        user_json = build_turn(
            heard=turn["heard"],
            vision=vision.snapshot(),
            event=turn["event"],
            notes=notes or None,
        )
        signals.pipe_input.emit(user_json)
        if _admin_console is not None:
            _admin_console.push_log(f"[turn] {user_json}")

        signals.pipe_brain.emit("")          # clear brain box for new turn
        with brain_lock:
            first = True
            for sentence in brain.respond_stream(user_json):
                if _cancel_gen != my_gen:
                    # Interrupted: drop the rest of this reply
                    if _admin_console is not None:
                        _admin_console.push_log("[brain] reply cancelled by interrupt")
                    break
                if not sentence.strip():
                    continue
                if first:
                    signals.status.emit("tts")
                    first = False
                dispatch_speech(sentence)
                _tts_queue.append(sentence)
                signals.pipe_brain.emit(sentence)
                signals.pipe_queue.emit(list(_tts_queue))
        if _cancel_gen != my_gen:
            signals.timing.emit(time.time() - t0)
            continue                                       # no gesture for a cancelled reply
        response = brain.last_response or {}
        if first:
            signals.pipe_brain.emit("(silent)")
            signals.status.emit("listening")
            if _admin_console is not None:
                _admin_console.push_log(f"[brain] stayed silent — raw: {response}")
        dispatch_gesture(response.get("gesture", "idle"))
        signals.timing.emit(time.time() - t0)

def _gesture_loop(signals):
    def tick(): time.sleep(TICK_RATE)
    while True:
        action_name = _gesture_q.get()
        signals.pipe_action.emit(action_name)
        try:
            mod = importlib.import_module(f"actions.scripts.{action_name}")
            fn  = getattr(mod, action_name)
            # idle is interruptable: stop as soon as a new action is queued
            if action_name == "idle":
                running_fn = lambda: _gesture_q.empty()
            else:
                running_fn = lambda: True
            fn(Sequence(robot, tick_fn=tick, running_fn=running_fn)).play()
        except Exception as e:
            print(f"[gesture] {action_name!r} failed: {e}")
        signals.pipe_action.emit("")  # clear when done

def _home_pose():
    def tick(): time.sleep(TICK_RATE)
    Sequence(robot, tick_fn=tick).pose(duration=1.5, lerp=True, **HOME).play()


def _cam_feed_loop(signals):
    """Pull MJPEG frames from the Jetson demo stream and emit as raw JPEG bytes."""
    if not JETSON_IP:
        return
    url = f"http://{JETSON_IP}:8081/stream"
    buf = b""
    while True:
        try:
            req = urllib.request.urlopen(url, timeout=5)
            while True:
                chunk = req.read(4096)
                if not chunk:
                    break
                buf += chunk
                # Scan for JPEG start/end markers
                s = buf.find(b"\xff\xd8")
                e = buf.find(b"\xff\xd9", s + 2) if s != -1 else -1
                if s != -1 and e != -1:
                    signals.cam_frame.emit(buf[s:e + 2])
                    buf = buf[e + 2:]
                    if len(buf) > 1 << 20:   # safety: drop if buffer bloats
                        buf = b""
        except Exception:
            buf = b""
            time.sleep(2)


def _perception_loop(signals):
    """Emit the latest perception string every 2 s."""
    while True:
        ctx = vision.context_string()
        if ctx:
            signals.perception.emit(ctx)
        time.sleep(2)


# Gestures that should proactively trigger a GLaDOS response when detected.
_TRIGGER_GESTURES = {"waving", "raised_hand", "pointing", "thumbs_up", "thumbs_down"}
_TRIGGER_COOLDOWN = 8.0   # seconds between proactive triggers

def _vision_trigger_loop(signals):
    """Watch for vision events and feed them as utterances when notable things happen."""
    if not JETSON_IP:
        return
    prev_faces   = 0
    prev_gesture = ""
    last_trigger = 0.0

    while True:
        time.sleep(1.0)
        snap = vision.latest()   # thread-safe copy of latest perception dict
        if not snap:
            continue

        now            = time.time()
        faces          = snap.get("face_count", 0)
        gestures       = snap.get("gestures", [])          # list of gesture strings
        active_gesture = gestures[0] if gestures else ""   # first/most prominent one
        cooldown_ok    = (now - last_trigger) > _TRIGGER_COOLDOWN

        # Someone new entered the frame
        if faces > 0 and prev_faces == 0 and cooldown_ok:
            submit_turn(signals, event="person_entered")
            last_trigger = now

        # A new deliberate gesture was detected
        elif (active_gesture
              and active_gesture not in ("idle", "none", "")
              and active_gesture != prev_gesture
              and cooldown_ok):
            submit_turn(signals, event="gesture")
            last_trigger = now

        prev_faces   = faces
        prev_gesture = active_gesture


# ── Interactive sim viewport ──────────────────────────────────────────────────

class SimView(QLabel):
    """QLabel that forwards mouse/scroll to the MuJoCo free camera."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(False)
        self._last: QPoint | None = None
        self._button: Qt.MouseButton | None = None

    def mousePressEvent(self, ev):
        self._last   = ev.position().toPoint()
        self._button = ev.button()

    def mouseReleaseEvent(self, ev):
        self._last   = None
        self._button = None

    def mouseMoveEvent(self, ev):
        if self._last is None:
            return
        import math
        cur = ev.position().toPoint()
        dx  = cur.x() - self._last.x()
        dy  = cur.y() - self._last.y()
        self._last = cur

        if self._button == Qt.LeftButton:
            # Orbit: left/right → azimuth, up/down → elevation (full range)
            cam.azimuth   = (cam.azimuth - dx * 0.4) % 360
            cam.elevation = max(-89.0, min(89.0, cam.elevation - dy * 0.4))

        elif self._button == Qt.RightButton:
            # Pan lookat point in camera-local plane:
            #   horizontal drag  → slide along camera-right axis (XY plane)
            #   vertical drag    → slide along world Z (up/down)
            pan = cam.distance * 0.0015
            az  = math.radians(cam.azimuth)
            # camera-right vector in world XY
            rx, ry = math.sin(az), -math.cos(az)
            cam.lookat[0] += rx * dx * pan
            cam.lookat[1] += ry * dx * pan
            cam.lookat[2] -= dy * pan   # Z is up in MuJoCo

    def wheelEvent(self, ev: QWheelEvent):
        delta = ev.angleDelta().y()
        cam.distance = max(0.3, cam.distance * (0.9 if delta > 0 else 1.1))


# ── Qt signals bridge ─────────────────────────────────────────────────────────

class Signals(QObject):
    status      = Signal(str)        # pipeline stage name
    timing      = Signal(float)      # total response seconds
    metrics     = Signal(str, str)   # (source, line) for llm/tts stats
    # data-flow boxes
    pipe_input  = Signal(str)        # STT result
    pipe_brain  = Signal(str)        # each LLM sentence (cumulative)
    pipe_queue  = Signal(list)       # current TTS queue contents
    pipe_speak  = Signal(str)        # sentence now being spoken
    pipe_done   = Signal()           # finished speaking
    pipe_action = Signal(str)        # current gesture/action name
    # camera
    cam_frame   = Signal(bytes)      # raw JPEG bytes from Jetson MJPEG stream
    perception  = Signal(str)        # latest perception context string

# ── Pipeline bar ──────────────────────────────────────────────────────────────

# Stage order and display names
PIPELINE = ["listen", "stt", "llm", "tts", "speak"]
STAGE_IDX = {
    "starting":  0,
    "listening": 0,
    "stt":       1,
    "thinking":  2,
    "tts":       3,
    "speaking":  4,
}
STAGE_COLORS = {
    "listen": "#4caf50",
    "stt":    "#8bc34a",
    "llm":    "#f5a623",
    "tts":    "#ab82d4",
    "speak":  "#2196f3",
}

class PipelineBar(QWidget):
    """Horizontal progress bar showing which pipeline stage is active + elapsed time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self._stage   = 0          # index into PIPELINE
        self._times   = [None] * len(PIPELINE)   # completion time (s) or None
        self._t_start = None       # when current stage started
        self._font    = QFont("Menlo", 9)

        self._tick = QTimer(self)
        self._tick.timeout.connect(self.update)   # repaint for live elapsed
        self._tick.start(100)

    def set_stage(self, status: str):
        import time as _time
        idx = STAGE_IDX.get(status, -1)
        if idx < 0:
            return
        if idx != self._stage:
            # record completion time for previous stage
            if self._t_start is not None and self._stage < len(PIPELINE):
                self._times[self._stage] = _time.time() - self._t_start
            # reset future stages if going backwards (new turn)
            if idx <= self._stage and idx == 0:
                self._times = [None] * len(PIPELINE)
            self._stage   = idx
            self._t_start = _time.time()
        self.update()

    def paintEvent(self, _ev):
        import time as _time
        p    = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setFont(self._font)

        w     = self.width()
        h     = self.height()
        n     = len(PIPELINE)
        seg_w = w / n
        gap   = 4

        for i, name in enumerate(PIPELINE):
            x  = int(i * seg_w) + gap
            sw = int(seg_w) - gap * 2
            sy = 4
            sh = h - 8

            color = STAGE_COLORS[name]

            if i < self._stage:
                # completed — solid, dim
                p.setBrush(QBrush(QColor(color).darker(180)))
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(x, sy, sw, sh, 4, 4)
                # show elapsed
                t = self._times[i]
                label = f"{t:.2f}s" if t else name
            elif i == self._stage:
                # active — bright fill + pulsing outline
                p.setBrush(QBrush(QColor(color)))
                p.setPen(QPen(QColor("white"), 1))
                p.drawRoundedRect(x, sy, sw, sh, 4, 4)
                elapsed = (_time.time() - self._t_start) if self._t_start else 0
                label = f"{name}  {elapsed:.1f}s"
            else:
                # pending — outline only
                p.setBrush(QBrush(QColor("#222")))
                p.setPen(QPen(QColor("#444"), 1))
                p.drawRoundedRect(x, sy, sw, sh, 4, 4)
                label = name

            p.setPen(QColor("white") if i == self._stage else QColor("#888"))
            p.drawText(x, sy, sw, sh, Qt.AlignCenter, label)

        p.end()

# ── Admin console ─────────────────────────────────────────────────────────────

class AdminConsole(QDialog):
    """Separate admin window — independent, no always-on-top."""

    # Signal so the log appender (called from any thread) is thread-safe
    _log_line = Signal(str)
    inject_consumed = Signal()   # emitted by brain thread after a one-shot inject is used

    def __init__(self, signals: "Signals", parent=None):
        super().__init__(parent)
        self.signals = signals
        self.setWindowTitle("GLaDOS Admin")
        self.resize(560, 720)
        # Qt.Window makes it a proper standalone window; no StaysOnTop
        self.setWindowFlags(Qt.Window)
        self.setStyleSheet("background: #0e0e0e; color: #ccc;")
        self._log_line.connect(self._append_log)
        self._build()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        mono8  = QFont("Menlo", 8)
        mono9  = QFont("Menlo", 9)
        mono10 = QFont("Menlo", 10)

        def _section(title):
            g = QGroupBox(title)
            g.setFont(mono8)
            g.setStyleSheet(
                "QGroupBox { color: #666; border: 1px solid #2a2a2a;"
                " border-radius: 4px; margin-top: 6px; padding-top: 6px; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
            )
            lay = QVBoxLayout(g)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.setSpacing(6)
            return g, lay

        def _btn(text, color="#333", fg="#ccc"):
            b = QPushButton(text)
            b.setFont(mono9)
            b.setFixedHeight(24)
            b.setStyleSheet(
                f"QPushButton {{ background: {color}; color: {fg}; border: 1px solid #444;"
                f" border-radius: 3px; padding: 0 8px; }}"
                f"QPushButton:pressed {{ background: #555; }}"
            )
            return b

        def _toggle(text, default=False):
            cb = QCheckBox(text)
            cb.setFont(mono9)
            cb.setChecked(default)
            cb.setStyleSheet("color: #aaa;")
            return cb

        # ── Manual input ──────────────────────────────────────────────────────
        sec0, lay0 = _section("MANUAL INPUT  (as if spoken by human)")
        self.manual_input = QLineEdit()
        self.manual_input.setFont(mono10)
        self.manual_input.setPlaceholderText("Type something and press Enter…")
        self.manual_input.setStyleSheet(
            "background: #141414; color: #ddd; border: 1px solid #333; border-radius: 3px; padding: 4px;"
        )
        self.manual_input.returnPressed.connect(self._on_manual_input)
        lay0.addWidget(self.manual_input)
        send_row = QHBoxLayout()
        send_row.addStretch()
        send_btn = _btn("▶ send", "#1a2a1a", "#4caf50")
        send_btn.clicked.connect(self._on_manual_input)
        send_row.addWidget(send_btn)
        lay0.addLayout(send_row)
        root.addWidget(sec0)

        # ── Mute / Deafen buttons ─────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        def _big_toggle(label, color):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setChecked(False)
            b.setFont(QFont("Menlo", 12, QFont.Bold))
            b.setFixedHeight(52)
            b.setStyleSheet(
                f"QPushButton {{ background: #1a1a1a; color: #555; border: 2px solid #333; border-radius: 6px; }}"
                f"QPushButton:checked {{ background: #2a0a0a; color: {color}; border: 2px solid {color}; }}"
                f"QPushButton:pressed {{ background: #2a0a0a; }}"
            )
            return b

        self.mute_btn = _big_toggle("🔇  MUTE", "#e57373")
        self.mute_btn.toggled.connect(self._on_mute_toggle)
        btn_row.addWidget(self.mute_btn)

        self.deaf_btn = _big_toggle("🎙 DEAFEN", "#ab82d4")
        self.deaf_btn.toggled.connect(self._on_deafen_toggle)
        btn_row.addWidget(self.deaf_btn)

        root.addLayout(btn_row)

        # ── Interrupt & pipeline ──────────────────────────────────────────────
        sec, lay = _section("PIPELINE")

        row1 = QHBoxLayout()
        self.cb_interrupt = _toggle("auto-interrupt on barge-in", default=True)
        self.cb_interrupt.toggled.connect(self._on_interrupt_toggle)
        row1.addWidget(self.cb_interrupt)
        row1.addStretch()
        self.btn_interrupt_now = _btn("⚡ interrupt now", "#3a1a1a", "#e57373")
        self.btn_interrupt_now.clicked.connect(self._on_interrupt_now)
        row1.addWidget(self.btn_interrupt_now)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_clear_hist = _btn("🗑 clear history", "#1a1a2a", "#7ec8e3")
        self.btn_clear_hist.clicked.connect(self._on_clear_history)
        row2.addWidget(self.btn_clear_hist)
        self.btn_home = _btn("⌂ home pose", "#1a2a1a", "#4caf50")
        self.btn_home.clicked.connect(lambda: threading.Thread(target=_home_pose, daemon=True).start())
        row2.addWidget(self.btn_home)
        self.btn_visuals = _btn("👁 input visuals", "#1a2233", "#2b8fcc")
        self.btn_visuals.clicked.connect(self._on_input_visuals)
        row2.addWidget(self.btn_visuals)
        row2.addStretch()
        lay.addLayout(row2)

        root.addWidget(sec)

        # ── System prompt inject ──────────────────────────────────────────────
        sec2, lay2 = _section("NEXT-TURN SYSTEM INJECT  (prepended to next user message)")
        self.inject_box = QTextEdit()
        self.inject_box.setFont(mono9)
        self.inject_box.setFixedHeight(70)
        self.inject_box.setPlaceholderText("[System: speak in rhymes from now on]")
        self.inject_box.setStyleSheet(
            "background: #141414; color: #ddd; border: 1px solid #333; border-radius: 3px;"
        )
        lay2.addWidget(self.inject_box)
        self.inject_box.textChanged.connect(self._sync_inject)
        self.inject_consumed.connect(self.inject_box.clear)
        row3 = QHBoxLayout()
        self.cb_inject_persist = _toggle("persist (every turn)", default=False)
        row3.addWidget(self.cb_inject_persist)
        self.cb_inject_persist.toggled.connect(self._sync_inject)
        row3.addStretch()
        self.btn_inject_clear = _btn("clear")
        self.btn_inject_clear.clicked.connect(self.inject_box.clear)
        row3.addWidget(self.btn_inject_clear)
        lay2.addLayout(row3)
        root.addWidget(sec2)

        # ── Force-speak ───────────────────────────────────────────────────────
        sec3, lay3 = _section("FORCE SPEAK  (bypass LLM)")
        self.force_input = QLineEdit()
        self.force_input.setFont(mono9)
        self.force_input.setPlaceholderText("Type something for GLaDOS to say…")
        self.force_input.setStyleSheet(
            "background: #141414; color: #ddd; border: 1px solid #333; border-radius: 3px; padding: 3px;"
        )
        self.force_input.returnPressed.connect(self._on_force_speak)
        lay3.addWidget(self.force_input)
        row4 = QHBoxLayout()
        row4.addStretch()
        self.btn_force = _btn("▶ speak", "#1a2a1a", "#4caf50")
        self.btn_force.clicked.connect(self._on_force_speak)
        row4.addWidget(self.btn_force)
        lay3.addLayout(row4)
        root.addWidget(sec3)

        # ── LLM knobs ─────────────────────────────────────────────────────────
        sec4, lay4 = _section("LLM")
        temp_row = QHBoxLayout()
        temp_lbl = QLabel("temperature:")
        temp_lbl.setFont(mono9)
        temp_row.addWidget(temp_lbl)
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 200)   # 0.00 – 2.00
        self.temp_slider.setValue(int(brain._temperature * 100))
        self.temp_slider.setFixedHeight(16)
        self.temp_slider.valueChanged.connect(self._on_temp_change)
        temp_row.addWidget(self.temp_slider)
        self.temp_val = QLabel(f"{brain._temperature:.2f}")
        self.temp_val.setFont(mono9)
        self.temp_val.setFixedWidth(36)
        temp_row.addWidget(self.temp_val)
        lay4.addLayout(temp_row)

        root.addWidget(sec4)

        # ── Actions ───────────────────────────────────────────────────────────
        sec_act, lay_act = _section("ACTIONS")
        actions = [
            "idle", "nod", "scan",
            "curious_peek_horiz", "curious_peek_vert",
            "confused_scan", "look_away",
            "eye_blink", "eye_extend", "eye_retract",
        ]
        grid = QHBoxLayout()
        grid.setSpacing(4)
        col = None
        for i, name in enumerate(actions):
            if i % 2 == 0:
                col = QVBoxLayout()
                col.setSpacing(4)
                grid.addLayout(col)
            label = name.replace("_", " ")
            b = _btn(label)
            b.clicked.connect(lambda checked=False, n=name: self._on_run_action(n))
            col.addWidget(b)
        lay_act.addLayout(grid)
        root.addWidget(sec_act)

        # ── Subprocess controls ───────────────────────────────────────────────
        sec5, lay5 = _section("SUBPROCESSES")
        proc_row = QHBoxLayout()
        self.btn_restart_daemon = _btn("↺ restart TTS daemon", "#2a1a2a", "#ab82d4")
        self.btn_restart_daemon.clicked.connect(self._on_restart_daemon)
        proc_row.addWidget(self.btn_restart_daemon)
        proc_row.addStretch()
        lay5.addLayout(proc_row)
        root.addWidget(sec5)

        # ── Raw log ───────────────────────────────────────────────────────────
        sec6, lay6 = _section("RAW LOG")
        self.raw_log = QTextEdit()
        self.raw_log.setReadOnly(True)
        self.raw_log.setFont(QFont("Menlo", 8))
        self.raw_log.setStyleSheet(
            "background: #0a0a0a; color: #555; border: 1px solid #222; border-radius: 3px;"
        )
        lay6.addWidget(self.raw_log)
        root.addWidget(sec6, 1)   # stretch

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _on_run_action(self, name: str):
        try:
            _gesture_q.get_nowait()
        except queue.Empty:
            pass
        _gesture_q.put_nowait(name)

    def _on_manual_input(self):
        text = self.manual_input.text().strip()
        if not text:
            return
        self.manual_input.clear()
        submit_turn(self.signals, heard=text, event="speech")

    def _on_mute_toggle(self, checked):
        if checked:
            _muted.set()
            _interrupt_speech(self.signals)
            self.mute_btn.setText("🔇  MUTED")
        else:
            _muted.clear()
            self.mute_btn.setText("🔇  MUTE")

    def _on_deafen_toggle(self, checked):
        if checked:
            _deafened.set()
            self.deaf_btn.setText("🎙 DEAFENED")
        else:
            _deafened.clear()
            self.deaf_btn.setText("🎙 DEAFEN")

    def _on_interrupt_toggle(self, checked):
        if checked:
            auto_interrupt.set()
        else:
            auto_interrupt.clear()

    def _on_interrupt_now(self):
        _interrupt_speech(self.signals)
        self.signals.status.emit("listening")

    def _on_input_visuals(self):
        """Ask GLaDOS to comment on what the camera sees right now."""
        if not vision.is_connected():
            self._log_line.emit("[admin] input visuals: Jetson NOT connected — vision will be null")
        submit_turn(self.signals, event="visual_check")

    def _on_clear_history(self):
        with brain_lock:
            brain._history.clear()
        self._log_line.emit("[admin] conversation history cleared")

    def _on_force_speak(self):
        text = self.force_input.text().strip()
        if not text:
            return
        dispatch_speech(text)
        _tts_queue.append(text)
        self.signals.pipe_queue.emit(list(_tts_queue))
        self.force_input.clear()
        self._log_line.emit(f"[admin] force-speak: {text!r}")

    def _on_temp_change(self, val):
        t = val / 100.0
        brain._temperature = t
        self.temp_val.setText(f"{t:.2f}")

    def _on_restart_daemon(self):
        global daemon_proc
        self._log_line.emit("[admin] restarting TTS daemon…")
        try:
            daemon_proc.terminate()
            daemon_proc.wait(timeout=3)
        except Exception:
            pass
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        p = subprocess.Popen(
            ["python3", "speech/glados_daemon.py"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        procs.append(p)
        daemon_proc = p
        threading.Thread(target=_tts_reader, args=(p, self.signals), daemon=True,
                         name="tts-reader-2").start()
        self._log_line.emit("[admin] TTS daemon restarted")

    def push_log(self, line: str):
        """Thread-safe: call from any thread to append to the raw log."""
        self._log_line.emit(line)

    def _append_log(self, line: str):
        cur = self.raw_log.textCursor()
        cur.movePosition(QTextCursor.End)
        self.raw_log.setTextCursor(cur)
        self.raw_log.insertPlainText(line + "\n")
        self.raw_log.ensureCursorVisible()

    def _sync_inject(self, *_):
        """Mirror the inject box into plain globals the brain thread can read safely."""
        global _inject_text, _inject_persist
        with _inject_lock:
            _inject_text    = self.inject_box.toPlainText()
            _inject_persist = self.cb_inject_persist.isChecked()


# ── Main window ───────────────────────────────────────────────────────────────

daemon_proc: "subprocess.Popen | None" = None  # set in main(); also used by admin


class GladosWindow(QMainWindow):
    def __init__(self, signals: Signals):
        super().__init__()
        self.signals = signals
        self.setWindowTitle("GLaDOS")
        self.setMinimumSize(640, 400)
        self.resize(SIM_W + 360, SIM_H + 320)
        self._setup_palette()
        self._build_ui()
        self._connect_signals()

        # Debounce renderer recreation on resize
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._rebuild_renderer)

        # Render timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._render_frame)
        self._timer.start(1000 // FPS)

        # Live vision box refresh (4 Hz)
        self._vision_timer = QTimer(self)
        self._vision_timer.timeout.connect(self._refresh_vision_box)
        self._vision_timer.start(250)

        # System usage refresh (1 Hz)
        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._refresh_sys_box)
        self._sys_timer.start(1000)

        # Home pose on startup
        threading.Thread(target=_home_pose, daemon=True).start()

    def _refresh_sys_box(self):
        def pct(v):
            if v is None:
                return "<span style='color:#555'>n/a</span>"
            c = "#5c5" if v < 60 else "#e5a00d" if v < 85 else "#c55"
            return f"<span style='color:{c}'>{v:.0f}%</span>"

        # ── Mac
        try:
            import psutil
            vm  = psutil.virtual_memory()
            mac = (f"<b>MAC</b>&nbsp;&nbsp;cpu {pct(psutil.cpu_percent(None))}"
                   f"&nbsp; ram {vm.used/2**30:.1f}/{vm.total/2**30:.0f}G {pct(vm.percent)}")
            parts = []
            for p in procs:
                if p.poll() is not None:
                    continue
                label = "llama" if "llama" in str(p.args) else "tts"
                try:
                    pp = psutil.Process(p.pid)
                    parts.append(f"{label} {pp.memory_info().rss/2**30:.1f}G")
                except psutil.Error:
                    pass
            me = psutil.Process()
            parts.append(f"gui {me.memory_info().rss/2**30:.1f}G")
            mac += "<br><span style='color:#666'>&nbsp;&nbsp;" + " · ".join(parts) + "</span>"
        except Exception as e:
            mac = f"<b>MAC</b>&nbsp;&nbsp;<span style='color:#555'>psutil n/a ({e})</span>"

        # ── Jetson (from the demo's /status)
        j = vision.system if vision.is_connected() else {}
        if j:
            ram = ""
            if j.get("ram_used_gb") is not None and j.get("ram_total_gb"):
                ram_pct = 100 * j["ram_used_gb"] / j["ram_total_gb"]
                ram = f"&nbsp; ram {j['ram_used_gb']:.1f}/{j['ram_total_gb']:.0f}G {pct(ram_pct)}"
            temp = max([t for t in (j.get("cpu_temp_c"), j.get("gpu_temp_c")) if t is not None],
                       default=None)
            jet = (f"<b>JETSON</b>&nbsp;cpu {pct(j.get('cpu_pct'))}&nbsp; gpu {pct(j.get('gpu_pct'))}"
                   f"{ram}")
            extra = []
            if temp is not None:
                tc = "#5c5" if temp < 65 else "#e5a00d" if temp < 80 else "#c55"
                extra.append(f"<span style='color:{tc}'>{temp:.0f}°C</span>")
            if j.get("demo_rss_gb") is not None:
                extra.append(f"vision {j['demo_rss_gb']:.1f}G")
            if extra:
                jet += "<br><span style='color:#666'>&nbsp;&nbsp;" + " · ".join(extra) + "</span>"
        else:
            jet = "<b>JETSON</b>&nbsp;<span style='color:#555'>no stats (demo not reachable or old version)</span>"

        self.box_sys.setText(mac + "<br>" + jet)

    def _refresh_vision_box(self):
        if not JETSON_IP:
            self.box_vision.setText("<span style='color:#a55'>● off</span> — JETSON_IP not set")
            return
        s   = vision.latest()
        age = time.time() - vision.last_ok_t if vision.last_ok_t else None
        if not vision.is_connected():
            err = (vision.last_error or "waiting…")[:70]
            self.box_vision.setText(
                f"<span style='color:#c55'>● no data</span> from {JETSON_IP}<br>"
                f"<span style='color:#666'>{err}</span>"
            )
            return
        known  = ", ".join(s.get("known_faces") or []) or "unknown"
        gest   = ", ".join(s.get("gestures") or []) or "—"
        expr   = ", ".join(s.get("expressions") or []) or "—"
        dist   = s.get("nearest_person_m")
        fps    = s.get("fps")
        lines = [
            f"<span style='color:#5c5'>● {vision.source}</span>"
            f"<span style='color:#555'>  {age:.1f}s ago"
            f"{f'  · {fps:.0f} fps' if fps else ''}</span>",
            f"people: {s.get('face_count', 0)} ({known})"
            f"{f'  · {dist:.1f} m' if dist is not None else ''}",
            f"gesture: <span style='color:#ffd479'>{gest}</span>",
            f"expr: {expr}",
        ]
        self.box_vision.setText("<br>".join(lines))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Rebuild the offscreen renderer 150ms after the last resize event
        self._resize_timer.start(150)

    def _rebuild_renderer(self):
        global renderer
        w = max(64, self.sim_label.width())
        h = max(64, self.sim_label.height())
        # Cap to offscreen buffer size (set to 2048 at model load)
        w = min(w, 2048)
        h = min(h, 2048)
        renderer = mujoco.Renderer(model, height=h, width=w)

    def _setup_palette(self):
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor("#111111"))
        pal.setColor(QPalette.WindowText, QColor("#e0e0e0"))
        pal.setColor(QPalette.Base, QColor("#1c1c1c"))
        pal.setColor(QPalette.Text, QColor("#e0e0e0"))
        QApplication.setPalette(pal)

    def _build_ui(self):
        mono   = QFont("Menlo", 11)
        small  = QFont("Menlo", 10)
        title  = QFont("Menlo", 13, QFont.Bold)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 8, 10, 10)
        root_layout.setSpacing(8)

        # Top bar
        top = QHBoxLayout()
        title_lbl = QLabel("G L A D O S")
        title_lbl.setFont(title)
        title_lbl.setStyleSheet("color: #f5a623;")
        top.addWidget(title_lbl)
        top.addStretch()

        self.admin_console = AdminConsole(self.signals, parent=self)
        admin_btn = QPushButton("⚙ admin")
        admin_btn.setFont(QFont("Menlo", 9))
        admin_btn.setFixedHeight(22)
        admin_btn.setStyleSheet(
            "QPushButton { background: #1a1a1a; color: #666; border: 1px solid #333;"
            " border-radius: 3px; padding: 0 8px; }"
            "QPushButton:pressed { color: #ccc; }"
        )
        admin_btn.clicked.connect(self._open_admin)
        top.addWidget(admin_btn)

        root_layout.addLayout(top)

        # Pipeline progress bar
        self.pipeline_bar = PipelineBar()
        root_layout.addWidget(self.pipeline_bar)

        # Main row: sim + pipeline panel (draggable splitter)
        row = QSplitter(Qt.Horizontal)
        row.setHandleWidth(6)
        row.setChildrenCollapsible(False)
        row.setStyleSheet("QSplitter::handle { background: #1a1a1a; }")

        # ── Sim display ───────────────────────────────────────────────────────
        sim_widget = QWidget()
        sim_col = QVBoxLayout(sim_widget)
        sim_col.setSpacing(2)
        sim_col.setContentsMargins(0, 0, 0, 0)

        self.sim_label = SimView()
        self.sim_label.setMinimumSize(320, 240)
        self.sim_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sim_label.setStyleSheet("background: #000;")
        self.sim_label.setCursor(Qt.CrossCursor)
        sim_col.addWidget(self.sim_label, stretch=1)

        self.action_lbl = QLabel("")
        self.action_lbl.setFont(QFont("Menlo", 9))
        self.action_lbl.setAlignment(Qt.AlignCenter)
        self.action_lbl.setFixedHeight(18)
        self.action_lbl.setStyleSheet("color: #555; background: transparent;")
        sim_col.addWidget(self.action_lbl)

        row.addWidget(sim_widget)

        # ── Pipeline data-flow panel ──────────────────────────────────────────
        panel = QFrame()
        panel.setMinimumWidth(240)
        panel.setStyleSheet("background: #111; border-radius: 6px;")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(0)

        def _pipe_box(label, color, min_h=70, expand=False):
            """Return (outer QFrame, content QLabel) for one pipeline stage box."""
            outer = QFrame()
            outer.setStyleSheet(
                f"QFrame {{ background: #1a1a1a; border: 1px solid {color}55;"
                f" border-radius: 5px; }}"
            )
            if expand:
                outer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            else:
                outer.setMinimumHeight(min_h)
                outer.setMaximumHeight(min_h)
            ol = QVBoxLayout(outer)
            ol.setContentsMargins(8, 4, 8, 4)
            ol.setSpacing(2)
            hdr = QLabel(label)
            hdr.setFont(QFont("Menlo", 8, QFont.Bold))
            hdr.setStyleSheet(f"color: {color}; background: transparent; border: none;")
            ol.addWidget(hdr)
            body = QLabel("—")
            body.setFont(QFont("Menlo", 10))
            body.setStyleSheet("color: #bbb; background: transparent; border: none;")
            body.setWordWrap(True)
            body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            ol.addWidget(body, 1)
            return outer, body

        def _arrow():
            a = QLabel("▼")
            a.setAlignment(Qt.AlignCenter)
            a.setFixedHeight(16)
            a.setStyleSheet("color: #333; font-size: 12px; border: none;")
            return a

        box_input_frame,  self.box_input  = _pipe_box("INPUT  (one JSON: heard + vision)",  "#7ec8e3", min_h=110)
        box_brain_frame,  self.box_brain  = _pipe_box("BRAIN  /  LLM",  "#f5a623", expand=True)
        box_queue_frame,  self.box_queue  = _pipe_box("TTS QUEUE",       "#ab82d4", min_h=80)
        box_speak_frame,  self.box_speak  = _pipe_box("SPEAKING",        "#4caf50")

        # Live vision data — refreshed constantly, independent of turns
        box_vision_frame, self.box_vision = _pipe_box("VISION  /  LIVE", "#2b8fcc", min_h=96)
        self.box_vision.setFont(QFont("Menlo", 9))
        panel_layout.addWidget(box_vision_frame)

        box_sys_frame, self.box_sys = _pipe_box("SYSTEM  /  USAGE", "#888888", min_h=92)
        self.box_sys.setFont(QFont("Menlo", 9))
        panel_layout.addSpacing(4)
        panel_layout.addWidget(box_sys_frame)
        panel_layout.addWidget(_arrow())

        panel_layout.addWidget(box_input_frame)
        panel_layout.addWidget(_arrow())
        panel_layout.addWidget(box_brain_frame, 1)
        panel_layout.addWidget(_arrow())
        panel_layout.addWidget(box_queue_frame)
        panel_layout.addWidget(_arrow())
        panel_layout.addWidget(box_speak_frame)

        # Footer row: interrupt toggle + latency
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 4, 0, 0)

        self.interrupt_btn = QPushButton("⚡ interrupt: on")
        self.interrupt_btn.setCheckable(True)
        self.interrupt_btn.setChecked(True)
        self.interrupt_btn.setFont(QFont("Menlo", 8))
        self.interrupt_btn.setFixedHeight(20)
        self.interrupt_btn.setStyleSheet(
            "QPushButton { background: #2a2a2a; color: #4caf50; border: 1px solid #333;"
            " border-radius: 3px; padding: 0 6px; }"
            "QPushButton:checked { background: #2a2a2a; color: #4caf50; }"
            "QPushButton:!checked { background: #2a2a2a; color: #555; }"
        )
        self.interrupt_btn.toggled.connect(self._toggle_interrupt)
        footer.addWidget(self.interrupt_btn)

        self.timing_lbl = QLabel("")
        self.timing_lbl.setFont(QFont("Menlo", 8))
        self.timing_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.timing_lbl.setStyleSheet("color: #444; border: none;")
        footer.addWidget(self.timing_lbl, 1)

        panel_layout.addLayout(footer)

        row.addWidget(panel)
        row.setStretchFactor(0, 1)
        row.setStretchFactor(1, 0)
        row.setSizes([SIM_W, 320])

        # Vertical splitter: (sim | pipeline) over camera
        self.v_split = QSplitter(Qt.Vertical)
        self.v_split.setHandleWidth(6)
        self.v_split.setChildrenCollapsible(False)
        self.v_split.setStyleSheet("QSplitter::handle { background: #1a1a1a; }")
        self.v_split.addWidget(row)

        # Rebuild the MuJoCo renderer when a splitter is dragged
        row.splitterMoved.connect(lambda *_: self._resize_timer.start(150))
        self.v_split.splitterMoved.connect(lambda *_: self._resize_timer.start(150))

        # ── Camera row (full width, bottom) ───────────────────────────────────
        cam_widget = QWidget()
        cam_row = QVBoxLayout(cam_widget)
        cam_row.setSpacing(0)
        cam_row.setContentsMargins(0, 4, 0, 0)

        cam_bar = QHBoxLayout()
        cam_bar.setContentsMargins(0, 0, 0, 0)
        cam_hdr = QLabel("JETSON CAM")
        cam_hdr.setFont(QFont("Menlo", 8, QFont.Bold))
        cam_hdr.setStyleSheet("color: #2b8fcc;")
        cam_bar.addWidget(cam_hdr)
        cam_bar.addStretch()
        self.perception_lbl = QLabel("")
        self.perception_lbl.setFont(QFont("Menlo", 8))
        self.perception_lbl.setStyleSheet("color: #2b5a7a;")
        cam_bar.addWidget(self.perception_lbl)
        cam_row.addLayout(cam_bar)

        self.cam_lbl = QLabel("waiting for Jetson stream…")
        self.cam_lbl.setAlignment(Qt.AlignCenter)
        # Explicit minimum so the scaled pixmap never blocks shrinking
        self.cam_lbl.setMinimumSize(160, 120)
        self.cam_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cam_lbl.setStyleSheet("background: #050505; color: #333; border-radius: 4px;")
        cam_row.addWidget(self.cam_lbl, stretch=1)

        self.v_split.addWidget(cam_widget)
        self.v_split.setStretchFactor(0, 1)
        self.v_split.setStretchFactor(1, 1)
        self.v_split.setSizes([SIM_H, 300])
        root_layout.addWidget(self.v_split, stretch=1)

    def _connect_signals(self):
        s = self.signals
        s.status.connect(self.pipeline_bar.set_stage)
        s.timing.connect(lambda t: self.timing_lbl.setText(f"↩ {t:.2f}s total"))
        s.metrics.connect(self._on_metrics)
        s.pipe_input.connect(self._on_input)
        s.pipe_brain.connect(self._on_brain)
        s.pipe_queue.connect(self._on_queue)
        s.pipe_speak.connect(self._on_speak)
        s.pipe_done.connect(self._on_done)
        s.pipe_action.connect(self._on_action)
        s.cam_frame.connect(self._on_cam_frame)
        s.perception.connect(self.perception_lbl.setText)

    # ── Box update handlers ────────────────────────────────────────────────────

    def _on_input(self, text: str):
        # text is the full JSON turn sent to the LLM — show it compactly
        try:
            t = json.loads(text)
            v = t.get("vision")
            vis = "camera offline" if v is None else json.dumps(v, separators=(",", ":"))
            shown = (f"[{t.get('event')}]  heard: {t.get('heard') or '—'}\n"
                     f"vision: {vis[:220]}")
            if t.get("notes"):
                shown += f"\nnotes: {'; '.join(t['notes'])[:120]}"
        except Exception:
            shown = text
        self.box_input.setText(shown)
        self.box_brain.setText("—")
        self.box_queue.setText("—")
        self.box_speak.setText("—")

    def _on_brain(self, sentence: str):
        if not sentence:
            self.box_brain.setText("…")
            return
        cur = self.box_brain.text()
        if cur in ("—", "…"):
            self.box_brain.setText(sentence)
        else:
            self.box_brain.setText(cur + "\n" + sentence)

    def _on_queue(self, items: list):
        if not items:
            self.box_queue.setText("(empty)")
        else:
            self.box_queue.setText("\n".join(f"• {s[:40]}{'…' if len(s)>40 else ''}"
                                             for s in items))

    def _on_speak(self, sentence: str):
        self.box_speak.setText(sentence)

    def _on_done(self):
        self.box_speak.setText("—")

    def _on_action(self, name: str):
        self.action_lbl.setText(name.replace("_", " ") if name else "")

    def _on_cam_frame(self, jpg: bytes):
        img = QImage.fromData(jpg, "JPG")
        if img.isNull():
            return
        w = max(1, self.cam_lbl.width())
        h = max(1, self.cam_lbl.height())
        pix = QPixmap.fromImage(img).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cam_lbl.setPixmap(pix)

    def _open_admin(self):
        self.admin_console.show()
        self.admin_console.raise_()

    def _toggle_interrupt(self, checked: bool):
        if checked:
            auto_interrupt.set()
            self.interrupt_btn.setText("⚡ interrupt: on")
        else:
            auto_interrupt.clear()
            self.interrupt_btn.setText("⚡ interrupt: off")

    def _on_metrics(self, source: str, line: str):
        # Append quietly to timing label so it doesn't take up space
        self.timing_lbl.setText(line)

    def _render_frame(self):
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        h, w, ch = pixels.shape
        qimg = QImage(pixels.data, w, h, w * ch, QImage.Format_RGB888)
        self.sim_label.setPixmap(QPixmap.fromImage(qimg))

    def _set_status(self, status: str):
        color = STATUS_COLORS.get(status, "#444")
        label = STATUS_LABELS.get(status, status)
        self.status_dot.setStyleSheet(f"color: {color};")
        self.status_lbl.setStyleSheet(f"color: {color};")
        self.status_lbl.setText(label)

# ── Subprocess helpers ────────────────────────────────────────────────────────

def clear_queues():
    """Remove stale speech/sim queue files from previous runs."""
    for d in ["/tmp/glados_queue", "/tmp/glados_sim_queue"]:
        p = pathlib.Path(d)
        if p.exists():
            for f in p.glob("*"):
                f.unlink(missing_ok=True)
    print("[launch] cleared stale queue files")

def start_proc(cmd, name, stdout=None):
    print(f"[launch] {name}...")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=stdout,
        stderr=subprocess.STDOUT if stdout else None,
        env=env,
    )
    procs.append(p)
    return p


def _llama_reader(proc: subprocess.Popen, signals: "Signals"):
    """Parse llama-server stdout and emit LLM timing metrics."""
    import re
    pending: dict = {}
    pat_timing = re.compile(
        r"print_timing.*task\s+(\d+)\s+\|\s+(prompt eval|eval|total)\s+time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+tokens"
        r"(?:\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\))?"
    )
    pat_graphs = re.compile(r"print_timing.*task\s+(\d+)\s+\|\s+graphs reused\s*=\s*(\d+)")
    for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        print(line)
        if _admin_console is not None:
            _admin_console.push_log(line)
        m = pat_timing.search(line)
        if m:
            task, kind, ms, ntok, tps = m.group(1), m.group(2), float(m.group(3)), int(m.group(4)), m.group(5)
            pending.setdefault(task, {})
            if kind == "prompt eval":
                pending[task]["prefill_ms"]   = ms
                pending[task]["prefill_tok"]  = ntok
                pending[task]["prefill_tps"]  = float(tps) if tps else 0
            elif kind == "eval":
                pending[task]["gen_ms"]  = ms
                pending[task]["gen_tok"] = ntok
                pending[task]["gen_tps"] = float(tps) if tps else 0
            elif kind == "total":
                pending[task]["total_ms"]  = ms
                pending[task]["total_tok"] = ntok
            continue
        m = pat_graphs.search(line)
        if m:
            task, graphs = m.group(1), int(m.group(2))
            d = pending.pop(task, {})
            prefill = f"{d.get('prefill_tps',0):.0f} tok/s ({d.get('prefill_tok',0)} tok)"
            gen     = f"{d.get('gen_tps',0):.0f} tok/s ({d.get('gen_tok',0)} tok)"
            total   = f"{d.get('total_ms',0):.0f}ms"
            hit_pct = f"{graphs}/{d.get('prefill_tok',1)+d.get('gen_tok',1)} cached"
            signals.metrics.emit("llm", f"llm  prefill {prefill}  gen {gen}  {total}  [{hit_pct}]")


def _tts_reader(proc: subprocess.Popen, signals: "Signals"):
    """Parse glados_daemon stdout, drive pipeline bar, and emit TTS timing metrics."""
    import re
    pat_rtf  = re.compile(r"\[(\d+)ms to first audio.*RTF ([\d.]+)x\]")
    pat_wait = re.compile(r"queue_wait\s+([\d.]+)s.*tts_first_segment\s+([\d.]+)s\s*=\s*([\d.]+)s")
    pat_play = re.compile(r"playback\s+([\d.]+)s\s*\(([\d.]+)s of audio\)")
    buf: dict = {}
    for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        print(line)
        if _admin_console is not None:
            _admin_console.push_log(line)
        m = pat_rtf.search(line)
        if m:
            buf["first_ms"] = int(m.group(1))
            buf["rtf"]      = float(m.group(2))
            _currently_speaking.set()
            signals.status.emit("speaking")   # audio playback has begun
            # Pop the next queued sentence and show it in the SPEAKING box
            if _tts_queue:
                speaking = _tts_queue.popleft()
                signals.pipe_speak.emit(speaking)
                signals.pipe_queue.emit(list(_tts_queue))
            continue
        m = pat_wait.search(line)
        if m:
            buf["q_wait"] = float(m.group(1))
            buf["tts"]    = float(m.group(2))
            continue
        m = pat_play.search(line)
        if m:
            buf["play_s"]  = float(m.group(1))
            buf["audio_s"] = float(m.group(2))
            q    = buf.get("q_wait", 0)
            tts  = buf.get("tts",    0)
            play = buf.get("play_s", 0)
            rtf  = buf.get("rtf", 0)
            signals.metrics.emit(
                "tts",
                f"tts  q {q:.2f}s  synth {tts:.2f}s  play {play:.1f}s  RTF {rtf:.2f}x"
            )
            # Sentence finished playing
            _currently_speaking.clear()
            signals.pipe_done.emit()
            # after last utterance finishes playing, go back to listen
            # (approximate — playback timing line comes after playback ends)
            if not _tts_queue:
                signals.status.emit("listening")
            buf.clear()

def wait_for_llama(timeout=60):
    print("[launch] waiting for llama-server", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://localhost:8080/health", timeout=1)
            print(" ✓")
            return
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1)
    print("\n[launch] WARNING: llama-server not responding, continuing anyway")

def shutdown():
    # Stop PyAudio stream first — must happen before Python GC teardown.
    # Without this, the CoreAudio IO thread calls PyGILState_Ensure during
    # shutdown and crashes with SIGTRAP in _os_workgroup_tsd_cleanup.
    global listener
    if listener is not None:
        try:
            listener.stop()
        except Exception:
            pass
        listener = None
    vision.stop()
    for p in reversed(procs):
        try: p.terminate()
        except Exception: pass
    # Skip Python's GC finalization — avoids the gc_collect_main assertion
    # crash that fires when daemon threads are still alive at exit.
    os._exit(0)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global daemon_proc, _admin_console

    clear_queues()
    vision.start()
    llama_proc  = start_proc([LLAMA_CMD, "-m", str(GGUF), "--port", "8080"], "llama-server",
                             stdout=subprocess.PIPE)
    wait_for_llama()
    daemon_proc = start_proc(["python3", "speech/glados_daemon.py"], "glados_daemon",
                             stdout=subprocess.PIPE)
    time.sleep(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    signals = Signals()

    threading.Thread(target=_llama_reader, args=(llama_proc,  signals), daemon=True, name="llama-reader").start()
    threading.Thread(target=_tts_reader,   args=(daemon_proc, signals), daemon=True, name="tts-reader").start()

    threading.Thread(target=_brain_loop,    args=(signals,), daemon=True, name="brain").start()
    threading.Thread(target=_gesture_loop,  args=(signals,), daemon=True, name="gesture").start()
    threading.Thread(target=_cam_feed_loop,     args=(signals,), daemon=True, name="cam-feed").start()
    threading.Thread(target=_perception_loop,   args=(signals,), daemon=True, name="perception").start()
    threading.Thread(target=_vision_trigger_loop, args=(signals,), daemon=True, name="vision-trigger").start()

    global listener
    listener = SpeechListener(
        on_transcription=lambda text: on_transcription(text, signals),
        language="en",
        model_final="distil-small.en",
    )
    threading.Thread(target=listener.start, daemon=True, name="listener").start()
    signals.status.emit("listening")

    win = GladosWindow(signals)
    _admin_console = win.admin_console
    win.show()

    signal.signal(signal.SIGINT, lambda *_: (shutdown(), app.quit()))
    app.aboutToQuit.connect(shutdown)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
