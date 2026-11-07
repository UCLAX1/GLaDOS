"""
demo_gui.py  —  GLaDOS Vision System launcher

A tkinter control panel to launch/stop each vision module.
Run from the GLaDOS/ root directory:
    python3 vision/demo_gui.py

Each module runs as a subprocess so the GUI stays responsive.
Press the button again to stop a running module.
"""

import tkinter as tk
from tkinter import font as tkfont
import subprocess
import sys
import os
import time
import threading

# ── module definitions ────────────────────────────────────────────────────────

MODULES = [
    {
        "id":    "face_memory",
        "label": "Face Recognition",
        "desc":  "Identify people via InsightFace embeddings",
        "file":  "vision/face_memory.py",
        "key":   "F",
    },
    {
        "id":    "expression",
        "label": "Expression Detection",
        "desc":  "8-class emotion recognition via HSEmotion",
        "file":  "vision/expression2.py",
        "key":   "E",
    },
    {
        "id":    "gesture",
        "label": "Gesture Detection",
        "desc":  "Hand + body gestures with confidence scores",
        "file":  "vision/gestures_revised.py",
        "key":   "G",
    },
    {
        "id":    "action",
        "label": "Action Recognition",
        "desc":  "Walking, sitting, approaching via pose velocity",
        "file":  "vision/actions.py",
        "key":   "A",
    },
    {
        "id":    "zed",
        "label": "Object Detection",
        "desc":  "YOLO + face recognition on ZED camera depth feed",
        "file":  "vision/object_detection.py",
        "key":   "Z",
    },
]

# ── colors + fonts ────────────────────────────────────────────────────────────

BG_BASE    = "#0a0c0f"
BG_SURFACE = "#131920"
BG_CARD    = "#1a2130"
BG_HOVER   = "#1f2a3a"

ORANGE     = "#f97316"
ORANGE_DIM = "#7c3910"
GREEN      = "#22c55e"
GREEN_DIM  = "#14532d"
RED        = "#ef4444"
MUTED      = "#4a5568"
TEXT_PRI   = "#e2e8f0"
TEXT_SEC   = "#64748b"
TEXT_MONO  = "#94a3b8"

FONT_FAMILY = "JetBrains Mono"
FONT_FALLBACK = "Courier New"


def mono(size, weight="normal"):
    return (FONT_FAMILY, size, weight)


# ── process manager ───────────────────────────────────────────────────────────

class ProcessManager:
    def __init__(self):
        self._procs: dict[str, subprocess.Popen] = {}

    def start(self, module_id: str, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        proc = subprocess.Popen(
            [sys.executable, filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._procs[module_id] = proc
        return True

    def stop(self, module_id: str):
        proc = self._procs.pop(module_id, None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def is_running(self, module_id: str) -> bool:
        proc = self._procs.get(module_id)
        if proc is None:
            return False
        if proc.poll() is not None:
            self._procs.pop(module_id, None)
            return False
        return True

    def stop_all(self):
        for mid in list(self._procs.keys()):
            self.stop(mid)


pm = ProcessManager()


# ── main window ───────────────────────────────────────────────────────────────

class GLaDOSDemo(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("GLaDOS — Vision System")
        self.configure(bg=BG_BASE)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._buttons: dict[str, tk.Button]    = {}
        self._status:  dict[str, tk.StringVar] = {}
        self._dots:    dict[str, tk.Label]      = {}

        self._build_ui()
        self._tick()   # start polling loop

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── header ────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=BG_BASE, pady=0)
        header.pack(fill="x", padx=0, pady=0)

        tk.Frame(header, bg=ORANGE, height=2).pack(fill="x")

        inner_header = tk.Frame(header, bg=BG_BASE)
        inner_header.pack(fill="x", padx=28, pady=(18, 4))

        tk.Label(
            inner_header,
            text="GLADOS",
            font=(FONT_FAMILY, 28, "bold"),
            fg=ORANGE, bg=BG_BASE,
            anchor="w",
        ).pack(side="left")

        tk.Label(
            inner_header,
            text="VISION CONTROL",
            font=(FONT_FAMILY, 10, "normal"),
            fg=TEXT_SEC, bg=BG_BASE,
            anchor="sw",
            pady=10,
        ).pack(side="left", padx=(10, 0))

        tk.Label(
            inner_header,
            text="X1 ROBOTICS  //  UCLA",
            font=(FONT_FAMILY, 9),
            fg=MUTED, bg=BG_BASE,
        ).pack(side="right")

        tk.Frame(header, bg=BG_SURFACE, height=1).pack(fill="x")

        # ── module list ───────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG_BASE)
        body.pack(fill="both", expand=True, padx=20, pady=(16, 0))

        for i, mod in enumerate(MODULES):
            self._build_module_row(body, mod, i)

        # ── footer ────────────────────────────────────────────────────────
        footer = tk.Frame(self, bg=BG_BASE, pady=14)
        footer.pack(fill="x", padx=28)

        tk.Frame(self, bg=BG_SURFACE, height=1).pack(fill="x")

        stop_all_btn = tk.Button(
            footer,
            text="STOP ALL",
            font=(FONT_FAMILY, 9, "bold"),
            fg=RED, bg=BG_SURFACE,
            activeforeground=RED,
            activebackground=BG_HOVER,
            relief="flat", bd=0,
            padx=16, pady=6,
            cursor="hand2",
            command=self._stop_all,
        )
        stop_all_btn.pack(side="right", pady=(14, 0))

        tk.Label(
            footer,
            text="modules run as subprocesses — close their window to stop",
            font=(FONT_FAMILY, 8),
            fg=MUTED, bg=BG_BASE,
        ).pack(side="left", pady=(14, 0))

    def _build_module_row(self, parent, mod: dict, idx: int):
        mid = mod["id"]

        # Card frame
        card = tk.Frame(parent, bg=BG_CARD, pady=0)
        card.pack(fill="x", pady=(0, 6))

        # Left accent bar (lights up when running)
        accent = tk.Frame(card, bg=MUTED, width=3)
        accent.pack(side="left", fill="y")
        self._dots[mid] = accent   # reuse as accent bar

        inner = tk.Frame(card, bg=BG_CARD, padx=16, pady=12)
        inner.pack(side="left", fill="both", expand=True)

        # Top row: label + key hint + button
        top = tk.Frame(inner, bg=BG_CARD)
        top.pack(fill="x")

        tk.Label(
            top,
            text=mod["label"].upper(),
            font=(FONT_FAMILY, 11, "bold"),
            fg=TEXT_PRI, bg=BG_CARD,
            anchor="w",
        ).pack(side="left")

        tk.Label(
            top,
            text=f"[{mod['key']}]",
            font=(FONT_FAMILY, 9),
            fg=MUTED, bg=BG_CARD,
        ).pack(side="left", padx=(8, 0))

        btn = tk.Button(
            top,
            text="LAUNCH",
            font=(FONT_FAMILY, 9, "bold"),
            fg=ORANGE, bg=BG_SURFACE,
            activeforeground=BG_BASE,
            activebackground=ORANGE,
            relief="flat", bd=0,
            padx=14, pady=4,
            cursor="hand2",
            command=lambda m=mod: self._toggle(m),
        )
        btn.pack(side="right")
        self._buttons[mid] = btn

        # Description + status
        bottom = tk.Frame(inner, bg=BG_CARD)
        bottom.pack(fill="x", pady=(4, 0))

        tk.Label(
            bottom,
            text=mod["desc"],
            font=(FONT_FAMILY, 9),
            fg=TEXT_SEC, bg=BG_CARD,
            anchor="w",
        ).pack(side="left")

        status_var = tk.StringVar(value="idle")
        self._status[mid] = status_var

        tk.Label(
            bottom,
            textvariable=status_var,
            font=(FONT_FAMILY, 9),
            fg=MUTED, bg=BG_CARD,
            anchor="e",
        ).pack(side="right")

        # Hover effects
        for widget in [card, inner, top, bottom]:
            widget.bind("<Enter>", lambda e, c=card, i=inner, t=top, b=bottom:
                        self._hover_on(c, i, t, b))
            widget.bind("<Leave>", lambda e, c=card, i=inner, t=top, b=bottom:
                        self._hover_off(c, i, t, b))

    def _hover_on(self, *frames):
        for f in frames:
            try:
                f.configure(bg=BG_HOVER)
            except Exception:
                pass

    def _hover_off(self, *frames):
        for f in frames:
            try:
                f.configure(bg=BG_CARD)
            except Exception:
                pass

    # ── actions ───────────────────────────────────────────────────────────────

    def _toggle(self, mod: dict):
        mid = mod["id"]
        if pm.is_running(mid):
            pm.stop(mid)
        else:
            ok = pm.start(mid, mod["file"])
            if not ok:
                self._status[mid].set(f"file not found: {mod['file']}")

    def _stop_all(self):
        pm.stop_all()

    # ── polling loop ──────────────────────────────────────────────────────────

    def _tick(self):
        for mod in MODULES:
            mid     = mod["id"]
            running = pm.is_running(mid)
            btn     = self._buttons[mid]
            accent  = self._dots[mid]

            if running:
                btn.configure(text="STOP", fg=RED, bg=BG_SURFACE,
                              activeforeground=BG_BASE, activebackground=RED)
                accent.configure(bg=GREEN)
                self._status[mid].set("● running")
            else:
                btn.configure(text="LAUNCH", fg=ORANGE, bg=BG_SURFACE,
                              activeforeground=BG_BASE, activebackground=ORANGE)
                accent.configure(bg=MUTED)
                if "not found" not in self._status[mid].get():
                    self._status[mid].set("idle")

        self.after(500, self._tick)

    def _on_close(self):
        pm.stop_all()
        self.destroy()


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = GLaDOSDemo()

    # Center window on screen
    app.update_idletasks()
    w = app.winfo_reqwidth()
    h = app.winfo_reqheight()
    x = (app.winfo_screenwidth()  - w) // 2
    y = (app.winfo_screenheight() - h) // 2
    app.geometry(f"+{x}+{y}")

    app.mainloop()