#!/usr/bin/env python3
"""
run_glados.py — Full GLaDOS pipeline.

  Mic → [Listener] → text → [Brain/LLM] → JSON → [Speech daemon] + [Motor action]

Prerequisites (3 separate steps):

  1. Start llama.cpp server (Bonsai-8B brain):
       llama-server -m central_ai/benchmarking_tools/Bonsai-8B-Q1_0.gguf --port 8080

  2. Start speech daemon:
       python3 speech/glados_daemon.py

  3. Run this script (from X1_GLaDOS/ root, with listening venv active):
       source listening/venv/bin/activate
       python3 run_glados.py

Flow:
  SpeechListener  ──transcription──►  GladosBrain  ──speech──►  /tmp/glados_request.json
                                           │
                                           └──gesture──►  mjpython actions/run_action.py <gesture>
"""

import json
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from central_ai.brain import GladosBrain
from listening.listener import SpeechListener

# ── Config ─────────────────────────────────────────────────────────────────────
SPEECH_QUEUE = Path("/tmp/glados_request.json")

# Maps brain gesture names → action script names in actions/scripts/
GESTURE_MAP = {
    "head_tilt":       "curious_peek_horiz",
    "recoil":          "confused_scan",
    "slow_sweep":      "scan",
    "lean_in":         "curious_peek_vert",
    "dismissive_turn": "look_away",
    "idle":            None,
}

# ── Globals ────────────────────────────────────────────────────────────────────
brain      = GladosBrain()
brain_lock = threading.Lock()   # one LLM request at a time


# ── Dispatch helpers ───────────────────────────────────────────────────────────

def dispatch_speech(text: str, speed: float = 1.0):
    """Write speech request for glados_daemon.py to pick up."""
    SPEECH_QUEUE.write_text(json.dumps({"text": text, "speed": speed}))


def dispatch_gesture(gesture: str):
    """Run the matching motor action (non-blocking, fire-and-forget)."""
    action = GESTURE_MAP.get(gesture)
    if action:
        subprocess.Popen(
            ["mjpython", "actions/run_action.py", action, "--once"],
        )


# ── Main callback ──────────────────────────────────────────────────────────────

def on_transcription(text: str):
    """Called by SpeechListener after each complete utterance."""
    print(f"\n[you]    {text}")

    with brain_lock:
        response = brain.respond(text)

    if response is None:
        print("[brain]  (no response — is the llama.cpp server running?)")
        return

    speech  = response.get("speech", "")
    gesture = response.get("gesture", "idle")
    look_at = response.get("look_at", "speaker")
    mood    = response.get("mood", {})

    print(f"[glados] {speech}")
    print(f"         gesture={gesture}  look_at={look_at}  mood={mood}")

    if speech:
        dispatch_speech(speech)

    dispatch_gesture(gesture)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  GLaDOS Pipeline")
    print("  Brain:   Bonsai-8B @ localhost:8080")
    print("  Speech:  glados_daemon  → /tmp/glados_request.json")
    print("  Mic:     default input device")
    print("=" * 60)
    print()

    listener = SpeechListener(
        on_transcription=on_transcription,
        language="en",
        model_final="distil-small.en",   # faster on CPU; swap for distil-large-v3 on Jetson
    )
    listener.start()   # blocks until Ctrl-C


if __name__ == "__main__":
    main()
