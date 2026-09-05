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
       source listening/.venv/bin/activate
       python3 run_glados.py

Flow (streamed end to end — each stage starts before the previous finishes):
  SpeechListener ──transcript──► GladosBrain ──sentence──► /tmp/glados_queue/
    (transcribes                  (yields each             (daemon speaks each
     during the VAD gap)           sentence as it            as it lands)
                                   is generated)
                                        │
                                        └──gesture──►  run_action.py <gesture>
                                           (mjpython, or python3 if GLADOS_HARDWARE=1)
"""

import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from central_ai.brain import GladosBrain
from listening.listener import SpeechListener
from speech.speech_queue import enqueue

# ── Config ─────────────────────────────────────────────────────────────────────
# Gestures run in the simulator by default. Set GLADOS_HARDWARE=1 to drive the
# real motors instead — run_action.py switches backend on the same variable.
HARDWARE = bool(os.environ.get("GLADOS_HARDWARE"))
ACTION_CMD = ["python3"] if HARDWARE else ["mjpython"]

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
listener   = None               # set in main(); read for per-utterance timings
turns: list[dict] = []          # per-turn stage timings, summarised on exit


# ── Dispatch helpers ───────────────────────────────────────────────────────────

def dispatch_speech(text: str, speed: float = 1.0):
    """Queue one utterance for glados_daemon.py.

    Appends rather than overwrites — a streamed reply arrives as several
    sentences, and the old single-file interface dropped every one after the
    first. The queued payload carries a t_request stamp so the daemon can
    report its own wait, closing the last gap in the latency picture.
    """
    enqueue(text, speed=speed)


def dispatch_gesture(gesture: str):
    """Run the matching motor action (non-blocking, fire-and-forget)."""
    action = GESTURE_MAP.get(gesture)
    if not action:
        return   # unmapped or "idle" — nothing to move
    try:
        subprocess.Popen(
            ACTION_CMD + ["actions/run_action.py", action, "--once"],
        )   # child inherits GLADOS_HARDWARE from os.environ
    except FileNotFoundError:
        # Missing interpreter must not kill the listener thread (see on_transcription).
        print(f"[gesture] {ACTION_CMD[0]} not found — skipped {action!r}. "
              f"Install mujoco in this venv: pip install mujoco")


# ── Main callback ──────────────────────────────────────────────────────────────

def on_transcription(text: str):
    """Called by SpeechListener after each complete utterance."""
    stt_timing = dict(listener.last_timing) if listener else {}
    print(f"\n[you]    {text}")

    # Queue each sentence the moment it exists. "speech" is the first field in
    # the schema, so GLaDOS starts talking while the model is still writing
    # gesture, look_at and mood.
    t0        = time.time()
    llm_first = None
    dispatch  = 0.0
    sentences = 0

    with brain_lock:
        for sentence in brain.respond_stream(text):
            t_disp = time.time()
            if llm_first is None:
                llm_first = t_disp - t0
            dispatch_speech(sentence)
            dispatch += time.time() - t_disp
            sentences += 1
            print(f"[glados] {sentence}")

    llm_total = time.time() - t0

    if llm_first is None:
        print("[brain]  (no speech produced — check the llama.cpp server)")
        return

    # gesture/look_at/mood only exist once the whole object has arrived.
    response = brain.last_response or {}
    gesture  = response.get("gesture", "idle")
    print(f"         gesture={gesture}  look_at={response.get('look_at', 'speaker')}  "
          f"mood={response.get('mood', {})}")

    t1 = time.time()
    dispatch_gesture(gesture)
    dispatch += time.time() - t1

    log_turn(stt_timing, llm_first, llm_total, dispatch, sentences)


# ── Latency accounting ─────────────────────────────────────────────────────────

def log_turn(stt_timing: dict, llm_first: float, llm_total: float,
             dispatch: float, sentences: int):
    """Record and print the delay breakdown for one turn.

    The headline number is time-to-first-speech: the user stops talking → the
    first sentence is queued for the daemon. Generating the rest of the reply
    overlaps with speaking it, so llm_total is reported alongside but is not
    part of the perceived delay.
    """
    t = {
        "vad_gap":   stt_timing.get("vad_gap", 0.0),
        "q_wait":    stt_timing.get("q_wait", 0.0),
        "stt":       stt_timing.get("stt", 0.0),
        "llm_first": llm_first,
        "dispatch":  dispatch,
    }
    t["total"]     = sum(t.values())
    t["llm_total"] = llm_total
    t["overlap"]   = max(0.0, llm_total - llm_first)
    turns.append(t)

    spec = "*" if stt_timing.get("speculative") else " "
    print(
        f"         [timing] vad_gap {t['vad_gap']:.2f} → q_wait {t['q_wait']:.2f} → "
        f"stt {t['stt']:.2f}{spec} → llm_1st {t['llm_first']:.2f}"
        f"  =  {t['total']:.2f}s to first speech"
    )
    print(
        f"                  (full generation {llm_total:.2f}s over {sentences} "
        f"sentence(s) — {t['overlap']:.2f}s of it overlapped with speaking)"
    )


def print_summary():
    """Per-stage distribution across the session. Printed on exit."""
    if not turns:
        return
    stages = ["vad_gap", "q_wait", "stt", "llm_first", "dispatch", "total"]
    width  = 68

    print()
    print("=" * width)
    print(f"  Latency to first speech over {len(turns)} turn(s)")
    print("=" * width)
    print(f"  {'stage':<11}{'mean':>8}{'median':>9}{'min':>8}{'max':>8}{'share':>9}")
    print("  " + "-" * (width - 4))

    mean_total = statistics.fmean(x["total"] for x in turns)
    for stage in stages:
        vals  = [x[stage] for x in turns]
        mean  = statistics.fmean(vals)
        share = "" if stage == "total" else f"{mean / mean_total * 100:5.1f}%"
        label = "TOTAL" if stage == "total" else stage
        print(f"  {label:<11}{mean:>7.2f}s{statistics.median(vals):>8.2f}s"
              f"{min(vals):>7.2f}s{max(vals):>7.2f}s{share:>9}")

    print("  " + "-" * (width - 4))
    spec = sum(1 for x in turns if x["stt"] < 0.01)
    over = statistics.fmean(x["overlap"] for x in turns)
    full = statistics.fmean(x["llm_total"] for x in turns)
    print(f"  Full generation averaged {full:.2f}s, of which {over:.2f}s ran")
    print(f"  while GLaDOS was already speaking.")
    print(f"  Transcription came free from the VAD gap on {spec}/{len(turns)} turn(s).")
    print("  Excludes TTS and playback — see the glados_daemon.py terminal.")
    print("=" * width)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  GLaDOS Pipeline")
    print("  Brain:   Bonsai-8B @ localhost:8080")
    print("  Speech:  glados_daemon  → /tmp/glados_request.json")
    print("  Mic:     default input device")
    print("=" * 60)
    print()

    global listener
    listener = SpeechListener(
        on_transcription=on_transcription,
        language="en",
        model_final="distil-small.en",   # faster on CPU; swap for distil-large-v3 on Jetson
    )
    listener.start()   # blocks until Ctrl-C
    print_summary()


if __name__ == "__main__":
    main()
