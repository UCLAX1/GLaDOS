#!/usr/bin/env python3
"""
launch.py — start the full GLaDOS pipeline in one command.

Usage:
    python3 launch.py

Starts (in order):
  1. llama-server     — Bonsai-8B brain
  2. glados_daemon.py — TTS / speech playback
  3. sim_daemon.py    — persistent MuJoCo viewer  (skipped if GLADOS_HARDWARE=1)
  4. run_glados.py    — mic → STT → brain → speech + gestures (foreground)

Ctrl-C shuts everything down cleanly.

Config — edit these if your paths differ:
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent
GGUF      = ROOT / "central_ai/benchmarking_tools/Bonsai-8B-Q1_0.gguf"
LLAMA_CMD = "llama-server"   # must be on PATH; or set full path e.g. "/opt/homebrew/bin/llama-server"
MJPYTHON  = "mjpython"       # must be on PATH
VENV_PY   = ROOT / "listening/venv/bin/python3"

HARDWARE  = bool(os.environ.get("GLADOS_HARDWARE"))

# ── Helpers ───────────────────────────────────────────────────────────────────

procs: list[subprocess.Popen] = []

def start(cmd: list, name: str, cwd=ROOT) -> subprocess.Popen:
    print(f"[launch] starting {name}...")
    p = subprocess.Popen(cmd, cwd=str(cwd))
    procs.append(p)
    return p

def wait_for_llama(timeout=60):
    """Poll localhost:8080 until llama-server is ready."""
    print("[launch] waiting for llama-server...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://localhost:8080/health", timeout=1)
            print(" ready.")
            return
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1)
    print("\n[launch] WARNING: llama-server not responding after 60s — continuing anyway")

def shutdown(sig=None, frame=None):
    print("\n[launch] shutting down...")
    for p in reversed(procs):
        try:
            p.terminate()
        except Exception:
            pass
    for p in reversed(procs):
        try:
            p.wait(timeout=3)
        except Exception:
            p.kill()
    sys.exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ── Launch sequence ───────────────────────────────────────────────────────────

# 1 — LLM brain
start([LLAMA_CMD, "-m", str(GGUF), "--port", "8080"], "llama-server")
wait_for_llama()

# 2 — TTS / speech daemon
start(["python3", "speech/glados_daemon.py"], "glados_daemon")
time.sleep(1)   # give it a moment to initialize

# 3 — Sim viewer (skip on hardware)
if not HARDWARE:
    start([MJPYTHON, "sim/sim_daemon.py"], "sim_daemon")
    time.sleep(1)

# 4 — Main pipeline (foreground — blocks until Ctrl-C)
print("[launch] all services up — starting pipeline\n")
venv_py = str(VENV_PY) if VENV_PY.exists() else sys.executable
pipeline = subprocess.Popen([venv_py, "run_glados.py"], cwd=str(ROOT))
procs.append(pipeline)
pipeline.wait()

shutdown()
