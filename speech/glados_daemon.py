#!/usr/bin/env python3
"""
glados_daemon.py — GLaDOS speech daemon for the X1 robot.

Loads the TTS model once at startup, then watches for speech requests
and plays them immediately with minimal latency.

Usage:
    python3 speech/glados_daemon.py

Interface (from any other process):
    Write JSON to /tmp/glados_request.json:
        {"text": "The cake is a lie."}
        {"text": "Firing neurotoxin.", "speed": 0.9}

    The daemon consumes the file, speaks, then goes back to waiting.
    For queued speech, write files sequentially — the daemon processes
    one at a time in the order it sees them.

Platform:
    Mac:    uses afplay
    Jetson: uses aplay (ALSA)
    Set AUDIO_PLAYER env var to override: AUDIO_PLAYER=aplay python3 glados_daemon.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

# Allow running from repo root: python3 speech/glados_daemon.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from glados_tts import GladosTTS

# ── config ─────────────────────────────────────────────────────────────────
QUEUE_FILE   = Path(os.environ.get("GLADOS_QUEUE", "/tmp/glados_request.json"))
POLL_INTERVAL = 0.02   # seconds between queue checks

# Auto-detect audio player
if "AUDIO_PLAYER" in os.environ:
    AUDIO_PLAYER = os.environ["AUDIO_PLAYER"].split()
elif sys.platform == "darwin":
    AUDIO_PLAYER = ["afplay"]
else:
    AUDIO_PLAYER = ["aplay"]


def play(audio, sample_rate: int = 24000) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        sf.write(tmp, audio.numpy(), sample_rate)
        subprocess.run(AUDIO_PLAYER + [tmp], check=True, capture_output=True)
    finally:
        Path(tmp).unlink(missing_ok=True)


def main():
    print("Loading GLaDOS TTS model…")
    tts = GladosTTS()
    print(f"GLaDOS daemon ready. Watching {QUEUE_FILE}")
    print(f"Audio player: {' '.join(AUDIO_PLAYER)}")
    print()

    while True:
        if QUEUE_FILE.exists():
            try:
                raw  = QUEUE_FILE.read_text()
                QUEUE_FILE.unlink()   # consume immediately so nothing re-queues it
                data  = json.loads(raw)
                text  = data.get("text", "").strip()
                speed = float(data.get("speed", 1.0))
                if text:
                    t0    = time.time()
                    audio = tts.speak(text, speed=speed)
                    gen_t = time.time() - t0
                    dur   = len(audio) / 24000
                    print(f"[{gen_t*1000:.0f}ms gen, {dur:.1f}s audio, RTF {gen_t/dur:.2f}x] {text[:60]}")
                    play(audio)
            except json.JSONDecodeError as e:
                print(f"bad JSON in queue: {e}")
            except Exception as e:
                print(f"error: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
