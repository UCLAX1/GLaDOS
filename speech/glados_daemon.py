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

import numpy as np
import soundfile as sf

# Allow running from repo root: python3 speech/glados_daemon.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from glados_tts import GladosTTS, SAMPLE_RATE
from speech_queue import QUEUE_DIR, LEGACY_FILE, next_request

# ── config ─────────────────────────────────────────────────────────────────
POLL_INTERVAL = 0.02   # seconds between queue checks

# Auto-detect audio player
if "AUDIO_PLAYER" in os.environ:
    AUDIO_PLAYER = os.environ["AUDIO_PLAYER"].split()
elif sys.platform == "darwin":
    AUDIO_PLAYER = ["afplay"]
else:
    AUDIO_PLAYER = ["aplay"]


def play_via_subprocess(audio, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a temp wav and hand it to afplay/aplay. One process per segment."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        sf.write(tmp, audio.numpy(), sample_rate)
        subprocess.run(AUDIO_PLAYER + [tmp], check=True, capture_output=True)
    finally:
        Path(tmp).unlink(missing_ok=True)


class Player:
    """
    Audio sink for streamed segments.

    Uses sounddevice on macOS so playback always follows the current system
    default output device (AirPods, speakers, etc.) — no restart needed when
    you switch audio devices. Falls back to afplay/aplay subprocess if
    sounddevice is unavailable.
    """

    def __init__(self):
        self._sd = None
        try:
            import sounddevice as _sd
            self._sd = _sd
            self.backend = "sounddevice (follows system default)"
        except Exception as e:
            self.backend = f"{' '.join(AUDIO_PLAYER)} (sounddevice unavailable: {e})"

    def play(self, audio) -> None:
        data = np.ascontiguousarray(audio.numpy(), dtype=np.float32)
        if self._sd is not None:
            try:
                self._sd.play(data, samplerate=SAMPLE_RATE, blocking=True)
                return
            except Exception:
                pass   # fall through to subprocess
        play_via_subprocess(audio)

    def close(self) -> None:
        if self._sd is not None:
            try:
                self._sd.stop()
            except Exception:
                pass


def speak_request(tts: GladosTTS, player: Player, req: dict) -> None:
    """Synthesise and play one utterance, streaming segment by segment."""
    text  = req.get("text", "").strip()
    speed = float(req.get("speed", 1.0))
    if not text:
        return

    # Stamped by the pipeline so we can report the wait this process added.
    t_req = req.get("t_request")
    t0    = time.time()

    first_gen = None          # generation time for the opening segment
    t_first   = None          # when sound actually started
    samples   = 0

    for audio in tts.speak_stream(text, speed=speed):
        if first_gen is None:
            first_gen = time.time() - t0
            t_first   = time.time()
        samples += len(audio)
        player.play(audio)

    if first_gen is None:
        return                # nothing synthesised (empty after normalisation)

    total = time.time() - t0
    dur   = samples / SAMPLE_RATE
    print(f"[{first_gen*1000:.0f}ms to first audio, {dur:.1f}s audio, "
          f"RTF {total/dur:.2f}x] {text[:60]}")
    if t_req is not None:
        q_wait = t0 - float(t_req)
        print(f"  [timing] queue_wait {q_wait:.2f}s → tts_first_segment {first_gen:.2f}s"
              f"  =  {q_wait + first_gen:.2f}s from request to first audio")
    print(f"  [timing] playback {time.time() - t_first:.2f}s ({dur:.1f}s of audio)")


def main():
    print("Loading GLaDOS TTS model…")
    tts    = GladosTTS()
    player = Player()
    print(f"GLaDOS daemon ready. Watching {QUEUE_DIR}/  (and {LEGACY_FILE})")
    print(f"Audio output: {player.backend}")
    print()

    try:
        while True:
            req = next_request()
            if req is None:
                time.sleep(POLL_INTERVAL)
                continue
            try:
                speak_request(tts, player, req)
            except Exception as e:
                print(f"error: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        player.close()


if __name__ == "__main__":
    main()
