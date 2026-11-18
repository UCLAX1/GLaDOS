#!/usr/bin/env python3
"""
glados_daemon.py — GLaDOS speech daemon for the X1 robot.

Loads the TTS model once at startup, then watches the speech queue and speaks
each request with minimal latency.

Usage:
    python3 speech/glados_daemon.py

Interface (from any other process):
    from speech.speech_queue import enqueue, clear
    enqueue("The cake is a lie.")
    clear()          # barge-in: drop the queue AND stop what is playing now

Latency design
--------------
Two threads, so synthesis never waits for playback:

    synth thread   queue file → Kokoro → audio segments ─┐
                                                         ▼  (bounded buffer)
    player thread  ◄──────────────────────────── plays segments back to back

While sentence 1 is playing, sentence 2 is already being synthesised, so
sentences run together without a gap and the first sound starts as soon as the
first segment exists.

Cancel (speech_queue.clear) is honoured within ~50 ms: audio is written in
small blocks and the player checks for a cancel between blocks.

The output stream is opened fresh for each burst of speech, so it always uses
the *current* system default device (AirPods, speakers, …).

Machine-readable events on stdout (one per line, for the GUI):
    [speak-start] {"text": ..., "wait": seconds from request to first sound}
    [speak-end]   {"text": ..., "audio": seconds, "synth_first": seconds, "rtf": x}
    [cancelled]   {"dropped": n}

Platform:
    Uses sounddevice (PortAudio). Falls back to afplay (Mac) / aplay (Linux),
    or set AUDIO_PLAYER to override the fallback command.
"""

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# Allow running from repo root: python3 speech/glados_daemon.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from speech_queue import QUEUE_DIR, LEGACY_FILE, next_request, take_cancel

# ── config ─────────────────────────────────────────────────────────────────
POLL_INTERVAL   = 0.01     # seconds between queue checks
BLOCK_FRAMES    = 1200     # 50 ms at 24 kHz — cancel granularity during playback
BUFFER_SEGMENTS = 6        # how far synthesis may run ahead of playback
STREAM_IDLE_S   = 0.6      # close the output stream after this much silence

if "AUDIO_PLAYER" in os.environ:
    AUDIO_PLAYER = os.environ["AUDIO_PLAYER"].split()
elif sys.platform == "darwin":
    AUDIO_PLAYER = ["afplay"]
else:
    AUDIO_PLAYER = ["aplay"]


def emit(tag: str, **payload) -> None:
    print(f"[{tag}] {json.dumps(payload, ensure_ascii=False)}", flush=True)


# ── shared cancel state ────────────────────────────────────────────────────

class CancelState:
    """Everything stamped before `before` has been cancelled."""

    def __init__(self):
        self.before = 0.0
        self._lock  = threading.Lock()

    def poll(self) -> bool:
        """Pick up a cancel marker if one appeared. True if a new one was seen."""
        t = take_cancel()
        if t is None:
            return False
        with self._lock:
            self.before = max(self.before, t)
        return True

    def is_cancelled(self, t_request: float) -> bool:
        return t_request < self.before


# ── player ─────────────────────────────────────────────────────────────────

class Player(threading.Thread):
    """
    Plays (kind, t_request, payload) items from a queue:
        ("start", t, {"text": ...})
        ("audio", t, float32 ndarray)
        ("end",   t, {...stats})
    """

    def __init__(self, sample_rate: int, cancel: CancelState):
        super().__init__(daemon=True, name="player")
        self.sample_rate = sample_rate
        self.cancel      = cancel
        self.q: queue.Queue = queue.Queue(maxsize=BUFFER_SEGMENTS)
        self._stream     = None
        self._sd         = None
        try:
            import sounddevice as sd
            self._sd = sd
            self.backend = "sounddevice (pipelined, follows system default device)"
        except Exception as e:
            self.backend = f"{' '.join(AUDIO_PLAYER)} (sounddevice unavailable: {e})"

    # -- stream management --------------------------------------------------
    def _open(self):
        if self._stream is None and self._sd is not None:
            try:
                self._stream = self._sd.OutputStream(
                    samplerate=self.sample_rate, channels=1, dtype="float32")
                self._stream.start()
            except Exception as e:
                print(f"[player] could not open output stream: {e}", flush=True)
                self._stream = None
                self._sd = None          # fall back to subprocess playback

    def _close(self, abort: bool = False):
        if self._stream is not None:
            try:
                self._stream.abort() if abort else self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _play(self, audio: np.ndarray, t_request: float) -> bool:
        """Play one segment. Returns False if cancelled part-way."""
        self._open()
        if self._stream is None:
            return self._play_subprocess(audio, t_request)
        for i in range(0, len(audio), BLOCK_FRAMES):
            self.cancel.poll()
            if self.cancel.is_cancelled(t_request):
                self._close(abort=True)      # drop what is still buffered
                return False
            self._stream.write(audio[i:i + BLOCK_FRAMES])
        return True

    def _play_subprocess(self, audio: np.ndarray, t_request: float) -> bool:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, audio, self.sample_rate)
            proc = subprocess.Popen(AUDIO_PLAYER + [tmp],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while proc.poll() is None:
                self.cancel.poll()
                if self.cancel.is_cancelled(t_request):
                    proc.terminate()
                    return False
                time.sleep(0.02)
            return True
        finally:
            Path(tmp).unlink(missing_ok=True)

    # -- main loop ----------------------------------------------------------
    def run(self):
        started = None      # (t_request, text, t_sound_start) of the utterance in progress
        while True:
            try:
                kind, t_req, payload = self.q.get(timeout=STREAM_IDLE_S)
            except queue.Empty:
                self._close()            # idle: release the device
                continue

            if self.cancel.is_cancelled(t_req):
                if started and started[0] == t_req:
                    started = None
                continue

            if kind == "start":
                started = None
                self._pending_start = payload
            elif kind == "audio":
                if started is None:
                    text = getattr(self, "_pending_start", {}).get("text", "")
                    started = (t_req, text, time.time())
                    emit("speak-start", text=text, wait=round(time.time() - t_req, 3))
                if not self._play(payload, t_req):
                    emit("cancelled", dropped=1)
                    started = None
            elif kind == "end":
                if started is not None:
                    emit("speak-end", text=started[1], **payload)
                started = None


# ── synthesis loop ─────────────────────────────────────────────────────────

def main():
    print("Loading GLaDOS TTS model…", flush=True)
    from glados_tts import GladosTTS, SAMPLE_RATE
    tts    = GladosTTS()
    cancel = CancelState()
    player = Player(SAMPLE_RATE, cancel)
    player.start()
    print(f"GLaDOS daemon ready. Watching {QUEUE_DIR}/  (and {LEGACY_FILE})", flush=True)
    print(f"Audio output: {player.backend}\n", flush=True)

    def put(item) -> bool:
        """Blocking put that gives up if the item gets cancelled while waiting."""
        while True:
            cancel.poll()
            if cancel.is_cancelled(item[1]):
                return False
            try:
                player.q.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue

    try:
        while True:
            if cancel.poll():
                # Drop anything already synthesised but not yet played
                dropped = 0
                try:
                    while True:
                        player.q.get_nowait()
                        dropped += 1
                except queue.Empty:
                    pass
                if dropped:
                    emit("cancelled", dropped=dropped)

            req = next_request()
            if req is None:
                time.sleep(POLL_INTERVAL)
                continue

            text  = (req.get("text") or "").strip()
            speed = float(req.get("speed", 1.0))
            t_req = float(req.get("t_request") or time.time())
            if not text or cancel.is_cancelled(t_req):
                continue

            try:
                t0, first, samples, ok = time.time(), None, 0, True
                if not put(("start", t_req, {"text": text})):
                    continue
                for audio in tts.speak_stream(text, speed=speed):
                    if first is None:
                        first = time.time() - t0
                    seg = np.ascontiguousarray(audio.cpu().numpy(), dtype=np.float32)
                    samples += len(seg)
                    if not put(("audio", t_req, seg)):
                        ok = False
                        break
                if ok and first is not None:
                    dur = samples / SAMPLE_RATE
                    put(("end", t_req, {
                        "audio":       round(dur, 2),
                        "synth_first": round(first, 3),
                        "rtf":         round((time.time() - t0) / max(dur, 1e-6), 2),
                    }))
            except Exception as e:
                print(f"error: {e}", flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
