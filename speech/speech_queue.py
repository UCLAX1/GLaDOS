#!/usr/bin/env python3
"""
speech_queue.py — filesystem queue between the pipeline and the speech daemon.

One JSON file per utterance, named so that lexical order is chronological
order. Writes are atomic (write a temp name, then rename), so the daemon can
never read a half-written file.

This replaces the single overwritable /tmp/glados_request.json, which lost an
utterance whenever a second one was written during playback — unavoidable once
the brain streams a reply sentence by sentence.

Deliberately dependency-free: run_glados.py imports this without pulling in
torch and kokoro via glados_tts.

    from speech.speech_queue import enqueue
    enqueue("The cake is a lie.", speed=0.95)
"""

import itertools
import json
import os
import tempfile
import time
from pathlib import Path

# /tmp on Mac/Linux (as documented); the user's temp folder on Windows.
_TMP = Path("/tmp") if os.name != "nt" else Path(tempfile.gettempdir())

QUEUE_DIR = Path(os.environ.get("GLADOS_QUEUE_DIR", str(_TMP / "glados_queue")))

# Single-shot interface documented in speech/README.md. Still honoured so
# existing callers and the one-liner in the docs keep working.
LEGACY_FILE = Path(os.environ.get("GLADOS_QUEUE", str(_TMP / "glados_request.json")))


def _unlink(f: Path) -> None:
    """Delete, ignoring 'already gone' and (Windows) 'still open in the other process'."""
    try:
        f.unlink()
    except OSError:
        pass

_seq = itertools.count()


def enqueue(text: str, speed: float = 1.0, queue_dir: Path = QUEUE_DIR) -> Path:
    """Append an utterance. Returns the queued file path."""
    queue_dir.mkdir(parents=True, exist_ok=True)
    payload = {"text": text, "speed": speed, "t_request": time.time()}
    # timestamp orders it; pid+counter keeps concurrent writers from colliding
    name = f"{time.time():.6f}-{os.getpid()}-{next(_seq):04d}.json"
    tmp  = queue_dir / (name + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, queue_dir / name)   # atomic within one filesystem (also on Windows)
    return queue_dir / name


def next_request(queue_dir: Path = QUEUE_DIR,
                 legacy_file: Path = LEGACY_FILE) -> dict | None:
    """Oldest pending request, or None. Consumes whatever it returns."""
    if queue_dir.is_dir():
        for f in sorted(p for p in queue_dir.iterdir() if p.suffix == ".json"):
            try:
                data = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                _unlink(f)                    # corrupt or vanished — drop it
                continue
            _unlink(f)
            return data

    if legacy_file.exists():
        try:
            raw = legacy_file.read_text()
            _unlink(legacy_file)
            return json.loads(raw)
        except (OSError, json.JSONDecodeError):
            _unlink(legacy_file)

    return None


CANCEL_NAME = "cancel.marker"   # not .json, so next_request() never returns it


def clear(queue_dir: Path = QUEUE_DIR) -> int:
    """
    Drop everything pending (e.g. on barge-in) AND tell the daemon to stop what
    it is already synthesising / playing. Returns how many queued files were dropped.

    The marker carries a timestamp: the daemon discards every request stamped
    earlier than it, so sentences enqueued right after clear() are unaffected.
    """
    queue_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in queue_dir.iterdir():
        _unlink(f)
        n += 1
    tmp = queue_dir / (CANCEL_NAME + ".tmp")
    tmp.write_text(f"{time.time():.6f}")
    os.replace(tmp, queue_dir / CANCEL_NAME)
    return n


def take_cancel(queue_dir: Path = QUEUE_DIR) -> float | None:
    """Daemon side: consume the cancel marker. Returns its timestamp, or None."""
    f = queue_dir / CANCEL_NAME
    try:
        t = float(f.read_text())
    except (OSError, ValueError):
        return None
    _unlink(f)
    return t
