#!/usr/bin/env python3
"""
vision_client.py — Mac-side perception client for X1 GLaDOS.

Polls the Jetson's vision_server.py at ~5 Hz in a background thread and
exposes the latest perception state to the rest of the pipeline.

Usage:
    from vision.vision_client import VisionClient

    client = VisionClient("100.73.92.3")   # Jetson's IP
    client.start()

    # anywhere in the pipeline:
    ctx = client.context_string()
    # e.g. "Gestures: right: waving | Faces in view: 1 | Nearest person: 1.2 m"

    client.stop()

If the Jetson is unreachable the client silently returns empty context — the
pipeline degrades gracefully rather than crashing.

Set env var JETSON_IP to skip passing the IP explicitly:
    JETSON_IP=100.73.92.3 python3 launch.py
"""

import os
import threading
import time
from typing import Optional

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False


class VisionClient:
    """
    Background polling client for the Jetson perception server.

    Parameters
    ----------
    jetson_ip : str
        IP or hostname of the Jetson running vision_server.py.
    port : int
        Port the server listens on (default 5555).
    poll_hz : float
        How often to poll (default 5 Hz — plenty for gesture context).
    timeout : float
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        jetson_ip: Optional[str] = None,
        port: int = 5555,
        poll_hz: float = 5.0,
        timeout: float = 1.0,
    ):
        self._ip      = jetson_ip or os.environ.get("JETSON_IP", "")
        self._url     = f"http://{self._ip}:{port}/perception" if self._ip else ""
        self._interval = 1.0 / poll_hz
        self._timeout  = timeout

        self._state: dict = {}
        self._lock         = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._ok      = False   # True once at least one successful poll

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        """Start the background polling thread."""
        if not self._ip:
            print("[vision_client] JETSON_IP not set — vision context disabled")
            return
        if not _REQUESTS_OK:
            print("[vision_client] 'requests' not installed — vision context disabled")
            return
        self._running = True
        self._thread  = threading.Thread(target=self._poll_loop, daemon=True, name="VisionClient")
        self._thread.start()
        print(f"[vision_client] polling {self._url} at {1 / self._interval:.0f} Hz")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    # ── State access ───────────────────────────────────────────────────────────

    def latest(self) -> dict:
        """Return a copy of the most recent perception snapshot."""
        with self._lock:
            return dict(self._state)

    def is_connected(self) -> bool:
        return self._ok

    def context_string(self) -> str:
        """
        One-line summary suitable for injecting into the LLM system prompt.
        Returns an empty string if no data has arrived yet.
        """
        s = self.latest()
        if not s:
            return ""

        parts = []

        gestures = s.get("gestures", [])
        if gestures:
            parts.append("Gestures: " + ", ".join(gestures))

        face_count = s.get("face_count", 0)
        known      = s.get("known_faces", [])
        if face_count:
            who = ", ".join(known) if known else "unknown"
            parts.append(f"Faces in view: {face_count} ({who})")

        depth = s.get("nearest_person_m")
        if depth is not None:
            parts.append(f"Nearest person: {depth:.1f} m")

        return " | ".join(parts) if parts else ""

    # ── Poll loop ──────────────────────────────────────────────────────────────

    def _poll_loop(self):
        consecutive_errors = 0
        while self._running:
            t0 = time.time()
            try:
                resp = _requests.get(self._url, timeout=self._timeout)
                resp.raise_for_status()
                data = resp.json()
                with self._lock:
                    self._state = data
                self._ok = True
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors == 1:
                    print(f"[vision_client] connection error: {e}")
                elif consecutive_errors == 10:
                    print(f"[vision_client] Jetson unreachable — will keep retrying silently")
                self._ok = False

            elapsed = time.time() - t0
            time.sleep(max(0.0, self._interval - elapsed))
