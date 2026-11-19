#!/usr/bin/env python3
"""
vision_client.py — Mac-side perception client for X1 GLaDOS.

Polls the Jetson's vision demo (vision/demo/run_demo.py, GET :8081/status) at
~5 Hz in a background thread and exposes what it sees to the rest of the
pipeline. The demo owns the camera and already fuses tracking, faces,
expressions, gestures and depth, so there is no separate perception server.

Usage:
    from vision.vision_client import VisionClient

    client = VisionClient("100.73.92.3")   # Jetson's IP
    client.start()

    # anywhere in the pipeline:
    client.snapshot()   # {"people": [{"name": ..., "expression": ..., "gestures": [...], "distance": ...}]}

    client.stop()

If the Jetson is unreachable the client silently returns empty context — the
pipeline degrades gracefully rather than crashing.

Set env var JETSON_IP to skip passing the IP explicitly:
    JETSON_IP=100.73.92.3 ./start.sh
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
        IP or hostname of the Jetson running the vision demo.
    port : int or None
        Port of the demo's web server (default: $VISION_DEMO_PORT or 8081).
    poll_hz : float
        How often to poll (default 5 Hz — plenty for gesture context).
    timeout : float
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        jetson_ip: Optional[str] = None,
        port: Optional[int] = None,
        poll_hz: float = 5.0,
        timeout: float = 1.0,
    ):
        self._ip      = jetson_ip or os.environ.get("JETSON_IP", "")
        demo_port      = int(port or os.environ.get("VISION_DEMO_PORT", "8081"))
        self._demo_url = f"http://{self._ip}:{demo_port}/status" if self._ip else ""
        self._url      = self._demo_url
        self.source    = ""   # "demo" once the Jetson has answered
        self.last_ok_t = 0.0  # time of last successful poll
        self.last_error = ""  # most recent poll error (for the GUI)
        self._interval = 1.0 / poll_hz
        self._timeout  = timeout

        self._state: dict = {}
        self._raw:   dict = {}   # untouched JSON from the Jetson (for the LLM)
        self.system: dict = {}   # Jetson usage stats (GUI only — never sent to the LLM)
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
        print(f"[vision_client] polling {self._demo_url} "
              f"at {1 / self._interval:.0f} Hz")

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

    def snapshot(self) -> Optional[dict]:
        """
        What GLaDOS sees, for the per-turn LLM input (None if the camera is offline).
        Observations only — camera stats, track IDs and confidence scores are
        dropped so the model reacts to people, not to numbers.
        """
        if not self._ok:
            return None
        with self._lock:
            raw = dict(self._raw)
        people = []
        for p in raw.get("people", []) or []:
            person = {"name": p.get("name") or "unknown"}
            e = p.get("expr")
            if e and (len(e) < 2 or float(e[1]) >= 0.4):   # skip low-confidence guesses
                person["expression"] = str(e[0]).lower()
            g = [x for x in (p.get("gestures") or []) if x]
            if g:
                person["gestures"] = g
            d = p.get("dist")
            if d is not None:
                d = float(d)
                person["distance"] = ("very close" if d < 0.8 else
                                      "close" if d < 1.8 else
                                      "across the room" if d < 4 else "far away")
            people.append(person)
        return {"people": people}

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
            parts.append(f"People in view: {face_count} ({who})")

        exprs = s.get("expressions", [])
        if exprs:
            parts.append("Expressions: " + ", ".join(exprs))

        depth = s.get("nearest_person_m")
        if depth is not None:
            parts.append(f"Nearest person: {depth:.1f} m")

        return " | ".join(parts) if parts else ""

    # ── Poll loop ──────────────────────────────────────────────────────────────

    @staticmethod
    def _from_demo(status: dict) -> dict:
        """Convert run_demo.py /status JSON into the perception schema."""
        people = status.get("people", []) or []
        gestures, names, exprs, dists = [], [], [], []
        for p in people:
            gestures += [g for g in (p.get("gestures") or []) if g]
            if p.get("name"):
                names.append(p["name"])
            e = p.get("expr")
            if e:
                exprs.append(f"{p.get('name') or 'person'}: {e[0]}")
            if p.get("dist") is not None:
                dists.append(float(p["dist"]))
        return {
            "gestures":         gestures,
            "face_count":       len(people),
            "known_faces":      names,
            "expressions":      exprs,
            "nearest_person_m": min(dists) if dists else None,
            "fps":              (status.get("camera") or {}).get("fps"),
            "ts":               time.time(),
        }

    def _fetch(self) -> dict:
        r = _requests.get(self._demo_url, timeout=self._timeout)
        r.raise_for_status()
        self.source = "demo"
        raw = r.json()
        if raw.get("error"):
            raise RuntimeError(f"Jetson /status error: {raw['error']}")
        self.system = raw.get("system") or {}
        with self._lock:
            self._raw = {"people": raw.get("people", [])}   # id, name, expr, gestures, dist
        return self._from_demo(raw)

    def _poll_loop(self):
        consecutive_errors = 0
        while self._running:
            t0 = time.time()
            try:
                data = self._fetch()
                with self._lock:
                    self._state = data
                self._ok = True
                self.last_ok_t = time.time()
                self.last_error = ""
                consecutive_errors = 0
            except Exception as e:
                self.last_error = str(e)
                consecutive_errors += 1
                if consecutive_errors == 1:
                    print(f"[vision_client] cannot reach the Jetson: {e}")
                elif consecutive_errors % 150 == 0:   # every ~30 s at 5 Hz
                    print(f"[vision_client] still cannot reach {self._demo_url} "
                          f"({consecutive_errors} failed polls)")
                self._ok = False

            elapsed = time.time() - t0
            time.sleep(max(0.0, self._interval - elapsed))
