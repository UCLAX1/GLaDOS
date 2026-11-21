"""
presence.py — auto-idle: run the heavy modules only when someone is there.

No extra model. Person tracking (the cheapest module, ~20 ms on the Jetson GPU)
is the watcher, plus a room check on an 80x45 grey copy of the frame:
    motion   none / low / medium / high   (mean frame-to-frame change; the
             benchmark's `optical_flow` key)
    light    ok / dark / bright           (mean brightness)
    blocked  lens covered or camera blind (almost no texture)

Modes (DEMO_IDLE=0 or --no-idle turns this off; the self-test runs without it):
    idle     nobody seen for IDLE_AFTER_S. Only tracking runs, at IDLE_TRACK_HZ;
             any motion brings it to full speed at once, so wake-up is fast
    aware    someone is in view but further than ENGAGE_M: tracking + depth
    engaged  someone within ENGAGE_M (or no depth: anyone in view): everything
Leaving uses wider margins and delays than entering (DISENGAGE_M, ENGAGED_HOLD_S),
so a person at the boundary doesn't flip it on and off.

It only pauses modules (Worker.paused_for = "idle: ..."), never switches them
off, so a feature turned off by hand stays off, and a feature paused by face
registration is left alone. Pauses are renewed every PAUSE_LEASE_S: if this
thread ever stops, everything resumes by itself.

Usage (standalone: the room check on a video or camera, no models):
    vision/demo/.venv/bin/python vision/demo/presence.py [video | webcam:0 | zed]
"""

import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

IDLE_AFTER_S = float(os.environ.get("DEMO_IDLE_AFTER", "10"))   # nobody seen this long -> idle
ENGAGE_M = float(os.environ.get("DEMO_ENGAGE_M", "2.0"))          # closer than this -> engaged
DISENGAGE_M = ENGAGE_M + 1.2
ENGAGED_HOLD_S = 5.0          # nobody within DISENGAGE_M this long -> aware
IDLE_TRACK_HZ = 2.0
MOTION_BOOST_S = 3.0          # full-speed tracking this long after motion while idle
PAUSE_LEASE_S = 5.0
CHECK_HZ = 5
SMALL = (80, 45)

# mean absolute grey change per pixel (0-255) between checks, 5 Hz
MOTION_LEVELS = ((1.2, "none"), (3.0, "low"), (8.0, "medium"), (1e9, "high"))
MOTION_WAKE = 1.2             # above "none": speed tracking up
DARK, BRIGHT, BLOCKED_STD = 35, 225, 6.0

PAUSED_IN = {"idle": ("gesture", "face", "expr", "depth"),
             "aware": ("gesture", "face", "expr"),
             "engaged": ()}


def room_stats(small_grey, prev_grey):
    mean, std = float(small_grey.mean()), float(small_grey.std())
    motion = 0.0 if prev_grey is None else float(cv2.absdiff(small_grey, prev_grey).mean())
    level = next(name for limit, name in MOTION_LEVELS if motion < limit)
    light = "dark" if mean < DARK else "bright" if mean > BRIGHT else "ok"
    return {"motion": round(motion, 2), "optical_flow": level, "light": light,
            "brightness": round(mean), "blocked": std < BLOCKED_STD}


class Presence:
    def __init__(self, source, hub: dict, enabled=True):
        self.source, self.hub = source, hub
        self.enabled = enabled
        self.mode = "engaged"            # start engaged: everything loads and shows at once
        self.since = time.time()
        self.reason = "starting"
        self.room = {}
        self.events = deque(maxlen=8)    # (time, text) of mode changes, newest last
        self.last_seen = time.time()     # someone in view
        self.last_near = time.time()     # someone within DISENGAGE_M
        self.last_motion = 0.0
        self.people, self.nearest = 0, None
        self._prev = None
        self._unmeasured_t = None        # since when someone in view has no depth reading
        self._track_hz = None            # tracking's own max_hz, restored when awake
        self._running = False

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        threading.Thread(target=self._run, name="presence", daemon=True).start()
        return self

    def stop(self):
        self._running = False
        self._release()

    def toggle(self) -> str:
        self.enabled = not self.enabled
        if not self.enabled:
            self._release()
            self._set_mode("engaged", "auto-idle turned off")
        else:
            self.last_seen = self.last_near = time.time()   # don't drop to idle at once
            self._set_mode("engaged", "auto-idle turned on")
        return f"auto-idle {'on' if self.enabled else 'off'}"

    def status(self) -> dict:
        return {"enabled": self.enabled, "mode": self.mode if self.enabled else "off",
                "since_s": round(time.time() - self.since), "reason": self.reason,
                "person_detected": self.people > 0, "person_count": self.people,
                "nearest_m": None if self.nearest is None else round(self.nearest, 2),
                "room": self.room,
                "events": [f"{time.strftime('%H:%M:%S', time.localtime(t))} {e}" for t, e in self.events]}

    def line(self) -> str:
        """One line for the panel."""
        if not self.enabled:
            mode = "auto-idle off"
        else:
            mode = self.mode.upper() + (f" ({self.reason})" if self.reason else "")
        r = self.room
        room = (f"motion {r.get('optical_flow', '?')}, light {r.get('light', '?')}"
                + (", CAMERA BLOCKED" if r.get("blocked") else ""))
        return f"{mode} | {room}"

    # ── Loop ──────────────────────────────────────────────────────────────────

    def _run(self):
        while self._running:
            t0 = time.time()
            try:
                self._tick()
            except Exception as e:                     # never take the demo down
                print(f"[presence] error: {e}", file=sys.stderr)
                self._release()
                time.sleep(1.0)
            time.sleep(max(0.0, 1 / CHECK_HZ - (time.time() - t0)))
        self._release()

    def _tick(self):
        now = time.time()
        frame = self.source.latest()
        if frame is not None:
            small = cv2.cvtColor(cv2.resize(frame.image, SMALL, interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2GRAY)
            self.room = room_stats(small, self._prev)
            self._prev = small
            if self.room["motion"] > MOTION_WAKE:
                self.last_motion = now

        track = self.hub.get("track")
        tracking = track is not None and track.state == "running" and track.enabled
        people = (track.fresh_result(max_age_s=1.5) or []) if tracking else []
        depth = self.hub.get("depth")
        depth_on = depth is not None and depth.state == "running" and depth.enabled
        dists = (depth.fresh_result(max_age_s=1.5) or {}) if depth_on else {}
        known = [dists[p["id"]] for p in people if p["id"] in dists]
        self.people = len(people)
        self.nearest = min(known) if known else None
        # someone without a distance: wait up to 1 s for depth (it was paused while idle);
        # with no depth at all (no ZED, depth off) or still none after that, count them as close
        if len(known) < len(people):
            self._unmeasured_t = self._unmeasured_t or now
        else:
            self._unmeasured_t = None
        unknown_close = bool(people) and (not depth_on or (self._unmeasured_t is not None
                                                          and now - self._unmeasured_t > 1.0))
        close = unknown_close or bool(known and min(known) <= ENGAGE_M)
        if people:
            self.last_seen = now
            if unknown_close or (known and min(known) <= DISENGAGE_M):
                self.last_near = now
        if not self.enabled:
            return

        face = self.hub.get("face")
        if not tracking:                               # nothing to watch with: stay fully on
            self._set_mode("engaged", "tracking is off")
        elif face is not None and getattr(face, "registering", None):
            self._set_mode("engaged", "registering a face")
        elif close:
            self._set_mode("engaged", self._who(people, dists))
        elif self.mode == "engaged" and now - self.last_near < ENGAGED_HOLD_S:
            pass                                       # just stepped back: hold a moment
        elif people:
            self._set_mode("aware", self._who(people, dists))
        elif self.mode != "idle" and now - self.last_seen < IDLE_AFTER_S:
            pass                                       # just left (or turned away): hold
        else:
            self._set_mode("idle", "nobody around")
        self._apply(now)

    def _who(self, people, dists):
        near = min(dists.values()) if dists else None
        s = f"{len(people)} {'person' if len(people) == 1 else 'people'}"
        return s + (f", nearest {near:.1f} m" if near is not None else "")

    def _set_mode(self, mode, reason):
        self.reason = reason
        if mode != self.mode:
            text = f"{self.mode} -> {mode}: {reason}"
            self.events.append((time.time(), text))
            print(f"[presence] {text}")
            self.mode, self.since = mode, time.time()

    def _apply(self, now):
        paused = PAUSED_IN[self.mode]
        why = {"idle": "idle: nobody around", "aware": "idle: nobody close"}.get(self.mode, "")
        for key, w in self.hub.items():
            if key in paused:
                if not w.paused_for or w.paused_for.startswith("idle"):
                    w.paused_for, w.paused_until = why, now + PAUSE_LEASE_S
            elif w.paused_for.startswith("idle"):
                w.paused_for = ""
        track = self.hub.get("track")
        if track is not None:
            if self._track_hz is None:
                self._track_hz = (type(track).max_hz,)
            slow = self.mode == "idle" and now - self.last_motion > MOTION_BOOST_S
            track.max_hz = IDLE_TRACK_HZ if slow else self._track_hz[0]

    def _release(self):
        """Undo everything this controller did (turned off, or on error)."""
        for w in self.hub.values():
            if w.paused_for.startswith("idle"):
                w.paused_for = ""
        track = self.hub.get("track")
        if track is not None and self._track_hz is not None:
            track.max_hz = self._track_hz[0]


# ── Standalone: room check only ────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from camera import FrameSource
    src = FrameSource(sys.argv[1] if len(sys.argv) > 1 else "auto").start()
    prev, t_end = None, time.time() + 20
    try:
        while time.time() < t_end:
            f = src.latest()
            if f is not None:
                small = cv2.cvtColor(cv2.resize(f.image, SMALL, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                t = time.perf_counter()
                stats = room_stats(small, prev)
                prev = small
                print(f"{stats}  ({1000 * (time.perf_counter() - t):.2f} ms)", flush=True)
            time.sleep(1 / CHECK_HZ)
    finally:
        src.stop()
