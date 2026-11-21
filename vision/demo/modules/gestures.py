"""
gestures.py — arm and hand gestures from vision/gestures_revised.py.

Runs MediaPipe Pose and Hands in parallel (as that script does) and applies
its detect_arm_gestures() rules, which give one label per arm. The camera
source has already cropped the ZED to the left lens.

Two modes (GESTURE_MODE, default crop):
  frame  Pose + Hands on the whole frame: one body (whoever MediaPipe locks
         onto), attached to a tracked person by its shoulder midpoint.
  crop   Pose + Hands on a square crop around each of the GESTURE_PEOPLE
         nearest tracked people (YOLO boxes, padded so outstretched arms fit).
         Every person gets their own models and wave history, so gestures
         belong to a track ID directly; hands get more pixels, which matters
         at a distance. Square crops keep x and y on one scale for the rules.
         Falls back to frame mode while tracking has no people / is off, and
         uses the whole frame for a single close person (taller than
         GESTURE_CLOSE_FRAC of the frame), where a crop adds no pixels.

Lite models on the Jetson (CPU-bound), full models elsewhere; override with
GESTURE_COMPLEXITY=0/1.

Result: {"gestures": ["right: waving"], "pose": [(x, y, vis), ...] | None,
         "hands": [[(x, y), ...], ...], "anchor": (x, y) | None,
         "people": [{"track": 3, "gestures", "pose", "hands", "anchor"}, ...]}
The top-level fields are the first (nearest) person; "people" is empty in frame mode.
anchor = shoulder midpoint in pixels, used to attach frame-mode gestures to a tracked person.

Usage (score both modes on a labelled recording, also shrunk to look further away):
    vision/demo/.venv/bin/python vision/demo/modules/gestures.py vision/gesture_eval/recordings/<run> [--shrink 1,0.5,0.33]
"""

import os
import platform
import sys
import time
from collections import deque

import cv2
import numpy as np

if __name__ == "__main__":                       # run directly: make the demo's modules importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workers import Worker  # noqa: E402

JETSON = platform.machine() == "aarch64" and os.path.exists("/etc/nv_tegra_release")
MODE = os.environ.get("GESTURE_MODE", "crop")
MAX_PEOPLE = int(os.environ.get("GESTURE_PEOPLE", "2"))
PAD_X, PAD_TOP = 0.35, 0.10       # box padding: arms stretch sideways, a raised hand goes above the head
MAX_SIDE = 640                    # crops larger than this are scaled down (MediaPipe resizes anyway)
# one person this tall (fraction of the frame) gets the whole frame instead: a crop adds no
# pixels then, and close up the full frame scored better (73% vs 64% on the labelled clip)
CLOSE_FRAC = float(os.environ.get("GESTURE_CLOSE_FRAC", "0.5"))
FORGET_S = 2.0                    # drop a person's models this long after their track is gone


# ── Per-frame logic (shared by the worker and the evaluation) ─────────────────

def _empty():
    return {"gestures": [], "pose": None, "hands": [], "anchor": None, "people": []}


def _person_result(g, pose, hands, to_px, gestures):
    hand_pts = [[to_px(p.x, p.y) for p in hl.landmark] for hl in hands.multi_hand_landmarks or []]
    if not pose.pose_landmarks:
        return {"gestures": [], "pose": None, "hands": hand_pts, "anchor": None}
    lm = pose.pose_landmarks.landmark
    pts = [to_px(p.x, p.y) + (p.visibility,) for p in lm]
    ls, rs = pts[11], pts[12]                   # left / right shoulder
    return {"gestures": gestures, "pose": pts, "hands": hand_pts,
            "anchor": ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)}


class FrameGestures:
    """gestures_revised's own models on the whole frame (one body)."""

    def __init__(self, g):
        self.g = g
        self.hist = {side: deque(maxlen=20) for side in g.ARMS}

    def __call__(self, image, tracks=None):
        g = self.g
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hands_future = g.hands_pool.submit(g.hands.process, rgb)
        pose = g.pose.process(rgb)
        hands = hands_future.result()
        g.wrist_history = self.hist             # the rules keep wave history in a module global
        if pose.pose_landmarks:
            gestures = g.detect_arm_gestures(pose.pose_landmarks.landmark, hands)
        else:
            g.clear_wrist_history()
            gestures = []
        out = _empty()
        out.update(_person_result(g, pose, hands, lambda x, y: (int(x * w), int(y * h)), gestures))
        return out


class _P:
    """A landmark moved to whole-frame coordinates (same fields the rules read)."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z, visibility):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class _Landmarks:
    __slots__ = ("landmark",)

    def __init__(self, pts):
        self.landmark = pts


class _Hands:
    __slots__ = ("multi_hand_landmarks",)

    def __init__(self, hands):
        self.multi_hand_landmarks = hands


def to_frame(lms, sx, sy, side, unit):
    """
    Crop-normalised landmarks -> whole-frame pixels / unit (unit = frame width, for x AND y).
    The rules then see the wrist move in the room, not relative to a crop window that
    shifts as an arm stretches the person's box, and x / y share one scale.
    """
    return [_P((sx + p.x * side) / unit, (sy + p.y * side) / unit, p.z,
               getattr(p, "visibility", 1.0)) for p in lms]


class _Person:
    """Models + wave history for one tracked person (MediaPipe tracks between frames)."""

    def __init__(self, g):
        c = g.MODEL_COMPLEXITY
        self.pose = g.mp_pose.Pose(model_complexity=c)
        self.hands = g.mp_hands.Hands(max_num_hands=2, model_complexity=min(c, 1),
                                      min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.hist = {side: deque(maxlen=20) for side in g.ARMS}
        self.seen = time.time()
        self.win = None                         # smoothed crop window (cx, cy, side)

    def window(self, sx, sy, side):
        """Smooth the crop: follow the person, grow at once (arm out), shrink slowly."""
        cx, cy = sx + side / 2, sy + side / 2
        if self.win is None:
            self.win = (cx, cy, side)
        else:
            pcx, pcy, ps = self.win
            new_side = side if side > ps else 0.9 * ps + 0.1 * side
            self.win = (0.6 * pcx + 0.4 * cx, 0.6 * pcy + 0.4 * cy, new_side)
        cx, cy, side = self.win
        side = int(side)
        return int(cx - side / 2), int(cy - side / 2), side

    def close(self):
        self.pose.close()
        self.hands.close()


def square_crop(box, w, h, pad_x=PAD_X, pad_top=PAD_TOP):
    """Square region around a person box (may extend past the image: padded with grey)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - pad_x * bw, x2 + pad_x * bw
    y1 = y1 - pad_top * bh
    side = int(max(x2 - x1, y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sx, sy = int(cx - side / 2), int(cy - side / 2)
    return sx, sy, side


def cut(image, sx, sy, side):
    h, w = image.shape[:2]
    out = np.full((side, side, 3), 114, np.uint8)
    ax1, ay1, ax2, ay2 = max(sx, 0), max(sy, 0), min(sx + side, w), min(sy + side, h)
    if ax2 > ax1 and ay2 > ay1:
        out[ay1 - sy:ay2 - sy, ax1 - sx:ax2 - sx] = image[ay1:ay2, ax1:ax2]
    return out


class CropGestures:
    """Pose + Hands per tracked person, on a square crop of their (padded) box."""

    def __init__(self, g, max_people=MAX_PEOPLE):
        self.g, self.max_people = g, max_people
        self.people = {}                        # track id -> _Person
        self.frame = FrameGestures(g)           # when there are no tracks to crop

    def __call__(self, image, tracks, dists=None):
        g = self.g
        if not tracks:
            return self.frame(image)
        dists = dists or {}
        h, w = image.shape[:2]
        if len(tracks) == 1 and tracks[0]["box"][3] - tracks[0]["box"][1] > CLOSE_FRAC * h:
            out = self.frame(image)              # one close person: whole frame is best
            out["people"] = [{**{k: out[k] for k in ("gestures", "pose", "hands", "anchor")},
                              "track": tracks[0]["id"]}]
            return out
        # nearest first: depth when known, else the biggest box
        area = lambda t: (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1])
        chosen = sorted(tracks, key=lambda t: (dists.get(t["id"], 1e9), -area(t)))[:self.max_people]
        now = time.time()
        results = []
        for t in chosen:
            p = self.people.get(t["id"])
            if p is None:
                p = self.people[t["id"]] = _Person(g)
            p.seen = now
            sx, sy, side = p.window(*square_crop(t["box"], w, h))
            crop = cut(image, sx, sy, side)
            k = min(1.0, MAX_SIDE / side)
            if k < 1.0:
                crop = cv2.resize(crop, (MAX_SIDE, MAX_SIDE), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            hands_future = g.hands_pool.submit(p.hands.process, rgb)
            pose = p.pose.process(rgb)
            hands = hands_future.result()
            g.wrist_history = p.hist
            if pose.pose_landmarks:
                lm = to_frame(pose.pose_landmarks.landmark, sx, sy, side, w)
                hl = _Hands([_Landmarks(to_frame(x.landmark, sx, sy, side, w))
                             for x in hands.multi_hand_landmarks or []])
                gestures = g.detect_arm_gestures(lm, hl)
            else:
                g.clear_wrist_history()
                gestures = []
            r = _person_result(g, pose, hands, lambda x, y: (int(sx + x * side), int(sy + y * side)), gestures)
            r["track"] = t["id"]
            results.append(r)
        for tid in [k for k, p in self.people.items() if now - p.seen > FORGET_S]:
            self.people.pop(tid).close()
        out = _empty()
        out.update({k: v for k, v in results[0].items() if k != "track"})
        out["people"] = results
        return out


# ── Worker ─────────────────────────────────────────────────────────────────────

class GestureWorker(Worker):
    key, label, hotkey = "gesture", "Gestures", "2"
    # waving needs ~10+ Hz of wrist history; more than 12 on the Jetson only starves tracking
    max_hz = 12 if JETSON else None

    def load(self):
        os.environ.setdefault("GESTURE_COMPLEXITY", "0" if JETSON else "1")
        import gestures_revised as g            # builds Pose + Hands on import
        self.g = g
        self.pose_connections = list(g.mp_pose.POSE_CONNECTIONS)
        self.hand_connections = list(g.mp_hands.HAND_CONNECTIONS)
        self.crop = MODE == "crop"
        self.run = CropGestures(g) if self.crop else FrameGestures(g)
        self.message = f"complexity {g.MODEL_COMPLEXITY}, " + (
            f"per person (crops, up to {MAX_PEOPLE})" if self.crop else "whole frame")

    def process(self, frame):
        if not self.crop:
            return self.run(frame.image)
        track = self.hub.get("track")
        depth = self.hub.get("depth")
        tracks = (track.fresh_result(max_age_s=0.5) if track else None) or []
        dists = (depth.fresh_result(max_age_s=1.0) if depth else None) or {}
        return self.run(frame.image, tracks, dists)


# ── Standalone: score frame vs crop mode on a labelled recording ──────────────

def _score(frames, labels):
    fps = labels["fps"] or 30.0
    rows = []
    for seg in labels["segments"]:
        if seg["key"] == "transition":
            continue
        first = seg["start"] + int((seg["settle_s"] or 0) * fps)
        fs = [f for f in frames if first <= f["frame"] < seg["end"]]
        if not fs:
            continue
        n = len(fs)
        row = {"key": seg["key"], "n": n}
        if seg["expect"]:
            (side, want), = seg["expect"].items()
            row["ok"] = 100 * sum(f"{side}: {want}" in f["gestures"] for f in fs) / n
        else:
            row["false"] = 100 * sum(bool(f["gestures"]) for f in fs) / n
        row["body"] = 100 * sum(f["body"] for f in fs) / n
        row["hand"] = 100 * sum(f["hands"] > 0 for f in fs) / n
        rows.append(row)
    return rows


def _shrink(img, s):
    if s >= 1:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    out = np.full_like(img, 114)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


if __name__ == "__main__":
    import argparse
    import json
    from modules.tracking import TrackerWorker

    ap = argparse.ArgumentParser(description="score frame vs crop gestures on a labelled recording")
    ap.add_argument("run", help="recording folder with video.avi + labels.json")
    ap.add_argument("--shrink", default="1,0.5,0.33", help="scales to test (0.5 ~ twice as far away)")
    ap.add_argument("--stride", type=int, default=2, help="every Nth frame (2: ~15 fps, the live rate)")
    args = ap.parse_args()
    labels = json.load(open(os.path.join(args.run, "labels.json")))
    os.environ.setdefault("GESTURE_COMPLEXITY", "0" if JETSON else "1")
    import gestures_revised as g

    summary = {}
    for s in [float(x) for x in args.shrink.split(",")]:
        tracker = TrackerWorker(source=None, hub={})
        tracker.load()
        modes = {"frame": FrameGestures(g), "crop": CropGestures(g)}
        frames = {m: [] for m in modes}
        ms = {m: 0.0 for m in modes}
        cap = cv2.VideoCapture(os.path.join(args.run, "video.avi"))
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if idx % args.stride == 0:
                img = _shrink(img, s)
                tracks = tracker.process(type("F", (), {"image": img})())
                for m, fn in modes.items():
                    t = time.perf_counter()
                    r = fn(img, tracks) if m == "crop" else fn(img)
                    ms[m] += time.perf_counter() - t
                    frames[m].append({"frame": idx, "gestures": r["gestures"],
                                      "body": r["pose"] is not None, "hands": len(r["hands"])})
            idx += 1
        cap.release()
        n = len(frames["frame"])
        rows = {m: _score(frames[m], labels) for m in modes}
        print(f"\n── scale {s:g} (person {1 / s:.1f}x further away) ── per frame: "
              + ", ".join(f"{m} {1000 * ms[m] / n:.0f} ms" for m in modes))
        print(f"{'segment':<14} {'expected':<22} {'frame ✓/false':>13} {'crop ✓/false':>13}   body f/c   hand f/c")
        for rf, rc in zip(rows["frame"], rows["crop"]):
            seg = next(x for x in labels["segments"] if x["key"] == rf["key"])
            exp = " ".join(f"{k}: {v}" for k, v in seg["expect"].items()) or "nothing"
            val = lambda r: f"{r['ok']:.0f}%" if "ok" in r else f"({r['false']:.0f}% false)"
            print(f"{rf['key']:<14} {exp:<22} {val(rf):>13} {val(rc):>13}   "
                  f"{rf['body']:3.0f}/{rc['body']:3.0f}   {rf['hand']:3.0f}/{rc['hand']:3.0f}")
        def mean(rs, k):
            v = [r[k] for r in rs if k in r]
            return sum(v) / len(v) if v else 0.0
        summary[s] = {m: (mean(rows[m], "ok"), mean(rows[m], "false"), 1000 * ms[m] / n) for m in modes}
    print("\n── summary: mean correct % on gesture segments / mean false-alarm % on idle segments / ms ──")
    for s, v in summary.items():
        print(f"scale {s:<5g} " + "   ".join(f"{m}: {a:5.1f}% ✓ {b:5.1f}% false {c:4.0f} ms" for m, (a, b, c) in v.items()))
    os._exit(0)
