#!/usr/bin/env python3
"""
eval_gestures.py — score gesture code against a labelled recording.

Replays a record_gestures.py recording through MediaPipe Pose + Hands and
applies two rule sets to the same landmarks:
    original  gestures_revised.py as committed (single label, right arm only)
    fixed     the local copy with LOCAL TEST fixes (one label per arm)
for each model size (lite = model_complexity 0, full = 1).

Per gesture it reports how often the model saw a body / a hand (model misses)
and how often the rules then gave the expected answer (rule misses), plus
false alarms in segments where nothing should be detected.

Frames are processed every --stride frames (default 2: 30 fps recording ->
15 fps, about the live rate, so waving history spans the same time as live).

Usage:
    source ~/Documents/GLaDOS-gesture/vision/.venv/bin/activate
    python vision/gesture_eval/eval_gestures.py vision/gesture_eval/recordings/<run>
    python vision/gesture_eval/eval_gestures.py <run> --complexity 0      # lite only

Output: printed tables + <run>/eval_<HHMMSS>.json (per-frame outputs + summary).
"""

import argparse
import json
import os
import subprocess
import time
import types
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp

GESTURE_REPO = Path.home() / "Documents" / "GLaDOS-gesture"
FIXED_PATH   = GESTURE_REPO / "vision" / "gestures_revised.py"


# ── Loading the rule sets ──────────────────────────────────────────────────────

class _NoCamera:
    """Stands in for cv2.VideoCapture so a script's top-level loop exits at once."""
    def set(self, *a): return True
    def read(self): return False, None
    def release(self): pass


def load_script(source: str, name: str) -> types.SimpleNamespace:
    """Execute a gesture script's top level (functions, state) without a camera."""
    real_capture, real_imshow = cv2.VideoCapture, cv2.imshow
    cv2.VideoCapture, cv2.imshow = (lambda *a: _NoCamera()), (lambda *a: None)
    try:
        ns = {"__name__": f"gesture_eval_{name}"}
        exec(compile(source, name, "exec"), ns)
    finally:
        cv2.VideoCapture, cv2.imshow = real_capture, real_imshow
    return types.SimpleNamespace(**ns)


def decide_original(o, pose_results, hand_results) -> str:
    """The committed main loop's decision logic, verbatim apart from the o. prefix."""
    gesture = "none"
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand = hand_landmarks.landmark
                hand_gesture = o.detect_hand_gesture(hand)
                pointing = o.pointing_direction(landmarks, hand)
                if o.hand_is_raised(landmarks, hand):
                    gesture = "hand raised"
                elif o.is_waving(landmarks):
                    gesture = "waving"
                elif pointing:
                    gesture = pointing
                if hand_gesture:
                    gesture = hand_gesture
    return gesture


def decide_fixed(f, pose_results, hand_results) -> list:
    if not pose_results.pose_landmarks:
        f.clear_wrist_history()
        return []
    return f.detect_arm_gestures(pose_results.pose_landmarks.landmark, hand_results)


# ── Running ────────────────────────────────────────────────────────────────────

def run_models(video: Path, complexity: int, stride: int, original, fixed) -> list:
    """One pass over the video with one model size; both rule sets per frame."""
    pose = mp.solutions.pose.Pose(model_complexity=complexity)
    hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=min(complexity, 1),
                                     min_detection_confidence=0.7, min_tracking_confidence=0.7)
    original.wrist_history.clear()
    fixed.clear_wrist_history()

    cap = cv2.VideoCapture(str(video))
    frames, idx, t0 = [], 0, time.time()
    pose_ms = hands_ms = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = time.perf_counter(); pr = pose.process(rgb); pose_ms += time.perf_counter() - t
            t = time.perf_counter(); hr = hands.process(rgb); hands_ms += time.perf_counter() - t
            frames.append({
                "frame": idx,
                "pose": pr.pose_landmarks is not None,
                "hands": len(hr.multi_hand_landmarks or []),
                "original": decide_original(original, pr, hr),
                "fixed": decide_fixed(fixed, pr, hr),
            })
        idx += 1
    cap.release()
    pose.close()
    hands.close()
    n = len(frames)
    print(f"  complexity={complexity}: {n} frames in {time.time() - t0:.0f}s "
          f"(pose {1000 * pose_ms / max(n, 1):.0f} ms, hands {1000 * hands_ms / max(n, 1):.0f} ms per frame)")
    return frames


# ── Scoring ────────────────────────────────────────────────────────────────────

def score(frames: list, labels: dict) -> list:
    fps = labels["fps"] or 30.0
    rows = []
    for seg in labels["segments"]:
        if seg["key"] == "transition":
            continue
        first = seg["start"] + int((seg["settle_s"] or 0) * fps)
        fs = [f for f in frames if first <= f["frame"] < seg["end"]]
        if not fs:
            continue
        expect = seg["expect"]
        n = len(fs)
        row = {
            "key": seg["key"], "frames": n,
            "pose_pct": 100 * sum(f["pose"] for f in fs) / n,
            "hand_pct": 100 * sum(f["hands"] > 0 for f in fs) / n,
            "original_top": Counter(f["original"] for f in fs).most_common(2),
            "fixed_top": Counter(" | ".join(f["fixed"]) or "none" for f in fs).most_common(2),
        }
        if expect:
            (side, want), = expect.items()
            row["expect"] = f"{side}: {want}"
            # original has no arm sides: count the gesture name alone
            row["original_pct"] = 100 * sum(f["original"] == want for f in fs) / n
            row["fixed_any_arm_pct"] = 100 * sum(any(g.endswith(": " + want) for g in f["fixed"])
                                                 for f in fs) / n
            row["fixed_pct"] = (row["fixed_any_arm_pct"] if side == "any" else
                                100 * sum(f"{side}: {want}" in f["fixed"] for f in fs) / n)
        else:
            row["expect"] = "nothing"
            row["original_false_pct"] = 100 * sum(f["original"] != "none" for f in fs) / n
            row["fixed_false_pct"] = 100 * sum(bool(f["fixed"]) for f in fs) / n
        rows.append(row)
    return rows


def _top(pairs, n_frames):
    return ", ".join(f"{name} {100 * c / n_frames:.0f}%" for name, c in pairs)


def print_table(complexity: int, rows: list):
    name = {0: "LITE", 1: "FULL"}.get(complexity, str(complexity))
    print(f"\n── {name} models (complexity={complexity}) ──")
    print("| gesture | expected | body seen | hand seen | original ✓ | fixed ✓ (expected arm / any arm) | fixed: most common output |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if "fixed_pct" in r:
            o, f = f"{r['original_pct']:.0f}%", f"{r['fixed_pct']:.0f}% / {r['fixed_any_arm_pct']:.0f}%"
        else:
            o, f = f"{r['original_false_pct']:.0f}% false", f"{r['fixed_false_pct']:.0f}% false"
        print(f"| {r['key']} | {r['expect']} | {r['pose_pct']:.0f}% | {r['hand_pct']:.0f}% | {o} | {f} | "
              f"{_top(r['fixed_top'], r['frames'])} |")


# ── Standalone ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="recording folder (contains video.avi + labels.json)")
    ap.add_argument("--complexity", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fixed", default=str(FIXED_PATH), help="fixed gestures_revised.py")
    args = ap.parse_args()

    run = Path(args.run)
    labels = json.loads((run / "labels.json").read_text())
    original_src = subprocess.run(
        ["git", "-C", str(GESTURE_REPO), "show", "HEAD:vision/gestures_revised.py"],
        check=True, capture_output=True, text=True).stdout
    original = load_script(original_src, "original")
    fixed = load_script(Path(args.fixed).read_text(), "fixed")
    print(f"[eval] {run.name}: {labels['frames']} frames at {labels['fps']} fps, stride {args.stride}")

    results = {}
    for c in args.complexity:
        frames = run_models(run / "video.avi", c, args.stride, original, fixed)
        rows = score(frames, labels)
        results[c] = {"summary": rows, "frames": frames}
    for c in args.complexity:
        print_table(c, results[c]["summary"])

    out = run / f"eval_{time.strftime('%H%M%S')}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"labels": labels, "stride": args.stride,
                               "results": {str(c): r for c, r in results.items()}}, indent=1))
    os.replace(tmp, out)
    print(f"\n[eval] wrote {out}")
