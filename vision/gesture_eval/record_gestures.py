#!/usr/bin/env python3
"""
record_gestures.py — prompted, labelled gesture recording for testing gesture code.

Shows the camera with big on-screen prompts ("POINT TO YOUR LEFT", "WAVE RIGHT
ARM", ...) and records the ZED Mini's left lens exactly as the gesture scripts
see it (V4L2, unrectified, 1280x720), plus which gesture was prompted on every
frame. eval_gestures.py then replays the recording through the gesture code.

Stand ~1.5-2 m from the camera, whole upper body in view, facing it. Point
with whichever hand is natural; use your RIGHT hand for the other hand gestures. The preview is
mirrored (like a selfie camera); the recording is not.

Usage (needs a display: the monitor or remote desktop, not plain SSH):
    source ~/Documents/GLaDOS-gesture/vision/.venv/bin/activate
    python vision/gesture_eval/record_gestures.py            # ~80 s
    python vision/gesture_eval/record_gestures.py --list     # just print the script
    python vision/gesture_eval/record_gestures.py --quick    # 5x faster dry run

Output: vision/gesture_eval/recordings/<YYYYmmdd_HHMMSS>/
    video.avi     MJPG 1280x720, every camera frame, no overlays
    labels.json   segments: gesture key, prompt, expected output, frame range

Press Q to stop early (what was recorded so far is kept).
Recordings show faces — they are gitignored; don't share them outside the team.
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2

# ── Script ─────────────────────────────────────────────────────────────────────
# (key, prompt on screen, expected per-arm output, seconds, settle seconds)
# expect: arm ("right"/"left"/"any") -> gesture name as the gesture code prints it;
# {} = nothing expected.
# settle: start of the segment that is not scored (reaction time; waving also
# needs ~1.3 s of wrist history before it can fire).
SCRIPT = [
    ("idle",          "STAND STILL, arms down",                {},                          4, 0.5),
    ("point_left",    "POINT TO YOUR LEFT (either hand)",      {"any": "pointing left"},    3, 0.7),
    ("point_right",   "POINT TO YOUR RIGHT (either hand)",     {"any": "pointing right"},   3, 0.7),
    ("point_up",      "POINT UP (either hand)",                {"any": "pointing up"},      3, 0.7),
    ("point_down",    "POINT DOWN (either hand)",              {"any": "pointing down"},    3, 0.7),
    ("point_camera",  "POINT AT THE CAMERA (either hand)",     {},                          3, 0.7),
    ("wave_right",    "WAVE your RIGHT arm (hand at face height)", {"right": "waving"},     5, 1.5),
    ("wave_left",     "WAVE your LEFT arm (hand at face height)",  {"left": "waving"},      5, 1.5),
    ("raise_right",   "RAISE your RIGHT hand above your head, palm open", {"right": "hand raised"}, 3, 0.7),
    ("thumbs_up",     "RIGHT hand: THUMBS UP",                 {"right": "thumbs up"},      3, 0.7),
    ("thumbs_down",   "RIGHT hand: THUMBS DOWN",               {"right": "thumbs down"},    3, 0.7),
    ("peace",         "RIGHT hand: PEACE SIGN",                {"right": "peace sign"},     3, 0.7),
    ("hands_visible", "Hands open in front of your chest, palms out, still", {},            4, 0.7),
    ("talking",       "Talk with your hands (natural motion)", {},                          4, 0.7),
    ("idle_end",      "STAND STILL, arms down",                {},                          3, 0.5),
]
GET_READY_S = 2.0
COUNTDOWN_S = 5.0

OUT_DIR = Path(__file__).resolve().parent / "recordings"


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_prompt(frame, title, subtitle, progress, color):
    """Big prompt banner at the top, progress bar at the bottom."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.putText(frame, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    cv2.putText(frame, subtitle, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
    cv2.rectangle(frame, (0, h - 20), (int(w * progress), h), color, -1)


# ── Recording ──────────────────────────────────────────────────────────────────

def record(args):
    scale = 0.2 if args.quick else 1.0
    run_dir = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)    # ZED Mini: both lenses side by side
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read from the camera (is something else using it?)")
    h, w = frame.shape[:2]
    left_w = w // 2
    print(f"[record] camera {w}x{h}, recording left lens {left_w}x{h} -> {run_dir}")

    writer = cv2.VideoWriter(str(run_dir / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                             30, (left_w, h))
    window = "GLaDOS gesture recording"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def read_left():
        ok, f = cap.read()
        return ok, (f[:, :left_w] if ok else None)

    # ── countdown (not recorded) ──
    t0 = time.time()
    while time.time() - t0 < COUNTDOWN_S * scale:
        ok, f = read_left()
        if not ok:
            break
        show = cv2.flip(f, 1)
        left = COUNTDOWN_S * scale - (time.time() - t0)
        draw_prompt(show, f"Starting in {left:.0f}...", "Step back so your upper body is in view",
                    1 - left / (COUNTDOWN_S * scale), (0, 200, 255))
        cv2.imshow(window, show)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── scripted segments ──
    segments, frame_times, n, aborted = [], [], 0, False
    plan = []
    for i, (key, prompt, expect, secs, settle) in enumerate(SCRIPT):
        plan.append(("transition", f"GET READY: {prompt}", {}, GET_READY_S, None))
        plan.append((key, prompt, expect, secs, settle))

    for key, prompt, expect, secs, settle in plan:
        start, t_seg = n, time.time()
        dur = secs * scale
        is_ready = key == "transition"
        print(f"[record] {'   ' if is_ready else '>> '}{prompt}")
        while time.time() - t_seg < dur:
            ok, f = read_left()
            if not ok:
                aborted = True
                break
            writer.write(f)                        # raw frame, no overlays
            frame_times.append(time.time())
            n += 1
            show = cv2.flip(f, 1)
            draw_prompt(show, prompt.replace("GET READY: ", "Next: ") if is_ready else prompt,
                        "get ready..." if is_ready else f"{dur - (time.time() - t_seg):.1f} s left",
                        (time.time() - t_seg) / dur, (0, 200, 255) if is_ready else (0, 255, 0))
            cv2.imshow(window, show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                aborted = True
                break
        segments.append({"key": key, "prompt": prompt, "expect": expect,
                         "start": start, "end": n,
                         "settle_s": None if settle is None else settle * scale})
        if aborted:
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()

    fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
    labels = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "ZED Mini left lens via V4L2 (unrectified)",
        "frame_size": [left_w, h],
        "fps": round(fps, 2),
        "frames": n,
        "quick": args.quick,
        "aborted": aborted,
        "segments": segments,
    }
    tmp = run_dir / "labels.json.tmp"
    tmp.write_text(json.dumps(labels, indent=2))
    os.replace(tmp, run_dir / "labels.json")
    print(f"[record] {n} frames at {fps:.1f} fps{' (stopped early)' if aborted else ''} -> {run_dir}")


# ── Standalone ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--quick", action="store_true", help="5x shorter segments, for a dry run")
    ap.add_argument("--list", action="store_true", help="print the gesture script and exit")
    args = ap.parse_args()

    if args.list:
        total = COUNTDOWN_S
        for key, prompt, expect, secs, settle in SCRIPT:
            total += GET_READY_S + secs
            print(f"  {secs:>2}s  {prompt:<55} expect {expect or 'nothing'}")
        print(f"  total ~{total:.0f} s")
    else:
        record(args)
