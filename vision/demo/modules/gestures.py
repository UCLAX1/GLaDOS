"""
gestures.py — arm and hand gestures from vision/gestures_revised.py.

Runs MediaPipe Pose and Hands in parallel (as that script does) and applies
its detect_arm_gestures() rules, which give one label per arm. The camera
source has already cropped the ZED to the left lens.

Lite models on the Jetson (CPU-bound), full models elsewhere; override with
GESTURE_COMPLEXITY=0/1.

Result: {"gestures": ["right: waving"], "pose": [(x, y, vis), ...] | None,
         "hands": [[(x, y), ...], ...], "anchor": (x, y) | None}
anchor = shoulder midpoint in pixels, used to attach gestures to a tracked person.
"""

import os
import platform

import cv2

from workers import Worker

JETSON = platform.machine() == "aarch64" and os.path.exists("/etc/nv_tegra_release")


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
        self.message = f"complexity {g.MODEL_COMPLEXITY}"

    def process(self, frame):
        g = self.g
        h, w = frame.image.shape[:2]
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        hands_future = g.hands_pool.submit(g.hands.process, rgb)
        pose = g.pose.process(rgb)
        hands = hands_future.result()

        hand_pts = [[(int(p.x * w), int(p.y * h)) for p in hl.landmark]
                    for hl in hands.multi_hand_landmarks or []]
        if not pose.pose_landmarks:
            g.clear_wrist_history()
            return {"gestures": [], "pose": None, "hands": hand_pts, "anchor": None}

        lm = pose.pose_landmarks.landmark
        gestures = g.detect_arm_gestures(lm, hands)
        pts = [(int(p.x * w), int(p.y * h), p.visibility) for p in lm]
        ls, rs = pts[11], pts[12]               # left / right shoulder
        anchor = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        return {"gestures": gestures, "pose": pts, "hands": hand_pts, "anchor": anchor}
