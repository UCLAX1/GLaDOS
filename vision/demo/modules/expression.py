"""
expression.py — facial expression per face, from vision/expression2.py.

Uses that script's MediaPipe face detector + HSEmotion classifier and its
padding / smoothing settings. Smoothing is keyed by a face's position
(nearest face from the last call) instead of its index in the frame, so
two people's scores don't swap when the detector lists them in another order.

Result: [{"box": (x1, y1, x2, y2), "emotion": "Happiness", "conf": 0.71}]
"""

import os
from collections import defaultdict, deque

import cv2
import numpy as np

from workers import JETSON, Worker, onnx_threads, sessions_to_gpu

MAX_HZ = float(os.environ.get("DEMO_EXPR_HZ", "6"))


class ExpressionWorker(Worker):
    key, label, hotkey = "expr", "Expression", "4"
    max_hz = MAX_HZ

    def load(self):
        import mediapipe as mp
        with onnx_threads(1 if JETSON else 0):
            import expression2 as ex             # builds detector + HSEmotion on import
        self.mp, self.ex = mp, ex
        # HSEmotion asks for CPU; the GPU runs it ~7x faster (CPU fallback)
        self.message = "HSEmotion " + sessions_to_gpu([ex.fer.ort_session], self.key)
        self.hist = defaultdict(lambda: deque(maxlen=ex.SMOOTHING_FRAMES))
        self.prev = []                           # [(key, cx, cy, w)]
        self.next_key = 0
        self.last_ts = 0

    def _key_for(self, cx, cy, w, used):
        best = None
        for k, px, py, pw in self.prev:
            d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if k not in used and d < 0.6 * max(w, pw) and (best is None or d < best[0]):
                best = (d, k)
        if best:
            return best[1]
        self.next_key += 1
        return self.next_key

    def process(self, frame):
        ex, mp = self.ex, self.mp
        fh, fw = frame.image.shape[:2]
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        ts = max(int(frame.t * 1000), self.last_ts + 1)   # must strictly increase
        self.last_ts = ts
        det = ex.face_detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        out, now, used = [], [], set()
        for d in det.detections or []:
            bb = d.bounding_box
            x1, y1, x2, y2 = ex.pad_bbox(bb.origin_x, bb.origin_y, bb.width, bb.height,
                                         ex.FACE_PAD, fw, fh)
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            _, scores = ex.fer.predict_emotions(crop, logits=False)
            key = self._key_for((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, used)
            used.add(key)
            now.append((key, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1))
            self.hist[key].append(np.asarray(scores, dtype=np.float32))
            mean = np.mean(self.hist[key], axis=0)
            i = int(np.argmax(mean))
            out.append({"box": (x1, y1, x2, y2), "emotion": ex.EMOTION_LABELS[i], "conf": float(mean[i])})
        for k in set(self.hist) - used - {p[0] for p in now}:
            if k not in {p[0] for p in self.prev}:
                del self.hist[k]                 # face gone for two calls: forget it
        self.prev = now
        return out
