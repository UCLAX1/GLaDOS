"""
expression.py
Multi-face expression detection for GLaDOS.
- MediaPipe FaceDetector   : fast, accurate bounding boxes for N faces
- HSEmotion ONNX           : neural-net expression classifier (8 classes)
Expressions: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
pip install hsemotion-onnx onnxruntime
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import defaultdict, deque

from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# tuning

MODEL_NAME        = "enet_b0_8_best_afew"   # fastest accurate HSEmotion model
SMOOTHING_FRAMES  = 8                        # rolling average window per face
DISPLAY_THRESHOLD = 0.25                     # min confidence to show secondary label
AMBIGUITY_WINDOW  = 0.20                     # show labels within this of top score
FACE_PAD          = 0.20                     # fraction to pad face crop for better accuracy

#  MediaPipe face detector setup 

detector_options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(
        model_asset_path="vision/face_detector.tflite"
    ),
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.5,
)
face_detector = vision.FaceDetector.create_from_options(detector_options)

#  HSEmotion classifier 

fer = HSEmotionRecognizer(model_name=MODEL_NAME)
EMOTION_LABELS = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise"
]

#  per-face score smoothing 
# Keyed by face index (order in frame). Not identity-aware — for identity-aware
# smoothing, integrate with face_memory.py and key by recognised name instead.

score_histories: dict = defaultdict(lambda: {
    label: deque(maxlen=SMOOTHING_FRAMES) for label in EMOTION_LABELS
})


def smooth_scores(face_idx: int, raw_scores: np.ndarray) -> dict:
    hist = score_histories[face_idx]
    for i, label in enumerate(EMOTION_LABELS):
        hist[label].append(float(raw_scores[i]))
    return {label: float(np.mean(hist[label])) for label in EMOTION_LABELS}


#  helpers 

def pad_bbox(x, y, w, h, pad, frame_w, frame_h):
    """Expand bounding box by pad fraction, clamped to frame."""
    px = int(w * pad)
    py = int(h * pad)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(frame_w, x + w + px)
    y2 = min(frame_h, y + h + py)
    return x1, y1, x2, y2


def top_emotions(scores: dict, threshold=DISPLAY_THRESHOLD,
                 window=AMBIGUITY_WINDOW) -> list:
    """Return (emotion, confidence) pairs to display, sorted by confidence."""
    sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not sorted_s or sorted_s[0][1] < threshold:
        return [("Neutral", 1.0)]
    top_conf = sorted_s[0][1]
    return [
        (name, conf) for name, conf in sorted_s
        if conf >= threshold and top_conf - conf <= window
    ]


def conf_color(conf: float) -> tuple:
    return (0, int(conf * 220), int((1 - conf) * 220))


BOX_COLORS = [
    (0,   255, 100),
    (100, 180, 255),
    (255, 180,   0),
    (200,   0, 255),
    (0,   220, 220),
]

#  main loop 

cap          = cv2.VideoCapture(0)
timestamp_ms = 0

print("Press Q to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_ms += 33
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        det_result   = face_detector.detect_for_video(mp_image, timestamp_ms)

        detections = det_result.detections if det_result.detections else []

        cv2.putText(frame, f"Faces: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        for i, detection in enumerate(detections):
            bb    = detection.bounding_box
            x1, y1, x2, y2 = pad_bbox(
                bb.origin_x, bb.origin_y, bb.width, bb.height,
                FACE_PAD, fw, fh
            )

            face_crop = rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            try:
                _, raw_scores = fer.predict_emotions(face_crop, logits=False)
                # raw_scores may be list or array
                raw_scores = np.array(raw_scores, dtype=np.float32)
            except Exception as e:
                print(f"HSEmotion error on face {i}: {e}")
                continue

            smoothed  = smooth_scores(i, raw_scores)
            top_name  = max(smoothed, key=smoothed.get)
            top_conf  = smoothed[top_name]
            box_color = BOX_COLORS[i % len(BOX_COLORS)]

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Single emotion label above the box
            label = f"{top_name} {int(top_conf * 100)}%"
            cv2.putText(frame, label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color(top_conf), 2)

            # Mini confidence bars inside the box bottom
            bar_x  = x1 + 4
            bar_y  = y2 - 4 - len(EMOTION_LABELS) * 14
            if bar_y > y1 + 30:
                for label in EMOTION_LABELS:
                    conf    = smoothed[label]
                    bar_len = int(conf * min(80, x2 - x1 - 8))
                    cv2.putText(frame, label[:3], (bar_x, bar_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                                (180, 180, 180), 1)
                    cv2.rectangle(frame,
                                  (bar_x + 28, bar_y - 9),
                                  (bar_x + 28 + bar_len, bar_y - 1),
                                  conf_color(conf), -1)
                    bar_y += 14

        cv2.imshow("GLaDOS Expression Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    face_detector.close()
    cv2.destroyAllWindows()