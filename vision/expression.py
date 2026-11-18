import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# landmark indices (left/right = the 33 / 263 sides of the face mesh)
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
LEFT_EYE_TOP = 159
RIGHT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_BOTTOM = 374
LEFT_BROW = 105
RIGHT_BROW = 334
LEFT_INNER_BROW = 107
RIGHT_INNER_BROW = 336
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
UPPER_LIP = 13
LOWER_LIP = 14

# the landmarks used for the measurements, and the lines drawn between them
TRACKED_LANDMARKS = [
    LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_EYE_INNER, RIGHT_EYE_INNER,
    LEFT_EYE_TOP, RIGHT_EYE_TOP, LEFT_EYE_BOTTOM, RIGHT_EYE_BOTTOM,
    LEFT_BROW, RIGHT_BROW, LEFT_INNER_BROW, RIGHT_INNER_BROW,
    LEFT_MOUTH, RIGHT_MOUTH, UPPER_LIP, LOWER_LIP
]

TRACKED_CONNECTIONS = [
    (LEFT_EYE_OUTER, RIGHT_EYE_OUTER),
    (LEFT_EYE_TOP, LEFT_EYE_BOTTOM),
    (RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM),
    (LEFT_BROW, LEFT_EYE_TOP),
    (RIGHT_BROW, RIGHT_EYE_TOP),
    (LEFT_INNER_BROW, LEFT_EYE_INNER),
    (RIGHT_INNER_BROW, RIGHT_EYE_INNER),
    (LEFT_INNER_BROW, RIGHT_INNER_BROW),
    (LEFT_MOUTH, UPPER_LIP),
    (UPPER_LIP, RIGHT_MOUTH),
    (LEFT_MOUTH, LOWER_LIP),
    (LOWER_LIP, RIGHT_MOUTH)
]

# how far a feature must move from the neutral baseline (in interocular
# distances) for an expression to score 1.0
SMILE_WIDTH = 0.10
SMILE_LIFT = 0.05
FROWN_LIFT = 0.03
INNER_BROW_RAISE = 0.04
BROW_LOWER = 0.04
BROW_PINCH = 0.05
EYE_SQUINT = 0.02
BROW_ASYMMETRY = 0.04
MOUTH_TILT = 0.03

EXPRESSIONS = ["happy", "sad", "angry", "confused"]
EXPRESSION_THRESHOLD = 0.4
CALIBRATION_FRAMES = 45

"""
TODO:
Automatic calibration
 -find a frame where the face is neutral (frontal, mouth closed, brows
  steady for a few frames) and calibrate from it instead of pressing C
 -keep the C key as a manual recalibrate
 -maybe save the baseline per person (see face_memory.py)
"""

baseline = None
calibration_samples = []
calibrating = False

# recent scores for each expression, averaged to stop flickering
score_history = {
    name: deque(maxlen=10) for name in EXPRESSIONS
}


def face_points(landmarks, w, h):
    # pixel positions of every landmark, rotated so the eye line is level
    points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

    left = points[LEFT_EYE_OUTER]
    right = points[RIGHT_EYE_OUTER]
    angle = np.arctan2(right[1] - left[1], right[0] - left[0])
    center = (left + right) / 2

    cos, sin = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[cos, -sin], [sin, cos]])

    return (points - center) @ rotation.T


def extract_features(landmarks, w, h):
    p = face_points(landmarks, w, h)

    # everything is divided by the distance between the eyes so the
    # measurements don't change with distance from the camera
    iod = np.linalg.norm(p[RIGHT_EYE_OUTER] - p[LEFT_EYE_OUTER]) + 1e-9

    # smaller y = higher on screen, so these are flipped to be positive when up
    mouth_center_y = (p[UPPER_LIP][1] + p[LOWER_LIP][1]) / 2
    left_lift = (mouth_center_y - p[LEFT_MOUTH][1]) / iod
    right_lift = (mouth_center_y - p[RIGHT_MOUTH][1]) / iod

    left_brow = (p[LEFT_EYE_TOP][1] - p[LEFT_BROW][1]) / iod
    right_brow = (p[RIGHT_EYE_TOP][1] - p[RIGHT_BROW][1]) / iod

    left_inner = (p[LEFT_EYE_INNER][1] - p[LEFT_INNER_BROW][1]) / iod
    right_inner = (p[RIGHT_EYE_INNER][1] - p[RIGHT_INNER_BROW][1]) / iod

    left_eye = (p[LEFT_EYE_BOTTOM][1] - p[LEFT_EYE_TOP][1]) / iod
    right_eye = (p[RIGHT_EYE_BOTTOM][1] - p[RIGHT_EYE_TOP][1]) / iod

    return {
        "mouth_width": np.linalg.norm(p[RIGHT_MOUTH] - p[LEFT_MOUTH]) / iod,
        "mouth_lift": (left_lift + right_lift) / 2,
        "mouth_tilt": abs(left_lift - right_lift),
        "brow_height": (left_brow + right_brow) / 2,
        "brow_asymmetry": abs(left_brow - right_brow),
        "inner_brow_height": (left_inner + right_inner) / 2,
        "brow_gap": abs(p[RIGHT_INNER_BROW][0] - p[LEFT_INNER_BROW][0]) / iod,
        "eye_openness": (left_eye + right_eye) / 2
    }


def make_baseline(samples):
    return {
        name: float(np.mean([sample[name] for sample in samples]))
        for name in samples[0]
    }


def ramp(value, full):
    # 0 when the feature hasn't moved, 1 when it moved by `full` or more
    return max(0.0, min(1.0, value / full))


def happy_score(d):
    return (
        ramp(d["mouth_width"], SMILE_WIDTH)
        + ramp(d["mouth_lift"], SMILE_LIFT)
    ) / 2


def sad_score(d):
    # mouth corners pulled down + inner brows pulled up
    return (
        ramp(-d["mouth_lift"], FROWN_LIFT)
        + ramp(d["inner_brow_height"], INNER_BROW_RAISE)
    ) / 2


def angry_score(d):
    # brows lowered and pinched together, eyes narrowed
    return (
        ramp(-d["brow_height"], BROW_LOWER)
        + ramp(-d["brow_gap"], BROW_PINCH)
        + ramp(-d["eye_openness"], EYE_SQUINT)
    ) / 3


def confused_score(d):
    # one brow higher than the other, mouth pulled to one side
    return (
        ramp(d["brow_asymmetry"], BROW_ASYMMETRY) * 0.7
        + ramp(d["mouth_tilt"], MOUTH_TILT) * 0.3
    )


def detect_expression(features):
    delta = {name: features[name] - baseline[name] for name in features}

    scores = {
        "happy": happy_score(delta),
        "sad": sad_score(delta),
        "angry": angry_score(delta),
        "confused": confused_score(delta)
    }

    smoothed = {}
    for name, score in scores.items():
        score_history[name].append(score)
        smoothed[name] = sum(score_history[name]) / len(score_history[name])

    best = max(smoothed, key=smoothed.get)
    if smoothed[best] < EXPRESSION_THRESHOLD:
        return "neutral", smoothed

    return best, smoothed


def draw_tracked_landmarks(frame, landmarks, w, h):
    points = {
        i: (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in TRACKED_LANDMARKS
    }

    for a, b in TRACKED_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (224, 224, 224), 2)

    for point in points.values():
        cv2.circle(frame, point, 3, (0, 0, 255), -1)


def clear_history():
    for history in score_history.values():
        history.clear()


print("Relax your face and press C to calibrate")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status = "no face"
    scores = None

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        features = extract_features(face_landmarks.landmark, w, h)

        draw_tracked_landmarks(frame, face_landmarks.landmark, w, h)

        if calibrating:
            calibration_samples.append(features)
            status = f"calibrating {len(calibration_samples)}/{CALIBRATION_FRAMES}"

            if len(calibration_samples) >= CALIBRATION_FRAMES:
                baseline = make_baseline(calibration_samples)
                calibrating = False
                clear_history()
                print("Calibrated")

        elif baseline is None:
            status = "press C with a relaxed face"

        else:
            status, scores = detect_expression(features)
    else:
        clear_history()

    cv2.putText(
        frame,
        f"Expression: {status}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # score bars
    if scores:
        for i, name in enumerate(EXPRESSIONS):
            y = 90 + i * 30
            cv2.putText(
                frame,
                name,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            cv2.rectangle(
                frame,
                (130, y - 14),
                (130 + int(scores[name] * 200), y),
                (0, 255, 0),
                -1
            )

    cv2.imshow("GLaDOS Expression Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        calibration_samples = []
        calibrating = True
    elif key == ord("q"):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
