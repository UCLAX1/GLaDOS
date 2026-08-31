import cv2
import mediapipe as mp
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

# Store recent right-wrist x positions for wave detection
wrist_history = deque(maxlen=20)


def hand_is_raised(landmarks):
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # smaller y = higher on screen
    return wrist.y < shoulder.y


def is_waving(landmarks):
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    wrist_history.append(wrist.x)
    if len(wrist_history) < wrist_history.maxlen:
        return False

    movement = max(wrist_history) - min(wrist_history)
    return hand_is_raised(landmarks) and movement > 0.12


def pointing_direction(landmarks):
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    # Arm should be roughly horizontal
    wrist_shoulder_vertical_diff = abs(wrist.y - shoulder.y)
    elbow_shoulder_vertical_diff = abs(elbow.y - shoulder.y)
    if wrist_shoulder_vertical_diff > 0.15:
        return None
    if elbow_shoulder_vertical_diff > 0.15:
        return None

    # Arm must also be extended
    horizontal_extension = abs(wrist.x - shoulder.x)
    if horizontal_extension < 0.20:
        return None


    if wrist.x > shoulder.x:
        return "pointing right"
    if wrist.x < shoulder.x:
        return "pointing left"

    return None


while True:
    ret, frame = cap.read()

    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    gesture = "none"
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS 
       )
        pointing = pointing_direction(landmarks)

        if is_waving(landmarks):
            gesture = "waving"
        elif pointing:
            gesture = pointing
        elif hand_is_raised(landmarks):
            gesture = "hand raised"

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("GLaDOS Gesture Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
pose.close()
cv2.destroyAllWindows()