import cv2
import mediapipe as mp
from collections import deque

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

wrist_history = deque(maxlen=20)


def hand_is_raised(landmarks):
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # smaller y = higher on screen
    return wrist.y < shoulder.y


def is_waving(landmarks):
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    wrist_history.append(wrist.x)
    if len(wrist_history) < wrist_history.maxlen:
        return False
    movement = max(wrist_history) - min(wrist_history)

    # wrist must be above shoulder but below head
    correct_height = shoulder.y > wrist.y > nose.y
    return correct_height and movement > 0.12
#fix this so waving is only below the head 

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


def fingers_extended(hand):
    
    # returns whether index, middle, ring, and pinky are extended
    # smaller y = higher on screen
  

    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    finger_pips = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]

    return [
        hand[tip].y < hand[pip].y
        for tip, pip in zip(finger_tips, finger_pips)
    ]


def thumbs_up(hand):
    thumb_tip = hand[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand[mp_hands.HandLandmark.THUMB_IP]
    wrist = hand[mp_hands.HandLandmark.WRIST]
    fingers = fingers_extended(hand)
    thumb_vertical = (
        thumb_tip.y < thumb_ip.y
        and thumb_tip.y < wrist.y
    )

    fingers_folded = not any(fingers)
    return thumb_vertical and fingers_folded


def thumbs_down(hand):
    thumb_tip = hand[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand[mp_hands.HandLandmark.THUMB_IP]
    wrist = hand[mp_hands.HandLandmark.WRIST]
    fingers = fingers_extended(hand)

    thumb_vertical = (
        thumb_tip.y > thumb_ip.y
        and thumb_tip.y > wrist.y
    )
    fingers_folded = not any(fingers)
    return thumb_vertical and fingers_folded

def detect_hand_gesture(hand):

    if thumbs_up(hand):
        return "thumbs up"

    if thumbs_down(hand):
        return "thumbs down"

    return None
    


while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    gesture = "none"

    # body gestures
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark

        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        pointing = pointing_direction(landmarks)
        if is_waving(landmarks):
            gesture = "waving"
        elif pointing:
            gesture = pointing
        elif hand_is_raised(landmarks):
            gesture = "hand raised"

    # hand gestures
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hand = hand_landmarks.landmark
            hand_gesture = detect_hand_gesture(hand)

            # hand gesture overrides body gesture
            if hand_gesture:
                gesture = hand_gesture

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            
    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Glados Gesture Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
pose.close()
hands.close()
cv2.destroyAllWindows()