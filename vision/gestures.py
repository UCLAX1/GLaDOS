
import cv2
import mediapipe as mp
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="vision/pose_landmarker_lite.task"
    ),
    num_poses=5,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = vision.PoseLandmarker.create_from_options(options)

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

    return (
        wrist.visibility > 0.5
        and shoulder.visibility > 0.5
        and wrist.y < shoulder.y
    )


def is_waving(landmarks):
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    if min(wrist.visibility, shoulder.visibility, nose.visibility) < 0.5:
        wrist_history.clear()
        return False

    # Waving must be below the head and above the shoulder.
    correct_height = nose.y < wrist.y < shoulder.y

    if not correct_height:
        wrist_history.clear()
        return False

    wrist_history.append(wrist.x)

    if len(wrist_history) < wrist_history.maxlen:
        return False

    movement = max(wrist_history) - min(wrist_history)

    return movement > 0.12


def pointing_direction(landmarks):
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    if min(wrist.visibility, elbow.visibility, shoulder.visibility) < 0.5:
        return None

    if abs(wrist.y - shoulder.y) > 0.15:
        return None

    if abs(elbow.y - shoulder.y) > 0.15:
        return None

    horizontal_extension = abs(wrist.x - shoulder.x)

    if horizontal_extension < 0.20:
        return None

    if wrist.x > shoulder.x:
        return "pointing right"

    if wrist.x < shoulder.x:
        return "pointing left"

    return None


def fingers_extended(hand):
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
        hand[tip.value].y < hand[pip.value].y
        for tip, pip in zip(finger_tips, finger_pips)
    ]


def thumbs_up(hand):
    thumb_tip = hand[mp_hands.HandLandmark.THUMB_TIP.value]
    thumb_ip = hand[mp_hands.HandLandmark.THUMB_IP.value]
    wrist = hand[mp_hands.HandLandmark.WRIST.value]

    fingers = fingers_extended(hand)

    thumb_vertical = (
        thumb_tip.y < thumb_ip.y
        and thumb_tip.y < wrist.y
    )

    fingers_folded = not any(fingers)

    return thumb_vertical and fingers_folded


def thumbs_down(hand):
    thumb_tip = hand[mp_hands.HandLandmark.THUMB_TIP.value]
    thumb_ip = hand[mp_hands.HandLandmark.THUMB_IP.value]
    wrist = hand[mp_hands.HandLandmark.WRIST.value]

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


def draw_pose(frame, landmarks):
    height, width = frame.shape[:2]

    for start, end in mp_pose.POSE_CONNECTIONS:
        first = landmarks[start]
        second = landmarks[end]

        if (
            getattr(first, "visibility", 1.0) < 0.5
            or getattr(second, "visibility", 1.0) < 0.5
        ):
            continue

        p1 = (int(first.x * width), int(first.y * height))
        p2 = (int(second.x * width), int(second.y * height))

        cv2.line(frame, p1, p2, (255, 180, 0), 2)

    for point in landmarks:
        if getattr(point, "visibility", 1.0) < 0.5:
            continue

        x = int(point.x * width)
        y = int(point.y * height)

        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


timestamp_ms = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_ms += 33

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        pose_results = pose.detect_for_video(
            mp_image,
            timestamp_ms
        )

        hand_results = hands.process(rgb_frame)

        gestures = []

        # Body gestures
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks[0]

            draw_pose(frame, landmarks)

            if is_waving(landmarks):
                gestures.append("Waving")

            pointing = pointing_direction(landmarks)

            if pointing:
                gestures.append(pointing.capitalize())

            if hand_is_raised(landmarks):
                gestures.append("Hand raised")
        else:
            wrist_history.clear()

        # Hand gestures
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand = hand_landmarks.landmark

                hand_gesture = detect_hand_gesture(hand)

                if hand_gesture:
                    gestures.append(hand_gesture.capitalize())

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # Remove duplicate labels
        gestures = list(dict.fromkeys(gestures))

        if not gestures:
            gestures = ["None"]

        # Display all detected gestures simultaneously
        cv2.putText(
            frame,
            "Detected Gestures:",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        for i, label in enumerate(gestures):
            cv2.putText(
                frame,
                label,
                (30, 75 + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Glados Gesture Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    pose.close()
    hands.close()
    cv2.destroyAllWindows()
