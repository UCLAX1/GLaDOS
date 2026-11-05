import cv2
import mediapipe as mp
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_pose   = mp.solutions.pose
mp_hands  = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ── pose setup ────────────────────────────────────────────────────────────────
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="vision/pose_landmarker_lite.task"
    ),
    num_poses=5,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose  = vision.PoseLandmarker.create_from_options(options)
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
cap = cv2.VideoCapture(0)

# ── tuning ────────────────────────────────────────────────────────────────────
DISPLAY_THRESHOLD = 0.45   # minimum confidence to show a label at all
AMBIGUITY_WINDOW  = 0.20   # show all gestures within this of the top score
HOLD_FRAMES       = 6      # frames a gesture must be stable before confirming
WAVE_TRAVEL       = 0.10   # min horizontal wrist travel to count as wave
WAVE_HISTORY      = 20     # rolling frame window for waving

wrist_histories: dict = {}     # keyed by hand label
hand_state:      dict = {}     # debounce state per hand


# ── geometry helpers ──────────────────────────────────────────────────────────

def dist(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def signed_cross(A, B, P) -> float:
    return (B.x - A.x) * (P.y - A.y) - (B.y - A.y) * (P.x - A.x)


def point_in_triangle_score(A, B, C, P) -> float:
    """
    Soft containment: ~1.0 = P well inside ABC, ~0.0 = clearly outside.
    """
    d1 = signed_cross(A, B, P)
    d2 = signed_cross(B, C, P)
    d3 = signed_cross(C, A, P)
    inside = (d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0)
    margin = min(abs(d1), abs(d2), abs(d3))
    area   = max(abs(signed_cross(A, B, C)) / 2, 1e-6)
    score  = min(margin / area, 1.0)
    return (0.5 + score * 0.5) if inside else (0.5 - score * 0.5)


def finger_curl_score(hand, tip_lm) -> float:
    """1.0 = fully curled into palm, 0.0 = fully extended."""
    index_mcp = hand[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = hand[mp_hands.HandLandmark.PINKY_MCP]
    wrist     = hand[mp_hands.HandLandmark.WRIST]
    tip       = hand[tip_lm]
    return point_in_triangle_score(index_mcp, pinky_mcp, wrist, tip)


def all_curl_scores(hand) -> dict:
    return {
        "index":  finger_curl_score(hand, mp_hands.HandLandmark.INDEX_FINGER_TIP),
        "middle": finger_curl_score(hand, mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
        "ring":   finger_curl_score(hand, mp_hands.HandLandmark.RING_FINGER_TIP),
        "pinky":  finger_curl_score(hand, mp_hands.HandLandmark.PINKY_TIP),
    }


def finger_extension_score(hand, tip_lm, pip_lm, mcp_lm) -> float:
    """1.0 = tip well above PIP (extended), 0.0 = curled."""
    tip = hand[tip_lm]
    pip = hand[pip_lm]
    mcp = hand[mcp_lm]
    finger_len = dist(mcp, pip) + 1e-6
    extension  = (pip.y - tip.y) / finger_len
    return max(0.0, min(1.0, extension))


def thumb_raise_score(hand) -> float:
    """1.0 = thumb clearly pointing upward above wrist."""
    thumb_tip = hand[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip  = hand[mp_hands.HandLandmark.THUMB_IP]
    wrist     = hand[mp_hands.HandLandmark.WRIST]
    hand_height = abs(wrist.y - thumb_tip.y) + 1e-6
    tip_above_ip    = max(0.0, (thumb_ip.y  - thumb_tip.y) / hand_height)
    tip_above_wrist = max(0.0, (wrist.y     - thumb_tip.y) / hand_height)
    return min(1.0, (tip_above_ip + tip_above_wrist) / 2)


def thumb_lower_score(hand) -> float:
    """1.0 = thumb clearly pointing downward below wrist."""
    thumb_tip = hand[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip  = hand[mp_hands.HandLandmark.THUMB_IP]
    wrist     = hand[mp_hands.HandLandmark.WRIST]
    hand_height = abs(wrist.y - thumb_tip.y) + 1e-6
    tip_below_ip    = max(0.0, (thumb_tip.y - thumb_ip.y)  / hand_height)
    tip_below_wrist = max(0.0, (thumb_tip.y - wrist.y)     / hand_height)
    return min(1.0, (tip_below_ip + tip_below_wrist) / 2)


# ── gesture scoring ───────────────────────────────────────────────────────────

def score_thumbs_up(hand) -> float:
    curls = all_curl_scores(hand)
    fingers_closed = (
        curls["middle"] * 0.35 + curls["ring"] * 0.35
        + curls["pinky"] * 0.20 + curls["index"] * 0.10
    )
    thumb_up = thumb_raise_score(hand)
    thumb_ip = hand[mp_hands.HandLandmark.THUMB_IP]
    other_tips = [hand[lm] for lm in [
        mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,  mp_hands.HandLandmark.PINKY_TIP,
    ]]
    fingers_below = sum(max(0.0, min(1.0, (t.y - thumb_ip.y) / 0.05))
                        for t in other_tips) / 4
    return thumb_up * 0.40 + fingers_closed * 0.35 + fingers_below * 0.25


def score_thumbs_down(hand) -> float:
    curls = all_curl_scores(hand)
    fingers_closed = (
        curls["middle"] * 0.35 + curls["ring"] * 0.35
        + curls["pinky"] * 0.20 + curls["index"] * 0.10
    )
    thumb_down = thumb_lower_score(hand)
    thumb_ip   = hand[mp_hands.HandLandmark.THUMB_IP]
    other_tips = [hand[lm] for lm in [
        mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,  mp_hands.HandLandmark.PINKY_TIP,
    ]]
    fingers_above = sum(max(0.0, min(1.0, (thumb_ip.y - t.y) / 0.05))
                        for t in other_tips) / 4
    return thumb_down * 0.40 + fingers_closed * 0.35 + fingers_above * 0.25


def score_open_palm(hand) -> float:
    """All fingers extended and spread — stop / halt."""
    exts = [
        finger_extension_score(hand, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                               mp_hands.HandLandmark.INDEX_FINGER_PIP,
                               mp_hands.HandLandmark.INDEX_FINGER_MCP),
        finger_extension_score(hand, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        finger_extension_score(hand, mp_hands.HandLandmark.RING_FINGER_TIP,
                               mp_hands.HandLandmark.RING_FINGER_PIP,
                               mp_hands.HandLandmark.RING_FINGER_MCP),
        finger_extension_score(hand, mp_hands.HandLandmark.PINKY_TIP,
                               mp_hands.HandLandmark.PINKY_PIP,
                               mp_hands.HandLandmark.PINKY_MCP),
    ]
    avg_ext = sum(exts) / len(exts)
    tips    = [hand[lm] for lm in [
        mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,  mp_hands.HandLandmark.PINKY_TIP,
    ]]
    spread_score = max(0.0, min(1.0, dist(tips[0], tips[-1]) / 0.15))
    return avg_ext * 0.6 + spread_score * 0.4


def score_fist(hand) -> float:
    curls = all_curl_scores(hand)
    return (curls["index"] + curls["middle"] + curls["ring"] + curls["pinky"]) / 4


def score_pointing(hand) -> tuple:
    """Returns (confidence, direction_string)."""
    curls     = all_curl_scores(hand)
    others_curled = (curls["middle"] + curls["ring"] + curls["pinky"]) / 3
    index_ext = finger_extension_score(
        hand, mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP)

    # Index should be more extended than middle to distinguish from peace sign
    middle_ext = finger_extension_score(
        hand, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
    index_dominance = max(0.0, min(1.0, (index_ext - middle_ext) / 0.2 + 0.5))

    confidence = index_ext * 0.45 + others_curled * 0.35 + index_dominance * 0.20

    index_tip = hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    dy = index_tip.y - index_mcp.y
    dx = index_tip.x - index_mcp.x
    if abs(dy) > abs(dx):
        direction = "Pointing down" if dy > 0 else "Pointing up"
    else:
        direction = "Pointing left" if dx > 0 else "Pointing right"

    return confidence, direction


def score_peace_sign(hand) -> float:
    index_ext = finger_extension_score(
        hand, mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP)
    middle_ext = finger_extension_score(
        hand, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
    curls = all_curl_scores(hand)
    others_curled = (curls["ring"] + curls["pinky"]) / 2

    index_tip  = hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    index_mcp  = hand[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    v_score    = max(0.0, min(1.0,
        (dist(index_tip, middle_tip) - dist(index_mcp, middle_mcp)) / 0.05))

    # Both fingers should be similarly extended (not one much higher)
    balance = 1.0 - min(abs(index_ext - middle_ext) / 0.3, 1.0)

    return (index_ext + middle_ext) / 2 * 0.35 + others_curled * 0.25 + v_score * 0.25 + balance * 0.15


def score_finger_gun(hand) -> float:
    """Index + thumb extended sideways; middle/ring/pinky curled."""
    index_ext = finger_extension_score(
        hand, mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP)
    curls = all_curl_scores(hand)
    others_curled = (curls["middle"] + curls["ring"] + curls["pinky"]) / 3

    # Thumb horizontal: more x-travel than y-travel
    thumb_tip = hand[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip  = hand[mp_hands.HandLandmark.THUMB_IP]
    dx = abs(thumb_tip.x - thumb_ip.x)
    dy = abs(thumb_tip.y - thumb_ip.y) + 1e-6
    thumb_horizontal = max(0.0, min(1.0, dx / dy))

    return index_ext * 0.40 + others_curled * 0.40 + thumb_horizontal * 0.20


# ── pose / body gesture scoring ───────────────────────────────────────────────

def score_hand_raised(pose_lm, hand_lm) -> float:
    wrist     = pose_lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_eye  = pose_lm[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = pose_lm[mp_pose.PoseLandmark.RIGHT_EYE.value]
    eye_y     = (left_eye.y + right_eye.y) / 2
    height_score = max(0.0, min(1.0, (eye_y - wrist.y) / 0.08))
    palm_score   = score_open_palm(hand_lm)
    return height_score * 0.5 + palm_score * 0.5


def score_waving(pose_lm, hand_label: str) -> float:
    wrist    = pose_lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    shoulder = pose_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose     = pose_lm[mp_pose.PoseLandmark.NOSE.value]

    if min(wrist.visibility, shoulder.visibility, nose.visibility) < 0.5:
        wrist_histories.pop(hand_label, None)
        return 0.0

    if not (nose.y < wrist.y < shoulder.y):
        wrist_histories.pop(hand_label, None)
        return 0.0

    hist = wrist_histories.setdefault(hand_label, deque(maxlen=WAVE_HISTORY))
    hist.append(wrist.x)

    if len(hist) < WAVE_HISTORY:
        return 0.0

    travel = max(hist) - min(hist)
    return max(0.0, min(1.0, travel / WAVE_TRAVEL))


# ── debounce state machine ────────────────────────────────────────────────────

def debounce_update(hand_label: str, best_name: str, best_conf: float) -> bool:
    """
    Returns True once the gesture has been held for HOLD_FRAMES.
    Resets the counter whenever the top gesture changes.
    """
    if hand_label not in hand_state:
        hand_state[hand_label] = {"gesture": best_name, "hold": 0}

    state = hand_state[hand_label]
    if best_name == state["gesture"]:
        state["hold"] = min(state["hold"] + 1, HOLD_FRAMES * 2)
    else:
        state["gesture"] = best_name
        state["hold"]    = 1

    return state["hold"] >= HOLD_FRAMES


# ── evaluate all gestures for one hand ───────────────────────────────────────

def evaluate_hand(hand, hand_label: str, pose_lm=None) -> list:
    """
    Returns list of (name, confidence, confirmed) sorted by confidence.
    Shows multiple labels when scores are close (within AMBIGUITY_WINDOW).
    """
    pointing_conf, pointing_dir = score_pointing(hand)

    candidates = [
        ("Thumbs up",   score_thumbs_up(hand)),
        ("Thumbs down", score_thumbs_down(hand)),
        ("Open palm",   score_open_palm(hand)),
        ("Fist",        score_fist(hand)),
        ("Peace sign",  score_peace_sign(hand)),
        ("Finger gun",  score_finger_gun(hand)),
        (pointing_dir,  pointing_conf),
    ]

    if pose_lm is not None:
        candidates += [
            ("Hand raised", score_hand_raised(pose_lm, hand)),
            ("Waving",      score_waving(pose_lm, hand_label)),
        ]

    candidates.sort(key=lambda x: x[1], reverse=True)

    if not candidates or candidates[0][1] < DISPLAY_THRESHOLD:
        return []

    top_conf = candidates[0][1]
    visible  = [(name, conf) for name, conf in candidates
                if conf >= DISPLAY_THRESHOLD and top_conf - conf <= AMBIGUITY_WINDOW]

    confirmed = debounce_update(hand_label, visible[0][0], visible[0][1])

    return [(name, conf, confirmed) for name, conf in visible]


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    for start, end in mp_pose.POSE_CONNECTIONS:
        a, b = landmarks[start], landmarks[end]
        if getattr(a, "visibility", 1.0) < 0.5 or getattr(b, "visibility", 1.0) < 0.5:
            continue
        cv2.line(frame,
                 (int(a.x * w), int(a.y * h)),
                 (int(b.x * w), int(b.y * h)),
                 (255, 180, 0), 2)
    for pt in landmarks:
        if getattr(pt, "visibility", 1.0) < 0.5:
            continue
        cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, (0, 255, 0), -1)


def conf_color(conf: float) -> tuple:
    """Interpolate green → orange by confidence."""
    r = int((1.0 - conf) * 220)
    g = int(conf * 220)
    return (0, g, r)


# ── main loop ─────────────────────────────────────────────────────────────────

timestamp_ms = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_ms += 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_results = pose.detect_for_video(mp_image, timestamp_ms)
        hand_results = hands.process(rgb)

        pose_lm = None
        if pose_results.pose_landmarks:
            pose_lm = pose_results.pose_landmarks[0]
            draw_pose(frame, pose_lm)

        per_hand: dict = {}

        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness,
            ):
                hand_label = handedness.classification[0].label  # "Left" / "Right"
                hand       = hand_landmarks.landmark

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                results = evaluate_hand(hand, hand_label, pose_lm)
                if results:
                    per_hand[hand_label] = results

        # ── display ───────────────────────────────────────────────────────────
        y = 40
        if not per_hand:
            cv2.putText(frame, "No gesture", (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 2)
        else:
            for hand_label, gestures in per_hand.items():
                cv2.putText(frame, f"{hand_label} hand:", (30, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
                y += 30
                for name, conf, confirmed in gestures:
                    pct   = int(conf * 100)
                    color = conf_color(conf)
                    # Dim unconfirmed gestures (still building hold)
                    if not confirmed:
                        color = tuple(c // 2 for c in color)
                    label = f"  {name}  {pct}%"
                    cv2.putText(frame, label, (30, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                    y += 30
                y += 10

        cv2.imshow("GLaDOS Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    pose.close()
    hands.close()
    cv2.destroyAllWindows()