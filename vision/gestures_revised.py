import cv2
import mediapipe as mp
from collections import deque
import math
import os  # LOCAL TEST
import threading  # LOCAL TEST
import time  # LOCAL TEST
from concurrent.futures import ThreadPoolExecutor  # LOCAL TEST

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# LOCAL TEST: lite models by default (pose 83 -> 62 ms, hands 75 -> 65 ms on the
# Jetson CPU). Compare with the full models: GESTURE_COMPLEXITY=1 python vision/gestures_revised.py
MODEL_COMPLEXITY = int(os.environ.get("GESTURE_COMPLEXITY", "0"))
print(f"MediaPipe model_complexity={MODEL_COMPLEXITY}")
pose = mp_pose.Pose(model_complexity=MODEL_COMPLEXITY)

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=min(MODEL_COMPLEXITY, 1),  # hands only has 0 and 1
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# LOCAL TEST: hands runs on this thread while pose runs on the main one
hands_pool = ThreadPoolExecutor(max_workers=1)


class LatestFrame:
    """
    LOCAL TEST: reads the camera on a background thread and keeps only the
    newest frame. Without it, frames queue up while the models run and the
    picture lags ~0.5 s behind real life.
    """

    def __init__(self, cap):
        self.cap = cap
        self.ok, self.frame = True, None
        self.lock = threading.Lock()
        self.fresh = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, frame
            self.fresh.set()
            if not ok:
                break

    def read(self):
        """Wait for a frame newer than the last one returned."""
        self.fresh.wait()
        self.fresh.clear()
        with self.lock:
            return self.ok, self.frame

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


cap = cv2.VideoCapture(0)
# LOCAL TEST: ZED Mini gives both lenses side by side; ask for 2x 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap = LatestFrame(cap)  # LOCAL TEST
frame_times = deque(maxlen=30)  # LOCAL TEST: for the FPS counter

# LOCAL TEST: both arms, each with its own wrist history (was right arm only)
ARMS = {
    "right": {
        "wrist": mp_pose.PoseLandmark.RIGHT_WRIST.value,
        "shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    },
    "left": {
        "wrist": mp_pose.PoseLandmark.LEFT_WRIST.value,
        "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    },
}
wrist_history = {side: deque(maxlen=20) for side in ARMS}
MIN_VISIBILITY = 0.5  # pose landmarks below this are probably off-screen or hidden


def shoulder_width(landmarks):
    """Body-size yardstick, so thresholds work near and far from the camera."""
    left = landmarks[ARMS["left"]["shoulder"]]
    right = landmarks[ARMS["right"]["shoulder"]]
    return max(abs(left.x - right.x), 0.02)


def update_wrist_history(landmarks):
    """LOCAL TEST: record both wrists every frame, whether or not a hand is detected."""
    for side, ids in ARMS.items():
        wrist = landmarks[ids["wrist"]]
        if wrist.visibility < MIN_VISIBILITY:
            wrist_history[side].clear()  # arm left the frame: drop stale movement
        else:
            wrist_history[side].append(wrist.x)


def clear_wrist_history():
    for history in wrist_history.values():
        history.clear()


def assign_hands_to_arms(landmarks, hand_list):
    """
    LOCAL TEST: match each detected hand to the nearest pose wrist, so we know
    which arm it belongs to. Hands far from both wrists (another person's)
    are ignored. Returns {"right": hand, "left": hand}.
    """
    max_dist = shoulder_width(landmarks)
    pairs = []
    for i, hand in enumerate(hand_list):
        hand_wrist = hand[mp_hands.HandLandmark.WRIST]
        for side, ids in ARMS.items():
            d = distance_two_point(hand_wrist, landmarks[ids["wrist"]])
            if d < max_dist:
                pairs.append((d, i, side))

    by_arm, used = {}, set()
    for d, i, side in sorted(pairs):  # closest pairs first
        if side not in by_arm and i not in used:
            by_arm[side] = hand_list[i]
            used.add(i)
    return by_arm

"""
FIX HAND IS RAISED GESUTRE TO NOT OVERRIDE
"""
def hand_is_raised(landmarks,hand,side):

    #make it so it checks for like a extending
    #hand thing so it aint override other hand gestures
    index_finger_tip = hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip =hand[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip =hand[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = hand[mp_hands.HandLandmark.THUMB_TIP]

    index_finger_mcp = hand[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp =hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_finger_mcp =hand[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp =hand[mp_hands.HandLandmark.PINKY_MCP]
    thumb_mcp = hand[mp_hands.HandLandmark.THUMB_MCP]

    wrist = landmarks[ARMS[side]["wrist"]]  # LOCAL TEST: either arm
    #shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    # LOCAL TEST: was "wrist above the eyes" (6-16% on the recording). With the
    # camera looking up, a hand raised beside the head shows up just below the
    # nose (median 0.19 shoulder-widths below), so allow down to ~mouth level.
    # Waving happens at the same height; it is told apart by motion and is
    # checked first in detect_arm_gestures().
    wrist_over_eye = wrist.y < nose.y + 0.35 * shoulder_width(landmarks)

    hand_palm_vertical= (
        (index_finger_mcp.y>index_finger_tip.y) and (middle_finger_mcp.y>middle_finger_tip.y) and
        (ring_finger_mcp.y>ring_finger_tip.y) and (pinky_mcp.y>pinky_tip.y) and
        (thumb_mcp.y>thumb_tip.y)
    )

    # smaller y = higher on screen
    return wrist_over_eye and hand_palm_vertical
#seems alright, only parameter is hand must be above eye/head

def is_waving(landmarks, side):
    # LOCAL TEST: either arm; history is filled by update_wrist_history() every
    # frame; needs real back-and-forth (not one sweep); thresholds scale with
    # body size instead of a fixed 0.12 of the image width
    wrist = landmarks[ARMS[side]["wrist"]]
    shoulder = landmarks[ARMS[side]["shoulder"]]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    history = wrist_history[side]
    if len(history) < history.maxlen:
        return False
    xs = list(history)
    width = shoulder_width(landmarks)
    movement = max(xs) - min(xs)

    # count changes of direction, ignoring jitter under 5% of shoulder width
    # (waving from the wrist moves it only ~0.25 shoulder-widths in total)
    reversals, direction, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        step = x - anchor
        if abs(step) < 0.05 * width:
            continue
        new_direction = 1 if step > 0 else -1
        if direction and new_direction != direction:
            reversals += 1
        direction, anchor = new_direction, x

    # wrist must be above shoulder but below head
    correct_height = shoulder.y > wrist.y > nose.y
    # measured: ~0.25 shoulder-widths of wrist travel when waving (was > 0.5)
    return correct_height and movement > 0.15 * width and reversals >= 2
#fix this so waving is only below the head
#hand must be extenteded upright similar parameter as hand is raised
#also what is "waving" an action/series of movement or just hand near ur face?
#add smt about probability

def distance_two_point(landmark_1, landmark_2):
    return (abs((landmark_1.y-landmark_2.y))**2+abs((landmark_1.x-landmark_2.x))**2)**(1/2)


# LOCAL TEST: finger straightness, with thresholds measured on a labelled
# recording (vision/gesture_eval in the vision-depth worktree).
# ratio = tip-to-knuckle / middle-joint-to-knuckle distance in the image.
# Straight fingers measured ~1.9-2.2; curled fingers 0.6-1.4 (higher when the
# hand is seen side-on, which is why the old "< 1.0" missed most pointing).
CURLED_MAX = 1.6
STRAIGHT_MIN = 1.8
FOUR_FINGERS = ("INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY")


def finger_ratio(hand, finger):
    tip = hand[getattr(mp_hands.HandLandmark, f"{finger}_TIP")]
    pip = hand[getattr(mp_hands.HandLandmark, f"{finger}_PIP")]
    mcp = hand[getattr(mp_hands.HandLandmark, f"{finger}_MCP")]
    return distance_two_point(tip, mcp) / max(distance_two_point(pip, mcp), 1e-6)


def fingers_curled(hand, fingers):
    return all(finger_ratio(hand, f) < CURLED_MAX for f in fingers)

#ask how varaible types work

def closed_finger_cross_test(trig_point_A, trig_point_B, trig_point_C, point_to_check):# The Cross Product / Edge Side Test
        check1 = (point_to_check.x-trig_point_B.x)*(trig_point_A.y-trig_point_B.y)-(trig_point_A.x-trig_point_B.x)*(point_to_check.y-trig_point_B.y)
        check2 = (point_to_check.x-trig_point_C.x)*(trig_point_B.y-trig_point_C.y)-(trig_point_B.x-trig_point_C.x)*(point_to_check.y-trig_point_C.y)
        check3 = (point_to_check.x-trig_point_A.x)*(trig_point_C.y-trig_point_A.y)-(trig_point_C.x-trig_point_A.x)*(point_to_check.y-trig_point_A.y)
        if (check1>=0 and check2>=0 and check3>=0) or (check1<=0 and check2<=0 and check3<=0):
            return True
        return False

"""
using dot product
measure distance between finger tip and wrist, and then wrist and mcp, and check for if finger tip and riwst is smaller than wrist and mcp

use pythag theorm for distnace

"""

def closed_fist(hand): #also ur ear is
    index_finger_tip = hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip =hand[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip =hand[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = hand[mp_hands.HandLandmark.THUMB_TIP]

    index_finger_mcp = hand[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp =hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_finger_mcp =hand[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp =hand[mp_hands.HandLandmark.PINKY_MCP]
    thumb_mcp = hand[mp_hands.HandLandmark.THUMB_MCP]

    index_finger_pip =hand[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_finger_pip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_finger_pip =hand[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip =hand[mp_hands.HandLandmark.PINKY_PIP]
    thumb_ip = hand[mp_hands.HandLandmark.THUMB_IP]

    wrist = hand[mp_hands.HandLandmark.WRIST]

    closed_fist1 = (
    (distance_two_point(index_finger_tip, wrist)<distance_two_point(index_finger_mcp,wrist))
    and (distance_two_point(middle_finger_tip, wrist)<distance_two_point(middle_finger_mcp,wrist))
    and (distance_two_point(ring_finger_tip, wrist)<distance_two_point(ring_finger_mcp,wrist))
    and (distance_two_point(pinky_tip, wrist)<distance_two_point(pinky_mcp,wrist))
    and (distance_two_point(pinky_mcp, thumb_tip) < distance_two_point(thumb_mcp, pinky_mcp))

    )
    closed_fist2 = (
        closed_finger_cross_test(index_finger_mcp,pinky_mcp, wrist, index_finger_tip)
        and closed_finger_cross_test(index_finger_mcp,pinky_mcp, wrist, middle_finger_tip)
        and closed_finger_cross_test(index_finger_mcp,pinky_mcp, wrist, ring_finger_tip)
        and closed_finger_cross_test(index_finger_mcp,pinky_mcp, wrist, pinky_tip)
    )
    closed_fist3 = (
        (distance_two_point(index_finger_tip,index_finger_mcp)<distance_two_point(index_finger_pip,index_finger_mcp))
        and (distance_two_point(middle_finger_tip,middle_finger_mcp)<distance_two_point(middle_finger_pip,middle_finger_mcp))
        and (distance_two_point(ring_finger_tip,ring_finger_mcp)<distance_two_point(ring_finger_pip,ring_finger_mcp))
        and (distance_two_point(pinky_tip,pinky_mcp)<distance_two_point(pinky_pip,pinky_mcp))
    )

    closed_fist4= (
        (index_finger_tip.y<index_finger_pip.y and index_finger_tip.y>index_finger_mcp.y)
        and (middle_finger_tip.y<middle_finger_pip.y and middle_finger_tip.y>middle_finger_mcp.y)
        and (ring_finger_tip.y<ring_finger_pip.y and ring_finger_tip.y>ring_finger_mcp.y)
        and (pinky_tip.y<pinky_pip.y and pinky_tip.y>pinky_mcp.y)
        and ((thumb_tip.x>pinky_mcp.x and thumb_tip.x<index_finger_mcp.x) or (thumb_tip.x<pinky_mcp.x and thumb_tip.x>index_finger_mcp.x))

    )

    """
    print(f"closed_fist1: {closed_fist1}")
    print(f"closed_fist2: {closed_fist2}")
    print(f"closed_fist3: {closed_fist3}")
    print(f"closed_fist4: {closed_fist4}")
    """
    if (closed_fist1 or closed_fist2 or closed_fist3 or closed_fist4):
        return True
    else:
        return False


def direction(landmark_1, landmark_2):#landmark 2 is esseitially, pivot point, landmark 1 is the direction u want it to relate to
    if ((abs(landmark_1.y-landmark_2.y) > abs(landmark_1.x-landmark_2.x))):
        print("vertical")
        if (landmark_1.y>landmark_2.y):
            return "down"
        else:
            return "up"
    else:
        print("horizontal")
        if (landmark_1.x>landmark_2.x):
            return "left"
        else:
            return "right"


def pointing_direction(hand):
    """
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
    """
    index_finger_tip = hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip =hand[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip =hand[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = hand[mp_hands.HandLandmark.THUMB_TIP]

    index_finger_mcp = hand[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp =hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_finger_mcp =hand[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp =hand[mp_hands.HandLandmark.PINKY_MCP]
    thumb_mcp = hand[mp_hands.HandLandmark.THUMB_MCP]

    index_finger_pip =hand[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_finger_pip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_finger_pip =hand[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip =hand[mp_hands.HandLandmark.PINKY_PIP]
    thumb_ip = hand[mp_hands.HandLandmark.THUMB_IP]

    wrist = hand[mp_hands.HandLandmark.WRIST]


    """
    #Must fix so its slope distance, not only x or y
    """
    # LOCAL TEST: thumb no longer has to be curled (people often point with it out);
    # measured thresholds, see finger_ratio()
    closed_fingers = fingers_curled(hand, ("MIDDLE_FINGER", "RING_FINGER", "PINKY"))
    index_extended = finger_ratio(hand, "INDEX_FINGER") > STRAIGHT_MIN
    # LOCAL TEST: previously any visible hand returned left/right, so
    # "pointing" showed constantly; now it needs an actual pointing hand
    if not (closed_fingers and index_extended):
        return None

    # Directions are from the pointer's point of view: the ZED image is not
    # mirrored, so a finger toward the image's right points to their left.
    if abs(index_finger_tip.y-index_finger_mcp.y) > abs(index_finger_tip.x-index_finger_mcp.x):
        if (index_finger_tip.y>index_finger_mcp.y):
            return "pointing down"
            #all finger other than thumb should point up, thumb points right(according to ur pov)
        else:
            return "pointing up"
            #all finger other than thumb should point down, thumb points left(according to ur pov)
    else:
        if (index_finger_tip.x>index_finger_mcp.x):
            return "pointing left"
            #all finger other than thumb should point right, thumb points down(according to ur pov)
        else:
            return "pointing right"
            #all finger other than thumb should point left, thumb points down(according to ur pov)


    """
    #max dif u can have between fingeer points(0.045) for thumb, to account for thumb being impossible to bend like other fingers
    print(f"distance_two_point(thumb_tip, thumb_mcp)<(distance_two_point(thumb_pip,thumb_mcp)+0.045): {distance_two_point(thumb_tip, thumb_mcp)<(distance_two_point(thumb_pip,thumb_mcp)+0.045)}")
    print(f"distance_two_point(thumb_tip, thumb_mcp): {distance_two_point(thumb_tip, thumb_mcp)}")
    print(f"distance_two_point(thumb_pip,thumb_mcp): {distance_two_point(thumb_pip,thumb_mcp)}")
    """
    return None
#Change to include hand in addition to landmark
#figure out how to hand closed recognized, also figureout why index alone doesn't show on camera
#thumb dont have pip only ip acts differentl than other fginers
#change so index must be fully extended
#when palm up pointing in direction of the arm's direction relative to body


def thumbs_up(hand):
    index_finger_mcp = hand[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp =hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_finger_mcp =hand[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp =hand[mp_hands.HandLandmark.PINKY_MCP]

    index_finger_pip = hand[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_finger_pip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_finger_pip =hand[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip =hand[mp_hands.HandLandmark.PINKY_PIP]



    index_finger_tip = hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip =hand[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip =hand[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = hand[mp_hands.HandLandmark.THUMB_TIP]

    thumb_ip = hand[mp_hands.HandLandmark.THUMB_IP]
    wrist = hand[mp_hands.HandLandmark.WRIST]
    #fingers = fingers_extended(hand)
    thumb_vertical = (
        thumb_tip.y < thumb_ip.y
        and thumb_tip.y < wrist.y
    )
    other_fingers_under_thumb = (
        thumb_ip.y<index_finger_tip.y and thumb_ip.y<middle_finger_tip.y and thumb_ip.y<ring_finger_tip.y
        and thumb_ip.y<pinky_tip.y
    )
    # LOCAL TEST: was an x-only check that never passed for thumbs down (0% on the
    # recording); now the same distance-based curl test as pointing
    closed_fingers = fingers_curled(hand, FOUR_FINGERS)
    #closed_fingers = (
    #maybe go get abs value of mcp and pip and make sure its larger than abs value of dip and tip
    #problem would be it would be a constant, maybe tip mcp vs pip mcp, for flexable points of refernce
    #)
    #orientation of upright poential if thumb tip.y is in a rough parameter around wrist.y is considered "upright?"
    """
    print(f"abs(index_finger_tip.x-index_finger_mcp.x): {abs(index_finger_tip.x-index_finger_mcp.x)}")
    print(f"abs(index_finger_pip.x-index_finger_mcp.x): {abs(index_finger_pip.x-index_finger_mcp.x)}")

    print(f"thumb_tip.y {thumb_tip.y}")
    print(f"thumb_ip.y {thumb_ip.y}")
    print(f"wrist.y {wrist.y}")
    """


    #fingers_folded = not any(fingers)
    return thumb_vertical and other_fingers_under_thumb and closed_fingers#and fingers_folded

def thumbs_down(hand):

    index_finger_mcp = hand[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp =hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_finger_mcp =hand[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp =hand[mp_hands.HandLandmark.PINKY_MCP]

    index_finger_pip = hand[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_finger_pip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_finger_pip =hand[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip =hand[mp_hands.HandLandmark.PINKY_PIP]


    index_finger_tip = hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip =hand[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip =hand[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = hand[mp_hands.HandLandmark.THUMB_TIP]

    thumb_ip = hand[mp_hands.HandLandmark.THUMB_IP]
    wrist = hand[mp_hands.HandLandmark.WRIST]


    thumb_vertical = (
        thumb_tip.y > thumb_ip.y
        and thumb_tip.y > wrist.y
    )
    other_fingers_above_thumb = (
            thumb_ip.y>index_finger_tip.y and thumb_ip.y>middle_finger_tip.y and thumb_ip.y>ring_finger_tip.y
            and thumb_ip.y>pinky_tip.y
        )

    # LOCAL TEST: was an x-only check that never passed for thumbs down (0% on the
    # recording); now the same distance-based curl test as pointing
    closed_fingers = fingers_curled(hand, FOUR_FINGERS)

    return thumb_vertical and closed_fingers and other_fingers_above_thumb
#does wont at some angels specificly when palm is facing camera often

def peace_sign(hand):
    #add parameters so that fingers must be under index_finger_tip
    index_finger_tip = hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    thumb_tip =hand[mp_hands.HandLandmark.THUMB_TIP]
    ring_finger_tip =hand[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip =hand[mp_hands.HandLandmark.PINKY_TIP]

    index_finger_mcp = hand[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp =hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    index_finger_pip = hand[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_finger_pip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

    middle_finger_pip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

    wrist = hand[mp_hands.HandLandmark.WRIST]

    is_top_ = (
        abs(index_finger_tip.x-middle_finger_tip.x)>abs(index_finger_mcp.x-middle_finger_mcp.x)#check for V
        and abs(index_finger_tip.x-middle_finger_tip.x)>abs(index_finger_pip.x-middle_finger_pip.x)#addiational check for V
        and (middle_finger_tip.y<middle_finger_mcp.y) and (index_finger_tip.y<index_finger_mcp.y)#check for extented upright
        and (index_finger_tip.y<wrist.y) and (ring_finger_tip.y>middle_finger_pip.y)#check for wrist upright(2nd check) & check for down fingers
        and (thumb_tip.y>middle_finger_pip.y) and (pinky_tip.y>middle_finger_pip.y)#check for down fingers
        and (pinky_tip.y<wrist.y)# check for anti leaning pointing being a peace sign
        #and (abs(index_finger_tip.y-middle_finger_tip.y)>0.1)
        #check that peace sign abs values tip and mcp are horizontally align using parameter
    )
    #print(f"wrist.y {wrist.y}")
    """
    print(f"abs(index_finger_tip.x-middle_finger_tip.x)>abs(index_finger_mcp.x-middle_finger_mcp.x): {abs(index_finger_tip.x-middle_finger_tip.x)>abs(index_finger_mcp.x-middle_finger_mcp.x)}")
    print(f"abs(index_finger_tip.x-middle_finger_tip.x)>abs(index_finger_pip.x-middle_finger_pip.x): {abs(index_finger_tip.x-middle_finger_tip.x)>abs(index_finger_pip.x-middle_finger_pip.x)}")
    print(f"(middle_finger_tip.y<middle_finger_mcp.y): {(middle_finger_tip.y<middle_finger_mcp.y)}")
    print(f"(index_finger_tip.y<index_finger_mcp.y): {(index_finger_tip.y<index_finger_mcp.y)}")
    print(f"(index_finger_tip.y<wrist.y): {(index_finger_tip.y<wrist.y)}")
    print(f"(ring_finger_tip.y>middle_finger_pip.y): {(ring_finger_tip.y>middle_finger_pip.y)}")
    print(f"(thumb_tip.y>middle_finger_pip.y): {(thumb_tip.y>middle_finger_pip.y)}")
    print(f"(pinky_tip.y>middle_finger_pip.y): {(pinky_tip.y>middle_finger_pip.y)}")
    print(f"(pinky_tip.y<wrist.y): {(pinky_tip.y<wrist.y)}")
    print(f"(abs(index_finger_tip.y-middle_finger_tip.y)>0.1): {(abs(index_finger_tip.y-middle_finger_tip.y)>0.1)}")
    """


    return is_top_
#seems alright, bigh be a bit buggy with the middle and ring finger?

def siccor(hand):
#add parameters so that fingers must be under index_finger_tip
    index_finger_tip = hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    thumb_tip =hand[mp_hands.HandLandmark.THUMB_TIP]
    ring_finger_tip =hand[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip =hand[mp_hands.HandLandmark.PINKY_TIP]

    index_finger_mcp = hand[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp =hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_finger_mcp = hand[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand[mp_hands.HandLandmark.PINKY_MCP]
    thumb_mcp = hand[mp_hands.HandLandmark.THUMB_MCP]

    index_finger_pip = hand[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_finger_pip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_finger_pip = hand[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand[mp_hands.HandLandmark.PINKY_PIP]
    thumb_ip = hand[mp_hands.HandLandmark.THUMB_IP]

    wrist = hand[mp_hands.HandLandmark.WRIST]

    is_V = (
        abs(middle_finger_tip.y-index_finger_tip.y)>abs(middle_finger_mcp.y-index_finger_mcp.y)
        and abs(middle_finger_tip.y-index_finger_tip.y)>abs(middle_finger_pip.y-index_finger_pip.y)
    )
    finger_close = (
        (distance_two_point(ring_finger_tip, wrist)<distance_two_point(ring_finger_mcp,wrist))
        and (distance_two_point(pinky_tip, wrist)<distance_two_point(pinky_mcp,wrist))
        and (distance_two_point(pinky_mcp, thumb_tip) < distance_two_point(thumb_mcp, pinky_mcp))
    )
    finger_close1 = (
            (distance_two_point(ring_finger_mcp, ring_finger_tip)<distance_two_point(ring_finger_mcp,ring_finger_pip))
            and (distance_two_point(pinky_tip, pinky_mcp)<distance_two_point(pinky_mcp,pinky_pip))
            and (distance_two_point(ring_finger_pip, thumb_tip) < distance_two_point(thumb_mcp, ring_finger_mcp))
        )


    print(f"finger_close: {finger_close}")
    print(f"is_V: {is_V}")
    print(f"finger_close1: {finger_close1}")
    #print(f": {}")


    return is_V and finger_close
"""
Maybe add rock(fist) paper(extended hand under sholder?) sissors (peace sign but rotate 90degree)
"""


def detect_hand_gesture(hand):

    if thumbs_up(hand):
        return "thumbs up"

    if thumbs_down(hand):
        return "thumbs down"

    if peace_sign(hand):
        return "peace sign"

    return None


def detect_arm_gestures(landmarks, hand_results):
    """
    LOCAL TEST: one gesture per arm instead of "last hand wins".
    Each hand is matched to an arm via the nearest pose wrist.
    Returns e.g. ["right: thumbs up", "left: waving"]; empty if nothing.
    """
    update_wrist_history(landmarks)
    hand_list = [h.landmark for h in hand_results.multi_hand_landmarks or []]
    hands_by_arm = assign_hands_to_arms(landmarks, hand_list)

    arm_gestures = []
    for side in ("right", "left"):
        hand = hands_by_arm.get(side)
        arm_gesture = None

        # waving first: it is the only gesture defined by motion, and a waving
        # open hand otherwise reads as "hand raised" or "peace sign". It only
        # needs the pose: a moving hand is often too blurry for the hand model.
        if is_waving(landmarks, side):
            arm_gesture = "waving"
        if not arm_gesture and hand:
            #hand gesture
            # hand gesture overrides body gesture
            arm_gesture = detect_hand_gesture(hand)
            #body gesture
            if not arm_gesture and hand_is_raised(landmarks, hand, side):
                arm_gesture = "hand raised"
        if not arm_gesture and hand:
            arm_gesture = pointing_direction(hand)

        if arm_gesture:
            arm_gestures.append(f"{side}: {arm_gesture}")
    return arm_gestures


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = frame[:, :frame.shape[1] // 2]  # LOCAL TEST: left lens only
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_future = hands_pool.submit(hands.process, rgb_frame)  # LOCAL TEST: parallel
    pose_results = pose.process(rgb_frame)
    hand_results = hands_future.result()
    gesture = "none"



    # body gestures
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark

        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        arm_gestures = detect_arm_gestures(landmarks, hand_results)  # LOCAL TEST
        if arm_gestures:
            gesture = " | ".join(arm_gestures)

        # LOCAL TEST: draw every hand, not only ones making a hand gesture
        for hand_landmarks in hand_results.multi_hand_landmarks or []:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    else:
        clear_wrist_history()  # LOCAL TEST: person left the frame






    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    # LOCAL TEST: the original call had no text argument and raised on every
    # frame; it now shows the processing FPS instead
    frame_times.append(time.time())
    fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Glados Gesture Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hands_pool.shutdown()  # LOCAL TEST
pose.close()
hands.close()
cv2.destroyAllWindows()
