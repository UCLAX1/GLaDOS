import cv2
import mediapipe as mp
from collections import deque
import math

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

"""
FIX HAND IS RAISED GESUTRE TO NOT OVERRIDE
"""
def hand_is_raised(landmarks,hand):
    
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

    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    #shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye= landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

    wrist_over_eye = (
        (wrist.y < left_eye.y) and (wrist.y < right_eye.y)
    )

    hand_palm_vertical= (
        (index_finger_mcp.y>index_finger_tip.y) and (middle_finger_mcp.y>middle_finger_tip.y) and 
        (ring_finger_mcp.y>ring_finger_tip.y) and (pinky_mcp.y>pinky_tip.y) and 
        (thumb_mcp.y>thumb_tip.y)
    )

    # smaller y = higher on screen
    return wrist_over_eye and hand_palm_vertical
#seems alright, only parameter is hand must be above eye/head

def is_waving(landmarks):

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

    index_finger_pip = hand[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_finger_pip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_finger_pip =hand[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip =hand[mp_hands.HandLandmark.PINKY_PIP]
    thumb_ip = hand[mp_hands.HandLandmark.THUMB_IP]

    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye= landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

    wrist_under_eye = (
        (wrist.y > left_eye.y) and (wrist.y > right_eye.y)
    )

    hand_palm_vertical= (
        (index_finger_mcp.y>index_finger_tip.y) and (middle_finger_mcp.y>middle_finger_tip.y) and 
        (ring_finger_mcp.y>ring_finger_tip.y) and (pinky_mcp.y>pinky_tip.y) and 
        (thumb_mcp.y>thumb_tip.y)
        and (index_finger_pip.y>index_finger_tip.y) and (middle_finger_pip.y>middle_finger_tip.y) and 
        (ring_finger_pip.y>ring_finger_tip.y) and (pinky_pip.y>pinky_tip.y)
    )
    wrist_over_sholder = (
        (wrist.y < shoulder.y) and (wrist.y < shoulder.y)
    )
    
    # smaller y = higher on screen
    return wrist_under_eye and hand_palm_vertical  and wrist_over_sholder

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
#hand must be extenteded upright similar parameter as hand is raised
#also what is "waving" an action/series of movement or just hand near ur face?
#add smt about probability
def distance_two_point(landmark_1, landmark_2):
    return (abs((landmark_1.y-landmark_2.y))**2+abs((landmark_1.x-landmark_2.x))**2)**(1/2)
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
    closed_fingers = (
        #math can be used to account for distanc gaps and difs and etc, just dont ever use constant variables wihtout reason aka adding x
        #box ratio mcp pinky and index, then either pink or index to wrist

        #(distance_two_point(thumb_tip, thumb_mcp)<(distance_two_point(thumb_pip,thumb_mcp)))
        (distance_two_point(middle_finger_tip, middle_finger_mcp)<distance_two_point(middle_finger_pip,middle_finger_mcp))
        and (distance_two_point(ring_finger_tip, ring_finger_mcp)<distance_two_point(ring_finger_pip,ring_finger_mcp))
        and (distance_two_point(pinky_tip, pinky_mcp)<distance_two_point(pinky_pip,pinky_mcp))
    )
    """
    #closed_finger_cross_test use function
    closed_finger = (
    (distance_two_point(middle_finger_tip, wrist)<distance_two_point(middle_finger_mcp,wrist))
    and (distance_two_point(ring_finger_tip, wrist)<distance_two_point(ring_finger_mcp,wrist))
    and (distance_two_point(pinky_tip, wrist)<distance_two_point(pinky_mcp,wrist))
    and (distance_two_point(pinky_mcp, thumb_tip) < distance_two_point(thumb_mcp, pinky_mcp))
    #closed_finger_cross_test(index_finger_mcp,pinky_mcp, wrist, index_finger_tip)
    #and closed_finger_cross_test(index_finger_mcp,pinky_mcp, wrist, middle_finger_tip)
    #and closed_finger_cross_test(index_finger_mcp,pinky_mcp, wrist, ring_finger_tip)
    #and closed_finger_cross_test(index_finger_mcp,pinky_mcp, wrist, pinky_tip)
    )

    closed_finger2 = (
        (distance_two_point(middle_finger_tip,middle_finger_mcp)<distance_two_point(middle_finger_pip,middle_finger_mcp))
        and (distance_two_point(ring_finger_tip,ring_finger_mcp)<distance_two_point(ring_finger_pip,ring_finger_mcp))
        and (distance_two_point(pinky_tip,pinky_mcp)<distance_two_point(pinky_pip,pinky_mcp))
        and (distance_two_point(pinky_mcp, thumb_tip) < distance_two_point(thumb_mcp, pinky_mcp))
    )



    """
    print(f"(distance_two_point(middle_finger_tip, middle_finger_mcp)<distance_two_point(middle_finger_pip,middle_finger_mcp)): {(distance_two_point(middle_finger_tip, middle_finger_mcp)<distance_two_point(middle_finger_pip,middle_finger_mcp))}")
    print(f"(distance_two_point(ring_finger_tip, ring_finger_mcp)<distance_two_point(ring_finger_pip,ring_finger_mcp)): {(distance_two_point(ring_finger_tip, ring_finger_mcp)<distance_two_point(ring_finger_pip,ring_finger_mcp))}")
    print(f"(distance_two_point(pinky_tip, pinky_mcp)<distance_two_point(pinky_pip,pinky_mcp)): {(distance_two_point(pinky_tip, pinky_mcp)<distance_two_point(pinky_pip,pinky_mcp))}")
    """
    
    """
    print(f"direction(middle_finger_tip,middle_finger_mcp): { direction(middle_finger_tip,middle_finger_mcp)}")
    print(f"direction(ring_finger_tip,ring_finger_mcp): {direction(ring_finger_tip,ring_finger_mcp)}")
    print(f"direction(pinky_tip,pinky_mcp): {direction(pinky_tip,pinky_mcp)}")
    print(f"direction(thumb_tip,thumb_mcp): {direction(thumb_tip,thumb_mcp)}")
    """

    
    
    """
    print(f"(distance_two_point(middle_finger_tip, wrist)<distance_two_point(middle_finger_mcp,wrist)): {(distance_two_point(middle_finger_tip, wrist)<distance_two_point(middle_finger_mcp,wrist))}")
    print(f"(distance_two_point(ring_finger_tip, wrist)<distance_two_point(ring_finger_mcp,wrist)): {(distance_two_point(ring_finger_tip, wrist)<distance_two_point(ring_finger_mcp,wrist))}")
    print(f"(distance_two_point(pinky_tip, wrist)<distance_two_point(pinky_mcp,wrist)): {(distance_two_point(pinky_tip, wrist)<distance_two_point(pinky_mcp,wrist))}")
    print(f"(distance_two_point(pinky_mcp, thumb_tip) < distance_two_point(thumb_mcp, pinky_mcp)): {(distance_two_point(pinky_mcp, thumb_tip) < distance_two_point(thumb_mcp, pinky_mcp))}")
    
    print(f"closed_finger: {closed_finger}")
    print(f"closed_finger2: {closed_finger2}")
    """
    """
    JUST add directional check for the fingers potentialy like yifan, suggested, all remember thumbs up/down still use iffy code for "closedfinger" parameter
    direction of wrist can be measured with wrist and some mcp
    check if all finger under index direcrtion maybe
    account for differfnet angels
    rember u can add bunch of extra checks it doesnt need to be one check for each finger closed
    """

    if ((abs(index_finger_tip.y-index_finger_mcp.y) > abs(index_finger_tip.x-index_finger_mcp.x)) and (closed_finger or closed_finger2)):
        print("vertical")
        if (index_finger_tip.y>index_finger_mcp.y):
            return "pointing down"
            #all finger other than thumb should point up, thumb points right(according to ur pov)
        else:
            return "pointing up"
            #all finger other than thumb should point down, thumb points left(according to ur pov)
    elif(closed_finger or closed_finger2): #i forgot to add closed finger check -_-
        print("horizontal")
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
    closed_fingers = (
        (abs(index_finger_tip.x-index_finger_mcp.x)<abs(index_finger_pip.x-index_finger_mcp.x))
        and (abs(middle_finger_tip.x-middle_finger_mcp.x)<abs(middle_finger_pip.x-middle_finger_mcp.x))
        and (abs(ring_finger_tip.x-ring_finger_mcp.x)<abs(ring_finger_pip.x-ring_finger_mcp.x))
        and (abs(pinky_tip.x-pinky_mcp.x)<abs(pinky_pip.x-pinky_mcp.x))
    )
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
    
    closed_fingers = (
        (abs(index_finger_tip.x-index_finger_mcp.x)<abs(index_finger_pip.x-index_finger_mcp.x))
        and (abs(middle_finger_tip.x-middle_finger_mcp.x)<abs(middle_finger_pip.x-middle_finger_mcp.x))
        and (abs(ring_finger_tip.x-ring_finger_mcp.x)<abs(ring_finger_pip.x-ring_finger_mcp.x))
        and (abs(pinky_tip.x-pinky_mcp.x)<abs(pinky_pip.x-pinky_mcp.x))
    )

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
        
        
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
                )
                hand = hand_landmarks.landmark
                
               
                

                pointing = pointing_direction(hand)
                #body gesture
                if hand_is_raised(landmarks,hand):
                    gesture = "hand raised"
                elif is_waving(landmarks):
                    gesture = "waving"
                elif (pointing == "pointing down"):
                    gesture = pointing
                elif (pointing == "pointing up"):
                    gesture = pointing
                elif (pointing == "pointing left"):
                    gesture = pointing
                elif (pointing == "pointing right"):
                    gesture = pointing
                
                #hand gesture
                if siccor(hand):
                    gesture = "siccor"
                if closed_fist(hand):
                    gesture = "closed fist"
                if thumbs_up(hand):
                    gesture = "thumbs up"
                
                if thumbs_down(hand):
                    gesture = "thumbs down"
                    
                if peace_sign(hand):
                    gesture = "peace sign"

             
                # hand gesture overrides body gesture
                
            
            
            

    
    
    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    #more text
    # cv2.putText(
    #     frame,
    #     f"cAKE: piyhgf",
    #     (60, 90),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (0, 255, 0),
    #     2
    # )

    cv2.imshow("Glados Gesture Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
pose.close()
hands.close()
cv2.destroyAllWindows()