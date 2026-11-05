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

def distance_two_point(landmark_1, landmark_2):
    return (abs((landmark_1.y-landmark_2.y))**2+abs((landmark_1.x-landmark_2.x))**2)**(1/2)
    
#ask how varaible types work

def pointing_direction(landmarks,hand):
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

    middle_finger_pip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_finger_pip =hand[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip =hand[mp_hands.HandLandmark.PINKY_PIP]
    thumb_pip = hand[mp_hands.HandLandmark.THUMB_IP]
        

    #Must fix so its slope distance, not only x or y
    closed_fingers = (
        (distance_two_point(thumb_tip, thumb_mcp)<distance_two_point(thumb_pip,thumb_mcp))
        and (distance_two_point(middle_finger_tip, middle_finger_mcp)<distance_two_point(middle_finger_pip,middle_finger_mcp))
        and (distance_two_point(ring_finger_tip, ring_finger_mcp)<distance_two_point(ring_finger_pip,ring_finger_mcp))
        and (distance_two_point(pinky_tip, pinky_mcp)<distance_two_point(pinky_pip,pinky_mcp))
    )
    

    if ((abs(index_finger_tip.y-index_finger_mcp.y) > abs(index_finger_tip.x-index_finger_mcp.x)) and closed_fingers):
        if (index_finger_tip.y>index_finger_mcp.y):
            return "pointing down"
        else:
            return "pointing up"
    else:
        if (index_finger_tip.x>index_finger_mcp.x):
            return "pointing left"
        else:
            return "pointing right"
        
    """
    print(f"index_finger_tip.y: {index_finger_tip.y}")
    print(f"index_finger_mcp.y: {index_finger_mcp.y}")
    print(f"index_finger_tip.y>index_finger_mcp.x: {index_finger_tip.y>index_finger_mcp.y}")#pointing down
   
    print(f"index_finger_tip.x: {index_finger_tip.x}")
    print(f"index_finger_mcp.x: {index_finger_mcp.x}")
    print(f"index_finger_tip.x>index_finger_mcp.x: {index_finger_tip.x>index_finger_mcp.x}")#pointing left
    """
    return None
#Change to include hand in addition to landmark
#figure out how to hand closed recognized, also figureout why index alone doesn't show on camera
#thumb dont have pip only ip acts differentl than other fginers
#change so index must be fully extended

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
    #print(f"Index tip: {hand[mp_hands.HandLandmark.INDEX_FINGER_TIP].y}")
    #print(f"Index pip: {hand[mp_hands.HandLandmark.INDEX_FINGER_PIP].y}")

    return [#faulty code because it only concerns the y axis for extensions, additioanlyl to 
        #only care for certain fingers maybe? when index tip > index pip, it returned true
        hand[tip].y < hand[pip].y
        for tip, pip in zip(finger_tips, finger_pips)
    ]
#lwk, could get rid

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

    middle_finger_pip =hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

    wrist = hand[mp_hands.HandLandmark.WRIST]
   
    is_top_ = (
        abs(index_finger_tip.x-middle_finger_tip.x)>abs(index_finger_mcp.x-middle_finger_mcp.x)#check for V
        and (middle_finger_tip.y<middle_finger_mcp.y) and (index_finger_tip.y<index_finger_mcp.y)#check for extented upright
        and (index_finger_tip.y<wrist.y) and (ring_finger_tip.y>middle_finger_pip.y)#check for wrist upright(2nd check) & check for down fingers
        and (thumb_tip.y>middle_finger_pip.y) and (pinky_tip.y>middle_finger_pip.y)#check for down fingers
    )
    #print(f"wrist.y {wrist.y}")
    #print(f"index_finger_tip.y {index_finger_tip.y}")
    #print(f"index_finger_mcp.y {index_finger_mcp.y}")
    


  
    return is_top_ 
#seems alright, bigh be a bit buggy with the middle and ring finger?

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
                hand = hand_landmarks.landmark
                hand_gesture = detect_hand_gesture(hand)

                pointing = pointing_direction(landmarks,hand)
                #body gesture
                if hand_is_raised(landmarks,hand):
                    gesture = "hand raised"
                elif is_waving(landmarks):
                    gesture = "waving"
                elif pointing:
                    gesture = pointing
                
                #hand gesture
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
    #more text
    cv2.putText(
        frame,
        (60, 90),
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