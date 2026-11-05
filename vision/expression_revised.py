#pip install hsemotion-onnx onnxruntime
"""
import cv2
import sys
print("EXACT PYTHON PATH IS:", sys.executable)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
"""

import cv2
import sys
import os

# 1. FIX THE LIBRARY BUG MANUALLY HERE:
import urllib.request

# Now import the library safely
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

print("EXACT PYTHON PATH IS:", sys.executable)

# 2. Initialize the ONNX HSEmotion detector
model_name = 'enet_b0_8_best_afew'
fer = HSEmotionRecognizer(model_name=model_name)

# 3. Robust path loading for OpenCV's Haar Cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print(f"ERROR: Cascade file not found at {cascade_path}")
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)

print("\n--- Starting Webcam Loop ---")
print("Press 'q' in the VIDEO WINDOW to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break
        
    # Convert images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Basic tracking status indicator on the CV2 window
    cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    for (x, y, w, h) in faces:
        # Crop the face
        face_img = rgb_frame[y:y+h, x:x+w]
        
        if face_img.size == 0:
            continue
            
        try:
            # Predict emotion (ONNX implementation)
            emotion, scores = fer.predict_emotions(face_img, logits=False)
            
            # Handle formats safely
            if isinstance(scores, dict):
                confidence_pct = scores.get(emotion, 0) * 100
            else:
                confidence_pct = max(scores) * 100
            
            # TERMINAL PRINTING
            print(f"Face Found! Emotion: {emotion} ({confidence_pct:.1f}%)")
            
            # CV2 OVERLAYS
            display_text = f"{emotion} ({confidence_pct:.1f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"HSEmotion ONNX Analysis Error: {e}")

    # Display video
    cv2.imshow("Emotion Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()