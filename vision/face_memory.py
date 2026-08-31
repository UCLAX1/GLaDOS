
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis


FACES_DIR = "vision/known_faces"
os.makedirs(FACES_DIR, exist_ok=True)

# Load face detector & ArcFace embedding model
#buffalo_s is a smaller model, use buffalo_l for larger model
app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1)
def get_face_embedding(frame):
    faces = app.get(frame)
    if len(faces) == 0:
        return None

    # Use largest detected face
    face = max(
        faces,
        key=lambda f: (
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )
    )

    return face.normed_embedding
def register_person(name, frame):
    embedding = get_face_embedding(frame)
    if embedding is None:
        print("No face detected.")
        return

    np.save(
        f"{FACES_DIR}/{name}.npy",
        embedding
    )
    print(f"Saved face for {name}")


def recognize_person(frame, threshold=0.5):
    embedding = get_face_embedding(frame)
    if embedding is None:
        return "unknown"
    best_name = "unknown"
    best_score = 0
    for filename in os.listdir(FACES_DIR):
        if not filename.endswith(".npy"):
            continue

        known_embedding = np.load(
            os.path.join(FACES_DIR, filename)
        )
        score = np.dot(
            embedding,
            known_embedding
        )
        if score > best_score:
            best_score = score
            best_name = filename.replace(".npy", "")
    if best_score >= threshold:
        return best_name
    return "unknown"


cap = cv2.VideoCapture(0)
print("Press R to register your face")
print("Press SPACE to recognize")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow(
        "GLaDOS Face Recognition",
        frame
    )
    key = cv2.waitKey(1) & 0xFF
    # register the face 
    if key == ord("r"):
        name = input("Enter name: ")
        register_person(
            name,
            frame
        )

    elif key == ord(" "):
        name = recognize_person(frame)
        print(
            f"Recognized: {name}"
        )

cap.release()
cv2.destroyAllWindows()