
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
    faces = app.get(frame)
    results = []

    for face in faces:
        embedding = face.normed_embedding
        best_name = "unknown"
        best_score = -1

        for filename in os.listdir(FACES_DIR):
            if not filename.endswith(".npy"):
                continue

            known_embedding = np.load(
                os.path.join(FACES_DIR, filename)
            )

            score = np.dot(embedding, known_embedding)

            if score > best_score:
                best_score = score
                best_name = filename.replace(".npy", "")

        if best_score < threshold:
            best_name = "unknown"

        bbox = face.bbox.astype(int)
        results.append((best_name, bbox, best_score))

    return results



cap = cv2.VideoCapture(0)
print("Press R to register your face")
print("Press SPACE to recognize")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Recognize every face in each frame
    results = recognize_person(frame)

    for name, bbox, score in results:
        x1, y1, x2, y2 = bbox

        color = (0, 255, 0) if name != "unknown" else (0, 165, 255)

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), color, 2
        )

        cv2.putText(
            frame,
            name,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("GLaDOS Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        name = input("Enter name: ")
        register_person(name, frame)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
