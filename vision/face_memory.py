
import cv2
import numpy as np
import os
import re
from insightface.app import FaceAnalysis

FACES_DIR = "vision/known_faces"
os.makedirs(FACES_DIR, exist_ok=True)

# Use buffalo_l for a larger model, buffalo_s for a smaller model
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))

THRESHOLD = 0.5
NUM_SAMPLES = 15


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


def register_person(name, cap):
    name = name.strip()

    # Keep filenames safe
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        print("Name must contain only letters, numbers, _ or -")
        return

    embeddings = []

    print(f"\nRegistering {name}...")
    print("Slowly turn your head left, right, up, and down.")
    print("Press Q to cancel registration.")

    while len(embeddings) < NUM_SAMPLES:
        ret, frame = cap.read()

        if not ret:
            print("Camera error.")
            break

        faces = app.get(frame)

        # Only register when exactly one face is visible
        if len(faces) == 1:
            embedding = faces[0].normed_embedding

            # Avoid saving nearly identical samples
            if (
                len(embeddings) == 0
                or max(
                    np.dot(embedding, old)
                    for old in embeddings
                ) < 0.995
            ):
                embeddings.append(embedding.copy())
                print(f"Captured {len(embeddings)}/{NUM_SAMPLES}")

        cv2.putText(
            frame,
            f"Registering {name}: {len(embeddings)}/{NUM_SAMPLES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("GLaDOS Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Registration canceled.")
            return

    if len(embeddings) < NUM_SAMPLES:
        print("Registration incomplete.")
        return

    embeddings = np.array(embeddings, dtype=np.float32)

    np.save(
        os.path.join(FACES_DIR, f"{name}.npy"),
        embeddings
    )

    print(f"Saved {len(embeddings)} face samples for {name}!")


def recognize_person(frame, threshold=THRESHOLD):
    faces = app.get(frame)
    results = []

    for face in faces:
        embedding = face.normed_embedding

        best_name = "unknown"
        best_score = -1.0

        for filename in os.listdir(FACES_DIR):
            if not filename.endswith(".npy"):
                continue

            known_embedding = np.load(
                os.path.join(FACES_DIR, filename),
                allow_pickle=False
            )

            # Support both old single-sample files
            # and new multi-sample files
            if known_embedding.ndim == 1:
                known_embedding = known_embedding.reshape(1, -1)

            # Compare against all saved face samples
            scores = known_embedding @ embedding
            score = float(np.max(scores))

            if score > best_score:
                best_score = score
                best_name = os.path.splitext(filename)[0]

        if best_score < threshold:
            best_name = "unknown"

        bbox = face.bbox.astype(int)
        results.append((best_name, bbox, best_score))

    return results


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press R to register a person")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Recognize every face in the frame
    results = recognize_person(frame)

    for name, bbox, score in results:
        x1, y1, x2, y2 = bbox

        color = (
            (0, 255, 0)
            if name != "unknown"
            else (0, 165, 255)
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = f"{name} ({score:.2f})"

        cv2.putText(
            frame,
            label,
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
        register_person(name, cap)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
