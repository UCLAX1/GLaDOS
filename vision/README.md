# Vision

Software related to visual input and human interaction

## Gesture Detection

`gestures.py` uses MediaPipe Pose with a webcam to detect body landmarks and recognize gestures.

Currently detects:
- Hand raised
- Waving
- Pointing left
- Pointing right

Run:

```bash
pip install -r vision/requirements.txt
python vision/gestures.py
```

## Face Memory

`face_memory.py` uses InsightFace with an ArcFace-based recognition model to recognize previously registered people.

A detected face is converted into a face embedding and saved in `vision/known_faces/`. New face embeddings are compared with saved embeddings to determine whether the person is known.

Run:

```bash
python vision/face_memory.py
```

Controls:
- `R`: register a person's face
- `SPACE`: recognize the current person

```text
Camera
→ Face detection
→ ArcFace embedding
→ Compare with saved embeddings
→ Person identity
```