# Vision

Software related to visual input and human interaction

## Setup

Requires Python 3.12. MediaPipe 0.10.21, the last release with the `mp.solutions` API used here, has no wheels for newer Python versions.

```bash
python3.12 -m venv vision/.venv
source vision/.venv/bin/activate
pip install -r vision/requirements.txt
```

## Gesture Detection

`gestures.py` uses MediaPipe Pose with a webcam to detect body landmarks and recognize gestures.

Currently detects:
- Hand raised
- Waving
- Pointing left
- Pointing right

Run:

```bash
python vision/gestures.py
```

## Expression Detection

`expression.py` uses MediaPipe Face Mesh landmarks with a webcam to recognize facial expressions.

Measurements such as mouth width, mouth corner lift, brow height and brow asymmetry are taken from the face landmarks and divided by the distance between the eyes. They are compared with a neutral baseline calibrated on the current person, scored per expression, and averaged over the last 10 frames to avoid flickering.

Currently detects:
- Happy
- Sad
- Angry
- Confused
- Neutral (no expression above threshold)

Run:

```bash
python vision/expression.py
```

Controls:
- `C`: calibrate the neutral baseline (hold a relaxed face)
- `Q`: quit

```text
Camera
→ Face mesh landmarks
→ Normalized measurements
→ Difference from neutral baseline
→ Expression scores
→ Smoothing
→ Expression
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