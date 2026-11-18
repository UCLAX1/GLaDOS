import sys
import time

import cv2
import torch
from ultralytics import YOLO

# Model to test, e.g. python vision/object_tracker.py yolo26n-pose.pt
#   yolo26n.pt       boxes only
#   yolo26n-pose.pt  boxes + 17 body keypoints per person
#   yolo26n-seg.pt   boxes + a pixel mask per person
#   yolo11n.pt       boxes only, older and always uses NMS
# Swap n for s/m/l/x for bigger, slower, more accurate models.
# Weights download automatically on first run.
MODEL = sys.argv[1] if len(sys.argv) > 1 else "yolo26n.pt"

PERSON = 0  # COCO class id
CONFIDENCE = 0.4

# YOLO26 models have two output heads. The NMS-free one is trained to give
# exactly one box per person. The other gives several overlapping guesses per
# person and relies on NMS to delete the extras, which can also delete a real
# person standing close to someone else. Models without an NMS-free head
# (YOLO11) always use NMS.
NMS_FREE = True

# NMS keeps only the most confident of any boxes that overlap by more than
# this (intersection over union). Too low and one of two people standing close
# together disappears, too high and one person gets two boxes.
NMS_IOU = 0.7
NMS_IOU_STEP = 0.05

# ultralytics only uses Apple's GPU when asked, None lets it pick CUDA or CPU
DEVICE = "mps" if torch.backends.mps.is_available() else None

model = YOLO(MODEL)


def find_people(frame, nms_free, nms_iou, tracking):
  options = {
    "classes": [PERSON],
    "nms": False if nms_free else None,
    "iou": nms_iou,
    "device": DEVICE,
    "verbose": False
  }

  if not tracking:
    return model.predict(frame, conf=CONFIDENCE, **options)[0]

  # ByteTrack also uses low-confidence boxes: a half-hidden person scores low,
  # and those boxes keep their track alive until they're fully visible again.
  # So conf is left at track()'s default of 0.1 and ByteTrack does the filtering.
  result = model.track(
    frame,
    persist=True,
    tracker="bytetrack.yaml",
    **options
  )[0]

  # with nothing tracked, track() hands back the raw low-confidence boxes
  if result.boxes.id is None:
    return result[:0]

  return result


def draw_status(frame, lines):
  for i, line in enumerate(lines):
    cv2.putText(
      frame,
      line,
      (30, 50 + i * 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (0, 255, 0),
      2
    )


if __name__ == "__main__":
  cap = cv2.VideoCapture(0)
  print("Press T to toggle tracking")
  print("Press N to toggle the NMS-free head")
  print("Press [ or ] to lower or raise the NMS overlap threshold")
  print("Press Q to quit")

  nms_free = NMS_FREE
  nms_iou = NMS_IOU
  tracking = False
  fps = 0
  last_frame = time.time()

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    result = find_people(frame, nms_free, nms_iou, tracking)

    # a color per person (per track ID when tracking) so overlapping people
    # are easy to tell apart
    annotated = result.plot(color_mode="instance")

    now = time.time()
    fps = 0.9 * fps + 0.1 / max(now - last_frame, 1e-6)
    last_frame = now

    # read back what the model actually used, since models without an
    # NMS-free head ignore nms_free
    if model.predictor.model.end2end:
      nms_status = "NMS: off (NMS-free head)"
    else:
      nms_status = f"NMS IoU: {nms_iou:.2f}"

    draw_status(annotated, [
      f"People: {len(result.boxes)}",
      f"Model: {MODEL}",
      nms_status,
      f"Tracking: {'on' if tracking else 'off'}",
      f"FPS: {fps:.0f}"
    ])

    cv2.imshow("GLaDOS Person Tracker", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("t"):
      tracking = not tracking
    elif key == ord("n"):
      nms_free = not nms_free
    elif key == ord("["):
      nms_iou = max(0.05, nms_iou - NMS_IOU_STEP)
    elif key == ord("]"):
      nms_iou = min(0.95, nms_iou + NMS_IOU_STEP)
    elif key == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()
