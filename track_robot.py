from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

NUM_ROBOTS = 4
DETECTION_MIN_AREA = 500
DETECTION_MAX_AREA = 40000
CONTAINED_MARGIN = 12
FIELD_BOUNDS = np.array([
  [100, 50],
  [540, 50],
  [600, 590],
  [40, 590],
], dtype=np.int32)

def get_detections(frame, model):
  results = model(frame, verbose=False)[0]
  detections = sv.Detections.from_ultralytics(results)
  return detections

def filter_detections(detections):
  filter_reasons = {}
  bounds_mask = np.array([
    cv2.pointPolygonTest(FIELD_BOUNDS, ((d[0]+d[2])/2, (d[1]+d[3])/2), False) >= 0
    for d in detections.xyxy
  ], dtype=bool)
  for i in np.where(~bounds_mask)[0]:
    filter_reasons[i] = "bounds"
  kept_indices = list(np.where(bounds_mask)[0])
  detections = detections[bounds_mask]

  areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
  area_mask = (areas >= DETECTION_MIN_AREA) & (areas <= DETECTION_MAX_AREA)
  for i in np.where(~area_mask)[0]:
    filter_reasons[kept_indices[i]] = "area"
  kept_indices = [kept_indices[i] for i in np.where(area_mask)[0]]

  detections = detections[area_mask]
  if len(detections) > 1:
    keep = np.ones(len(detections), dtype=bool)
    for i in range(len(detections)):
      if not keep[i]:
        continue
      for j in range(len(detections)):
        if i == j or not keep[j]:
          continue
        if (detections.xyxy[j][0] >= detections.xyxy[i][0] - CONTAINED_MARGIN and
            detections.xyxy[j][1] >= detections.xyxy[i][1] - CONTAINED_MARGIN and
            detections.xyxy[j][2] <= detections.xyxy[i][2] + CONTAINED_MARGIN and
            detections.xyxy[j][3] <= detections.xyxy[i][3] + CONTAINED_MARGIN):
          keep[j] = False
          filter_reasons[kept_indices[j]] = "contained"
    kept_indices = [kept_indices[i] for i in range(len(keep)) if keep[i]]
    detections = detections[keep]
  
  if len(detections) > NUM_ROBOTS:
    top_indices = detections.confidence.argsort()[-NUM_ROBOTS:]
    for i in range(len(detections)):
      if i not in top_indices:
        filter_reasons[kept_indices[i]] = "confidence"
    kept_indices = [kept_indices[i] for i in top_indices]
    detections = detections[top_indices]
  kept_indices = set(kept_indices)

  return detections, filter_reasons, kept_indices