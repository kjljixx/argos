import cv2
import numpy as np

COLOR_DIFF_THRESHOLD = 20
MIN_AREA_THRESHOLD = 70

def get_purpleness(frame):
  b = frame[:, :, 0].astype(np.int16)
  g = frame[:, :, 1].astype(np.int16)
  r = frame[:, :, 2].astype(np.int16)
  return np.clip(np.min([r, b], axis=0) - g, 0, 255).astype(np.uint8)

def get_detections(frame, prev_frame):
  purpleness = get_purpleness(frame)
  prev_purpleness = get_purpleness(prev_frame)
  
  diff = purpleness.astype(np.int16) - prev_purpleness.astype(np.int16)
  mask = diff > COLOR_DIFF_THRESHOLD

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
  mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
  contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA_THRESHOLD]

  return contours