import cv2
import numpy as np

COLOR_DIFF_THRESHOLD = 20
MIN_AREA_THRESHOLD = 20

def get_purpleness(frame, xs, ys):
  hue_diff = np.abs(frame[ys, xs, 0].astype(np.int16) - 145)
  hue_dist = np.minimum(hue_diff, 180 - hue_diff)
  hue_score = np.clip(1.0 - (hue_dist / 30.0), 0.0, 1.0)
  lightness_diff = np.abs(frame[ys, xs, 1].astype(np.int16) - 144)
  lightness_score = np.clip(1.0 - (lightness_diff / 50.0), 0.0, 1.0)
  saturation_diff = np.abs(frame[ys, xs, 2].astype(np.int16) - 46)
  saturation_score = np.clip(1.0 - (saturation_diff / 50.0), 0.0, 1.0)
  return (hue_score * lightness_score * saturation_score * 255).astype(np.uint8)

def get_purple_mask(frame, prev_frame):
  hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
  hls_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HLS)

  lower = np.array([145-25, 144-40, 46-70])
  upper = np.array([145+25, 144+40, 46+70])
  mask = cv2.inRange(hls_frame, lower, upper)

  ys, xs = np.where(mask > 0)
  purpleness = get_purpleness(hls_frame, xs, ys)
  prev_purpleness = get_purpleness(hls_prev_frame, xs, ys)
  
  diff = purpleness.astype(np.int16) - prev_purpleness.astype(np.int16)
  mask = mask & (diff > COLOR_DIFF_THRESHOLD)
  return mask

def get_greenness(frame, xs, ys):
  hue_diff = np.abs(frame[ys, xs, 0].astype(np.int16) - 83)
  hue_dist = np.minimum(hue_diff, 180 - hue_diff)
  hue_score = np.clip(1.0 - (hue_dist / 30.0), 0.0, 1.0)
  lightness_diff = np.abs(frame[ys, xs, 1].astype(np.int16) - 97)
  lightness_score = np.clip(1.0 - (lightness_diff / 50.0), 0.0, 1.0)
  saturation_diff = np.abs(frame[ys, xs, 2].astype(np.int16) - 127)
  saturation_score = np.clip(1.0 - (saturation_diff / 50.0), 0.0, 1.0)
  return (hue_score * lightness_score * saturation_score * 255).astype(np.uint8)

def get_green_mask(frame, prev_frame):
  hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
  hls_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HLS)

  lower = np.array([83-25, 97-40, 127-70])
  upper = np.array([83+25, 97+40, 127+70])
  mask = cv2.inRange(hls_frame, lower, upper)

  ys, xs = np.where(mask > 0)
  greenness = get_greenness(hls_frame, xs, ys)
  prev_greenness = get_greenness(hls_prev_frame, xs, ys)
  
  diff = greenness.astype(np.int16) - prev_greenness.astype(np.int16)
  mask[ys, xs] = (diff > COLOR_DIFF_THRESHOLD)
  return mask

def get_purple_detections(frame, prev_frame):
  mask = get_purple_mask(frame, prev_frame)

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
  mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
  purple_contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  purple_contours = [c for c in purple_contours if cv2.contourArea(c) > MIN_AREA_THRESHOLD]

  return purple_contours

def get_green_detections(frame, prev_frame):
  mask = get_green_mask(frame, prev_frame)

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
  mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
  green_contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  green_contours = [c for c in green_contours if cv2.contourArea(c) > MIN_AREA_THRESHOLD]

  return green_contours