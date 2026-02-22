import tracking
import track_robot

import cv2
import numpy as np
import supervision as sv
import time
from ultralytics import YOLO

video_path = r"C:\Users\kjlji\Videos\Captures\2025-2026 Season_ Bensalem Area Qualifier - YouTube — Zen Browser 2026-02-18 20-27-14.mp4"

generator = sv.get_video_frames_generator(video_path)

original_info = sv.VideoInfo.from_video_path(video_path)
output_info = sv.VideoInfo(width=original_info.width, height=original_info.height, fps=15)

model = YOLO("models/best3.pt")
tracker = tracking.Tracker(velocity_alpha=0.0, max_lost_frames=5000000, max_distance=100, max_ids=4)

palette_rgb = [
  (230, 25, 75),
  (60, 180, 75),
  (0, 130, 200),
  (255, 225, 25),
]
palette = sv.ColorPalette([sv.Color(*c) for c in palette_rgb])
box_annotator = sv.RoundBoxAnnotator(color=palette, color_lookup=sv.annotators.utils.ColorLookup.TRACK, thickness=2)

with sv.VideoSink(f"output/{time.time()}.mp4", output_info) as sink:
  for frame_index, frame in enumerate(generator):
    if frame_index % round(original_info.fps / 15) > 0:
      continue
    
    print(f"Processing frame {frame_index}")

    frame = cv2.resize(frame, (640, 640))
    detections = track_robot.get_detections(frame, model)

    filtered_detections, filter_reasons, kept_indices = track_robot.filter_detections(detections)

    detections_centers = np.array([
      [(d[0] + d[2]) / 2, (d[1] + d[3]) / 2]
      for d in filtered_detections.xyxy
    ])

    filtered_detections.tracker_id = tracker.update(detections_centers, frame_index)

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=filtered_detections)
    annotated_frame = cv2.resize(annotated_frame, (original_info.width, original_info.height))

    sink.write_frame(annotated_frame)