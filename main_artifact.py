import tracking
import track_artifact

import cv2
import numpy as np
import supervision as sv
import time

video_path = r"C:\Users\kjlji\Downloads\coftc-10tele.mp4"

generator = sv.get_video_frames_generator(video_path)

original_info = sv.VideoInfo.from_video_path(video_path)
output_info = sv.VideoInfo(width=original_info.width, height=original_info.height, fps=15)

tracker = tracking.Tracker(velocity_alpha=0.8, max_lost_frames=0)

with sv.VideoSink(f"output/{time.time()}.mp4", output_info) as sink:
  prev_frame = None
  for frame_index, frame in enumerate(generator):
    if frame_index % round(original_info.fps / 15) > 0:
      continue
    
    print(f"Processing frame {frame_index}")

    frame = cv2.resize(frame, (640, 640))
    detections = track_artifact.get_detections(frame, prev_frame if prev_frame is not None else frame)

    moments = [cv2.moments(c) for c in detections]
    detections_centers = np.array([
      [m['m10'] / m['m00'], m['m01'] / m['m00']]
      for m in moments
    ])

    tracker_ids = tracker.update(detections_centers, frame_index)

    annotated_frame = frame.copy()
    for c, tracker_id in zip(detections, tracker_ids):
      color = (int(tracker_id) * 50 % 256, int(tracker_id) * 80 % 256, int(tracker_id) * 110 % 256)
      cv2.drawContours(annotated_frame, [c], -1, color, 2)
      center = (int(c[0][0][0]), int(c[0][0][1]))
      cv2.putText(annotated_frame, str(tracker_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    annotated_frame = cv2.resize(annotated_frame, (original_info.width, original_info.height))

    sink.write_frame(annotated_frame)
    prev_frame = frame