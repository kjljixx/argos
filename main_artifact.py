import tracking
import track_artifact

import cv2
import numpy as np
import supervision as sv
import time

video_path = r"C:\Users\kjlji\Videos\Captures\2025-2026 Season_ Bensalem Area Qualifier - YouTube — Zen Browser 2026-02-18 20-27-14.mp4"

generator = sv.get_video_frames_generator(video_path)

original_info = sv.VideoInfo.from_video_path(video_path)
output_info = sv.VideoInfo(width=original_info.width, height=original_info.height, fps=15)

tracker = tracking.Tracker(velocity_alpha=0.8, max_lost_frames=0, max_distance=50)

GOAL_ZONES = [
  np.array([
    [67, 384],
    [126, 343],
    [139, 345],
    [160, 432],
  ])
]


with sv.VideoSink(f"output/{time.time()}.mp4", output_info) as sink:
  prev_frames = []
  prev_tracker_ids = None
  prev_tracks = None
  goal_scores = [0 for _ in GOAL_ZONES]
  for frame_index, frame in enumerate(generator):
    if frame_index % round(original_info.fps / 30) > 0:
      continue
    
    print(f"Processing frame {frame_index}")
    print(f"Current goal scores: {goal_scores}")

    frame = cv2.resize(frame, (640, 640))
    detections = track_artifact.get_detections(frame, prev_frames[0] if len(prev_frames) > 0 else frame)

    moments = [cv2.moments(c) for c in detections]
    detections_centers = np.array([
      [m['m10'] / m['m00'], m['m01'] / m['m00']]
      for m in moments
    ])

    tracker_ids = tracker.update(detections_centers, frame_index)

    if prev_tracker_ids is not None:
      assert prev_tracks is not None
      for tracker_id in prev_tracker_ids:
        if tracker_id not in tracker_ids:
          for goal_index, goal in enumerate(GOAL_ZONES):
            in_goal = cv2.pointPolygonTest(
              goal, (prev_tracks[int(tracker_id)]["coords"][0], prev_tracks[int(tracker_id)]["coords"][1]), False
            ) >= 0
            if in_goal:
              goal_scores[goal_index] += 1
      to_remove = []
      for tracker_id in tracker_ids:
        if tracker_id not in prev_tracker_ids:
          for goal_index, goal in enumerate(GOAL_ZONES):
            in_goal = cv2.pointPolygonTest(
              goal, (tracker.tracks[int(tracker_id)]["coords"][0], tracker.tracks[int(tracker_id)]["coords"][1]), False
            ) >= 0
            if in_goal:
              to_remove.append(tracker_id)
      for tracker_id in to_remove:
        del tracker.tracks[tracker_id]
      detections = [c for c, tracker_id in zip(detections, tracker_ids) if tracker_id not in to_remove]
      tracker_ids = np.array([tracker_id for tracker_id in tracker_ids if tracker_id not in to_remove])

    annotated_frame = frame.copy()
    for goal in GOAL_ZONES:
      cv2.polylines(annotated_frame, [goal], isClosed=True, color=(0, 255, 255), thickness=2)
    for c, tracker_id in zip(detections, tracker_ids):
      color = (int(tracker_id) * 50 % 256, int(tracker_id) * 80 % 256, int(tracker_id) * 110 % 256)
      cv2.drawContours(annotated_frame, [c], -1, color, 2)
      center = (int(c[0][0][0]), int(c[0][0][1]))
      cv2.putText(annotated_frame, str(tracker_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    annotated_frame = cv2.resize(annotated_frame, (original_info.width, original_info.height))
    annotated_frame = cv2.putText(annotated_frame, f"{goal_scores}", (300, 800), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if frame_index % round(original_info.fps / 30) > 0:
      pass
    else:
      sink.write_frame(annotated_frame)
    if len(prev_frames) < 2:
      prev_frames.append(frame)
    else:
      prev_frames = prev_frames[1:] + [frame]
    prev_tracker_ids = tracker_ids
    prev_tracks = tracker.tracks.copy()