import tracking
import track_artifact
import track_robot

import cv2
import numpy as np
import supervision as sv
import time
from ultralytics import YOLO

video_path = r"C:\Users\kjlji\Videos\Captures\2025-2026 Season_ Bensalem Area Qualifier - YouTube — Zen Browser 2026-02-18 20-27-14.mp4"

generator = sv.get_video_frames_generator(video_path)

original_info = sv.VideoInfo.from_video_path(video_path)
output_info = sv.VideoInfo(width=original_info.width, height=original_info.height, fps=30)

green_tracker = tracking.Tracker(velocity_alpha=0.95, max_lost_frames=0, max_distance=50)

GOAL_ZONES = [
  np.array([
    [67, 384],
    [126, 343],
    [139, 345],
    [160, 432],
  ])
]

LAUNCH_ZONES = [
  np.array([
    [254, 150],
    [259, 215],
    [327, 267],
    [394, 216],
    [399, 153]
  ]),
  np.array([[145, 563],
       [183, 378],
       [326, 271],
       [468, 366],
       [513, 563]])
]

#TO DO: Add launch zone where trackers can only be created there
#TO DO: remove velocity tracking for tracker ids in goal zone (bc they could get stuck on a wall)

model = YOLO("models/best3.pt")
robot_tracker = tracking.Tracker(velocity_alpha=0.0, max_lost_frames=5000000, max_distance=100, max_ids=4)

palette_rgb = [
  (230, 25, 75),
  (60, 180, 75),
  (0, 130, 200),
  (255, 225, 25),
]
palette = sv.ColorPalette([sv.Color(*c) for c in palette_rgb])
box_annotator = sv.RoundBoxAnnotator(color=palette, color_lookup=sv.annotators.utils.ColorLookup.TRACK, thickness=2)

with sv.VideoSink(f"output/{time.time()}.mp4", output_info) as sink:
  prev_frames = []
  prev_tracker_ids = None
  prev_tracks = None
  goal_scores = [0 for _ in GOAL_ZONES]
  robot_scores = [0 for _ in range(4)]
  artifact_id_to_robot_id = {}
  filtered_detections = None
  for frame_index, frame in enumerate(generator):
    if frame_index % round(original_info.fps / 30) > 0:
      continue
    
    print(f"{time.time()} Processing frame {frame_index}")
    print(f"{time.time()} Current goal scores: {goal_scores}")

    frame = cv2.resize(frame, (640, 640))
    detections = track_artifact.get_green_detections(frame, prev_frames[0] if len(prev_frames) > 0 else frame)

    print(f"{time.time()} Got {len(detections)} artifact detections")

    moments = [cv2.moments(c) for c in detections]
    detections_centers = np.array([
      [m['m10'] / m['m00'], m['m01'] / m['m00']]
      for m in moments
    ])

    tracker_ids = green_tracker.update(detections_centers, frame_index)

    print(f"{time.time()} Updated tracker, now tracking {len(green_tracker.tracks)} artifacts")

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
              if tracker_id in artifact_id_to_robot_id:
                robot_scores[artifact_id_to_robot_id[tracker_id]] += 1
      to_remove = set()
      for tracker_id in tracker_ids:
        if tracker_id not in prev_tracker_ids:
          closest_robot_id = None
          closest_robot_dist = float('inf')
          for robot_id, robot_track in robot_tracker.tracks.items():
            dist = np.linalg.norm(robot_track["coords"] - green_tracker.tracks[int(tracker_id)]["coords"])
            if dist < closest_robot_dist:
              closest_robot_dist = dist
              closest_robot_id = robot_id
          if closest_robot_id is not None and closest_robot_dist < 100:
            artifact_id_to_robot_id[tracker_id] = closest_robot_id

          to_remove.add(tracker_id)
          for lz_index, launch_zone in enumerate(LAUNCH_ZONES):
            in_launch_zone = cv2.pointPolygonTest(
              launch_zone, (green_tracker.tracks[int(tracker_id)]["coords"][0], green_tracker.tracks[int(tracker_id)]["coords"][1]), False
            ) >= 0
            if in_launch_zone:
              to_remove.remove(tracker_id)
      for tracker_id in to_remove:
        del green_tracker.tracks[tracker_id]
      detections = [c for c, tracker_id in zip(detections, tracker_ids) if tracker_id not in to_remove]
      tracker_ids = np.array([tracker_id for tracker_id in tracker_ids if tracker_id not in to_remove])

    print(f"{time.time()}: Scored/Filtered")

    annotated_frame = frame.copy()
    if len(prev_frames) > 0:
      greenness = track_artifact.get_greenness(annotated_frame, np.arange(annotated_frame.shape[1]), np.arange(annotated_frame.shape[0])[:, None])
      prev_greenness = track_artifact.get_greenness(prev_frames[0], np.arange(prev_frames[0].shape[1]), np.arange(prev_frames[0].shape[0])[:, None])
      diff = greenness.astype(np.int16) - prev_greenness.astype(np.int16)
      dim_mask = (diff <= track_artifact.COLOR_DIFF_THRESHOLD)
      overlay = np.zeros_like(annotated_frame)
      overlay[:] = (30, 30, 30)
      annotated_frame = np.where(dim_mask[..., None], cv2.addWeighted(annotated_frame, 0.5, overlay, 0.5, 0), annotated_frame)
    for goal in GOAL_ZONES:
      cv2.polylines(annotated_frame, [goal], isClosed=True, color=(0, 255, 255), thickness=2)
    for launch_zone in LAUNCH_ZONES:
      cv2.polylines(annotated_frame, [launch_zone], isClosed=True, color=(255, 0, 255), thickness=2)
    for c, tracker_id in zip(detections, tracker_ids):
      color = (int(tracker_id) * 50 % 256, int(tracker_id) * 80 % 256, int(tracker_id) * 110 % 256)
      cv2.drawContours(annotated_frame, [c], -1, color, 2)
      center = (int(c[0][0][0]), int(c[0][0][1]))
      cv2.circle(annotated_frame, center+green_tracker.tracks[int(tracker_id)]["velocity"].astype(int), 1, (0, 255), -1)
      cv2.putText(annotated_frame, str(tracker_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    print(f"{time.time()}: Annotated")

    if len(prev_frames) < 2:
      prev_frames.append(frame)
    else:
      prev_frames = prev_frames[1:] + [frame]
    prev_tracker_ids = tracker_ids
    prev_tracks = green_tracker.tracks.copy()

    if frame_index % round(original_info.fps / 15) == 0:
      print(f"{time.time()} Processing frame {frame_index}")

      detections = track_robot.get_detections(frame, model)

      print(f"{time.time()} Got {len(detections)} robot detections")

      filtered_detections, filter_reasons, kept_indices = track_robot.filter_detections(detections)

      print(f"{time.time()} Filtered to {len(filtered_detections)} robot detections")

      detections_centers = np.array([
        [(d[0] + d[2]) / 2, (d[1] + d[3]) / 2]
        for d in filtered_detections.xyxy
      ])

      filtered_detections.tracker_id = robot_tracker.update(detections_centers, frame_index)

      print(f"{time.time()} Updated robot tracker, now tracking {len(robot_tracker.tracks)} robots")

    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=filtered_detections)
    
    print(f"{time.time()}: Annotated robots")

    annotated_frame = cv2.resize(annotated_frame, (original_info.width, original_info.height))
    annotated_frame = cv2.putText(annotated_frame, f"{goal_scores}", (300, 800), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    annotated_frame = cv2.putText(annotated_frame, f"{robot_scores}", (300, 850), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    sink.write_frame(annotated_frame)

    print(f"{time.time()}: Wrote frame")