import tracking
import track_artifact
import track_robot

import cv2
import numpy as np
import supervision as sv
import time
from ultralytics import YOLO

DEBUG = True

VIDEO_PATH = r"C:\Users\kjlji\Videos\Captures\2025-2026 Season_ Bensalem Area Qualifier - YouTube — Zen Browser 2026-02-18 20-27-14 decimated.mp4"

original_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
output_info = sv.VideoInfo(width=original_info.width, height=original_info.height, fps=30)

purple_tracker = tracking.Tracker(velocity_alpha=0.95, max_lost_frames=0, max_distance=50)
green_tracker = tracking.Tracker(velocity_alpha=0.95, max_lost_frames=0, max_distance=50)

GOAL_ZONES = [
  np.array([[ 52, 373],
       [ 64, 373],
       [ 69, 389],
       [162, 434],
       [140, 346],
       [125, 346],
       [125, 330],
       [ 65, 365],
       [ 57, 356]])
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

def process_video():
  generator = sv.get_video_frames_generator(VIDEO_PATH)

  prev_frames = []
  prev_purple_tracker_ids = None
  prev_green_tracker_ids = None
  prev_purple_tracks = None
  prev_green_tracks = None
  scores = []
  artifact_id_to_robot_id = {}
  filtered_detections = None
  for frame_index, frame in enumerate(generator):
    if frame_index % round(original_info.fps / 30) > 0:
      continue
    
    print(f"{time.time()} Processing frame {frame_index}")

    frame = cv2.resize(frame, (640, 640))
    purple_detections = track_artifact.get_purple_detections(frame, prev_frames[0] if len(prev_frames) > 0 else frame)
    green_detections = track_artifact.get_green_detections(frame, prev_frames[0] if len(prev_frames) > 0 else frame)

    print(f"{time.time()} Got {len(purple_detections)} purple detections and {len(green_detections)} green detections")

    purple_moments = [cv2.moments(c) for c in purple_detections]
    purple_detections_centers = np.array([
      [m['m10'] / m['m00'], m['m01'] / m['m00']]
      for m in purple_moments
    ])
    purple_tracker_ids = purple_tracker.update(purple_detections_centers, frame_index)

    green_moments = [cv2.moments(c) for c in green_detections]
    green_detections_centers = np.array([
      [m['m10'] / m['m00'], m['m01'] / m['m00']]
      for m in green_moments
    ])
    green_tracker_ids = green_tracker.update(green_detections_centers, frame_index)

    print(f"{time.time()} Updated tracker, now tracking {len(green_tracker.tracks)} artifacts")

    if prev_purple_tracker_ids is not None:
      assert prev_purple_tracks is not None
      for tracker_id in prev_purple_tracker_ids:
        if tracker_id not in purple_tracker_ids:
          for goal_index, goal in enumerate(GOAL_ZONES):
            in_goal = cv2.pointPolygonTest(
              goal, (prev_purple_tracks[int(tracker_id)]["coords"][0], prev_purple_tracks[int(tracker_id)]["coords"][1]), False
            ) >= 0
            if in_goal:
              scores.append((frame_index, "purple", goal_index, artifact_id_to_robot_id[("purple", tracker_id)] if ("purple", tracker_id) in artifact_id_to_robot_id else None))
      to_remove = set()
      for tracker_id in purple_tracker_ids:
        if tracker_id not in prev_purple_tracker_ids:
          closest_robot_id = None
          closest_robot_dist = float('inf')
          for robot_id, robot_track in robot_tracker.tracks.items():
            dist = np.linalg.norm(robot_track["coords"] - purple_tracker.tracks[int(tracker_id)]["coords"])
            if dist < closest_robot_dist:
              closest_robot_dist = dist
              closest_robot_id = robot_id
          if closest_robot_id is not None and closest_robot_dist < 100:
            artifact_id_to_robot_id[("purple", tracker_id)] = closest_robot_id

          to_remove.add(tracker_id)
          for lz_index, launch_zone in enumerate(LAUNCH_ZONES):
            in_launch_zone = cv2.pointPolygonTest(
              launch_zone, (purple_tracker.tracks[int(tracker_id)]["coords"][0], purple_tracker.tracks[int(tracker_id)]["coords"][1]), False
            ) >= 0
            if in_launch_zone:
              to_remove.remove(tracker_id)
      for tracker_id in to_remove:
        del purple_tracker.tracks[tracker_id]
      purple_detections = [c for c, tracker_id in zip(purple_detections, purple_tracker_ids) if tracker_id not in to_remove]
      purple_tracker_ids = np.array([tracker_id for tracker_id in purple_tracker_ids if tracker_id not in to_remove])

    if prev_green_tracker_ids is not None:
      assert prev_green_tracks is not None
      for tracker_id in prev_green_tracker_ids:
        if tracker_id not in green_tracker_ids:
          for goal_index, goal in enumerate(GOAL_ZONES):
            in_goal = cv2.pointPolygonTest(
              goal, (prev_green_tracks[int(tracker_id)]["coords"][0], prev_green_tracks[int(tracker_id)]["coords"][1]), False
            ) >= 0
            if in_goal:
              scores.append((frame_index, "green", goal_index, artifact_id_to_robot_id[("green", tracker_id)] if ("green", tracker_id) in artifact_id_to_robot_id else None))
      to_remove = set()
      for tracker_id in green_tracker_ids:
        if tracker_id not in prev_green_tracker_ids:
          closest_robot_id = None
          closest_robot_dist = float('inf')
          for robot_id, robot_track in robot_tracker.tracks.items():
            dist = np.linalg.norm(robot_track["coords"] - green_tracker.tracks[int(tracker_id)]["coords"])
            if dist < closest_robot_dist:
              closest_robot_dist = dist
              closest_robot_id = robot_id
          if closest_robot_id is not None and closest_robot_dist < 100:
            artifact_id_to_robot_id[("green", tracker_id)] = closest_robot_id

          to_remove.add(tracker_id)
          for lz_index, launch_zone in enumerate(LAUNCH_ZONES):
            in_launch_zone = cv2.pointPolygonTest(
              launch_zone, (green_tracker.tracks[int(tracker_id)]["coords"][0], green_tracker.tracks[int(tracker_id)]["coords"][1]), False
            ) >= 0
            if in_launch_zone:
              to_remove.remove(tracker_id)
      for tracker_id in to_remove:
        del green_tracker.tracks[tracker_id]
      green_detections = [c for c, tracker_id in zip(green_detections, green_tracker_ids) if tracker_id not in to_remove]
      green_tracker_ids = np.array([tracker_id for tracker_id in green_tracker_ids if tracker_id not in to_remove])

    print(f"{time.time()}: Scored/Filtered")

    if len(prev_frames) < 2:
      prev_frames.append(frame)
    else:
      prev_frames = prev_frames[1:] + [frame]
    prev_purple_tracker_ids = purple_tracker_ids
    prev_green_tracker_ids = green_tracker_ids
    prev_purple_tracks = purple_tracker.tracks.copy()
    prev_green_tracks = green_tracker.tracks.copy()

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
  return scores

palette_rgb = [
  (230, 25, 75),
  (60, 180, 75),
  (0, 130, 200),
  (255, 225, 25),
]
palette = sv.ColorPalette([sv.Color(*c) for c in palette_rgb])
box_annotator = sv.RoundBoxAnnotator(color=palette, color_lookup=sv.annotators.utils.ColorLookup.TRACK, thickness=2)

with sv.VideoSink(f"output/{time.time()}.mp4", output_info) as sink:
  scores = process_video()
  print(scores)
  generator = sv.get_video_frames_generator(VIDEO_PATH)

  score_index = 0
  for frame_index, frame in enumerate(generator):
    if frame_index % round(original_info.fps / 30) > 0:
      continue
    
    print(f"{time.time()} Annotating frame {frame_index}")

    frame = cv2.resize(frame, (640, 640))

    annotated_frame = frame.copy()
    for goal in GOAL_ZONES:
      cv2.polylines(annotated_frame, [goal], isClosed=True, color=(0, 255, 255), thickness=2)
    for launch_zone in LAUNCH_ZONES:
      cv2.polylines(annotated_frame, [launch_zone], isClosed=True, color=(255, 0, 255), thickness=2)
    annotated_frame = cv2.resize(annotated_frame, (original_info.width, original_info.height))

    pos_idx = 0
    while score_index < len(scores) and scores[score_index][0] <= frame_index:
        score = scores[score_index]
        text = f"{score[1]} scored in goal {score[2]}" + (f" by robot {score[3]}" if score[3] is not None else "")
        cv2.putText(annotated_frame, text, (10, 30 + pos_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        score_index += 1
        pos_idx += 1
    sink.write_frame(annotated_frame)
  print(scores)