import numpy as np
import scipy.optimize

class Tracker:
  def __init__(self, velocity_alpha=0.8, max_lost_frames=5):
    self.curr_id_counter = 0
    self.tracks = {}
    self.velocity_alpha = velocity_alpha
    self.max_lost_frames = max_lost_frames
  
  def update(self, detection_coords, frame_num):
    if len(detection_coords) == 0:
      for tracker_id in list(self.tracks.keys()):
        if frame_num - self.tracks[tracker_id]['last_seen'] < self.max_lost_frames:
          continue
        del self.tracks[tracker_id]
      return np.array([], dtype=int)

    curr_ids = np.array(list(self.tracks.keys()))
    curr_coords = np.array([self.tracks[tracker_id]['coords'] for tracker_id in curr_ids])
    curr_velocities = np.array([self.tracks[tracker_id]['velocity'] for tracker_id in curr_ids])

    if len(curr_coords) > 0:
      pred_coords = curr_coords + curr_velocities

      cost = np.sum((detection_coords[:, None] - pred_coords[None]) ** 2, axis=2)
      row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)

      new_ids = np.full(len(detection_coords), -1, dtype=int)
      new_ids[row_ind] = curr_ids[col_ind]
    else:
      new_ids = np.full(len(detection_coords), -1, dtype=int)

    created_ids = new_ids == -1
    num_new = np.sum(created_ids)
    new_ids[created_ids] = np.arange(self.curr_id_counter, self.curr_id_counter + num_new)
    self.curr_id_counter += num_new

    updated_tracks = {}
    for tracker_id, coords in zip(new_ids, detection_coords):
      if tracker_id not in self.tracks:
        vel = np.zeros(2)
      else:
        vel = self.velocity_alpha*(coords - self.tracks[tracker_id]['coords']) + (1-self.velocity_alpha)*self.tracks[tracker_id]['velocity']
      updated_tracks[tracker_id] = {'coords': coords,
                                    'velocity': vel,
                                    'last_seen': frame_num}

    unseen_ids = curr_ids[~np.isin(curr_ids, new_ids)]
    for tracker_id in unseen_ids:
      if frame_num - self.tracks[tracker_id]['last_seen'] < self.max_lost_frames:
        updated_tracks[tracker_id] = self.tracks[tracker_id]

    self.tracks = updated_tracks
    return new_ids