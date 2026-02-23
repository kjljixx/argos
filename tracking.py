import numpy as np
import scipy.optimize

class Tracker:
  def __init__(self, velocity_alpha=0.8, max_lost_frames=5, max_distance=None, max_ids=None):
    self.curr_id_counter = 0
    self.tracks = {}
    self.velocity_alpha = velocity_alpha
    self.max_lost_frames = max_lost_frames
    self.max_distance = max_distance
    self.max_ids = max_ids

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
    curr_is_new = np.array([self.tracks[tracker_id]['is_new'] for tracker_id in curr_ids])

    if len(curr_coords) > 0:
      pred_coords = curr_coords + curr_velocities
      cost = np.sum((detection_coords[:, None] - pred_coords[None]) ** 2, axis=2)
      if self.max_distance is not None:
        cost[(np.sqrt(cost) > self.max_distance) &
             (~(curr_is_new[None, :] & (np.sqrt(cost) <= self.max_distance*2)))] = 1e9
      row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
      if self.max_distance is not None:
        valid = (
          (np.sqrt(np.sum((detection_coords[row_ind] - pred_coords[col_ind]) ** 2, axis=1)) <= self.max_distance) |
          (curr_is_new[col_ind] & (np.sqrt(np.sum((detection_coords[row_ind] - pred_coords[col_ind]) ** 2, axis=1)) <= self.max_distance*2))
        )
        row_ind, col_ind = row_ind[valid], col_ind[valid]
      new_ids = np.full(len(detection_coords), -1, dtype=int)
      new_ids[row_ind] = curr_ids[col_ind]
    else:
      new_ids = np.full(len(detection_coords), -1, dtype=int)
    
    if self.max_ids is not None:
      created_ids = (new_ids == -1)[:(self.max_ids - self.curr_id_counter)]
      if len(created_ids) < len(new_ids):
        created_ids = np.concatenate([created_ids, np.full(len(new_ids) - len(created_ids), False)]) 
    else:
      created_ids = (new_ids == -1)
    num_new = np.sum(created_ids)
    new_ids[created_ids] = np.arange(self.curr_id_counter, self.curr_id_counter + num_new)
    self.curr_id_counter += num_new

    updated_tracks = {}
    for tracker_id, coords in zip(new_ids, detection_coords):
      if tracker_id < 0:
        continue
      if tracker_id not in self.tracks:
        vel = np.zeros(2)
      else:
        vel = self.velocity_alpha*(coords - self.tracks[tracker_id]['coords']) + (1-self.velocity_alpha)*self.tracks[tracker_id]['velocity']
      updated_tracks[tracker_id] = {'coords': coords,
                                    'velocity': vel,
                                    'is_new': True if tracker_id not in self.tracks else False,
                                    'last_seen': frame_num}
    
    unseen_ids = curr_ids[~np.isin(curr_ids, new_ids)]
    for tracker_id in unseen_ids:
      if frame_num - self.tracks[tracker_id]['last_seen'] < self.max_lost_frames:
        updated_tracks[tracker_id] = self.tracks[tracker_id]
        updated_tracks[tracker_id]['is_new'] = False
    self.tracks = updated_tracks
    return new_ids