import numpy as np
import scipy.optimize

class Tracker:
  def __init__(self, num_objects, velocity_alpha=0.8):
    self.curr_id_counter = 0
    self.curr_ids = np.empty((0,))
    self.curr_coords = np.empty((0, 2))
    self.curr_velocities = np.empty((0, 2))
    self.velocity_alpha = velocity_alpha
    self.num_objects = num_objects
  
  def update(self, detection_coords):
    pred_coords = self.curr_coords + self.curr_velocities

    cost = np.sum((detection_coords[:, None] - pred_coords[None]) ** 2, axis=2)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)

    new_ids = np.full(len(detection_coords), -1, dtype=int)
    new_ids[row_ind] = col_ind

    created_ids = new_ids == -1
    num_new = np.sum(created_ids)
    new_ids[created_ids] = np.arange(self.curr_id_counter, self.curr_id_counter + num_new)
    self.curr_id_counter += num_new

    self.curr_ids = new_ids
    self.curr_coords[col_ind] = detection_coords[row_ind]
    self.curr_coords = np.concatenate([self.curr_coords, detection_coords[created_ids]], axis=0)
    self.curr_velocities[col_ind] = (detection_coords[row_ind] - self.curr_coords[col_ind])*self.velocity_alpha + self.curr_velocities[col_ind]*(1-self.velocity_alpha)
    self.curr_velocities = np.concatenate([self.curr_velocities, np.zeros((num_new, 2))], axis=0)

    return new_ids