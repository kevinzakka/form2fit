"""Compute various pose estimation metrics.

Ref: https://github.com/yuxng/PoseCNN/blob/master/lib/utils/pose_error.py
"""

import cv2
import numpy as np

from form2fit.code.utils.pointcloud import transform_xyz


def rotational_error(R1, R2):
    r, _ = cv2.Rodrigues(R1.dot(R2.T))
    return np.degrees(np.linalg.norm(r))


def translational_error(t1, t2):
    return np.linalg.norm(t1 - t2)


def compute_ADD(pose_true, pred_pose, obj_xyz):
    """Computes the Average Distance Metric (ADD) [1].

    [1]: https://arxiv.org/pdf/1711.00199.pdf
    """
    obj_xyz_pred = transform_xyz(obj_xyz, pred_pose)
    obj_xyz_true = transform_xyz(obj_xyz, pose_true)
    return np.linalg.norm(obj_xyz_pred - obj_xyz_true, axis=1).mean()


def reprojection_error(pose_true, pred_pose, obj_xyz, view_bounds, pixel_size):
    obj_xyz_pred = transform_xyz(obj_xyz, pred_pose)
    obj_xyz_true = transform_xyz(obj_xyz, pose_true)
    obj_xyz_pred[:, 0] = (obj_xyz_pred[:, 0] - view_bounds[0, 0]) / pixel_size
    obj_xyz_pred[:, 1] = (obj_xyz_pred[:, 1] - view_bounds[1, 0]) / pixel_size
    obj_idx_pred = obj_xyz_pred[:, [1, 0]]
    obj_xyz_true[:, 0] = (obj_xyz_true[:, 0] - view_bounds[0, 0]) / pixel_size
    obj_xyz_true[:, 1] = (obj_xyz_true[:, 1] - view_bounds[1, 0]) / pixel_size
    obj_idx_true = obj_xyz_true[:, [1, 0]]
    return np.linalg.norm(obj_idx_true - obj_idx_pred, axis=1).mean()