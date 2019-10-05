"""Pointcloud creation and manipulation functions.
"""

from functools import reduce

import numpy as np


def transform_xyzrgb(xyzrgb, transform):
    """Applies a rigid transform to a colored pointcloud.

    Args:
        xyzrgb: (ndarray) The colored pointcloud of shape (N, 6).
        transform: (ndarray) The rigid transform of shape (4, 4).

    Returns:
        xyzrgb_t: (ndarray) The transformed colored pointcloud.
    """
    num_pts = xyzrgb.shape[0]
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:]
    xyz_h = np.hstack([xyz, np.ones((num_pts, 1))])
    xyz_t = (transform @ xyz_h.T).T
    xyzrgb_t = np.hstack([xyz_t[:, :3], rgb])
    return xyzrgb_t


def transform_xyzg(xyzg, transform):
    """Applies a rigid transform to a grayscale pointcloud.

    Args:
        xyzg: (ndarray) The grayscale pointcloud of shape (N, 4).
        transform: (ndarray) The rigid transform of shape (4, 4).

    Returns:
        xyzg_t: (ndarray) The transformed colored pointcloud.
    """
    num_pts = xyzg.shape[0]
    xyz = xyzg[:, :3]
    g = xyzg[:, 3:]
    xyz_h = np.hstack([xyz, np.ones((num_pts, 1))])
    xyz_t = (transform @ xyz_h.T).T
    xyzg_t = np.hstack([xyz_t[:, :3], g])
    return xyzg_t


def transform_xyz(xyz, transform):
    """Applies a rigid transform to a pointcloud.

    Args:
        xyz: (ndarray) The pointcloud of shape (N, 3).
        transform: (ndarray) The rigid transform of shape (4, 4).

    Returns:
        (ndarray) The transformed pointcloud.
    """
    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    xyz_t = (transform @ xyz_h.T).T
    xyz_t = xyz_t[:, :3]
    return xyz_t


def deproject(uv, depth, intr, extr):
    """2D -> 3D.
    """
    z = depth[uv[:, 0], uv[:, 1]]
    zero_zs = z == 0.0
    z = z[~zero_zs]
    uv = uv[~zero_zs]
    x = z * (uv[:, 1] - intr[0, 2]) / intr[0, 0]
    y = z * (uv[:, 0] - intr[1, 2]) / intr[1, 1]
    xyz = np.vstack([x, y, z]).T
    xyz_tr = transform_xyz(xyz, extr)
    return xyz_tr