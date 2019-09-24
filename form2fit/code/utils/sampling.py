"""Sampling functions for non-matches.
"""

import numpy as np

from form2fit.code.utils.misc import rotate_uv


def make1d(uv, num_cols):
    uv = np.round(uv).astype("int")
    linear = uv[:, 0] * num_cols + uv[:, 1]
    return linear.astype("int")


def make2d(linear, num_cols):
    v = linear % num_cols
    u = linear // num_cols
    return np.vstack([u, v]).T


def remove_outliers(uv_1d, inside_2d, num_cols):
    u_min, u_max = np.min(inside_2d[:, 0]), np.max(inside_2d[:, 0])
    v_min, v_max = np.min(inside_2d[:, 1]), np.max(inside_2d[:, 1])
    uv_2d = make2d(uv_1d, num_cols)
    u_cond = np.logical_or(uv_2d[:, 0] <= u_min, uv_2d[:, 0] > u_max)
    v_cond = np.logical_or(uv_2d[:, 1] >= v_max, uv_2d[:, 1] < v_min)
    valid_ind = np.logical_or(u_cond, v_cond)
    uv_2d = uv_2d[valid_ind]
    return make1d(uv_2d, num_cols)


def sample_non_matches(
    num_nm, shape, angle, mask_source=None, mask_target=None, rotate=True, cxcy=None,
):
    """Randomly generate negative correspondences between a source and target image.

    An optional mask can be provided for both the source and target to restrict
    the sampling space.

    Args:
        num_nm: (int) The number of negative correspondences to sample.
        shape: (tuple) The height and width of the source and target images.
        source_mask: (ndarray) ...
        target_mask: (ndarray) ...

    Returns:
        source_non_matches: (ndarray) The (u, v) coordinates of the negative
            correspondences in the source image.
        target_non_matches: (ndarray) The (u, v) coordinates of the positive
            correspondences in the target image.
    """
    # define sampling spaces boundaries
    source_ss = (np.arange(0, shape[0] * shape[1]) if mask_source is None else mask_source)
    target_ss = (np.arange(0, shape[0] * shape[1]) if mask_target is None else mask_target)

    # rotate source indices
    if rotate:
        source_ss = rotate_uv(make2d(source_ss, shape[1]), angle, *shape, cxcy=cxcy)
        source_ss[:, 0] = np.clip(source_ss[:, 0], 0, shape[0] - 1)
        source_ss[:, 1] = np.clip(source_ss[:, 1], 0, shape[1] - 1)
        source_ss = make1d(source_ss, shape[1])

    # subset sampling with replacement
    source_nm = np.random.choice(source_ss, size=num_nm, replace=True)
    target_nm = np.random.choice(target_ss, size=num_nm, replace=True)

    # convert to 2D coordinates
    source_nm = make2d(source_nm, shape[1])
    target_nm = make2d(target_nm, shape[1])

    return np.hstack([source_nm, target_nm])


def non_matches_from_matches(
    num_nm, shape, angle, mask_source, matches_source, matches_target, cxcy=None,
):
    """Create negative correspondences from positive ones.

    This ensures that we do not create negative correspondences that are
    actually positive.
    """
    matches_source_1d = make1d(matches_source, shape[1])
    matches_target_1d = make1d(matches_target, shape[1])

    source_ss = rotate_uv(make2d(mask_source, shape[1]), angle, *shape, cxcy=cxcy)
    source_ss[:, 0] = np.clip(source_ss[:, 0], 0, shape[0] - 1)
    source_ss[:, 1] = np.clip(source_ss[:, 1], 0, shape[1] - 1)
    mask_source = make1d(source_ss, shape[1])

    source_nm = []
    target_nm = []
    while len(source_nm) < num_nm:
        source_idx = np.random.choice(mask_source)
        source_loc = np.where(matches_source_1d == source_idx)[0]
        if source_loc.size != 0:
            actual_target = matches_target_1d[source_loc[0]]
            target_idx = np.random.choice(matches_target_1d)
            while target_idx == actual_target:
                target_idx = np.random.choice(matches_target_1d)
        else:
            target_idx = np.random.choice(matches_target_1d)
        source_nm.append(source_idx)
        target_nm.append(target_idx)
    source_nm = np.array(source_nm)
    target_nm = np.array(target_nm)

    # convert to 2D coordinates
    source_nm = make2d(source_nm, shape[1])
    target_nm = make2d(target_nm, shape[1])

    return np.hstack([source_nm, target_nm])
