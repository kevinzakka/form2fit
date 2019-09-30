import os

import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d

from skimage.draw import circle

from form2fit.code.utils.misc import rotate_img


def plot_rgbd(rgb, depth, gray=False, name=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes[0].imshow(rgb, cmap="gray" if gray else None)
    axes[1].imshow(depth)
    for ax in axes:
        ax.axis("off")
    if name is not None:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
        plt.savefig("./plots/{}.png".format(name), format="png", dpi=150)
    plt.show()


def plot_img(img, figsize=(30, 30), uv=None, gray=False):
    fig = plt.figure(figsize=figsize)
    plt.imshow(img, cmap="gray" if gray else None)
    if uv is not None:
        plt.scatter(uv[:, 1], uv[:, 0], c="b")
    plt.axis("off")
    plt.show()


def view_pc(xyzrgbs, frame=True, gray=False):
    """Displays a list of colored pointclouds.
    """
    pcs = []
    for xyzrgb in xyzrgbs:
        pts = xyzrgb[:, :3].copy().astype(np.float64)
        if gray:
            clrs = np.repeat((xyzrgb[:, 3:].copy()).astype(np.float64), 3, axis=1)
        else:
            clrs = xyzrgb[:, 3:].copy().astype(np.float64)
        pc = o3d.PointCloud()
        pc.points = o3d.Vector3dVector(pts)
        pc.colors = o3d.Vector3dVector(clrs)
        pcs.append(pc)
    if frame:
        pcs.append(o3d.create_mesh_coordinate_frame(size=0.1, origin=[0, 0, 0]))
    o3d.draw_geometries(pcs)


def plot_correspondences(
    height_s,
    height_t,
    label,
    matches=5,
    non_matches=5,
    num_rotations=20,
    name=None,
    gray=False,
):
    # store image size for later use
    H, W = height_s.shape[:2]

    source_uv = label[:, :2].numpy()
    target_uv = label[:, 2:].numpy()
    rot_indices = label[:, 4].numpy()
    is_match = label[:, 5].numpy()

    rot_step_size = 360 / num_rotations
    rotations = np.array([rot_step_size * i for i in range(num_rotations)])

    fig, axes = plt.subplots(
        5, 4, figsize=(40, 20), gridspec_kw={"wspace": 0, "hspace": 0}
    )
    for rot_idx, (rot, ax) in enumerate(zip(rotations, axes.flatten())):
        height_s_r = rotate_img(height_s, -rot)
        height_combined = np.hstack([height_s_r, height_t])
        ax.imshow(height_combined, cmap="gray" if gray else None)

        # plot matches
        mask = np.logical_and(is_match == 1.0, rot_indices == rot_idx)
        if mask.sum() > 0:
            f_s = source_uv[mask]
            f_t = target_uv[mask]
            rand_idxs = np.random.choice(len(f_s), replace=False, size=matches)
            vs = [f_s[rand_idxs, 1], f_t[rand_idxs, 1] + W]
            us = [f_s[rand_idxs, 0], f_t[rand_idxs, 0]]
            ax.plot(vs, us, "wo--", linewidth=3.0, alpha=0.4)

        # sample and plot non-matches
        non_match_idxs = np.random.choice(
            np.where(np.logical_and(rot_indices == rot_idx, is_match == 0.0))[0],
            replace=False,
            size=non_matches,
        )
        f_s = source_uv[non_match_idxs]
        f_t = target_uv[non_match_idxs]
        vs = [f_s[:, 1], f_t[:, 1] + W]
        us = [f_s[:, 0], f_t[:, 0]]
        ax.plot(vs, us, "ro--", linewidth=3.0, alpha=0.4)
        ax.axis("off")
    if name is not None:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
        plt.savefig("./plots/{}.png".format(name), format="png", dpi=150)
    plt.show()


def plot_suction(height_c, height_d, suction, name=None, gray=True):
    suction = suction.detach().cpu().numpy()
    pos_suction = suction[suction[:, 2] == 1]
    neg_suction = suction[suction[:, 2] != 1]
    fig, axes = plt.subplots(1, 2, figsize=(20, 40))
    for ax, im in zip(axes, [height_c, height_d]):
        ax.imshow(im)
        ax.scatter(pos_suction[:, 1], pos_suction[:, 0], color="b")
        ax.scatter(neg_suction[:, 1], neg_suction[:, 0], color="r")
        ax.axis("off")
    if name is not None:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
        plt.savefig("./plots/{}.png".format(name), format="png", dpi=150)
    plt.show()


def plot_placement(height_c, height_d, placement, name=None, gray=True):
    placement = placement.detach().cpu().numpy()
    pos_placement = placement[placement[:, 2] == 1]
    neg_placement = placement[placement[:, 2] != 1]
    fig, axes = plt.subplots(1, 2, figsize=(20, 40))
    for ax, im in zip(axes, [height_c, height_d]):
        ax.imshow(im)
        ax.scatter(pos_placement[:, 1], pos_placement[:, 0], color="b")
        ax.scatter(neg_placement[:, 1], neg_placement[:, 0], color="r")
        ax.axis("off")
    if name is not None:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
        plt.savefig("./plots/{}.png".format(name), format="png", dpi=150)
    plt.show()


def plot_loss(arr, window=50, figsize=(20, 10), name=None):
    def _rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    arr = np.asarray(arr)
    fig, ax = plt.subplots(figsize=figsize)
    rolling_mean = np.mean(_rolling_window(arr, 50), 1)
    rolling_std = np.std(_rolling_window(arr, 50), 1)
    plt.plot(range(len(rolling_mean)), rolling_mean, alpha=0.98, linewidth=0.9)
    plt.fill_between(
        range(len(rolling_std)),
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=0.5,
    )
    plt.grid()
    plt.xlabel("Iteration #")
    plt.ylabel("Loss")
    if name is not None:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
        plt.savefig("./plots/{}.png".format(name), format="png", dpi=150)
    plt.show()


def plot_losses(arr1, arr2, figsize=(20, 10), name=None):
    fig, ax = plt.subplots(figsize=figsize)
    for arr, lbl in zip([arr1, arr2], ["train", "test"]):
        ax.plot(arr, linewidth=0.9, label=lbl)
    plt.grid()
    plt.xlabel("Iteration #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    if name is not None:
        if not os.path.exists("./plots/"):
            os.makedirs("./plots/")
        plt.savefig("./plots/{}.png".format(name), format="png", dpi=150)
    else:
        u_min, v_min = misc.make2d(min_val, w)
        plt.show()