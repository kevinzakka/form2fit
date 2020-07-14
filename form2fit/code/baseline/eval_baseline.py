"""Run the ORB-PE baseline referenced in Section V, A.
"""

import argparse
import glob
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from form2fit import config
from form2fit.code.utils.pointcloud import transform_xyz
from form2fit.code.utils import common

from walle.utils.geometry import estimate_rigid_transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ORB-PE Baseline on Benchmark")
    parser.add_argument("--debug", type=lambda s: s.lower() in ["1", "true"], default=False)
    args, unparsed = parser.parse_known_args()

    # instantiate ORB detector
    detector = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    save_dir = os.path.join("../dump/")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    kit_poses = {}
    kit_dirs = glob.glob("../../benchmark/data/train" + "/*")
    for kit_idx, data_dir in enumerate(kit_dirs):
        print(data_dir.split("/")[-1])

        train_dir = os.path.join(data_dir, "train")
        test_dir = os.path.join(data_dir, "test")

        train_foldernames = glob.glob(train_dir + "/*")
        test_foldernames = glob.glob(test_dir + "/*")
        test_foldernames.sort(key=lambda x: int(x.split("/")[-1]))

        pred_poses = []
        for test_folder in tqdm(test_foldernames, leave=False):
            # load camera params
            intr = np.loadtxt(os.path.join(data_dir, "intr.txt"))
            extr = np.loadtxt(os.path.join(data_dir, "extr.txt"))

            # load test color and depth heightmaps
            color_test = common.colorload(os.path.join(test_folder, "final_color_height.png"))
            depth_test = common.depthload(os.path.join(test_folder, "final_depth_height.png"))

            # load object mask
            obj_idxs_test = np.load(os.path.join(test_folder, "curr_object_mask.npy")).astype("int")
            obj_mask_test = np.zeros_like(color_test)
            obj_mask_test[obj_idxs_test[:, 0], obj_idxs_test[:, 1]] = 1

            # load initial and final object pose
            init_pose_test = np.loadtxt(os.path.join(test_folder, "init_pose.txt"))
            final_pose_test = np.loadtxt(os.path.join(test_folder, "final_pose.txt"))

            # compute end-effector transform
            true_transform = np.linalg.inv(final_pose_test @ np.linalg.inv(init_pose_test))

            # find keypoints and descriptors for current image
            kps_test, des_test = detector.detectAndCompute(color_test, obj_mask_test)

            # loop through train and detect keypoint matches
            matches_train = []
            for i, train_folder in enumerate(train_foldernames):
                # load train data
                color_train = common.colorload(os.path.join(train_folder, "final_color_height.png"))
                obj_idxs_train = np.load(os.path.join(train_folder, "curr_object_mask.npy")).astype("int")
                obj_mask_train = np.zeros_like(color_train)
                obj_mask_train[obj_idxs_train[:, 0], obj_idxs_train[:, 1]] = 1

                # find keypoints in train image
                kps_train, des_train = detector.detectAndCompute(color_train, obj_mask_train)
                if des_train is None:
                    continue

                # brute force match
                matches = bf.match(des_test, des_train)
                if len(matches) < config.MIN_NUM_MATCH:
                    continue
                matches_train.append([i, matches])

            # in case we don't find any matches
            if len(matches_train) == 0:
                pred_poses.append(np.nan)
                continue

            # sort matches by lowest average match distance
            matches_train = sorted(matches_train, key=lambda x: np.mean([y.distance for y in x[1]]))

            # retrieve top match image from database
            idx = matches_train[0][0]
            matches = matches_train[0][1]
            color_train = common.colorload(os.path.join(train_foldernames[idx], "final_color_height.png"))
            depth_train = common.depthload(os.path.join(train_foldernames[idx], "final_depth_height.png"))
            obj_idxs_train = np.load(os.path.join(train_foldernames[idx], "curr_object_mask.npy")).astype("int")
            obj_mask_train = np.zeros_like(color_train)
            obj_mask_train[obj_idxs_train[:, 0], obj_idxs_train[:, 1]] = 1
            init_pose_train = np.loadtxt(os.path.join(train_foldernames[idx], "init_pose.txt"))
            final_pose_train = np.loadtxt(os.path.join(train_foldernames[idx], "final_pose.txt"))
            transform_train = np.linalg.inv(final_pose_train @ np.linalg.inv(init_pose_train))
            kps_train, des_train = detector.detectAndCompute(color_train, obj_mask_train)

            # plots descriptor matches between query and train
            if args.debug:
                img_debug = cv2.drawMatches(color_test, kps_test, color_train, kps_train, matches, None, flags=2)
                plt.imshow(img_debug)
                plt.show()

            src_pts  = np.float32([kps_test[m.queryIdx].pt for m in matches]).reshape(-1, 2).astype("int")
            dst_pts  = np.float32([kps_train[m.trainIdx].pt for m in matches]).reshape(-1, 2).astype("int")

            # estimate rigid transform from matches projected in 3D
            zs = depth_test[src_pts[:, 1], src_pts[:, 0]].reshape(-1, 1)
            src_xyz = np.hstack([src_pts, zs])
            src_xyz[:, 0] = (src_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]
            src_xyz[:, 1] = (src_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
            zs = depth_train[dst_pts[:, 1], dst_pts[:, 0]].reshape(-1, 1)
            dst_xyz = np.hstack([dst_pts, zs])
            dst_xyz[:, 0] = (dst_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]
            dst_xyz[:, 1] = (dst_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
            R, t = estimate_rigid_transform(src_xyz, dst_xyz)
            tr = np.eye(4)
            tr[:4, :4] = R
            tr[:3, 3] = t

            # compute estimated transform
            estimated_trans = transform_train @ tr
            pred_poses.append(estimated_trans)

            if args.debug:
                zs = depth_test[obj_idxs_test[:, 0], obj_idxs_test[:, 1]].reshape(-1, 1)
                mask_xyz = np.hstack([obj_idxs_test, zs])
                mask_xyz[:, [0, 1]] = mask_xyz[:, [1, 0]]
                mask_xyz[:, 0] = (mask_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]
                mask_xyz[:, 1] = (mask_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
                mask_xyz = transform_xyz(mask_xyz, estimated_trans)
                mask_xyz[:, 0] = (mask_xyz[:, 0] - config.VIEW_BOUNDS[0, 0]) / config.HEIGHTMAP_RES
                mask_xyz[:, 1] = (mask_xyz[:, 1] - config.VIEW_BOUNDS[1, 0]) / config.HEIGHTMAP_RES
                hole_idxs_est = mask_xyz[:, [1, 0]]
                mask_xyz = np.hstack([obj_idxs_test, zs])
                mask_xyz[:, [0, 1]] = mask_xyz[:, [1, 0]]
                mask_xyz[:, 0] = (mask_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]
                mask_xyz[:, 1] = (mask_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
                true_xyz = transform_xyz(mask_xyz, true_transform)
                true_xyz[:, 0] = (true_xyz[:, 0] - config.VIEW_BOUNDS[0, 0]) / config.HEIGHTMAP_RES
                true_xyz[:, 1] = (true_xyz[:, 1] - config.VIEW_BOUNDS[1, 0]) / config.HEIGHTMAP_RES
                hole_idxs_true = true_xyz[:, [1, 0]]
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(color_test)
                axes[0].scatter(hole_idxs_true[:, 1], hole_idxs_true[:, 0])
                axes[0].title.set_text('Ground Truth')
                axes[1].imshow(color_test)
                axes[1].scatter(hole_idxs_est[:, 1], hole_idxs_est[:, 0])
                axes[1].title.set_text('Predicted')
                plt.show()

        kit_poses[data_dir.split("/")[-1]] = pred_poses

    with open(os.path.join(save_dir, "ORB-PE_poses.pkl"), "wb") as fp:
        pickle.dump(kit_poses, fp)