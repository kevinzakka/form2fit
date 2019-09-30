"""The correspondence network dataloader.
"""

import glob
import multiprocessing
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from functools import reduce
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from form2fit.code.utils import misc, viz
from form2fit.code.utils import sampling
from form2fit import config
from walle.core import RotationMatrix, Orientation


class CorrespondenceDataset(Dataset):
    """The correspondence network dataset.
    """

    def __init__(
        self,
        root,
        sample_ratio,
        num_rotations,
        markovian,
        augment,
        background_subtract,
        num_channels,
    ):
        """Initializes the dataset.

        Args:
            root: (str) Root directory path.
            sample_ratio: (float) The ratio of negative to positive labels.
            num_rotations: (int) The number of discrete rotation levels to consider.
            markovian: (bool) If `True`, only consider correspondences
                from the current timestep. Else, use correspondences
                from all previous and current timestep.
            augment: (bool) Whether to apply data augmentation.
            background_subtract: (bool) Whether to apply background subtraction.
            num_channels: (int) 4 clones the grayscale image to produce an RGB image.
        """
        self._root = root
        self._num_rotations = num_rotations
        self._markovian = markovian
        self._sample_ratio = sample_ratio
        self._augment = augment
        self._background_subtract = background_subtract
        self._num_channels = num_channels

        # figure out how many data samples we have
        self._get_filenames()

        # generate rotation increments
        self._rot_step_size = 360 / num_rotations
        self._rotations = np.array([self._rot_step_size * i for i in range(num_rotations)])

        # load per-channel mean and std
        stats = pickle.load(open(os.path.join(Path(self._root).parent, "mean_std.p"), "rb"))
        self.stats = stats
        if num_channels == 2:
            self._c_norm = transforms.Normalize(mean=stats[0][0], std=stats[0][1])
        else:
            self._c_norm = transforms.Normalize(mean=stats[0][0]*3, std=stats[0][1]*3)
        self._d_norm = transforms.Normalize(mean=stats[1][0], std=stats[1][1])
        self._transform = transforms.ToTensor()

    def __len__(self):
        return len(self._filenames)

    def _get_filenames(self):
        """Returns a list of filenames to process.
        """
        self._filenames = glob.glob(os.path.join(self._root, "*/"))
        self._filenames.sort(key=lambda x: int(x.split("/")[-2]))

    def _load_state(self, name):
        # load poses
        pose_i = np.loadtxt(os.path.join(name, "init_pose.txt"))
        pose_f = np.loadtxt(os.path.join(name, "final_pose.txt"))

        # load visual
        c_height_i = np.asarray(Image.open(os.path.join(name, "init_color_height.png")))
        d_height_i = np.asarray(Image.open(os.path.join(name, "init_depth_height.png")))
        c_height_f = np.asarray(Image.open(os.path.join(name, "final_color_height.png")))
        d_height_f = np.asarray(Image.open(os.path.join(name, "final_depth_height.png")))

        # convert depth to meters
        d_height_i = (d_height_i * 1e-3).astype("float32")
        d_height_f = (d_height_f * 1e-3).astype("float32")

        # load masks
        if self._markovian:
            object_mask = np.load(os.path.join(name, "curr_object_mask.npy"), allow_pickle=True)
            hole_mask = np.load(os.path.join(name, "curr_hole_mask.npy"), allow_pickle=True)
            kit_minus_hole_mask = np.load(
                os.path.join(name, "curr_kit_minus_hole_mask.npy"), allow_pickle=True
            )
            kit_plus_hole_mask = np.load(
                os.path.join(name, "curr_kit_plus_hole_mask.npy"), allow_pickle=True
            )
        else:
            object_mask = np.load(os.path.join(name, "object_mask.npy"), allow_pickle=True)
            hole_mask = np.load(os.path.join(name, "hole_mask.npy"), allow_pickle=True)
            kit_minus_hole_mask = np.load(os.path.join(name, "kit_minus_hole_mask.npy"), allow_pickle=True)
            kit_plus_hole_mask = np.load(os.path.join(name, "kit_plus_hole_mask.npy"), allow_pickle=True)

        # load correspondences
        corrs = np.load(os.path.join(name, "corrs.npy"), allow_pickle=True)

        # fix weird npy behavior
        if corrs.ndim > 2:
            corrs = [corrs[0]]
        else:
            corrs = corrs.tolist()

        # compute all previous rotation quantizations indices
        transforms = np.load(os.path.join(name, "transforms.npy"), allow_pickle=True)
        rotations = [np.rad2deg(misc.rotz2angle(t)) for t in transforms]
        rot_quant_indices = [self._quantize_rotation(r) for r in rotations]

        # sort by increasing rotation index
        if rot_quant_indices:
            temp_argsort = np.argsort(rot_quant_indices)
            rot_quant_indices.sort()
            curr_corrs = corrs[-1:]
            prev_corrs = corrs[:-1]
            if prev_corrs:
                prev_corrs = [prev_corrs[i] for i in temp_argsort]
            corrs = prev_corrs + curr_corrs

        return (
            pose_i,
            pose_f,
            c_height_i,
            d_height_i,
            c_height_f,
            d_height_f,
            object_mask,
            hole_mask,
            kit_minus_hole_mask,
            kit_plus_hole_mask,
            corrs,
            rot_quant_indices,
        )

    def _split_heightmap(self, height):
        """Splits a heightmap into a source and target.
        """
        half = height.shape[1] // 2
        self._half = half
        height_s = height[:, half:].copy()
        height_t = height[:, :half].copy()
        return height_s, height_t

    def _compute_relative_rotation(self, pose_i, pose_f):
        """Computes the relative z-axis rotation between two poses.

        Returns:
            (float) The angle in degrees.
        """
        transform = pose_f @ np.linalg.inv(pose_i)
        rotation = np.rad2deg(misc.rotz2angle(transform))
        return rotation

    def _quantize_rotation(self, true_rot):
        """Bins the true rotation into one of `num_rotations`.
        Returns:
            (int) An index from 0 to `num_rotations` - 1.
        """
        angle = true_rot - (360 * np.floor(true_rot * (1 / 360)))
        # angle = (true_rot % 360 + 360) % 360

        # since 0 = 360 degrees, we need to remap
        # any indices in the last quantization
        # bracket to 0 degrees.
        if angle > (360 - (0.5 * self._rot_step_size)) and angle <= 360:
            return 0

        return np.argmin(np.abs(self._rotations - angle))

    def _process_correspondences(self, corrs, rot_idx, depth=None, append=True):
        """Processes correspondences for a given rotation.
        """
        # split correspondences into source and target
        source_corrs = corrs[:, 0:2]
        target_corrs = corrs[:, 2:4]

        # rotate source indices
        source_idxs = misc.rotate_uv(source_corrs, -self._rot_step_size * rot_idx, self._H, self._W, (self._uc, self._vc))
        target_idxs = np.round(target_corrs)

        # remove any repetitions
        _, unique_idxs = np.unique(source_idxs, return_index=True, axis=0)
        source_idxs_unique = source_idxs[unique_idxs]
        target_idxs_unique = target_idxs[unique_idxs]
        _, unique_idxs = np.unique(target_idxs_unique, return_index=True, axis=0)
        source_idxs_unique = source_idxs_unique[unique_idxs]
        target_idxs_unique = target_idxs_unique[unique_idxs]

        # remove indices that exceed image bounds
        valid_idxs = np.logical_and(
            target_idxs_unique[:, 0] < self._H,
            np.logical_and(
                target_idxs_unique[:, 1] < self._W,
                np.logical_and(
                    target_idxs_unique[:, 0] >= 0, target_idxs_unique[:, 1] >= 0
                ),
            ),
        )
        target_idxs = target_idxs_unique[valid_idxs]
        source_idxs = source_idxs_unique[valid_idxs]
        valid_idxs = np.logical_and(
            source_idxs[:, 0] < self._H,
            np.logical_and(
                source_idxs[:, 1] < self._W,
                np.logical_and(source_idxs[:, 0] >= 0, source_idxs[:, 1] >= 0),
            ),
        )
        target_idxs = target_idxs[valid_idxs].astype("int")
        source_idxs = source_idxs[valid_idxs].astype("int")

        # if depth is not None:
        #     depth_vals = depth[target_idxs[:, 0], target_idxs[:, 1]]
        #     valid_ds = depth_vals >= depth_vals.mean()
        #     mask = np.zeros_like(depth)
        #     mask[target_idxs[valid_ds][:, 0], target_idxs[valid_ds][:, 1]] = 1
        #     mask = misc.largest_cc(mask)
        #     valid_mask = np.vstack(np.where(mask == 1)).T
        #     tset = set([tuple(x) for x in valid_mask])
        #     for i in range(len(valid_ds)):
        #         is_valid = valid_ds[i]
        #         if is_valid:
        #             tidx = target_idxs[i]
        #             if tuple(tidx) not in tset:
        #                 valid_ds[i] = False
        #     target_idxs = target_idxs[valid_ds]
        #     source_idxs = source_idxs[valid_ds]

        if append:
            self._features_source.append(source_idxs)
            self._features_target.append(target_idxs)
            self._rot_idxs.append(np.repeat([rot_idx], len(source_idxs)))
            self._is_match.append(np.ones(len(source_idxs)))
        else:
            return np.hstack([source_idxs, target_idxs])

    def _get_valid_idxs(self, corr, rows, cols):
        positive_cond = np.logical_and(corr[:, 0] >= 0, corr[:, 1] >= 0)
        within_cond = np.logical_and(corr[:, 0] < rows, corr[:, 1] < cols)
        valid_idxs = reduce(np.logical_and, [positive_cond, within_cond])
        return valid_idxs

    def _sample_translation(self, corrz, angle):
        aff_1 = np.eye(3)
        aff_1[:2, 2] = [-self._uc, -self._vc]
        aff_2 = RotationMatrix.rotz(-angle)
        aff_3 = np.eye(3, 3)
        aff_3[:2, 2] = [self._uc, self._vc]
        affine = aff_3 @ aff_2 @ aff_1
        affine = affine[:2, :]
        corrs = []
        for corr in corrz:
            ones = np.ones((len(corr), 1))
            corrs.append((affine @ np.hstack((corr, ones)).T).T)
        max_vv = corrs[0][:, 1].max()
        max_vu = corrs[0][corrs[0][:, 1].argmax()][0]
        min_vv = corrs[0][:, 1].min()
        min_vu = corrs[0][corrs[0][:, 1].argmin()][0]
        max_uu = corrs[0][:, 0].max()
        max_uv = corrs[0][corrs[0][:, 0].argmax()][1]
        min_uu = corrs[0][:, 0].min()
        min_uv = corrs[0][corrs[0][:, 0].argmin()][1]
        for t in corrs[1:]:
            if t[:, 1].max() > max_vv:
                max_vv = t[:, 1].max()
                max_vu = t[t[:, 1].argmax()][0]
            if t[:, 1].min() < min_vv:
                min_vv = t[:, 1].min()
                min_vu = t[t[:, 1].argmin()][0]
            if t[:, 0].max() > max_uu:
                max_uu = t[:, 0].max()
                max_uv = t[t[:, 0].argmax()][1]
            if t[:, 0].min() < min_uu:
                min_uu = t[:, 0].min()
                min_uv = t[t[:, 0].argmin()][1]
        tu = np.random.uniform(-min_vv + 10, self._W - max_vv - 10)
        tv = np.random.uniform(-min_uu + 10, self._H - max_uu - 10)
        return tu, tv

    def __getitem__(self, idx):
        name = self._filenames[idx]

        # load states
        pose_i, pose_f, c_height_i, d_height_i, c_height_f, \
            d_height_f, object_mask, hole_mask, kit_minus_hole_mask, \
                kit_plus_hole_mask, all_corrs, rot_quant_indices = self._load_state(self._filenames[idx])

        # split heightmap into source and target
        c_height_s, c_height_t = self._split_heightmap(c_height_f)
        d_height_s, d_height_t = self._split_heightmap(d_height_f)
        self._H, self._W = c_height_t.shape[:2]

        hole_mask[:, 1] = hole_mask[:, 1] - self._half
        kit_minus_hole_mask[:, 1] = kit_minus_hole_mask[:, 1] - self._half
        kit_plus_hole_mask[:, 1] = kit_plus_hole_mask[:, 1] - self._half
        for corr in all_corrs:
            corr[:, 1] = corr[:, 1] - self._half

        if self._background_subtract is not None:
            # idxs = np.vstack(np.where(d_height_s > self._background_subtract[0])).T
            # mask = np.zeros_like(d_height_s)
            # mask[idxs[:, 0], idxs[:, 1]] = 1
            # mask = misc.largest_cc(np.logical_not(mask))
            # idxs = np.vstack(np.where(mask == 1)).T
            mask = np.zeros_like(d_height_s)
            mask[kit_plus_hole_mask[:, 0], kit_plus_hole_mask[:, 1]] = 1
            idxs = np.vstack(np.where(mask == 1)).T
            mask[int(idxs[:, 0].min()):int(idxs[:, 0].max()), int(idxs[:, 1].min()):int(idxs[:, 1].max())] = 1
            mask = np.logical_not(mask)
            idxs = np.vstack(np.where(mask == 1)).T
            c_height_s[idxs[:, 0], idxs[:, 1]] = 0
            d_height_s[idxs[:, 0], idxs[:, 1]] = 0
            idxs = np.vstack(np.where(d_height_t > self._background_subtract[1])).T
            mask = np.zeros_like(d_height_s)
            mask[idxs[:, 0], idxs[:, 1]] = 1
            mask = misc.largest_cc(np.logical_not(mask))
            idxs = np.vstack(np.where(mask == 1)).T
            c_height_t[idxs[:, 0], idxs[:, 1]] = 0
            d_height_t[idxs[:, 0], idxs[:, 1]] = 0

        # partition correspondences into current and previous
        curr_corrs = all_corrs[-1]
        prev_corrs = all_corrs[:-1]

        # compute rotation about z-axis using inital and final pose
        gd_truth_rot = self._compute_relative_rotation(pose_i, pose_f)

        # center of rotation is the center of the kit
        self._uc = int((kit_plus_hole_mask[:, 0].max() + kit_plus_hole_mask[:, 0].min()) // 2)
        self._vc = int((kit_plus_hole_mask[:, 1].max() + kit_plus_hole_mask[:, 1].min()) // 2)

        if self._augment:
            shape = (self._W, self._H)
            source_corrs = curr_corrs[:, 0:2].astype("float64")
            target_corrs = curr_corrs[:, 2:4].astype("float64")

            # determine bounds on translation for source and target
            all_corrz = []
            for i, corr in enumerate(all_corrs):
                all_corrz.append(corr)
            sources = [kit_plus_hole_mask]
            targets = [p[:, 2:4] for p in all_corrz]

            angle_s = np.radians(np.random.uniform(0, 360))
            tu_s, tv_s = self._sample_translation(sources, angle_s)
            aff_1 = np.eye(3)
            aff_1[:2, 2] = [-self._vc, -self._uc]
            aff_2 = RotationMatrix.rotz(angle_s)
            aff_2[:2, 2] = [tu_s, tv_s]
            aff_3 = np.eye(3, 3)
            aff_3[:2, 2] = [self._vc, self._uc]
            affine_s = aff_3 @ aff_2 @ aff_1
            affine_s = affine_s[:2, :]

            c_height_s = cv2.warpAffine(c_height_s, affine_s, shape, flags=cv2.INTER_NEAREST)
            d_height_s = cv2.warpAffine(d_height_s, affine_s, shape, flags=cv2.INTER_NEAREST)
            aff_1[:2, 2] = [-self._uc, -self._vc]
            aff_2 = RotationMatrix.rotz(-angle_s)
            aff_2[:2, 2] = [tv_s, tu_s]
            aff_3[:2, 2] = [self._uc, self._vc]
            affine_s = aff_3 @ aff_2 @ aff_1
            affine_s = affine_s[:2, :]
            source_corrs = (affine_s @ np.hstack((source_corrs, np.ones((len(source_corrs), 1)))).T).T

            # target affine transformation
            angle_t = 0
            tu_t, tv_t = 0, 0  # self._sample_translation(targets, angle_t)
            aff_1 = np.eye(3)
            aff_1[:2, 2] = [-self._vc, -self._uc]
            aff_2 = RotationMatrix.rotz(angle_t)
            aff_2[:2, 2] = [tu_t, tv_t]
            aff_3 = np.eye(3, 3)
            aff_3[:2, 2] = [self._vc, self._uc]
            affine_t = aff_3 @ aff_2 @ aff_1
            affine_t = affine_t[:2, :]

            c_height_t = cv2.warpAffine(c_height_t, affine_t, shape, flags=cv2.INTER_NEAREST)
            d_height_t = cv2.warpAffine(d_height_t, affine_t, shape, flags=cv2.INTER_NEAREST)
            aff_1[:2, 2] = [-self._uc, -self._vc]
            aff_2 = RotationMatrix.rotz(-angle_t)
            aff_2[:2, 2] = [tv_t, tu_t]
            aff_3[:2, 2] = [self._uc, self._vc]
            affine_t = aff_3 @ aff_2 @ aff_1
            affine_t = affine_t[:2, :]
            target_corrs = (affine_t @ np.hstack((target_corrs, np.ones((len(target_corrs), 1)))).T).T

            # remove invalid indices
            valid_target_idxs = self._get_valid_idxs(target_corrs, self._H, self._W)
            target_corrs = target_corrs[valid_target_idxs].astype("int64")
            source_corrs = source_corrs[valid_target_idxs].astype("int64")
            curr_corrs = np.hstack((source_corrs, target_corrs))

            # apply affine transformation to masks in source
            masks = [hole_mask, kit_plus_hole_mask, kit_minus_hole_mask]
            for i in range(len(masks)):
                ones = np.ones((len(masks[i]), 1))
                masks[i] = (affine_s @ np.hstack((masks[i], ones)).T).T
            hole_mask, kit_plus_hole_mask, kit_minus_hole_mask = masks

            # apply affine transformation to masks in target
            ones = np.ones((len(object_mask), 1))
            object_mask = (affine_t @ np.hstack((object_mask, ones)).T).T
            object_mask[:, 0] = np.clip(object_mask[:, 0], 0, self._H - 1)
            object_mask[:, 1] = np.clip(object_mask[:, 1], 0, self._W - 1)

        # reupdate kit mask center
        self._uc = int((kit_plus_hole_mask[:, 0].max() + kit_plus_hole_mask[:, 0].min()) // 2)
        self._vc = int((kit_plus_hole_mask[:, 1].max() + kit_plus_hole_mask[:, 1].min()) // 2)

        if self._augment:
            gd_truth_rot = gd_truth_rot + np.degrees(angle_t) - np.degrees(angle_s)

        # quantize rotation
        curr_rot_idx = self._quantize_rotation(gd_truth_rot)
        curr_rot = self._rotations[curr_rot_idx]

        self._features_source = []
        self._features_target = []
        self._rot_idxs = []
        self._is_match = []

        # sample matches from all previous timesteps if not markovian
        if not self._markovian:
            for rot_idx, corrs in zip(rot_quant_indices, prev_corrs):
                self._process_correspondences(corrs, rot_idx)

        # sample matches from the current timestep
        self._process_correspondences(curr_corrs, curr_rot_idx, depth=d_height_t)

        # determine the number of non-matches to sample per rotation
        num_matches = 0
        for m in self._is_match:
            num_matches += len(m)
        num_non_matches = int(self._sample_ratio * num_matches / self._num_rotations)

        # convert masks to linear indices for sampling
        all_idxs_1d = np.arange(0, self._H * self._W)
        object_target_1d = sampling.make1d(object_mask, self._W)
        background_target_1d = np.array(list((set(all_idxs_1d) - set(object_target_1d))))
        hole_source_1d = sampling.make1d(hole_mask, self._W)
        kit_minus_hole_source_1d = sampling.make1d(kit_minus_hole_mask, self._W)
        kit_plus_hole_source_1d = sampling.make1d(kit_plus_hole_mask, self._W)
        background_source_1d = np.array(list(set(all_idxs_1d) - set(kit_plus_hole_source_1d)))
        background_source_1d = sampling.remove_outliers(background_source_1d, kit_plus_hole_mask, self._W)

        # sample non-matches
        temp_idx = 0
        div_factor = 2
        for rot_idx in range(self._num_rotations):
            non_matches = []

            # # source: anywhere
            # # target: anywhere but the object
            # non_matches.append(sampling.sample_non_matches(
            #     1 * num_non_matches // div_factor,
            #     (self._H, self._W),
            #     -self._rotations[rot_idx],
            #     mask_target=background_target_1d,
            #     rotate=False)
            # )

            # # source: anywhere but the kit
            # # target: on the object
            # nm_idxs = sampling.sample_non_matches(
            #     1 * num_non_matches // div_factor,
            #     (self._H, self._W),
            #     -self._rotations[rot_idx],
            #     background_source_1d,
            #     object_target_1d,
            #     rotate=False,
            # )
            # non_matches.append(nm_idxs)

            # source: on the kit but not in the hole
            # target: on the object
            nm_idxs = sampling.sample_non_matches(
                1 * num_non_matches // div_factor,
                (self._H, self._W),
                -self._rotations[rot_idx],
                kit_minus_hole_source_1d,
                object_target_1d,
                cxcy=(self._uc, self._vc),
            )
            non_matches.append(nm_idxs)

            # # here, I want to explicity samples matches
            # # for the incorrect rotations to teach
            # # the network that in fact, this is
            # # the incorrect rotation.
            # # This is especially useful for the
            # # 180 degree rotated version of the
            # # correct rotation.
            # if rot_idx != curr_rot_idx:
            #     nm_idxs = self._process_correspondences(curr_corrs, rot_idx, False)
            #     subset_mask = np.random.choice(np.arange(len(nm_idxs)), replace=False, size=(1 * num_non_matches // div_factor))
            #     nm_idxs = nm_idxs[subset_mask]
            #     non_matches.append(nm_idxs)

            # source: in the hole
            # target: on the object
            if self._markovian:
                if rot_idx == curr_rot_idx:
                    nm_idxs = sampling.non_matches_from_matches(
                        1 * num_non_matches // div_factor,
                        (self._H, self._W),
                        -self._rotations[rot_idx],
                        hole_source_1d,
                        self._features_source[0],
                        self._features_target[0],
                        cxcy=(self._uc, self._vc),
                    )
                    non_matches.append(nm_idxs)
                else:
                    nm_idxs = sampling.sample_non_matches(
                        1 * num_non_matches // div_factor,
                        (self._H, self._W),
                        -self._rotations[rot_idx],
                        hole_source_1d,
                        object_target_1d,
                        cxcy=(self._uc, self._vc),
                    )
                    non_matches.append(nm_idxs)
            else:
                if rot_idx in rot_quant_indices:
                    non_matches.append(
                        sampling.non_matches_from_matches(
                            num_non_matches // div_factor,
                            (self._H, self._W),
                            -self._rotations[rot_idx],
                            hole_source_1d,
                            self._features_source[temp_idx],
                            self._features_target[temp_idx],
                        )
                    )
                    temp_idx += 1
                else:
                    non_matches.append(
                        sampling.sample_non_matches(
                            num_non_matches // div_factor,
                            (self._H, self._W),
                            -self._rotations[rot_idx],
                            hole_source_1d,
                            object_target_1d,
                        )
                    )
            non_matches = np.vstack(non_matches)
            self._features_source.append(non_matches[:, :2])
            self._features_target.append(non_matches[:, 2:])
            self._rot_idxs.append(np.repeat([rot_idx], len(non_matches)))
            self._is_match.append(np.repeat([0], len(non_matches)))

        # convert lists to numpy arrays
        self._features_source = np.concatenate(self._features_source)
        self._features_target = np.concatenate(self._features_target)
        self._rot_idxs = np.concatenate(self._rot_idxs)[..., np.newaxis]
        self._is_match = np.concatenate(self._is_match)[..., np.newaxis]

        # concatenate into 1 big array
        label = np.hstack(
            (
                self._features_source,
                self._features_target,
                self._rot_idxs,
                self._is_match,
            )
        )

        if self._num_channels == 2:
            c_height_s = c_height_s[..., np.newaxis]
            c_height_t = c_height_t[..., np.newaxis]
        else:  # clone the gray channel 3 times
            c_height_s = np.repeat(c_height_s[..., np.newaxis], 3, axis=-1)
            c_height_t = np.repeat(c_height_t[..., np.newaxis], 3, axis=-1)

        # ndarray -> tensor
        label_tensor = torch.LongTensor(label)

        # heightmaps -> tensor
        c_height_s = self._c_norm(self._transform(c_height_s))
        c_height_t = self._c_norm(self._transform(c_height_t))
        d_height_s = self._d_norm(self._transform(d_height_s[..., np.newaxis]))
        d_height_t = self._d_norm(self._transform(d_height_t[..., np.newaxis]))

        # concatenate height and depth into a 4-channel tensor
        source_img_tensor = torch.cat([c_height_s, d_height_s], dim=0)
        target_img_tensor = torch.cat([c_height_t, d_height_t], dim=0)

        # concatenate source and target into a 8-channel tensor
        img_tensor = torch.cat([source_img_tensor, target_img_tensor], dim=0)

        return img_tensor, label_tensor, (self._uc, self._vc)


def get_corr_loader(
    foldername,
    dtype="train",
    batch_size=1,
    shuffle=True,
    sample_ratio=1.0,
    num_rotations=20,
    markovian=True,
    augment=False,
    background_subtract=None,
    num_channels=2,
    num_workers=1,
):
    """Returns a dataloader over the correspondence dataset.

    Args:
        foldername: (str) The name of the folder containing the data.
        dtype: (str) Whether to use the train, validation or test partition.
        shuffle: (bool) Whether to shuffle the dataset at the end
            of every epoch.
        sample_ratio: (float) The ratio of negative to positive
            labels.
        num_rotations: (int) The number of discrete rotation levels
            to consider.
        markovian: (bool) If `True`, only consider correspondences
            from the current timestep. Else, use correspondences
            from all previous and current timestep.
        background_subtract: (bool) Whether to apply background subtraction.
        num_channels: (int) 4 clones the grayscale image to produce an RGB image.
        num_workers: (int) How many processes to use. Each workers
            is responsible for loading a batch.
    """

    def _collate_fn(batch):
        """A custom collate function.

        This is to support variable length correspondence labels.
        """
        imgs = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        centers = [b[2] for b in batch]
        # mask = [b[2] for b in batch]
        # kit_mask = [b[3] for b in batch]
        imgs = torch.stack(imgs, dim=0)
        max_num_label = labels[0].shape[0]
        for l in labels[1:]:
            if l.shape[0] > max_num_label:
                max_num_label = l.shape[0]
        new_labels = []
        for l in labels:
            if l.shape[0] < max_num_label:
                l_pad = torch.cat([l, torch.LongTensor([999]).repeat(max_num_label - l.shape[0], 6)], dim=0)
                new_labels.append(l_pad)
            else:
                new_labels.append(l)
        labels = torch.stack(new_labels, dim=0)
        return [imgs, labels, centers]

    num_workers = min(num_workers, multiprocessing.cpu_count())
    root = os.path.join(config.benchmark_dir, "train", foldername, dtype)

    dataset = CorrespondenceDataset(
        root,
        sample_ratio,
        num_rotations,
        markovian,
        augment,
        background_subtract,
        num_channels,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )

    return loader
