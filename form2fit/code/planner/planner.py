import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from itertools import product
from skimage.feature import peak_local_max

from form2fit.code.utils import misc


class Planner:
    def __init__(self, center):
        self.center = center

    def plan(self, suction_scores, place_scores, kit_descriptor_map, object_descriptor_map, suction_mask=None, place_mask=None, img1=None, img2=None):
        if suction_mask is not None:
            suction_scores[suction_mask[:, 0], suction_mask[:, 1]] = 0
        suction_coordinates = peak_local_max(suction_scores, min_distance=0, threshold_rel=0.1)

        if place_mask is not None:
            place_scores[place_mask[:, 0], place_mask[:, 1]] = 0
        place_coordinates = peak_local_max(place_scores, min_distance=0, threshold_rel=0.1)

        combinations = list(product(place_coordinates, suction_coordinates))
        num_rotations = len(kit_descriptor_map)
        B, D, H, W = object_descriptor_map.shape
        object_descriptor_map_flat = object_descriptor_map.view(B, D, H*W)

        distances = []
        rotation_idxs = []
        for place_uv, suction_uv in combinations:
            # index object descriptor map
            suction_uv_flat = torch.from_numpy(np.array((suction_uv[0]*W+suction_uv[1]))).long().cuda()
            object_descriptor = torch.index_select(object_descriptor_map_flat[0], 1, suction_uv_flat).unsqueeze(0)

            kit_descriptors = []
            for r in range(num_rotations):
                place_uv_rot = misc.rotate_uv(np.array([place_uv]), -(360/num_rotations)*r, H, W, cxcy=self.center)[0]
                place_uv_rot_flat = torch.from_numpy(np.array((place_uv_rot[0]*W+place_uv_rot[1]))).long().cuda()
                kit_descriptor_map_flat = kit_descriptor_map[r].view(kit_descriptor_map.shape[1], -1)
                kit_descriptors.append(torch.index_select(kit_descriptor_map_flat, 1, place_uv_rot_flat))
            kit_descriptors = torch.stack(kit_descriptors)

            # compute L2 distances
            diffs = object_descriptor - kit_descriptors
            l2_dists = diffs.pow(2).sum(1).sqrt()

            # store best across rotation
            best_rot_idx = l2_dists.argmin().item()
            l2_dist = l2_dists[best_rot_idx].item()
            distances.append(l2_dist)
            rotation_idxs.append(best_rot_idx)

        # compute best across candidates
        best_distance_idx = np.argmin(distances)
        best_place_uv, best_suction_uv = combinations[best_distance_idx]

        ret = {
            "best_distance_idx": best_distance_idx,
            "best_distance": distances[best_distance_idx],
            "best_rotation_idx": rotation_idxs[best_distance_idx],
            "best_place_uv": best_place_uv,
            "best_suction_uv": best_suction_uv,
        }

        return ret