"""Useful torch related helper functions.
"""

import random

import numpy as np
import torch


def tensor2ndarray(x, mu_sigma=None, color=True):
    """Converts a torch tensor to a numpy ndarray.

    If mu_sigma is provided, we additionally de-normalize
    by the respective color channel means and stds.

    Args:
        x: (Tensor) A torch image tensor.
        mu_sigma: (list) A list where the first
            element contains the channel means and
            the second element contains the channel
            standard deviations.

    Returns:
        img: (ndarray) A numpy image of shape (H, W, C).
    """
    img = x.cpu().detach().numpy().copy()
    channels = img.shape[0]
    if color:
        img = img.transpose(1, 2, 0)
        if mu_sigma is not None:
            means = mu_sigma[0]
            stds = mu_sigma[1]
            for c in range(channels):
                img[..., c] = (img[..., c] * stds[c]) + means[c]
        img = (img * 255).astype("uint8")
    else:
        img = img.squeeze()
        img = img[..., np.newaxis]
        if mu_sigma is not None:
            mean = mu_sigma[0]
            std = mu_sigma[1]
            for c in range(1):
                img[..., c] = (img[..., c] * std[c]) + mean[c]
        img = img.squeeze().astype("float32")
    return img


def seed_rng(seed):
    """Seeds the numpy and pytorch RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)