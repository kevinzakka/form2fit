"""Various loss function definitions.
"""

from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn.functional as F


def zero_loss(device):
    return torch.FloatTensor([0]).to(device)


def nanmean(x):
    """Computes the arithmetic mean, ignoring any NaNs.
    """
    return torch.mean(x[x == x])


def pixel2spatial(xs, H, W):
    """Converts a tensor of coordinates to a boolean spatial map. 
    """
    xs_spatial = []
    for x in xs:
        pos = x[x[:, 2] == 1][:, :2]
        pos_label = torch.zeros(1, 1, H, W)
        pos_label[:, :, pos[:, 0], pos[:, 1]] = 1
        xs_spatial.append(pos_label)
    xs_spatial = torch.cat(xs_spatial, dim=0).long()
    return xs_spatial


class BaseLoss(ABC):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class SuctionLoss(BaseLoss):
    """The loss for training a suction prediction network.
    """

    def __init__(self, sample_ratio, device, mean=True):
        self.sample_ratio = torch.FloatTensor([sample_ratio]).to(device)
        self._device = device
        self._mean = mean

    def forward(self, out, label, add_dice_loss=False):
        batch_size = len(label)
        if self._mean:
            loss = zero_loss(self._device)
            for i, l in enumerate(label):
                logits = out[i, :, l[:, 0], l[:, 1]]
                target = l[:, 2].unsqueeze(0).float()
                loss += F.binary_cross_entropy_with_logits(logits, target, self.sample_ratio)
            loss /= batch_size
        else:
            loss = []
            for i, l in enumerate(label):
                logits = out[i, :, l[:, 0], l[:, 1]]
                target = l[:, 2].unsqueeze(0).float()
                loss.append(F.binary_cross_entropy_with_logits(logits, target, self.sample_ratio, reduction="none"))
        
        if add_dice_loss:
            label_spatial = pixel2spatial(label, out.shape[2], out.shape[3])
            dice_loss = compute_dice_loss(label_spatial, out)
            loss = loss + (5 * dice_loss)

        return loss


PlacementLoss = SuctionLoss


class CorrespondenceLoss(BaseLoss):
    """The loss for training a correspondence prediction network.
    """

    def __init__(self, margin, num_rotations, hard_negative, device, sample_ratio=None):
        self.margin = margin
        self.num_rotations = num_rotations
        self.sample_ratio = sample_ratio
        self._device = device
        self._hard_neg = hard_negative

    def _sq_hinge_loss(self, source, target, reduction=True):
        diff = source - target
        l2_dist = diff.pow(2).sum(1).sqrt()
        margin_diff = self.margin - l2_dist
        loss = torch.clamp(margin_diff, min=0).pow(2)
        if reduction:
            return loss.mean()
        return loss

    def _sq_l2(self, source, target, reduction=True):
        diff = source - target
        l2_dist_sq = diff.pow(2).sum(1)
        if reduction:
            return l2_dist_sq.mean()
        return l2_dist_sq

    def _positive_corrs(self, outs_s, outs_t, labels):
        """Computes match loss.
        """
        batch_size = len(labels)
        match_loss = zero_loss(self._device)
        for b_idx, label in enumerate(labels):
            mask = torch.all(label == torch.LongTensor([999]).repeat(6).to(self._device), dim=1)
            label = label[~mask]
            source_idxs = label[:, 0:2]
            target_idxs = label[:, 2:4]
            rot_idx = label[:, 4]
            is_match = label[:, 5]
            correct_rot_idx = rot_idx[0]
            out_s = outs_s[b_idx]
            D, H, W = outs_t.shape[1:]
            out_t = outs_t[b_idx : b_idx + 1]
            out_t_flat = out_t.view(1, D, H*W).permute(0, 2, 1)
            mask = (is_match == 1) & (rot_idx == correct_rot_idx)
            s_idxs = source_idxs[mask]
            t_idxs = target_idxs[mask]
            s_idxs_f = s_idxs[:, 0] * W + s_idxs[:, 1]
            t_idxs_f = t_idxs[:, 0] * W + t_idxs[:, 1]
            out_s_flat = out_s[correct_rot_idx:correct_rot_idx+1].view(1, D, H*W).permute(0, 2, 1)
            match_s_descriptors = torch.index_select(out_s_flat, 1, s_idxs_f).squeeze(0)
            match_t_descriptors = torch.index_select(out_t_flat, 1, t_idxs_f).squeeze(0)
            match_loss += self._sq_l2(match_s_descriptors, match_t_descriptors)
        match_loss /= batch_size
        return match_loss

    def _negative_corrs(self, outs_s, outs_t, labels):
        """Computes non-match loss.
        """
        batch_size = len(labels)
        non_match_loss = zero_loss(self._device)
        for b_idx, label in enumerate(labels):
            mask = torch.all(label == torch.LongTensor([999]).repeat(6).to(self._device), dim=1)
            label = label[~mask]
            source_idxs = label[:, 0:2]
            target_idxs = label[:, 2:4]
            rot_idx = label[:, 4]
            is_match = label[:, 5]
            out_s = outs_s[b_idx]
            D, H, W = outs_t.shape[1:]
            out_t = outs_t[b_idx : b_idx + 1]
            out_t_flat = out_t.view(1, D, H*W).permute(0, 2, 1)
            non_match_s_descriptors = []
            non_match_t_descriptors = []
            for r_idx in range(self.num_rotations):
                mask = (is_match != 1) & (rot_idx == r_idx)
                s_idxs = source_idxs[mask]
                t_idxs = target_idxs[mask]
                s_idxs_f = s_idxs[:, 0] * W + s_idxs[:, 1]
                t_idxs_f = t_idxs[:, 0] * W + t_idxs[:, 1]
                out_s_flat = out_s[r_idx : r_idx + 1].view(1, D, H*W).permute(0, 2, 1)
                non_match_s_descriptors.append(torch.index_select(out_s_flat, 1, s_idxs_f).squeeze(0))
                non_match_t_descriptors.append(torch.index_select(out_t_flat, 1, t_idxs_f).squeeze(0))
            if self._hard_neg:
                for des_s, des_t in zip(non_match_s_descriptors, non_match_t_descriptors):
                    with torch.no_grad():
                        loss = self._sq_hinge_loss(des_s, des_t, False)
                    hard_negative_idxs = torch.nonzero(loss)
                    if hard_negative_idxs.size(0) != 0:
                        des_s = des_s[hard_negative_idxs.squeeze()]
                        des_t = des_t[hard_negative_idxs.squeeze()]
                        if des_s.ndimension() < 2:
                            des_s.unsqueeze_(0)
                            des_t.unsqueeze_(0)
                    non_match_loss += self._sq_hinge_loss(des_s, des_t)
            else:
                non_match_s_descriptors = torch.cat(non_match_s_descriptors, dim=0)
                non_match_t_descriptors = torch.cat(non_match_t_descriptors, dim=0)
                non_match_loss += self._sq_hinge_loss(non_match_s_descriptors, non_match_t_descriptors)
        non_match_loss /= batch_size
        return non_match_loss

    def forward(self, outs_s, outs_t, labels):
        """Computes contrastive loss.

        Args:
            outs_s: list of len B, rotations x D x H x W.
            outs_t: B x D x H x W
            labels: N x 6
        """
        match_loss = self._positive_corrs(outs_s, outs_t, labels)
        non_match_loss = self._negative_corrs(outs_s, outs_t, labels)
        if self.sample_ratio is not None:
            assert not np.isclose(self.sample_ratio, 0.0), "[!] Sample ratio cannot be zero."
            loss = (self.sample_ratio * match_loss) + non_match_loss
        return match_loss, non_match_loss


class TripletLoss(BaseLoss):
    """The triplet loss for training a correspondence prediction network.
    """

    def __init__(self, alpha, num_rotations, device):
        self.alpha = alpha
        self.num_rotations = num_rotations
        self._device = device

    def _triplet_loss(self, anchors, positives, negatives):
        diff_pos = (anchors - positives).pow(2).sum(1)
        diff_neg = (anchors - negatives).pow(2).sum(1)
        return torch.clamp(diff_pos - diff_neg + self.alpha, min=0).mean()

    def forward(self, outs_s, outs_t, labels):
        B, D, H, W = outs_t.shape
        batch_size = len(labels)
        loss = zero_loss(self._device)
        for b_idx, label in enumerate(labels):
            mask = torch.all(
                label == torch.LongTensor([999]).repeat(6).to(self._device), dim=1
            )
            label = label[~mask]
            source_idxs = label[:, 0:2]
            target_idxs = label[:, 2:4]
            rot_idx = label[:, 4]
            is_match = label[:, 5]
            r_idx = rot_idx[0]
            out_s = outs_s[b_idx]
            out_t = outs_t[b_idx : b_idx + 1]
            mask = (is_match == 1) & (rot_idx == r_idx)
            match_s_idxs = source_idxs[mask]
            match_t_idxs = target_idxs[mask]
            mask = is_match != 1
            non_match_s_idxs = source_idxs[mask]
            non_match_rot_idxs = rot_idx[mask]
            match_s_idxs_f = match_s_idxs[:, 0] * W + match_s_idxs[:, 1]
            out_s_flat = out_s[r_idx : r_idx + 1].view(1, D, H * W).permute(0, 2, 1)
            positives = torch.index_select(out_s_flat, 1, match_s_idxs_f).squeeze(0)
            match_t_idxs_f = match_t_idxs[:, 0] * W + match_t_idxs[:, 1]
            out_t_flat = out_t.view(1, D, H * W).permute(0, 2, 1)
            anchors = torch.index_select(out_t_flat, 1, match_t_idxs_f).squeeze(0)
            rand_idxs = np.random.choice(
                np.arange(len(non_match_rot_idxs)),
                replace=False,
                size=len(match_s_idxs_f),
            )
            negatives = []
            for idx in rand_idxs:
                nm_rot_idx = non_match_rot_idxs[idx]
                non_match_s_idx = non_match_s_idxs[idx]
                non_match_s_idx_f = non_match_s_idx[0] * W + non_match_s_idx[1]
                out_s_flat = (
                    out_s[nm_rot_idx : nm_rot_idx + 1]
                    .view(1, D, H * W)
                    .permute(0, 2, 1)
                )
                negatives.append(
                    torch.index_select(out_s_flat, 1, non_match_s_idx_f).squeeze(0)
                )
                negatives = torch.cat(negatives, dim=0)
            loss += self._triplet_loss(anchors, positives, negatives)
        loss /= batch_size
        return loss


def compute_bce_loss(true, logits, pos_weight=None):
    """Computes the weighted binary cross-entropy loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, 1, H, W]. Corresponds to
            the raw output or logits of the model.
        pos_weight: a scalar representing the weight attributed
            to the positive class. This is especially useful for
            an imbalanced dataset.
    Returns:
        bce_loss: the weighted binary cross-entropy loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits.float(),
        true.float(),
        pos_weight=pos_weight,
    )
    return bce_loss


def compute_ce_loss(true, logits, weights, ignore=255):
    """Computes the weighted multi-class cross-entropy loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        weight: a tensor of shape [C,]. The weights attributed
            to each class.
        ignore: the class index to ignore.
    Returns:
        ce_loss: the weighted multi-class cross-entropy loss.
    """
    ce_loss = F.cross_entropy(
        logits.float(),
        true.long(),
        ignore_index=ignore,
        weight=weights,
    )
    return ce_loss


def compute_dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def compute_jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(probas, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)


def compute_tversky_loss(true, logits, alpha, beta, eps=1e-7):
    """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(probas, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)