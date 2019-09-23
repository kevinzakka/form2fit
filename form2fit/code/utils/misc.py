import cv2
import numpy as np

from PIL import Image
from skimage.measure import label

from walle.core import RotationMatrix


def adjust_gamma(image, gamma=1.0):
    """https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def anglepie(num_rotations, use_rad):
    """Computes the rotation increments in a full rotation.

    Args:
        num_rotations: (int) The number of rotation increments.
        use_rad: (bool) Whether to return the angles in radians
            or not (degrees).

    Returns:
        A list containing the angle increments.
    """
    angles = [i * (360.0 / num_rotations) for i in range(num_rotations)]
    if use_rad:
        angles = np.deg2rad(angles).tolist()
    return angles


def rotz2angle(rotz):
    """Extracts z-rotation angle from rotation matrix.

    Args:
        rotz: (ndarray) The (3, 3) rotation about z.
    """
    return np.arctan2(rotz[1, 0], rotz[0, 0])


def clip_uv(uv, rows, cols):
    """Ensures pixel coordinates are within image bounds.
    """
    uv[:, 0] = np.clip(uv[:, 0], 0, rows - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, cols - 1)
    return uv


def scale_uv(uv, rows, cols):
    us = uv[:, 0]
    vs = uv[:, 1]
    us_s = ((2 * us) / rows) - 1
    vs_s = ((2 * vs) / cols) - 1
    return np.vstack([us_s, vs_s]).T


def descale_uv(uv, rows, cols):
    us = uv[:, 0]
    vs = uv[:, 1]
    us_s = 0.5 * ((us + 1.0) * rows)
    vs_s = 0.5 * ((vs + 1.0) * cols)
    return np.vstack([us_s, vs_s]).T


def rotate_uv(uv, angle, rows, cols, cxcy=None):
    """Finds the value of a pixel in an image after a rotation.

    Args:
        uv: (ndarray) The [u, v] image coordinates.
        angle: (float) The rotation angle in degrees.
    """
    txty = [cxcy[0], cxcy[1]] if cxcy is not None else [(rows // 2), (cols // 2)]
    txty = np.asarray(txty)
    uv = np.array(uv)
    aff_1 = np.eye(3)
    aff_3 = np.eye(3)
    aff_1[:2, 2] = -txty
    aff_2 = RotationMatrix.rotz(np.radians(angle))
    aff_3[:2, 2] = txty
    affine = aff_3 @ aff_2 @ aff_1
    affine = affine[:2, :]
    uv_rot = (affine @ np.hstack((uv, np.ones((len(uv), 1)))).T).T
    uv_rot = np.round(uv_rot).astype("int")
    uv_rot = clip_uv(uv_rot, rows, cols)
    return uv_rot


def xyz2uv(xyz, intrinsics):
    """Converts 3D points into 2D pixels.
    """
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    v = np.round((x * fx / z) + cx).astype("int")
    u = np.round((y * fy / z) + cy).astype("int")
    return np.vstack([u, v]).T


def make2d(idx, num_cols=224):
    """Make a linear image index 2D.
    """
    v = idx % num_cols
    u = idx // num_cols
    return u, v


def make1d(u, v, num_cols=224):
    """Make a 2D image index linear.
    """
    return (u * num_cols + v).astype("int")


def rotate_img(img, angle, center=None, txty=None):
    """Rotates an image represented by an ndarray.
    """
    img = Image.fromarray(img)
    img_r = img.rotate(angle, center=center, translate=txty)
    return np.array(img_r)


def process_mask(mask, erode=True, kernel_size=3):
    """Cleans up a binary mask.
    """
    mask = mask.astype("float32")
    kernel = np.ones((kernel_size, kernel_size), "uint8")
    if erode:
        mask = cv2.erode(mask, np.ones((3, 3), "uint8"), iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2).astype("float32")
    return mask


def largest_cc(mask):
    labels = label(mask)
    largest = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest


def mask2bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


class AverageMeter(object):
    """Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.arr = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.arr.append(val)
