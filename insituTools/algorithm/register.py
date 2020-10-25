import numpy as np
import cv2
from skimage.exposure import rescale_intensity, adjust_sigmoid
from skimage.util import invert, img_as_float, img_as_ubyte
from sklearn.decomposition import PCA


def get_angle(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.degrees(np.arccos(dot / norm))


def saturation_rectified_intensity(image):
    assert (
        type(image) is np.ndarray and image.dtype == np.uint8 and len(image.shape) == 3
    ), "The input image has to be a uint8 2D numpy array."
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = img_as_float(image_hsv[:, :, 1])
    intensity = img_as_float(image_hsv[:, :, 2])
    adjust = adjust_sigmoid(saturation, 0.08, 25)
    signal = invert(intensity)
    image_out = invert(adjust * signal)
    return img_as_ubyte(image_out)


def rescale_foreground(image, mask):
    assert (
        type(image) is np.ndarray and image.dtype == np.uint8 and len(image.shape) == 2
    ), "The input image has to be a uint8 2D numpy array."
    assert (
        type(mask) is np.ndarray and mask.dtype == np.uint8 and len(mask.shape) == 2
    ), "The input mask has to be a uint8 2D numpy array."
    fg_intensity = image[mask > 0]
    fg_range = np.min(fg_intensity), np.max(fg_intensity)
    return rescale_intensity(image, fg_range)


def affine_correct(image, mask):
    assert (
        type(image) is np.ndarray and image.dtype == np.uint8 and len(image.shape) == 2
    ), "The input image has to be a uint8 2D numpy array."
    assert (
        type(mask) is np.ndarray and mask.dtype == np.uint8 and len(mask.shape) == 2
    ), "The input mask has to be a uint8 2D numpy array."
    fg_pix = np.flip(mask.nonzero())
    pca = PCA(2)
    pca.fit(fg_pix.transpose())
    angle = get_angle(pca.components_[0, :], [1, 0])
    affine_mat = cv2.getRotationMatrix2D((0, 0), -angle, 1)
    new_pts = np.dot(affine_mat[:, :2], fg_pix)
    x_min, x_max = new_pts[0, :].min(), new_pts[0, :].max()
    y_min, y_max = new_pts[1, :].min(), new_pts[1, :].max()
    width, height = int(x_max - x_min), int(y_max - y_min)
    affine_mat[:, 2] = [-x_min, -y_min]
    image_out = cv2.warpAffine(image, affine_mat, (width, height), borderValue=255)
    mask_out = cv2.warpAffine(mask, affine_mat, (width, height), borderValue=0)
    return image_out, mask_out


def register(image, mask, size=None, rotate=True, rescale=True, rectify=False):
    assert (
        type(image) is np.ndarray and image.dtype == np.uint8
    ), "The input image has to be a uint8 numpy array."
    assert (
        type(mask) is np.ndarray and mask.dtype == np.uint8 and len(mask.shape) == 2
    ), "The input mask has to be a uint8 2D numpy array."
    assert (
        size is None or len(size) == 2
    ), "The 'size' should be a tuple of 2 integers, or does not exist."
    assert type(rotate) is bool
    assert type(rescale) is bool
    assert type(rectify) is bool
    if len(image.shape) > 2:
        image = (
            saturation_rectified_intensity(image)
            if rectify
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )
    if rescale:
        image = rescale_foreground(image, mask)
    if rotate:
        image, mask = affine_correct(image, mask)
    if size is not None:
        image = cv2.resize(image, size)
        mask = cv2.resize(mask, size)
    return image, mask
