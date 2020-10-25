import numpy as np
import cv2


def std_filter(image, kernel):
    assert (
        type(image) is np.ndarray and image.dtype == np.uint8 and len(image.shape) == 2
    ), "The input image has to be a uint8 2D numpy array."
    assert len(kernel) == 2, "The 'kernel' should be a tuple of 2 integers."
    image = image.astype(np.float)
    image = cv2.sqrt(cv2.blur(image ** 2, kernel) - cv2.blur(image, kernel) ** 2)
    return image


def fill_hole(mask):
    assert (
        type(mask) is np.ndarray and mask.dtype == np.uint8 and len(mask.shape) == 2
    ), "The input mask has to be a uint8 2D numpy array."
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    mask = cv2.drawContours(mask, contour, -1, 255, cv2.FILLED)
    return mask


def extract(image, filterSize=3, threshold=3, mask_only=False):
    assert (
        type(image) is np.ndarray and image.dtype == np.uint8
    ), "The input image has to be a uint8 numpy array."
    assert type(filterSize) is int
    assert type(mask_only) is bool
    kernel = (filterSize, filterSize)
    std = std_filter(
        image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        kernel,
    )
    thr = cv2.threshold(std, threshold, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    mask = fill_hole(thr)
    image = (
        None
        if mask_only
        else cv2.bitwise_or(
            image, cv2.bitwise_not(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        )
    )
    return image, mask
