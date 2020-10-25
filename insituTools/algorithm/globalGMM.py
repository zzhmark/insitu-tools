import numpy as np
from sklearn.mixture import GaussianMixture
from skimage.color import label2rgb
from skimage import img_as_ubyte
from skimage.measure import block_reduce
import cv2


def global_gmm(image, mask, patchSize=None, n=5, label_only=True):
    assert (
        type(image) is np.ndarray and image.dtype == np.uint8 and len(image.shape) == 2
    ), "The input image has to be a uint8 2D numpy array."
    assert (
        type(mask) is np.ndarray and mask.dtype == np.uint8 and len(mask.shape) == 2
    ), "The input mask has to be a uint8 2D numpy array."
    assert type(n) is int
    assert type(label_only) is bool
    assert type(patchSize) is int and patchSize > 0
    patch = (patchSize, patchSize)
    image_reduced = (
        image if patchSize == 1 else block_reduce(image, patch, np.mean, 255)
    )
    mask_reduced = mask if patchSize == 1 else block_reduce(mask, patch, np.min)
    global_mean = int(np.mean(image_reduced[mask_reduced > 0]))
    label = np.zeros(mask_reduced.shape, dtype=np.uint8)
    levels = []
    fg_ind = mask_reduced.nonzero()
    while True:
        data = image_reduced[fg_ind].reshape((-1, 1))
        gmm = GaussianMixture(n_components=min(n, len(data)), random_state=123)
        prediction, means = gmm.fit_predict(data), gmm.means_
        min_label, min_mean = np.argmin(means), np.min(means)
        if min_mean >= global_mean:
            break
        levels.append(min_mean)
        min_ind = prediction == min_label
        label[fg_ind[0][min_ind], fg_ind[1][min_ind]] = len(levels)
        fg_ind = fg_ind[0][~min_ind], fg_ind[1][~min_ind]
    label_resized = (
        label
        if patchSize == 1
        else cv2.resize(label, image.shape[::-1], interpolation=cv2.INTER_NEAREST)
    )
    image_labeled = (
        None
        if label_only
        else img_as_ubyte(label2rgb(label_resized, image, bg_label=0))
    )
    return image_labeled, mask_reduced, label, levels
