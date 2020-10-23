import numpy as np
from sklearn.mixture import GaussianMixture
from skimage.color import label2rgb
from skimage import img_as_ubyte


def global_gmm(image, mask, n=5, label_only=True):
    assert type(image) is np.ndarray
    assert len(image.shape) == 2
    assert type(mask) is np.ndarray
    assert len(mask.shape) == 2
    global_mean = int(np.mean(image[mask > 0]))
    label = np.zeros(mask.shape, dtype=np.uint8)
    levels = []
    fg_ind = mask.nonzero()
    while True:
        data = image[fg_ind]
        gmm = GaussianMixture(n_components=min(n, len(data)), random_state=123)
        prediction, means = gmm.fit_predict(data), gmm.means_
        min_label = np.argmin(means)
        if means[min_label] >= global_mean:
            break
        levels.append(means[min_label])
        min_ind = prediction == min_label
        label[fg_ind[0][min_ind], fg_ind[1][min_ind]] = len(levels)
        fg_ind = fg_ind[0][~min_ind], fg_ind[1][~min_ind]
    image = None if label_only else img_as_ubyte(label2rgb(label, image, bg_label=0))
    return image, label, levels
