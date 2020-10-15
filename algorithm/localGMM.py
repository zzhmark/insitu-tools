import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from skimage.color import label2rgb
from skimage import img_as_ubyte


def local_gmm(image, global_label, levels, n=10, label_only=True):
    local_label = np.zeros(global_label.shape, dtype=np.uint8)
    blob_levels = []
    for i in range(len(levels)):
        lvl_ind = (global_label == i).nonzero()
        data = np.transpose(lvl_ind)
        gmm = BayesianGaussianMixture(n_components=min(n, len(data)),
                                      random_state=123)
        prediction = gmm.fit_predict(data)
        for j in np.unique(prediction):
            blob_levels.append(levels[i])
            blob_pts = prediction == j
            local_label[lvl_ind[0][blob_pts], lvl_ind[1][blob_pts]] = len(blob_levels)
    if label_only:
        return local_label, blob_levels
    else:
        image = img_as_ubyte(label2rgb(local_label, image, bg_label=0))
        return image, local_label, blob_levels