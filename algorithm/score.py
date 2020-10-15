import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def local_score(label1, label2, blob_levels1, blob_levels2):
    n1, n2 = len(blob_levels1), len(blob_levels2)
    blobs1 = [label1 == i + 1 for i in range(n1)]
    blobs2 = [label2 == i + 1 for i in range(n2)]
    score_blobs = np.zeros((n1, n2))
    for i, blob1, level1 in zip(range(n1), blobs1, blob_levels1):
        for j, blob2, level2 in zip(range(n2), blobs2, blob_levels2):
                grad_term = 1 - np.abs(level1 - level2) / 255
                overlap_term = np.sum(blob1 & blob2) / np.sum(blob1 | blob2)
                score_blobs[i, j] = grad_term * overlap_term
    return np.max(score_blobs, axis=0).sum() + np.max(score_blobs, axis=1).sum()


def global_score(label1, label2, mask1, mask2):
    fg = (mask1 > 0) | (mask2 > 0)
    return normalized_mutual_info_score(label1[fg], label2[fg])


def global_match(mask1, mask2, label1, label2, flip=True):
    score = global_score(label1, label2, mask1, mask2)
    if flip:
        label1_flip_x = np.flip(label1, axis=0)
        label2_flip_y = np.flip(label2, axis=1)
        mask1_flip_x = np.flip(mask1, axis=0)
        mask2_flip_y = np.flip(mask2, axis=1)
        score_flip_x = global_score(label1_flip_x, label2, mask1_flip_x, mask2)
        score_flip_y = global_score(label1, label2_flip_y, mask1, mask2_flip_y)
        score_flip_xy = global_score(label1_flip_x, label2_flip_y, mask1_flip_x, mask2_flip_y)
        return score, score_flip_x, score_flip_y, score_flip_xy
    else:
        return score


def local_match(label1, label2, blob_levels1, blob_levels2, flip=True):
    score = local_score(label1, label2, blob_levels1, blob_levels2)
    if flip:
        label1_flip_x = np.flip(label1, axis=0)
        label2_flip_y = np.flip(label2, axis=1)
        score_flip_x = local_score(label1_flip_x, label2, blob_levels1, blob_levels2)
        score_flip_y = local_score(label1, label2_flip_y, blob_levels1, blob_levels2)
        score_flip_xy = local_score(label1_flip_x, label2_flip_y, blob_levels1, blob_levels2)
        return score, score_flip_x, score_flip_y, score_flip_xy
    else:
        return score
