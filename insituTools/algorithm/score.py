import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize


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


def hybrid_score(global_scores, local_scores):
    return np.max(np.array(global_scores) * np.array(local_scores))


def score(
    table,
    reference,
    masks,
    global_labels,
    local_labels,
    local_levels_list,
    global_score_cutoff=0,
    flip=True,
):
    assert (
        len(reference)
        <= len(masks)
        == len(global_labels)
        == len(local_labels)
        == len(local_levels_list)
    )
    n_row = len(reference)
    n_col = len(masks)
    for i in range(n_row):
        ref_masks = [masks[i]]
        ref_global_labels = [global_labels[i]]
        ref_local_labels = [local_labels[i]]
        if flip:
            ref_masks.append(np.flipud(masks[i]))
            ref_masks.append(np.fliplr(masks[i]))
            ref_masks.append(np.fliplr(ref_masks[1]))
            ref_global_labels.append(np.flipud(global_labels[i]))
            ref_global_labels.append(np.fliplr(global_labels[i]))
            ref_global_labels.append(np.fliplr(ref_global_labels[1]))
            ref_local_labels.append(np.flipud(local_labels[i]))
            ref_local_labels.append(np.fliplr(local_labels[i]))
            ref_local_labels.append(np.fliplr(ref_local_labels[1]))
        for j in range(n_col):
            global_scores = [
                global_score(mask, masks[j], label, global_labels[j])
                for mask, label in zip(ref_masks, ref_global_labels)
            ]
            if np.max(global_scores) > global_score_cutoff:
                table[i, j] = hybrid_score(
                    global_scores,
                    [
                        local_score(
                            label,
                            local_labels[j],
                            local_levels_list[i],
                            local_levels_list[j],
                        )
                        for label in ref_local_labels
                    ],
                )
    return normalize(table, norm="max")
