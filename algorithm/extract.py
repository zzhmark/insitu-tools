import numpy as np
import cv2


def std_filter(image, kernel):
    a = image.astype(np.float)
    return cv2.sqrt(cv2.blur(a ** 2, kernel) - cv2.blur(a, kernel) ** 2)


def fill_hole(mask):
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)[0]
    return cv2.drawContours(mask, contour, -1, 255, cv2.FILLED)


def extract(image, kernel=(3, 3), threshold=3, mask_only=False):
    std = std_filter(cv2.cvtColor(image,
                                  cv2.COLOR_BGR2GRAY),
                     kernel)
    thr = cv2.threshold(std, threshold, 255,
                        cv2.THRESH_BINARY)[1].astype(np.uint8)
    mask = fill_hole(thr)
    if mask_only:
        return mask
    else:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        image = cv2.bitwise_or(image, cv2.bitwise_not(mask_bgr))
        return image, mask
