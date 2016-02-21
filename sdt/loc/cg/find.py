import numpy as np
from scipy import ndimage


def find(image, search_radius, threshold):
    # reverse column order to convert image matrix indices to x, y coordinates
    return local_maxima(image, search_radius, threshold)[:, ::-1]


def local_maxima(image, search_radius, threshold):
    # create circular mask with radius `search_radius`
    x_sq = np.arange(-search_radius, search_radius + 1)**2
    mask = x_sq[:, np.newaxis] + x_sq[np.newaxis, :]
    mask = (mask <= search_radius**2)

    # use mask to dilate the image
    dil = ndimage.grey_dilation(image, footprint=mask)

    # if threshold is None, determine it from the data
    if threshold is None:
        threshold = np.percentile(image - image.min(), 70) + 1

    # wherever the image value is equal to the dilated value, we have a
    # candidate for a maximum
    candidates = np.nonzero(np.isclose(dil, image) & (image >= threshold))
    candidates = np.transpose(candidates)

    # discard maxima within `search_radius` pixels of the edges
    in_margin = np.any(
        (candidates < search_radius) |
        (candidates >= np.array(image.shape) - search_radius - 1), axis=1)
    candidates = candidates[~in_margin]

    # remove spurious maxima which result from flat peaks
    is_max = np.zeros(len(candidates), dtype=np.bool)
    candidate_list = candidates.T.tolist()
    max_img = np.zeros(image.shape)
    max_img[candidate_list] = image[candidate_list]

    for cnt, (i, j) in enumerate(candidates):
        roi = max_img[i-search_radius:i+search_radius+1,
                      j-search_radius:j+search_radius+1] * mask

        # complicated, but that's the way the original implementation does it
        # and we want to be compatible
        sort_val = np.sort(roi, axis=0)
        sort_idx = np.argsort(roi, axis=0)
        max_j = np.argmax(sort_val[-1, :])  # column of the max value
        max_i = sort_idx[-1, max_j]         # row of the max value
        if (max_i == search_radius) and (max_j == search_radius):
            # the current value is the maximum in the ROI, since
            # max_i == max_j == search_radius means that the max is in the
            # center of the ROI
            is_max[cnt] = True
        else:
            # delete from max_img
            max_img[i, j] = 0

    return candidates[is_max]
