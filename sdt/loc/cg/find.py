import numpy as np
from scipy import ndimage


def find(image, search_radius, threshold):
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
    candidates = np.nonzero(np.isclose(dil, image) & (image > threshold))
    candidates = np.transpose(candidates)

    # discard maxima within `search_radius` pixels of the edges
    in_margin = np.any(
        (candidates < search_radius) |
        (candidates > np.array(image.shape) - search_radius - 1), axis=1)
    candidates = candidates[~in_margin]

    # remove spurious maxima which result from flat peaks
    is_max = np.empty(len(candidates), dtype=np.bool)
    candidate_list = candidates.T.tolist()
    max_img = np.zeros(image.shape)
    max_img[candidate_list] = image[candidate_list]
    for cnt, (i, j) in enumerate(candidates):
        roi = max_img[i-search_radius:i+search_radius+1,
                      j-search_radius:j+search_radius+1]
        roi *= mask
        if np.sum(roi >= image[i, j]) == 1:
            # this is the only maximum (left), keep it
            is_max[cnt] = True
        else:
            # delete from max_img
            max_img[i, j] = 0

    return candidates[is_max]
