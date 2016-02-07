from ..daostorm_3d.algorithm import make_margin
from ..daostorm_3d.data import col_nums


def locate(raw_image, diameter, threshold, im_size, finder_class,
           fitter_class, fitter_margin=10):
    finder = finder_class(diameter, im_size)
    peaks = finder.find(raw_image, threshold)

    img_with_margin = make_margin(raw_image, fitter_margin)
    peaks[:, [col_nums.x, col_nums.y]] += fitter_margin

    fitter = fitter_class(img_with_margin, peaks, margin=fitter_margin)
    fitter.fit()

    peaks = fitter.peaks
    peaks[:, [col_nums.x, col_nums.y]] -= fitter_margin

    return peaks
