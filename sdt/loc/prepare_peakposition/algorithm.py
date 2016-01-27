def locate(raw_image, diameter, threshold, im_size, finder_class, fitter_class):
    finder = finder_class(raw_image, diameter, im_size)
    peaks = finder.find(raw_image, threshold)

    fitter = fitter_class(raw_image, peaks)
    fitter.fit()

    return peaks
