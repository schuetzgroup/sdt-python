def locate(raw_image, diameter, threshold, im_size, finder_class,
           fitter_class):
    finder = finder_class(diameter, im_size)
    peaks = finder.find(raw_image, threshold)

    fitter = fitter_class(raw_image, peaks, margin=finder.im_size)
    fitter.fit()

    return fitter.peaks
