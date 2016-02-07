def locate(raw_image, radius, threshold, im_size, finder_class,
           fitter_class, max_iterations):
    finder = finder_class(radius, im_size)
    peaks = finder.find(raw_image, threshold)

    fitter = fitter_class(raw_image, peaks, margin=finder.im_size)
    fitter.max_iterations = max_iterations
    fitter.fit()

    return fitter.peaks
