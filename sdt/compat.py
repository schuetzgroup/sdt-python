"""MATLAB tools compatibility layer

Allow for calling matlab tools from python. So far, the following tools are
supported:
- plotpdf
"""
def plotpdf(data, lim, f, matlab_engine, ax=None, mass_column=mass_column):
    """Call MATLAB `plotpdf`

    This is a wrapper around `plotpdf` using MATLAB engine for python

    Parameters
    ----------
    data : list or pandas.DataFrame
        List of molecule brightnesses. If it is a DataFrame, use the column
        named by the `mass_column` argument.
    lim : float
        Maximum brightness
    f : float
        Correction factor for sigma
    matlab_engine : MATLAB engine object
        as e.g. returned by `matlab.engine.start()`
    ax : axis object, optional
        `matplotlib` axes to use for plotting. If None, use `gca()`. Defaults
        to None.
    mass_column : str, optional
        Name of the column describing the integrated intensities ("masses") of
        the features. Defaults to the `mass_column` attribute of the module.
    """
    import matplotlib.pyplot as plt
    import matlab.engine

    if ax is None:
        ax = plt.gca()

    if isinstance(data, pd.DataFrame):
        data = data[mass_column]

    data = [[d] for d in data]
    x, y = matlab_engine.plotpdf(matlab.double(data), float(lim), float(f),
                                 "r", 0, nargout=2)
    ax.plot(x._data, y._data)
