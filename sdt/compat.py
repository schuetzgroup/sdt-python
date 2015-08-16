"""MATLAB tools compatibility layer

Allow for calling matlab tools from python. So far, the following tools are
supported:
- plotpdf

Attributes
----------
mass_column : str
    Name of the column describing the integrated intensities ("masses") of
    the features. Defaults to "mass".
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mass_column = "mass"


def plotpdf(data, lim, f, mlab, ax=None, mass_column=mass_column):
    """Call MATLAB ``plotpdf``

    This is a wrapper around ``plotpdf`` using `pymatbridge`

    Parameters
    ----------
    data : list or pandas.DataFrame
        List of molecule brightnesses. If it is a DataFrame, use the column
        named by the `mass_column` argument.
    lim : float
        Maximum brightness
    f : float
        Correction factor for sigma
    matlab_engine : pymatbridge.Matlab
        Matlab object with a running session
    ax : axis object, optional
        `matplotlib` axes to use for plotting. If None, use `gca()`. Defaults
        to None.
    mass_column : str, optional
        Name of the column describing the integrated intensities ("masses") of
        the features. Defaults to the `mass_column` attribute of the module.
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(data, pd.DataFrame):
        data = data[mass_column]

    res = mlab.run_func("plotpdf", data[:, np.newaxis], float(lim), float(f),
                        "r", 0, nargout=2)
    ax.plot(np.squeeze(res["result"][0]), np.squeeze(res["result"][1]))
