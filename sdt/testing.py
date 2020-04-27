# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def dist_sample(dist_func, range, nsamp, anchors=100, func_type="pdf"):
    """Create a sample whose distribution function is `dist_func`

    **This is not a random sample.** Instead, the sample's distribution
    function is accurately and smoothly given by `dist_func`. This can be used
    for testing purposes.

    Parameters
    ----------
    dist_func : callable
        PDF or CDF of the sample. See alse the `func_type` param.
    range : tuple of float
        Sample value range
    nsamp : int
        Sample size
    anchors : int, optional
        The `range` will be divided into `anchors` parts; between those, the
        returned sample's CDF will be linear (PDF will be constant).
        `nsamp` should be larger than `anchors` by at least an order of
        magnitude (more to get more precise results). Defaults to 100.
    func_type : {"pdf", "cdf"}, optional
        Whether `dist_func` is a PDF or a CDF. Defaults to "pdf".
    """
    obs = np.linspace(*range, anchors + 1)

    if func_type.lower() == "pdf":
        dx = (range[1] - range[0]) / (anchors + 1)
        pdf = dist_func(obs) * dx
        pdf = (pdf[:-1] + pdf[1:]) / 2
    elif func_type.lower() == "cdf":
        pdf = np.diff(dist_func(obs))

    ret = []
    for x1, x2, p in zip(obs[:-1], obs[1:], pdf):
        ret.append(np.linspace(x1, x2, int(round(p * nsamp))))

    return np.concatenate(ret)
