import itertools

import numpy as np
import matplotlib.pyplot as plt


def plot_changepoints(data, changepoints, time=None, segment_alpha=0.2,
                      style="shade", segment_colors=["#4286f4", "#f44174"],
                      ax=None):
    if ax is None:
        ax = plt.gca()
    if time is None:
        time = np.arange(len(data))

    if style == "shade":
        for s, e, c in zip(itertools.chain([0], changepoints),
                           itertools.chain(changepoints, [len(data)-1]),
                           itertools.cycle(segment_colors)):
            t_s = time[s]
            t_e = time[e]
            ax.axvspan(t_s, t_e, facecolor=c, alpha=segment_alpha)
    else:
        for c in changepoints:
            t_c = time[c]
            ax.axvline(x=t_c, linestyle="--", color="k")

    ax.plot(time, data)
    return ax
