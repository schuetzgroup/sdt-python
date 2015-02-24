import numpy as np
import pandas as pd
import scipy as sp


def vectors_cartesian(features, pos_columns = ["x", "y"]):
    dx = np.zeros((len(features), len(features)))
    dy = np.zeros((len(features), len(features)))

    #first column: x coordinate, second column: y coordinate
    data = features[pos_columns].as_matrix()
    for i in range(len(features)):
        diff = data - data[i]

        dx[i, :] = diff[:, 0]
        dy[i, :] = diff[:, 1]

    return dx, dy


def all_scores_cartesian_slow(vec1, vec2, tol_a, tol_r):
    dx1 = vec1[0]
    dy1 = vec1[1]
    dx2 = vec2[0]
    dy2 = vec2[1]

    score = np.zeros((len(dx1), len(dx2)), np.uint)

    for i, j in np.ndindex(dx1.shape):
        x1 = dx1[i, j]
        y1 = dy1[i, j]
        if x1 == 0 or y1 == 0:
            continue
        for k, l in np.ndindex(dx2.shape):
            x2 = dx2[k, l]
            y2 = dy2[k, l]
            if x2 == 0 or y2 == 0:
                continue
            dx = x2-x1
            dy = y2-y1
            if np.abs(dx/x1) < tol_r and np.abs(dy/y1) < tol_r:
                score[i, k] += 1

    return score


def _extend_array(a, new_shape, value=0):
    b = np.array(a)
    if len(new_shape) != len(b.shape):
        raise ValueError("new_shape must have the same dimensions as a.shape")
    if (np.array(new_shape) < np.array(b.shape)).any():
        raise ValueError("new_shape must be larger a.shape")

    return np.pad(b,
                  pad_width=[(0, n-o) for o, n in zip(b.shape, new_shape)],
                  mode="constant",
                  constant_values=[(0, value)]*len(new_shape))


def all_scores_cartesian(vec1, vec2, tol_rel, tol_abs):
    dx1 = vec1[0]
    dy1 = vec1[1]
    dx2 = vec2[0]
    dy2 = vec2[1]

    score = np.zeros((len(dx1), len(dx2)), np.uint)
    #TODO: sanity checking

    #pad the smaller matrix with infinity to the size of the larger matrix
    if len(dx1) > len(dx2):
        dx2 = _extend_array(dx2, (len(dx1), len(dx1)), np.inf)
        dy2 = _extend_array(dy2, (len(dy1), len(dy1)), np.inf)
    else:
        dx1 = _extend_array(dx1, (len(dx2), len(dx2)), np.inf)
        dy1 = _extend_array(dy1, (len(dy2), len(dy2)), np.inf)

    for i in range(len(dx1)**2):
        #check whether the coordinates are close enough
        #by roll()ing one matrix we shift the elements. If we do that
        #len(dx1)**2 times, we compare all elements of dx2 to those of dx1
        #same for y
        x = np.isclose(np.roll(dx2, -i), dx1, tol_rel, tol_abs)
        y = np.isclose(np.roll(dy2, -i), dy1, tol_rel, tol_abs)
        #x and y coordinates have to match
        total = x & y

        #indices of matches in dx1, dy1
        matches1 = total.nonzero()
        #indices of matches in dx2, dy2, need to be roll()ed back
        matches2 = np.roll(total, i).nonzero()
        for m1, m2 in zip(matches1[0], matches2[0]):
            score[m1, m2] += 1
        for m1, m2 in zip(matches1[1], matches2[1]):
            score[m1, m2] += 1

    return score


def pairs_from_score(feat1, feat2, score, score_cutoff=None,
                     pos_columns=["x", "y"],
                     feat_names=["channel1", "channel2"]):

    if score_cutoff is None:
        score_cutoff = 0.5*score.max()

    #TODO: detect doubles
    #TODO: do not solely rely on the maximum. If there are similarly high
    #scores, discard both because of ambiguity
    mi = pd.MultiIndex.from_product([feat_names, pos_columns])
    df = pd.DataFrame(columns=mi)

    indices = np.zeros(score.shape, bool)
    if score.shape[0] > score.shape[1]:
        indices = np.array([(i, j) for i, j in zip(np.argmax(score, axis=0),
                                          range(score.shape[1]))])
    else:
        indices = [(i, j) for i, j in zip(range(score.shape[0]),
                                          np.argmax(score, axis=1))]

    for i, ind in enumerate(indices):
        if score[ind] < score_cutoff:
            #score is too low
            continue
        df.loc[i] = (feat1.iloc[ind[0]][pos_columns[0]],
                     feat1.iloc[ind[0]][pos_columns[1]],
                     feat2.iloc[ind[1]][pos_columns[0]],
                     feat2.iloc[ind[1]][pos_columns[1]])
    return df


def find_pairs(feat1, feat2, tol_rel=0.1, tol_abs=0., score_cutoff=None,
               pos_columns=["x", "y"], feat_names=["channel1", "channel2"]):
    v1 = vectors_cartesian(feat1, pos_columns)
    v2 = vectors_cartesian(feat2, pos_columns)
    s = all_scores_cartesian(v1, v2, tol_rel, tol_abs)
    return pairs_from_score(feat1, feat2, s, score_cutoff, pos_columns,
                         feat_names)


def correction_parameters(pairs, pos_columns=["x", "y"],
                          feat_names=["channel1", "channel2"]):
    pars = []
    for p in pos_columns:
        r = sp.stats.linregress(pairs[feat_names[0]][p],
                                pairs[feat_names[1]][p])
        pars.append([r[i] for i in [0, 1, 4]])
    return pd.DataFrame(pars, columns=["slope", "intercept", "stderr"],
                        index=pos_columns)


def test(pairs, parms, pos_columns=["x", "y"],
         feat_names=["channel1", "channel2"]):
    import matplotlib.pyplot as plt

    for i, p in enumerate(pos_columns):
        plt.subplot(1, len(pos_columns), i+1, aspect=1)
        plt.scatter(pairs[feat_names[0], p], pairs[feat_names[1], p])
        plt.plot(pairs[feat_names[0]].sort(p)[p],
                 parms.loc[p, "slope"] * pairs[feat_names[0]].sort(p)[p] +
                 parms.loc[p, "intercept"])
        plt.title(p)