"""Correct chromatic aberrations of localizations.

When doing multi-color single molecule microscopy, chromatic aberrations pose
a problem. To circumvent this, one can record an image of beads, match the
localizations (as determined by a localization algorithm such as trackpy or
tracking2d) of the two channels and determine affine transformations for the
x and y coordinates.

Attributes:
    pos_colums (list of str): Names of the columns describing the x and the y
        coordinate of the features in pandas.DataFrames. Defaults to
        ["x", "y"].
    channel_names (list of str): Names of the two channels. Defaults to
        ["channel1", "channel2"].
"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.io


pos_columns = ["x", "y"]
channel_names = ["channel1", "channel2"]


class Corrector(object):
    """Convenience class for easy correction of chromatic aberrations

    This class provides an easy-to-use interface to the correction
    process.

    Attributes:
        pos_columns (list of str): Names of the columns describing the x and
            the y coordinate of the features.
        channel_names (list of str): Names of the channels.
        feat1 (pandas.DataFrame): Features of the first channel found by the
            localization algorithm. The x coordinate is in the column with name
            `pos_columns`[0], the y coordinate in `pos_columns`[1].
        feat2 (pandas.DataFrame): Features of the second channel found by the
            localization algorithm. The x coordinate is in the column with name
            `pos_columns`[0], the y coordinate in `pos_columns`[1].
        pairs (pandas.DataFrame): Contains the pairs found by
            `determine_parameters`.
        parameters (pandas.DataFrame): The parameters for the transformation as
            determined by `determine_parameters`.
    """
    def __init__(self, feat1, feat2, pos_columns=pos_columns,
                 channel_names=channel_names):
        """Constructor

        Args:
            feat1 (pandas.DataFrame): Sets the `feat1` attribute
            feat2 (pandas.DataFrame): Sets the `feat2` attribute
            pos_columns (list of str): Sets the `pos_columns` attribute.
                Defaults to the `pos_columns` attribute of the module.
            channel_names (list of str): Sets the `channel_names` attribute.
                Defaults to the `channel_names` attribute of the module.
        """
        self.feat1 = feat1
        self.feat2 = feat2
        self.pos_columns = pos_columns
        self.channel_names = channel_names
        self.pairs = None
        self.parameters = None

    def determine_parameters(self, tol_rel=0.1, tol_abs=0.):
        """Determine the parameters for the affine transformation

        This takes the localizations of `feat1` and tries to match them with
        those of `feat2`. Then a linear fit is used to determine the affine
        transformation to correct for chromatic aberrations.

        This is a convenience function that calls `find_pairs` and
        `fit_parameters`.

        Args:
            tol_rel (float): Relative tolerance parameter for
                `numpy.isclose()`. Defaults to 0.1.
            tol_abs (float): Absolute tolerance parameter for
                `numpy.isclose()`. Defaults to 0.
        """
        #TODO: score_cutoff
        self.find_pairs(tol_rel, tol_abs)
        self.fit_parameters()

    def find_pairs(self, tol_rel=0.1, tol_abs=0.):
        """Match features of `feat1` with features of `feat2`

        This is done by calculating the vectors from every
        feature all the other features in `feat1` and compare them to those of
        `feat2` using the `numpy.isclose` function on both the x and the y
        coordinate, where `tol_rel` and `tol_abs` are the relative and
        absolute tolerance parameters.


        Args:
            tol_rel (float): Relative tolerance parameter for
                `numpy.isclose()`. Defaults to 0.1.
            tol_abs (float): Absolute tolerance parameter for
                `numpy.isclose()`. Defaults to 0.
        """
        #TODO: score_cutoff
        v1 = self._vectors_cartesian(self.feat1)
        v2 = self._vectors_cartesian(self.feat2)
        s = self._all_scores_cartesian(v1, v2, tol_rel, tol_abs)
        self._pairs_from_score(s, None)

    def fit_parameters(self):
        """Determine parameters for the affine transformation

        A linear fit is used to map x and y coordinates of `feat1` to to those
        of `feat2`. This requires `find_pairs` to be run first.
        """
        pars = []
        for p in self.pos_columns:
            r = sp.stats.linregress(self.pairs[self.channel_names[0]][p],
                                    self.pairs[self.channel_names[1]][p])
            pars.append([r[i] for i in [0, 1, 4]])
        self.parameters = pd.DataFrame(pars,
                                       columns=["slope",
                                                "intercept",
                                                "stderr"],
                                       index=pos_columns)

    def __call__(self, features, channel=2):
        """Do chromatic correction on `features` coordinates

        This modifies the coordinates in place.

        Args:
            features (pandas.DataFrame): The features to be corrected
            channel (int, optional): If `features` are in the first channel
                (corresponding to the `feat1` arg of the constructor), set to
                1. If features are in the second channel, set to 2. Depending
                on this, a transformation will be applied to the coordinates of
                `features` to match the other channel (mathematically speaking
                depending on this parameter either the "original"
                transformation or its inverse are applied.)
        """
        if channel != 1 and channel != 2:
            raise ValueError("channel has to be either 1 or 2")

        x_col = self.pos_columns[0]
        y_col = self.pos_columns[1]
        xparm = self.parameters.loc[x_col]
        yparm = self.parameters.loc[y_col]

        if channel == 1:
            features[x_col] = features[x_col]*xparm.slope + xparm.intercept
            features[y_col] = features[y_col]*yparm.slope + yparm.intercept

        if channel == 2:
            features[x_col] = (features[x_col] - xparm.intercept)/xparm.slope
            features[y_col] = (features[y_col] - yparm.intercept)/yparm.slope

    def test(self):
        """Test validity of the correction parameters

        This plots the affine transformation functions and the coordinates of
        the pairs that were matched in the channels. If everything went well,
        the dots (i. e. pair coordinates) should lie on the line
        (transformation function).
        """
        import matplotlib.pyplot as plt

        for i, p in enumerate(self.pos_columns):
            plt.subplot(1, len(self.pos_columns), i+1, aspect=1)
            ax = plt.gca()
            ax.set_xlabel("{} ({})".format(p, self.channel_names[0]))
            ax.set_ylabel("{} ({})".format(p, self.channel_names[1]))
            ax.scatter(self.pairs[self.channel_names[0], p],
                       self.pairs[self.channel_names[1], p])
            ax.plot(self.pairs[self.channel_names[0]].sort(p)[p],
                    self.parameters.loc[p, "slope"] *
                    self.pairs[self.channel_names[0]].sort(p)[p] +
                    self.parameters.loc[p, "intercept"])
            ax.set_title(p)

    def to_hdf(self, path_or_buf, key="chromatic_correction_parameters"):
        """Save parameters to a HDF5 file

        This simply calls `parameters.to_hdf`(path_or_buf, key)
        """
        self.parameters.to_hdf(path_or_buf, key)

    def read_hdf(path_or_buf, key="chromatic_correction_parameters"):
        """Read paramaters from a HDF5 file and construct a corrector

        Args:
            path_or_buf: Passed to `pandas.read_hdf` as the first argument.
            key: Passed to `pandas.read_hdf` as the second argument.

        Returns:
            A `Corrector` instance with the parameters read from the HDF5
            file.
        """
        corr = Corrector(None, None)
        corr.parameters = pd.read_hdf(path_or_buf, key)
        corr.pos_columns = corr.parameters.index.tolist()
        return corr

    def to_wrp(self, path):
        """Save parameters to .wrp file

        Args:
            path (str): Path of the .wrp file to be created
        """
        k1 = 1./self.parameters.loc[self.pos_columns[0], "slope"]
        d1 = -self.parameters.loc[self.pos_columns[0], "intercept"]*k1
        k2 = 1./self.parameters.loc[self.pos_columns[1], "slope"]
        d2 = -self.parameters.loc[self.pos_columns[1], "intercept"]*k2

        S = dict(k1=k1, d1=d1, k2=k2, d2=d2)
        scipy.io.savemat(path, dict(S=S), appendmat=False)

    def read_wrp(path):
        """Read parameters from a .wrp file

        Construct a Corrector with those parameters

        Args:
            path (str): Path of the .wrp file

        Returns:
            A `Corrector` instance with the parameters read from the .wrp
            file.
        """
        corr = Corrector(None, None)
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        d = mat["S"]

        #data of the wrp file is for the inverse transformation
        parms = np.empty((2, 3))
        parms[0, :] = np.array([1./d.k1, -d.d1/d.k1, np.NaN])
        parms[1, :] = np.array([1./d.k2, -d.d2/d.k2, np.NaN])
        corr.parameters = pd.DataFrame(parms,
                                       columns=["slope", "intercept", "stderr"],
                                       index=pos_columns)
        return corr


    def _vectors_cartesian(self, features):
        """Calculate vectors of each point in features to every other point

        Args:
            features (pandas.DataFrame): Features with coordinates in columns
                `pos_columns`

        Returns:
            tuple of numpy.arrays: The first element of the tuple describes
            the x coordinate of the vectors, the second one the y coordinates.

            If we call the tuple (dx, dy), then dx[i, j] yields the x component
            of the vector pointing from the i-th feature in `features` to the
            j-th feature; same for dy and y coordinates.
        """
        dx = np.zeros((len(features), len(features)))
        dy = np.zeros((len(features), len(features)))

        #first column: x coordinate, second column: y coordinate
        data = features[self.pos_columns].as_matrix()
        for i in range(len(features)):
            diff = data - data[i]

            dx[i, :] = diff[:, 0]
            dy[i, :] = diff[:, 1]

        return dx, dy

    def _all_scores_cartesian(self, vec1, vec2, tol_rel, tol_abs):
        """Determine scores for all possible pairs of features

        Compare all elements of `vec1` and all of `vec2` using
        `numpy.isclose()` to determine which pairs of features have the most
        simalar vectors pointing to the other features.

        Args:
            vec1 (numpy.array): Vectors for the first set of features as
                determined by `_vectors_cartesian`.
            vec2 (numpy.array): Vectors for the second set of features as
                determined by `_vectors_cartesian`.
            tol_rel (float): Relative tolerance parameter for `numpy.isclose`
            tol_abs (float): absolute tolerance parameter for `numpy.isclose`

        Returns:
            numpy.array: A matrix of scores. scores[i, j] holds the number of
                matching vectors for the i-th entry of feat1 and the j-th
                entry of feat2.
        """
        dx1 = vec1[0]
        dy1 = vec1[1]
        dx2 = vec2[0]
        dy2 = vec2[1]

        score = np.zeros((len(dx1), len(dx2)), np.uint)

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
            #x and y coordinates both have to match
            total = x & y

            #if total[i, j] is not zero, that means that the vector pointing
            #from the i-th to the j-th feature in feat1 matches vector
            #pointing from the k-th to the l-th feature in feat2, were k and l
            #are the roll()ed-back indices i and j
            #indices of matches in dx1, dy1
            matches1 = total.nonzero()
            #indices of matches in dx2, dy2, need to be roll()ed back
            matches2 = np.roll(total, i).nonzero()
            #increase the score of features i and k
            for m1, m2 in zip(matches1[0], matches2[0]):
                score[m1, m2] += 1
            #increase the score of features j and l
            for m1, m2 in zip(matches1[1], matches2[1]):
                score[m1, m2] += 1

        return score

    def _pairs_from_score(self, score, score_cutoff=None):
        """Analyze the score matrix and determine what the pairs are

        Search for the maximum value of each feature pair in the score matrix.
        This finally sets the `pairs` attribute.

        Args:
            score (numpy.array): The score matrix as calculated by
                `_all_scores_cartesian`
            score_cutoff (float or None): In order to get rid of false matches
                a threshold for scores can be set. All scores below this
                threshold are discarded. If set to None, 0.5*score.max() will
                be used.
        """
        if score_cutoff is None:
            score_cutoff = 0.5*score.max()

        #TODO: detect doubles
        #TODO: do not solely rely on the maximum. If there are similarly high
        #scores, discard both because of ambiguity
        mi = pd.MultiIndex.from_product([self.channel_names, self.pos_columns])
        self.pairs = pd.DataFrame(columns=mi)

        indices = np.zeros(score.shape, bool)
        #always search through the axis with fewer localizations since
        #since otherwise there is bound to be double matches
        if score.shape[0] > score.shape[1]:
            indices = [(i, j) for i, j in zip(np.argmax(score, axis=0),
                                              range(score.shape[1]))]
        else:
            indices = [(i, j) for i, j in zip(range(score.shape[0]),
                                              np.argmax(score, axis=1))]

        for i, ind in enumerate(indices):
            if score[ind] < score_cutoff:
                #score is too low
                continue
            self.pairs.loc[i] = (self.feat1.iloc[ind[0]][self.pos_columns[0]],
                                 self.feat1.iloc[ind[0]][self.pos_columns[1]],
                                 self.feat2.iloc[ind[1]][self.pos_columns[0]],
                                 self.feat2.iloc[ind[1]][self.pos_columns[1]])


def _extend_array(a, new_shape, value=0):
    """Extend an array with constant values

    The indices of the old values remain the same. E. g. if the array is 2D,
    then rows are added at the bottom and columns at the right

    Args:
        a (numpy.array): Array to be extended
        new_shape (tuple of int): The new shape, must be larger than the old
            one
        value: value to be padded with

    Returns:
        The extended numpy.array
    """
    b = np.array(a)
    if len(new_shape) != len(b.shape):
        raise ValueError("new_shape must have the same dimensions as a.shape")
    if (np.array(new_shape) < np.array(b.shape)).any():
        raise ValueError("new_shape must be larger a.shape")

    return np.pad(b,
                  pad_width=[(0, n-o) for o, n in zip(b.shape, new_shape)],
                  mode="constant",
                  constant_values=[(0, value)]*len(new_shape))