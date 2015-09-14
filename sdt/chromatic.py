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
import scipy.io
import scipy.stats
import scipy.ndimage

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
        parameters1 (pandas.DataFrame): The parameters for the linear
            transformation to correct coordinates of channel 1.
        parameters2 (pandas.DataFrame): The parameters for the linear
            transformation to correct coordinates of channel 2.
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
        self.parameters1 = None
        self.parameters2 = None

    def determine_parameters(self, tol_rel=0.05, tol_abs=0.):
        """Determine the parameters for the affine transformation

        This takes the localizations of `feat1` and tries to match them with
        those of `feat2`. Then a linear fit is used to determine the affine
        transformation to correct for chromatic aberrations.

        This is a convenience function that calls `find_pairs` and
        `fit_parameters`.

        Args:
            tol_rel (float): Relative tolerance parameter for
                `numpy.isclose()`. Defaults to 0.05.
            tol_abs (float): Absolute tolerance parameter for
                `numpy.isclose()`. Defaults to 0.
        """
        #TODO: score_cutoff
        self.find_pairs(tol_rel, tol_abs)
        self.fit_parameters()

    def find_pairs(self, tol_rel=0.05, tol_abs=0.):
        """Match features of `feat1` with features of `feat2`

        This is done by calculating the vectors from every
        feature all the other features in `feat1` and compare them to those of
        `feat2` using the `numpy.isclose` function on both the x and the y
        coordinate, where `tol_rel` and `tol_abs` are the relative and
        absolute tolerance parameters.


        Args:
            tol_rel (float): Relative tolerance parameter for
                `numpy.isclose()`. Defaults to 0.05.
            tol_abs (float): Absolute tolerance parameter for
                `numpy.isclose()`. Defaults to 0.
        """
        #TODO: score_cutoff
        v1 = self._vectors_cartesian(self.feat1)
        v2 = self._vectors_cartesian(self.feat2)
        s = self._all_scores_cartesian(v1, v2, tol_rel, tol_abs)
        self.pairs = self._pairs_from_score(s, None)

    def fit_parameters(self):
        """Determine parameters for the affine transformation

        A linear fit is used to map x and y coordinates of `feat1` to to those
        of `feat2`. This requires `find_pairs` to be run first.
        """
        pars = []
        for p in self.pos_columns:
            r = scipy.stats.linregress(self.pairs[self.channel_names[0]][p],
                                       self.pairs[self.channel_names[1]][p])
            pars.append([r[i] for i in [0, 1, 4]])
        self.parameters1 = pd.DataFrame(pars,
                                        columns=["slope",
                                                 "intercept",
                                                 "stderr"],
                                        index=pos_columns)
        pars = []
        for p in self.pos_columns:
            r = scipy.stats.linregress(self.pairs[self.channel_names[1]][p],
                                       self.pairs[self.channel_names[0]][p])
            pars.append([r[i] for i in [0, 1, 4]])
        self.parameters2 = pd.DataFrame(pars,
                                        columns=["slope",
                                                 "intercept",
                                                 "stderr"],
                                        index=pos_columns)

    def __call__(self, data, channel=2, mode="constant", cval=0.0):
        """Do chromatic correction on `features` coordinates

        This modifies the coordinates in place.

        Args:
            data (pandas.DataFrame or ndarray): Either a DataFrame
                containing localization data or an ndarray with image data.
                Correction happens in place.
            channel (int, optional): If `features` are in the first channel
                (corresponding to the `feat1` arg of the constructor), set to
                1. If features are in the second channel, set to 2. Depending
                on this, a transformation will be applied to the coordinates of
                `features` to match the other channel (mathematically speaking
                depending on this parameter either the "original"
                transformation or its inverse are applied.)
            mode (str, optional): How to fill points outside of the uncorrected
                image boundaries. Possibilities are "constant", "nearest",
                "reflect" or "wrap". Defaults to "constant".
            cval (scalar, optional): What value to use for `mode="constant"`.
                Defaults to 0.0
        """
        x_col = self.pos_columns[0]
        y_col = self.pos_columns[1]
        if channel not in (1, 2):
            raise ValueError("channel has to be either 1 or 2")

        if isinstance(data, pd.DataFrame):
            if channel == 1:
                xparm = self.parameters1.loc[x_col]
                yparm = self.parameters1.loc[y_col]
            if channel == 2:
                xparm = self.parameters2.loc[x_col]
                yparm = self.parameters2.loc[y_col]
            data[x_col] = data[x_col]*xparm.slope + xparm.intercept
            data[y_col] = data[y_col]*yparm.slope + yparm.intercept
        elif isinstance(data, np.ndarray):
            if channel == 1:
                parms = self.parameters2
            if channel == 2:
                parms = self.parameters1
            #iloc[::-1] reverses the order since image coordinates and
            #matrix rows/columns have different order
            scipy.ndimage.affine_transform(data, parms.iloc[::-1]["slope"],
                                           parms.iloc[::-1]["intercept"],
                                           output=data, mode=mode, cval=cval)

    def test(self, ax=None):
        """Test validity of the correction parameters

        This plots the affine transformation functions and the coordinates of
        the pairs that were matched in the channels. If everything went well,
        the dots (i. e. pair coordinates) should lie on the line
        (transformation function).
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, len(pos_columns))

        for a, p in zip(ax, self.pos_columns):
            a.set_xlabel("{} ({})".format(p, self.channel_names[0]))
            a.set_ylabel("{} ({})".format(p, self.channel_names[1]))
            a.scatter(self.pairs[self.channel_names[0], p],
                       self.pairs[self.channel_names[1], p])
            a.plot(self.pairs[self.channel_names[0]].sort(p)[p],
                    self.parameters1.loc[p, "slope"] *
                    self.pairs[self.channel_names[0]].sort(p)[p] +
                    self.parameters1.loc[p, "intercept"])
            a.set_title(p)
            a.set_aspect(1)

        a.figure.tight_layout()

    def to_hdf(self, path_or_buf, key=("chromatic_corr_ch1",
                                       "chromatic_corr_ch2")):
        """Save parameters to a HDF5 file

        Save `parameters1` as `key`[0] and `parameters2` as `key`[1].

        Args:
            path_or_buf: HDF5 file to write to
            key (tuple of str, optional): Names of the parameter variables in
                the HDF5 file. Defaults to ("chromatic_corr_ch1",
                "chromatic_corr_ch2").
        """
        self.parameters1.to_hdf(path_or_buf, key[0], mode="w", format="t")
        self.parameters2.to_hdf(path_or_buf, key[1], mode="a", format="t")

    def read_hdf(path_or_buf, key=("chromatic_corr_ch1",
                                   "chromatic_corr_ch2")):
        """Read paramaters from a HDF5 file and construct a corrector

        Read `parameters1` from `key`[0] and `parameters2` from `key`[1].

        Args:
            path_or_buf: HDF5 file to load
            key (tuple of str, optional): Names of the parameter variables in
                the HDF5 file. Defaults to ("chromatic_corr_ch1",
                "chromatic_corr_ch2").

        Returns:
            A `Corrector` instance with the parameters read from the HDF5
            file.
        """
        corr = Corrector(None, None)
        corr.parameters1 = pd.read_hdf(path_or_buf, key[0])
        corr.pos_columns = corr.parameters1.index.tolist()
        corr.parameters2 = pd.read_hdf(path_or_buf, key[1])
        return corr

    def to_wrp(self, path):
        """Save parameters to .wrp file

        Warning: This only saves parameters2. In order not to lose parameters1,
        save to HDF5 using `to_hdf`().

        Args:
            path (str): Path of the .wrp file to be created
        """
        k1 = self.parameters2.loc[self.pos_columns[0], "slope"]
        d1 = self.parameters2.loc[self.pos_columns[0], "intercept"]
        k2 = self.parameters2.loc[self.pos_columns[1], "slope"]
        d2 = self.parameters2.loc[self.pos_columns[1], "intercept"]

        S = dict(k1=k1, d1=d1, k2=k2, d2=d2)
        scipy.io.savemat(path, dict(S=S), appendmat=False)

    def read_wrp(path):
        """Read parameters from a .wrp file

        Construct a Corrector with those parameters. Warning: The .wrp file
        only contains paramaters for the correction of channel 2. Parameters
        for channel 1 are calculated by inverting the transformation and may
        by not so accurate.

        Args:
            path (str): Path of the .wrp file

        Returns:
            A `Corrector` instance with the parameters read from the .wrp
            file.
        """
        corr = Corrector(None, None)
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        d = mat["S"]

        #data of the wrp file is for the channel1 transformation
        parms1 = np.empty((2, 3))
        parms1[0, :] = np.array([d.k1, d.d1, np.NaN])
        parms1[1, :] = np.array([d.k2, d.d2, np.NaN])
        corr.parameters2 = pd.DataFrame(parms1,
                                        columns=["slope", "intercept",
                                                 "stderr"],
                                        index=pos_columns)
        parms2 = np.empty((2, 3))
        parms2[0, :] = np.array([1./d.k1, -d.d1/d.k1, np.NaN])
        parms2[1, :] = np.array([1./d.k2, -d.d2/d.k2, np.NaN])
        corr.parameters1 = pd.DataFrame(parms2,
                                        columns=["slope", "intercept",
                                                 "stderr"],
                                        index=pos_columns)
        return corr

    def _vectors_cartesian(self, features):
        """Calculate vectors of each point in features to every other point

        Parameters
        ----------
        features : pandas.DataFrame
            Features with coordinates in columns `pos_columns`

        Returns
        -------
        vecs : numpy.ndarray
            The first axis specifies the coordinate component of the vectors,
            e. g. ``vecs[0]`` gives the x components. For each ``i``,
            ``vecs[i, j, k]`` yields the i-th component of the vector pointing
            from the j-th feature in `features` to the k-th feature.
        """
        # transpose so that data[1] gives x coordinates, data[2] y coordinates
        data = features[self.pos_columns].as_matrix().T
        # for each coordinate (the first ':'), calculate the differences
        # between all entries (thus 'np.newaxis, :' and ':, np.newaxis')
        return data[:, np.newaxis, :] - data[:, :, np.newaxis]

    def _all_scores_cartesian(self, vec1, vec2, tol_rel, tol_abs):
        """Determine scores for all possible pairs of features

        Compare all elements of `vec1` and all of `vec2` using
        `numpy.isclose()` to determine which pairs of features have the most
        simalar vectors pointing to the other features.

        Parameters
        ----------
        vec1 : numpy.ndarray
            Vectors for the first set of features as determined by
            `_vectors_cartesian`.
        vec2 : numpy.array
            Vectors for the second set of features as determined by
            `_vectors_cartesian`.
        tol_rel : float
            Relative tolerance parameter for `numpy.isclose`
        tol_abs : float
            Absolute tolerance parameter for `numpy.isclose`

        Returns
        -------
        numpy.array
            A matrix of scores. scores[i, j] holds the number of matching
            vectors for the i-th entry of feat1 and the j-th entry of feat2.
        """
        # vectors versions for broadcasting
        # The first index specifies the coordinate (x, y, z, …)
        # The np.newaxis are shifted for vec1 and vec2 so that calculating
        # vec1_b - vec2_b (as is done in np.isclose) results in a 5D array of
        # the differences of all vectors for all coordinates.
        vec1_b = vec1[:, np.newaxis, :, np.newaxis, :]
        vec2_b = vec2[:, :, np.newaxis, :, np.newaxis]

        # For each coordinate k the (i, j)-th 2D matrix
        # (vec1_b - vec2_b)[k, i, j] contains in the m-th row and n-th column
        # the difference between the vector from feature i to feature j in
        # feat1 and the vector from feature m to feature n in feat2
        # diff_small contains True or False depending on whether the vectors
        # are similar enough
        diff_small = np.isclose(vec1_b, vec2_b, tol_rel, tol_abs)

        # All coordinates have to be similar; this is like a logical AND along
        # only axis 0, the coordinate axis.
        all_small = np.all(diff_small, axis=0)

        # Sum up all True entries as the score. The more similar vectors two
        # points have, the more likely it is that they are the same.
        return np.sum(all_small, axis=(0, 1)).T

    def _pairs_from_score(self, score, score_cutoff=None,
                          ambiguity_factor=0.8):
        """Analyze the score matrix and determine what the pairs are

        For each feature, select the highest scoring corresponding feature.

        Parameters
        ----------
        score : numpy.array
            The score matrix as calculated by `_all_scores_cartesian`
        score_cutoff : float or None, optional
            In order to get rid of false matches a threshold for scores can
            be set. All scores below this threshold are discarded. If set to
            None, 0.5*score.max() will be used. Defaults to None.
        ambiguity_factor : float
            If there are two candidates as a partner for a feature, and the
            score of the lower scoring one is larger than `ambiguity_factor`
            times the score of the higher scorer, the pairs are considered
            ambiguous and therefore discarded. Defaults to 0.8.

        Returns
        -------
        pandas.DataFrame
            Each row of the DataFrame contains information of one feature pair.
            This information consists of lists of coordinates of the feature
            in each channel, i. e. the columns are a MultiIndex from the
            product of the `channel_names` and `pos_columns` class attributes.
        """
        if score_cutoff is None:
            score_cutoff = 0.5*score.max()

        # always search through the axis with fewer localizations since
        # since otherwise there is bound to be double matches
        if score.shape[0] > score.shape[1]:
            indices = np.array([np.argmax(score, axis=0),
                                np.arange(score.shape[1])])
            long_axis = 0
        else:
            indices = np.array([np.arange(score.shape[0]),
                                np.argmax(score, axis=1)])
            long_axis = 1

        # anything below score_cutoff is considered noise
        score[score < score_cutoff] = 0

        # now deal with ambiguities, i. e. if one feature has two similarly
        # likely partners
        # For each feature, calculate the threshold.
        amb_thresh = score[indices.tolist()]*ambiguity_factor
        # If there is more than one candidate's score above the threshold,
        # discard
        amb = score > np.expand_dims(amb_thresh, axis=long_axis)
        amb = np.sum(amb, axis=long_axis) > 1
        indices = indices[:, ~amb]

        pair_matrix = np.hstack((
            self.feat1.iloc[indices[0]][pos_columns],
            self.feat2.iloc[indices[1]][pos_columns]))

        mi = pd.MultiIndex.from_product([self.channel_names, self.pos_columns])
        return pd.DataFrame(pair_matrix, columns=mi)
