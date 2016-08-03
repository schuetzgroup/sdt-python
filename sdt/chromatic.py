"""This module allows for correction of chromatic aberrations.

When doing multi-color single molecule microscopy, chromatic aberrations pose
a problem. To circumvent this, one can record an image of beads, match the
localizations (as determined by a localization algorithm such as those in the
:py:mod:`sdt.loc` package) of the two channels and determine affine
transformations for the x and y coordinates.
"""

import numpy as np
import pandas as pd
import scipy.io
import scipy.stats
import scipy.ndimage

import slicerator

pos_columns = ["x", "y"]
"""Names of the columns describing the coordinates of the features in
    pandas.DataFrames.
"""
channel_names = ["channel1", "channel2"]
"""Names of the two channels."""


class Corrector(object):
    """Convenience class for easy correction of chromatic aberrations

    This class provides an easy-to-use interface to the correction
    process.

    Attributes
    ----------
    pos_columns : list of str
        Names of the columns describing the coordinates of the features.
    channel_names : list of str
        Names of the channels
    feat1, feat2 : list of pandas.DataFrames
        Bead positions of the first and second channel found by the
        localization algorithm. The x coordinate is in the column with name
        ``pos_columns[0]``, etc. Each DataFrame corresponds to one image
        (sequence), thus multiple bead images can be used to increase
        the accuracy.
    pairs : pandas.DataFrame
        The pairs found by `determine_parameters`.
    parameters1, parameters2 : pandas.DataFrame
        The parameters for the affine transformation to correct coordinates
        of channel 1 and channel 2 respectively, embedded in a vector space of
        higher dimension.
    """
    def __init__(self, feat1, feat2, pos_columns=pos_columns,
                 channel_names=channel_names):
        """Parameters
        ----------
        feat1, feat2 : list of pandas.DataFrame or pandas.DataFrame
            Set the `feat1` and `feat2` attribute (turning it into a list
            if it is a single DataFrame)
        pos_columns : list of str, optional
            Set the `pos_columns` attribute.
        channel_names : list of str, optional
            Set the `channel_names` attribute.
        """
        self.feat1 = [feat1] if isinstance(feat1, pd.DataFrame) else feat1
        self.feat2 = [feat2] if isinstance(feat2, pd.DataFrame) else feat2
        self.pos_columns = pos_columns
        self.channel_names = channel_names
        self.pairs = None
        self.parameters1 = np.eye(3)
        self.parameters2 = np.eye(3)

    def determine_parameters(self, tol_rel=0.05, tol_abs=0., score_cutoff=0.5,
                             ambiguity_factor=0.8):
        """Determine the parameters for the affine transformation

        This takes the localizations of :py:attr:`feat1` and tries to match
        them with those of :py:attr:`feat2`. Then a linear fit is used to
        determine the affine transformation to correct for chromatic
        aberrations.

        This is a convenience function that calls :py:meth:`find_pairs` and
        :py:meth:`fit_parameters`.

        Parameters
        ----------
        tol_rel : float
            Relative tolerance parameter for :py:func:`numpy.isclose()`.
            Defaults to 0.05.
        tol_abs : float
            Absolute tolerance parameter for :py:func:`numpy.isclose()`.
            Defaults to 0.
        score_cutoff : float, optional
            In order to get rid of false matches a threshold for scores can
            be set. All scores below this threshold are discarded. The
            threshold is calculated as ``score_cutoff*score.max()``. Defaults
            to 0.5.
        ambiguity_factor : float
            If there are two candidates as a partner for a feature, and the
            score of the lower scoring one is larger than `ambiguity_factor`
            times the score of the higher scorer, the pairs are considered
            ambiguous and therefore discarded. Defaults to 0.8.
        """
        self.find_pairs(tol_rel, tol_abs, score_cutoff, ambiguity_factor)
        self.fit_parameters()

    def find_pairs(self, tol_rel=0.05, tol_abs=0., score_cutoff=0.5,
                   ambiguity_factor=0.8):
        """Match features of `feat1` with features of `feat2`

        This is done by calculating the vectors from every
        feature all the other features in :py:attr:`feat1` and compare them to
        those of :py:attr:`feat2` using the :py:func:`numpy.isclose` function
        on both the x and the y
        coordinate, where `tol_rel` and `tol_abs` are the relative and
        absolute tolerance parameters.

        Parameters
        ----------
        tol_rel : float
            Relative tolerance parameter for `numpy.isclose()`. Defaults to
            0.05.
        tol_abs : float
            Absolute tolerance parameter for `numpy.isclose()`. Defaults to 0.
        score_cutoff : float, optional
            In order to get rid of false matches a threshold for scores can
            be set. All scores below this threshold are discarded. The
            threshold is calculated as score_cutoff * score.max(). Defaults to
            0.5.
        ambiguity_factor : float
            If there are two candidates as a partner for a feature, and the
            score of the lower scoring one is larger than `ambiguity_factor`
            times the score of the higher scorer, the pairs are considered
            ambiguous and therefore discarded. Defaults to 0.8.
        """
        p = []
        for f1, f2 in zip(self.feat1, self.feat2):
            if ("frame" in f1.columns and
                    "frame" in f2.columns):
                # If there is a "frame" column, split data according to
                # frame number since pairs can only be in the same frame
                data = ((f1[f1["frame"] == i], f2[f2["frame"] == i])
                        for i in f1["frame"].unique())
            else:
                # If there is no "frame" column, just take everything
                data = ((f1, f2),)
            for f1_frame_data, f2_frame_data in data:
                if f1_frame_data.empty or f2_frame_data.empty:
                    # Don't even try to operate on empty frames. Weird things
                    # are bound to happen
                    continue
                v1 = self._vectors_cartesian(f1_frame_data)
                v2 = self._vectors_cartesian(f2_frame_data)
                s = self._all_scores_cartesian(v1, v2, tol_rel, tol_abs)
                p.append(self._pairs_from_score(
                    f1_frame_data, f2_frame_data, s, score_cutoff,
                    ambiguity_factor))
        self.pairs = pd.concat(p)

    def fit_parameters(self):
        """Determine parameters for the affine transformation

        An affine transformation is used to map x and y coordinates of `feat1`
        to to those of `feat2`. This requires :py:meth:`find_pairs` to be run
        first or the :py:attr:`pairs` attribute to be set manually.
        The result is saved as :py:attr:`parameters1` (transform of channel 1
        coordinates to channel 2) and :py:attr:`parameters2` (transform of
        channel 2  coordinates to channel 1) attributes. In an ideal world,
        these would be inverse, but the world is hardly ever ideal.

        The transformations are calculated by a linear least squares fit
        of the embedding of the affine space into a higher dimensional vector
        space.
        """
        one_padding = np.ones((len(self.pairs), 1))
        ch1_name = self.channel_names[0]
        ch2_name = self.channel_names[1]
        affine_embedding_footer = np.zeros((1, len(self.pos_columns) + 1))
        affine_embedding_footer[-1, -1] = 1  # last column is translation

        # first transform
        coeff = np.hstack((self.pairs[ch1_name][pos_columns], one_padding))
        rhs = self.pairs[ch2_name][pos_columns].as_matrix()
        params = np.linalg.lstsq(coeff, rhs)[0].T
        self.parameters1 = np.vstack((params, affine_embedding_footer))

        # second transform
        coeff = np.hstack((self.pairs[ch2_name][pos_columns], one_padding))
        rhs = self.pairs[ch1_name][pos_columns].as_matrix()
        params = np.linalg.lstsq(coeff, rhs)[0].T
        self.parameters2 = np.vstack((params, affine_embedding_footer))

    def __call__(self, data, channel=2, inplace=False, mode="constant",
                 cval=0.0):
        """Correct for chromatic aberrations

        This can be done either on coordinates (e. g. resulting from feature
        localization) or directly on raw images.

        Parameters
        ----------
        data : pandas.DataFrame or slicerator.Slicerator or array-like
            data to be processed. If a pandas.Dataframe, the feature
            coordinates will be corrected. Otherwise, `slicerator.pipeline` is
            used to correct image data using an affine transformation (if
            available)
        channel : int, optional
            If `features` are in the first channel (corresponding to the
            `feat1` arg of the constructor), set to 1. If features are in the
            second channel, set to 2. Depending on this, a transformation will
            be applied to the coordinates of `features` to match the other
            channel (mathematically speaking depending on this parameter
            either the "original" transformation or its inverse are applied.)
        inplace : bool, optional
            Only has an effect if `data` is a DataFrame. If True, the
            feature coordinates will be corrected in place. Defaults to False.
        mode : str, optional
            How to fill points outside of the uncorrected image boundaries.
            Possibilities are "constant", "nearest", "reflect" or "wrap".
            Defaults to "constant".
        cval : scalar, optional
            What value to use for `mode="constant"`. Defaults to 0.0
        """
        if channel not in (1, 2):
            raise ValueError("channel has to be either 1 or 2")

        if isinstance(data, pd.DataFrame):
            if not inplace:
                data = data.copy()
            if channel == 1:
                corr_coords = np.dot(data[self.pos_columns].as_matrix(),
                                     self.parameters1[:-1, :-1].T)
                corr_coords += self.parameters1[:-1, -1]
            if channel == 2:
                corr_coords = np.dot(data[self.pos_columns].as_matrix(),
                                     self.parameters2[:-1, :-1].T)
                corr_coords += self.parameters2[:-1, -1]

            data[pos_columns] = corr_coords

            if not inplace:
                # copied previously, now return
                return data
        else:
            if channel == 1:
                parms = np.linalg.inv(self.parameters1)
            if channel == 2:
                parms = np.linalg.inv(self.parameters2)

            @slicerator.pipeline
            def corr(img):
                # transpose image since matrix axes are reversed compared to
                # coordinate axes
                ret = scipy.ndimage.affine_transform(
                    img.T, parms[:-1, :-1], parms[:-1, -1],
                    mode=mode, cval=cval)
                return ret.T  # transpose back
            return corr(data)

    def test(self, ax=None, safe_labels=False):
        """Test validity of the correction parameters

        This plots the affine transformation functions and the coordinates of
        the pairs that were matched in the channels. If everything went well,
        the dots (i. e. pair coordinates) should lie on the line
        (transformation function).

        Parameters
        ----------
        ax : tuple of matplotlib axes or None, optional
            Axes to use for plotting. The length of the tuple has to be 2.
            If None, allocate new axes using
            :py:func:`matplotlib.pyplot.subplots`. Defaults to None.
        safe_labels : bool, optional
            If True, do not use math mode for labels since that can cause
            crashes in GUI applications (at least with Qt4). Defaults to False.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            grid_kw = dict(width_ratios=(1, 2))
            fig, ax = plt.subplots(1, 2, gridspec_kw=grid_kw)

        c1 = self.pairs[self.channel_names[0]]
        c1_corr = self(c1, channel=1)
        diff1 = (c1_corr[self.pos_columns] -
                 self.pairs[self.channel_names[1]][self.pos_columns])
        diff1 = np.sqrt(np.sum(diff1**2, axis=1))

        c2 = self.pairs[self.channel_names[1]]
        c2_corr = self(c2, channel=2)
        diff2 = (c2_corr[self.pos_columns] -
                 self.pairs[self.channel_names[0]][self.pos_columns])
        diff2 = np.sqrt(np.sum(diff2**2, axis=1))

        ax[0].scatter(np.zeros(len(diff1)), diff1, marker="x", color="blue")
        ax[0].scatter(np.ones(len(diff2)), diff2, marker="+", color="green")
        ax[0].set_xticks([0, 1])
        if not safe_labels:
            ax[0].set_xticklabels([r"$1\rightarrow 2$", r"$2\rightarrow 1$"])
        else:
            ax[0].set_xticklabels(["1 > 2", "2 > 1"])
        ax[0].set_xlim(-0.5, 1.5)
        ax[0].set_ylim(0)
        ax[0].set_title("Error")

        ax[1].scatter(c1_corr[self.pos_columns[0]],
                      c1_corr[self.pos_columns[1]], marker="x", color="blue")
        ax[1].scatter(c2[self.pos_columns[0]],
                      c2[self.pos_columns[1]], marker="+", color="red")
        ax[1].set_aspect(1, adjustable="datalim")
        ax[1].set_title("Overlay")

        ax[0].figure.tight_layout()

    def save(self, file, fmt="npz", key=("chromatic_param1",
                                         "chromatic_param2")):
        """Save transformation parameters to file

        Parameters
        ----------
        file : str or file
            File name or an open file-like object to save data to.
        fmt : {"npz", "mat"}, optional
            Format to save data. Either numpy ("npz") or MATLAB ("mat").
            Defaults to "npz".

        Other parameters
        ----------------
        key : tuple of str, optional
            Name of the variables in the saved file. Defaults to
            ("chromatic_param1", "chromatic_param2").
        """
        vardict = {key[0]: self.parameters1, key[1]: self.parameters2}
        if fmt == "npz":
            np.savez(file, **vardict)
        elif fmt == "mat":
            scipy.io.savemat(file, vardict)
        else:
            raise ValueError("Unknown format: {}".format(fmt))

    @staticmethod
    def load(file, fmt="npz", key=("chromatic_param1", "chromatic_param2"),
             pos_columns=pos_columns):
        """Read paramaters from a file and construct a `Corrector`

        Parameters
        ----------
        file : str or file
            File name or an open file-like object to load data from.
        fmt : {"npz", "mat", "wrp"}, optional
            Format to save data. Either numpy ("npz") or MATLAB ("mat") or
            `determine_shiftstretch`'s wrp ("wrp"). Defaults to "npz".

        Returns
        -------
        Corrector
            A :py:class:`Corrector` instance with the parameters read from the
            file.

        Other parameters
        ----------------
        key : tuple of str, optional
            Name of the variables in the saved file (does not apply to "wrp").
            Defaults to ("chromatic_param1", "chromatic_param2").
        pos_columns : list of str, optional
            Sets the `pos_columns` attribute. Defaults to the `pos_columns`
            attribute of the module.
        """
        corr = Corrector(None, None, pos_columns=pos_columns)
        if fmt == "npz":
            npz = np.load(file)
            corr.parameters1 = npz[key[0]]
            corr.parameters2 = npz[key[1]]
        elif fmt == "mat":
            mat = scipy.io.loadmat(file)
            corr.parameters1 = mat[key[0]]
            corr.parameters2 = mat[key[1]]
        elif fmt == "wrp":
            d = scipy.io.loadmat(file, squeeze_me=True,
                                 struct_as_record=False)["S"]
            corr.parameters2 = np.array([[d.k1, 0., d.d1],
                                         [0., d.k2, d.d2],
                                         [0., 0., 1.]])
            corr.parameters1 = np.linalg.inv(corr.parameters2)

        else:
            raise ValueError("Unknown format: {}".format(fmt))

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

    def _pairs_from_score(self, feat1, feat2, score, score_cutoff=0.5,
                          ambiguity_factor=0.8):
        """Analyze the score matrix and determine what the pairs are

        For each feature, select the highest scoring corresponding feature.

        Parameters
        ----------
        feat1, feat2 : pandas.DataFrame
            Bead localizations
        score : numpy.array
            The score matrix as calculated by `_all_scores_cartesian`
        score_cutoff : float, optional
            In order to get rid of false matches a threshold for scores can
            be set. All scores below this threshold are discarded. The
            threshold is calculated as score_cutoff*score.max(). Defaults to
            0.5.
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
        # always search through the axis with fewer localizations since
        # since otherwise there is bound to be double matches
        if score.shape[0] > score.shape[1]:
            indices = np.array([np.argmax(score, axis=0),
                                np.arange(score.shape[1])])
        else:
            indices = np.array([np.arange(score.shape[0]),
                                np.argmax(score, axis=1)])

        # now deal with ambiguities, i. e. if one feature has two similarly
        # likely partners
        # For each feature, calculate the threshold.
        amb_thresh = score[indices.tolist()] * ambiguity_factor
        # look for high scorers along the 0 axis (rows)
        amb0 = (score[indices[0], :] > amb_thresh[:, np.newaxis])
        amb0 = (np.sum(amb0, axis=1) > 1)
        # look for high scorers along the 1 axis (columns)
        amb1 = (score[:, indices[1]] > amb_thresh[np.newaxis, :])
        amb1 = (np.sum(amb1, axis=0) > 1)
        # drop if there are more than one high scorers in any axis
        indices = indices[:, ~(amb0 | amb1)]

        # get rid of indices with low scores
        # TODO: Can this be gotten rid of now that the ambiguity stuff works?
        score_cutoff *= score.max()
        cutoff_mask = (score[indices.tolist()] >= score_cutoff)
        indices = indices[:, cutoff_mask]

        pair_matrix = np.hstack((
            feat1.iloc[indices[0]][pos_columns],
            feat2.iloc[indices[1]][pos_columns]))

        mi = pd.MultiIndex.from_product([self.channel_names, self.pos_columns])
        return pd.DataFrame(pair_matrix, columns=mi)
