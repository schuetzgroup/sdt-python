"""Overlay color channels
======================

Multi-color microscopy images are typically created by directing the light of
different wavelengths onto different regions of the camera chip or onto
different cameras.

In order to overlay the different channels, one can record images of
multi-color beads and identify each bead's position in each channel to
determine a transformation that transforms data from one channel to match the
other channel. The :py:class:`Corrector` class offers an easy way to do this.


Examples
--------

Let's assume that multiple images/sequences of flourescent beads have been
acquired with two color channels on in the same image. First, the beads
need to be localized (e.g. using the ``sdt.gui.locator`` application).
These need to be loaded:

>>> bead_loc = [sdt.io.load(b) for b in glob.glob("beads*.h5")]

Next, we define ROIs for the two channels using :py:class:`sdt.roi.ROI` and
choose the bead localizations with respect to the ROIs:

>>> r1 = sdt.roi.ROI((0, 0), size=(200, 100))
>>> r2 = sdt.roi.ROI((0, 200), size=(200, 100))
>>> beads_r1 = [r1(b) for b in bead_loc]
>>> beads_r2 = [r2(b) for b in bead_loc]

Now, calculate the transform that overlays the two channels using
:py:class:`Corrector`:

>>> corr = Corrector(beads_r1, beads_r2)
>>> corr.determine_parameters()
>>> corr.test()  # Plot results

This can now be used to transform i.e. image data from channel 1 so that it
can be overlaid with channel 1:

>>> img = pims.open("image.tif")[0]  # Load first frame (number 0)
>>> img_r1 = r1(img)  # Get channel 1 part of the image
>>> img_r2 = r2(img)  # Get channel 2 part of the image
>>> img_r1_corr = corr(img_r1, channel=1)  # Transform channel 1 image
>>> overlay = numpy.dstack([img_r1, img_r2, np.zeros_like(img_r1)])
>>> matplotlib.pyplot.imshow(overlay)  # Plot overlay

Similarly, coordinates of single molecule data can be transformed:

>>> loc = sdt.io.load("loc.h5")  # Load data
>>> loc_r1 = r1(loc)  # Get data from channel 1
>>> loc_r2 = r2(loc)  # Get data from channel 2
>>> loc_r1_corr = corr(loc_r1, channel=1)  # Transform channel 1 data
>>> matplotlib.pyplot.scatter(loc_r1_corr["x"], loc_r1_corr["y"], marker="+")
>>> matplotlib.pyplot.scatter(loc_r2["x"], loc_r2["y"], marker="x")

There is also support for saving and loading a :py:class:`Corrector` instance
to/from YAML:

>>> with open("output.yaml", "w") as f:
>>>     sdt.io.yaml.safe_dump(corr, f)
>>> with opon("output.yaml", "r") as f:
>>>     corr_loaded = sdt.io.yaml.safe_load(f)


Programming reference
---------------------

.. autoclass:: Corrector
    :members:
    :special-members: __call__
"""
from contextlib import suppress

import numpy as np
import pandas as pd
import scipy.io
import scipy.stats
import scipy.ndimage
import matplotlib as mpl

import slicerator

from . import roi, config


def _affine_trafo(params, loc):
    """Do an affine transformation

    Parameters
    ----------
    params : numpy.ndarray, shape(n, n+1) or shape(n+1, n+1)
        Transformation matrix. The top-left (n, n) block is used as the
        linear part of the transformation while the top-right column of n
        entries specifies the shift.
    loc : numpy.ndarray, shape(m, n)
        Array of m n-dimensional coordinate tuples to be transformed.

    Returns
    -------
    numpy.ndarray, shape(m, n)
        Transformed coordinate tuples
    """
    ndim = params.shape[1] - 1
    return loc @ params[:ndim, :ndim].T + params[:ndim, ndim]


class Corrector(object):
    """Class for easy overlay of two fluorescence microscopy channels

    This class provides an easy-to-use interface to the correction
    process.
    """
    yaml_tag = "!ChromaticCorrector"

    @config.set_columns
    def __init__(self, feat1=None, feat2=None, columns={},
                 channel_names=["channel1", "channel2"]):
        """Parameters
        ----------
        feat1, feat2 : list of pandas.DataFrame or pandas.DataFrame or None, optional
            Set the `feat1` and `feat2` attribute (turning it into a list
            if it is a single DataFrame). Can also be `None`, but in this case
            :py:meth:`find_pairs` and :py:meth:`determine_parameters` will
            not work. Defaults to `None`.

        Other parameters
        ----------------
        columns : dict, optional
            Override default column names as defined in
            :py:attr:`config.columns`. The only relevant name is `pos`.
            That means, if the DataFrames have coordinate columns "x" and "z",
            set ``columns={"pos": ["x", "z"]}``.
        channel_names : list of str, optional
            Set the `channel_names` attribute. Defaults to ``["channel1",
            "channel2"]``.
        """
        self.feat1 = [feat1] if isinstance(feat1, pd.DataFrame) else feat1
        """List of pandas.DataFrame; Positions of beads (as found by a
        localization algorithm) in the first channel. Each DataFrame
        corresponds to one image (sequence), thus multiple bead images can be
        used to increase the accuracy.
        """
        self.feat2 = [feat2] if isinstance(feat2, pd.DataFrame) else feat2
        """Same as :py:attr:`feat1`, but for the second channel"""
        self.pos_columns = columns["pos"]
        """List of names of the columns describing the coordinates of the
        features.
        """
        self.channel_names = channel_names
        """List of channel names"""
        self.pairs = None
        """:py:class:`pandas.DataFrame` containing the pairs found by
        :py:meth:`determine_parameters`.
        """
        self.parameters1 = np.eye(len(self.pos_columns) + 1)
        """Array describing the affine transformation of data from
        channel 1 to channel 2.
        """
        self.parameters2 = np.eye(len(self.pos_columns) + 1)
        """Array describing the affine transformation of data from
        channel 2 to channel 1.
        """

    def determine_parameters(self, tol_rel=0.05, tol_abs=0., score_cutoff=0.6,
                             ambiguity_factor=0.8):
        """Determine the parameters for the affine transformation

        This takes the localizations of :py:attr:`feat1` and tries to match
        them with those of :py:attr:`feat2`. Then a linear fit is used to
        determine the affine transformation to correct for chromatic
        aberrations.

        This is a convenience function that calls :py:meth:`find_pairs` and
        :py:meth:`fit_parameters`; see the documentation of the methods for
        details.

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

    def __call__(self, data, channel=2, inplace=False, mode="constant",
                 cval=0.0, columns={}):
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
        cval : scalar or callable, optional
            What value to use for `mode="constant"`. If this is callable, it
            should take a single argument (the uncorrected image) and return a
            scalar, which will be used as the fill value. Defaults to 0.0.

        Other parameters
        ----------------
        columns : dict, optional
            Override default column names in case `data` is a
            :py:class:`pandas.DataFrame`. The only relevant name is `pos`.
            That means, if the DataFrame has coordinate columns "x" and "z",
            set ``columns={"pos": ["x", "z"]}``.
        """
        if channel not in (1, 2):
            raise ValueError("channel has to be either 1 or 2")

        pos_columns = columns.get("pos", self.pos_columns)

        if isinstance(data, pd.DataFrame):
            if not inplace:
                data = data.copy()
            loc = data[pos_columns].values
            par = getattr(self, "parameters{}".format(channel))
            data[pos_columns] = _affine_trafo(par, loc)

            if not inplace:
                # copied previously, now return
                return data
        elif isinstance(data, roi.PathROI):
            t = mpl.transforms.Affine2D(
                getattr(self, "parameters{}".format(channel)))
            return roi.PathROI(t.transform_path(data.path),
                               buffer=data.buffer,
                               no_image=(data.image_mask is None))
        else:
            parms = getattr(
                self, "parameters{}".format(2 if channel == 1 else 1))

            @slicerator.pipeline
            def corr(img):
                # transpose image since matrix axes are reversed compared to
                # coordinate axes
                img_t = img.T

                if callable(cval):
                    cv = cval(img)
                else:
                    cv = cval

                # this way, the original subclass of np.ndarray is preserved
                ret = np.empty_like(img_t)
                scipy.ndimage.affine_transform(
                    img_t, parms[:-1, :-1], parms[:-1, -1], output=ret,
                    mode=mode, cval=cv)

                return ret.T  # transpose back
            return corr(data)

    def test(self, ax=None):
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

        ax[0].hist(diff1, bins=20)
        ax[0].set_title("Error")
        ax[0].set_xlabel("distance")
        ax[0].set_ylabel("# data points")

        ax[1].scatter(c1_corr[self.pos_columns[0]],
                      c1_corr[self.pos_columns[1]], marker="x", color="blue")
        ax[1].scatter(c2[self.pos_columns[0]],
                      c2[self.pos_columns[1]], marker="+", color="red")
        ax[1].set_aspect(1, adjustable="datalim")
        ax[1].set_title("Overlay")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")

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
             columns={}):
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
        sdt.chromatic.Corrector
            A :py:class:`Corrector` instance with the parameters read from the
            file.

        Other parameters
        ----------------
        columns : dict, optional
            Override default column names as defined in
            :py:attr:`config.columns`. The only relevant name is `pos`.
            That means, if the DataFrames have coordinate columns "x" and "z",
            set ``columns={"pos": ["x", "z"]}``.
        key : tuple of str, optional
            Name of the variables in the saved file (does not apply to "wrp").
            Defaults to ("chromatic_param1", "chromatic_param2").
        """
        corr = Corrector(None, None, columns=columns)
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

    def find_pairs(self, tol_rel=0.05, tol_abs=0., score_cutoff=0.6,
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
            if "frame" in f1.columns and "frame" in f2.columns:
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
        to to those of `feat2`. This requires :py:attr:`pairs` to be set, e.g.
        by running :py:meth:`find_pairs`.
        The result is saved as :py:attr:`parameters1` (transform of channel 1
        coordinates to channel 2) and :py:attr:`parameters2` (transform of
        channel 2  coordinates to channel 1) attributes.

        :py:attr:`parameters1` is calculated by determining the affine
        transformation between the :py:attr:`pairs` entries using a linear
        least squares fit. :py:attr:`parameters2` is its inverse. Therefore,
        results may be slightly different depending on what is channel1 and
        what is channel2.
        """
        ch1_name = self.channel_names[0]
        ch2_name = self.channel_names[1]

        loc1 = self.pairs[ch1_name][self.pos_columns].values
        loc2 = self.pairs[ch2_name][self.pos_columns].values
        ndim = loc1.shape[1]

        self.parameters1 = np.empty((ndim + 1,) * 2)
        loc1_embedded = np.hstack([loc1, np.ones((len(loc1), 1))])
        self.parameters1[:ndim, :] = np.linalg.lstsq(loc1_embedded, loc2)[0].T
        self.parameters1[ndim, :ndim] = 0.
        self.parameters1[ndim, ndim] = 1.

        self.parameters2 = np.linalg.inv(self.parameters1)

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
        data = features[self.pos_columns].values.T
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
            feat1.iloc[indices[0]][self.pos_columns],
            feat2.iloc[indices[1]][self.pos_columns]))

        mi = pd.MultiIndex.from_product([self.channel_names, self.pos_columns])
        return pd.DataFrame(pair_matrix, columns=mi)

    @classmethod
    def to_yaml(cls, dumper, data):
        """Dump as YAML

        Pass this as the `representer` parameter to your
        :py:class:`yaml.Dumper` subclass's `add_representer` method.
        """
        m = (("parameters1", data.parameters1),
             ("parameters2", data.parameters2),
             ("pos_columns", data.pos_columns))
        return dumper.represent_mapping(cls.yaml_tag, m)

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct from YAML

        Pass this as the `constructor` parameter to your
        :py:class:`yaml.Loader` subclass's `add_constructor` method.
        """
        m = loader.construct_mapping(node)
        cols = {} if "pos_columns" not in m else {"pos": m["pos_columns"]}
        ret = cls(columns=cols)
        ret.parameters1 = m["parameters1"]
        ret.parameters2 = m["parameters2"]
        return ret


with suppress(ImportError):
    from .io import yaml
    yaml.register_yaml_class(Corrector)
