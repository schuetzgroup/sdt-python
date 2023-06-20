# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import suppress
from pathlib import Path
from typing import (BinaryIO, Callable, Dict, Literal, Mapping, Optional,
                    Sequence, Tuple, TypeVar, Union, overload)

import cv2
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.io
import scipy.ndimage
import scipy.spatial

from .. import config, helper, roi


def _affine_trafo(params: np.ndarray, loc: np.ndarray) -> np.ndarray:
    """Perform an affine transformation

    Parameters
    ----------
    params
        Transformation matrix of shape (n, n+1) or (n+1, n+1). The top-left
        (n, n) block is used as the linear part of the transformation while the
        top-right column of n entries specifies the shift.
    loc
        Array of m n-dimensional coordinate tuples to be transformed.

    Returns
    -------
    Transformed coordinate tuples
    """
    ndim = params.shape[1] - 1
    return loc @ params[:ndim, :ndim].T + params[:ndim, ndim]


class Registrator:
    """Registration of two fluorescence microscopy emission channels

    This class provides an easy-to-use interface to determine maps between
    the channels' coordinates using localization data from fiducial markers. It
    is based on the algorithm published by Preibisch et al. [Preibisch2010]_.

    Examples
    --------
    Let's assume that multiple images/sequences of fluorescent beads have been
    acquired, which are visible in both emission channels. First, the beads
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
    :py:class:`Registrator`:

    >>> corr = Registrator(beads_r1, beads_r2)
    >>> corr.determine_parameters()
    >>> corr.test()  # Plot results

    This can now be used to transform i.e. image data from channel 1 so that it
    can be overlaid with channel 1:

    >>> with io.ImageSequence("image.tif") as s:
    ...     img = s[0]  # Load first frame (number 0)
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
    >>> matplotlib.pyplot.scatter(loc_r1_corr["x"], loc_r1_corr["y"],
    ...                           marker="+")
    >>> matplotlib.pyplot.scatter(loc_r2["x"], loc_r2["y"], marker="x")

    There is also support for saving and loading a :py:class:`Registrator`
    instance to/from YAML:

    >>> with open("output.yaml", "w") as f:
    >>>     sdt.io.yaml.safe_dump(corr, f)
    >>> with open("output.yaml", "r") as f:
    >>>     corr_loaded = sdt.io.yaml.safe_load(f)

    References
    ----------
    .. [Preibisch2010] Preibisch, S.; Saalfeld, S.; Schindelin, J. & Tomancak,
        P.: "Software for bead-based registration of selective plane
        illumination microscopy data", Nature Methods, Springer Science and
        Business Media LLC, 2010, 7, 418–419
    """
    yaml_tag = "!Registrator"

    feat1: Sequence[pd.DataFrame]
    """Positions of beads (as found by a localization algorithm) in the first
    channel. Each DataFrame corresponds to one image (sequence), thus multiple
    bead images can be used to increase the accuracy.
    """
    feat2: Sequence[pd.DataFrame]
    """Same as :py:attr:`feat1`, but for the second channel"""
    columns: Dict
    """Column names in :py:attr:`feat1` and :py:attr:`feat2`. Defaults are
    taken from :py:attr:`config.columns`.
    """
    channel_names: Sequence[str]
    """Channel names"""
    pairs: pd.DataFrame
    """Pairs found by :py:meth:`determine_parameters`."""
    parameters1: np.ndarray
    """Array describing the affine transformation of data from channel 1 to
    channel 2.
    """
    parameters2: np.ndarray
    """Array describing the affine transformation of data from channel 2 to
    channel 1.
    """

    @config.set_columns
    def __init__(self,
                 feat1: Optional[Union[Sequence[pd.DataFrame],
                                       pd.DataFrame]] = None,
                 feat2: Optional[Union[Sequence[pd.DataFrame],
                                       pd.DataFrame]] = None,
                 columns: Dict = {},
                 channel_names: Sequence[str] = ["channel1", "channel2"]):
        """Parameters
        ----------
        feat1, feat2
            Set the `feat1` and `feat2` attribute (turning it into a list
            if it is a single DataFrame). Can also be `None`, but in this case
            :py:meth:`find_pairs` and :py:meth:`determine_parameters` will
            not work. Defaults to `None`.

        Other parameters
        ----------------
        channel_names
            Set the `channel_names` attribute.
        columns
            Override default column names as defined in
            :py:attr:`config.columns`. Relevant name are `coords` and `time`.
            That means, if the DataFrames have coordinate columns "x" and "z",
            and a time column "alt_frame", set
            ``columns={"coords": ["x", "z"], "time": "alt_frame"}``. This is
            used to set the :py:attr:`columns` attribute.
        """
        self.feat1 = [feat1] if isinstance(feat1, pd.DataFrame) else feat1
        self.feat2 = [feat2] if isinstance(feat2, pd.DataFrame) else feat2
        self.columns = columns
        self.channel_names = channel_names
        self.pairs = None
        self.parameters1 = np.eye(len(columns["coords"]) + 1)
        self.parameters2 = np.eye(len(columns["coords"]) + 1)

    def determine_parameters(self, n_neighbors: int = 3,
                             ambiguity_factor: float = 5.0,
                             max_error: float = 1.0):
        """Determine the parameters for the affine transformation

        This takes the localizations of :py:attr:`feat1` and tries to match
        them with those of :py:attr:`feat2`. Then a fit is used to determine
        the affine transformation between the channels.

        This is a convenience function that calls :py:meth:`find_pairs` and
        :py:meth:`fit_parameters`; see the documentation of the methods for
        details.

        Parameters
        ----------
        n_neighbors
            Number of neighboring beads to consider to find matching features
            across channels.
        ambiguity_factor
            A low value (around 1) will accept pairs even if there are similar
            possible matches. The higher this value, the less are ambigous
            results accepted.
        max_error
            Maximum error (i.e., distance between transformed position from
            channel 1 and matched position in channel 2) to consider a feature
            pair not an outlier and thus remove it from the transformation fit.
        """
        self.find_pairs(n_neighbors, ambiguity_factor)
        self.fit_parameters(max_error)

    @overload
    def __call__(self, data: pd.DataFrame, channel: Union[int, str],
                 inplace: Literal[True], mode: str,
                 cval: Union[float, Callable[[np.ndarray], float]],
                 columns: Mapping): ...

    @overload
    def __call__(self, data: pd.DataFrame, channel: Union[int, str],
                 inplace: Literal[False], mode: str,
                 cval: Union[float, Callable[[np.ndarray], float]],
                 columns: Mapping) -> pd.DataFrame: ...

    @overload
    def __call__(self, data: Union[helper.Slicerator, helper.Pipeline],
                 channel: Union[int, str], inplace: Literal[False], mode: str,
                 cval: Union[float, Callable[[np.ndarray], float]],
                 columns: Mapping) -> helper.Pipeline: ...

    ImageOrROI = TypeVar("ImageOrROI",
                         bound=Union[roi.PathROI, np.ndarray])

    @config.set_columns
    def __call__(self,
                 data: ImageOrROI,
                 channel: Union[int, str], inplace: bool = False,
                 mode: str = "constant",
                 cval: Union[float, Callable[[np.ndarray], float]] = 0.0,
                 columns: Mapping = {}) -> ImageOrROI:
        """Apply transformation to data

        This can be done either on coordinates (e. g. resulting from feature
        localization) or directly on raw images.

        Parameters
        ----------
        data
            Data to be processed. If a pandas.Dataframe, the feature
            coordinates will be corrected. Otherwise,
            :py:class:`sdt.helper.Pipeline` is used to correct image data using
            an affine transformation.
        channel
            If `features` are in the first channel (corresponding to the
            `feat1` arg of the constructor), set to 1 or the first channel's
            name. If features are in the second channel, set to 2 or the
            second channel's name. Depending on this, a transformation will
            be applied to the coordinates of `features` to match the other
            channel (mathematically speaking depending on this parameter
            either the "original" transformation or its inverse are applied.)
        inplace
            Only has an effect if `data` is a DataFrame. If True, the
            feature coordinates will be corrected in place.
        mode
            How to fill points outside of the uncorrected image boundaries.
            Possibilities are "constant", "nearest", "reflect" or "wrap".
        cval
            What value to use for `mode="constant"`. If this is callable, it
            should take a single argument (the uncorrected image) and return a
            scalar, which will be used as the fill value.

        Other parameters
        ----------------
        columns
            Override default column names in case `data` is a
            :py:class:`pandas.DataFrame`. The only relevant name is `coords`.
            That means, if the DataFrame has coordinate columns "x" and "z",
            set ``columns={"coords": ["x", "z"]}``.
        """
        with suppress(ValueError):
            channel = self.channel_names.index(channel) + 1
        if channel not in (1, 2):
            valid = [1, 2, *self.channel_names[:2]]
            raise ValueError(
                "channel has to be one of "
                f"{', '.join(str(v) for v in valid)}")

        pos_columns = columns["coords"]

        if isinstance(data, pd.DataFrame):
            if not inplace:
                data = data.copy()
            loc = data[pos_columns].values
            par = getattr(self, f"parameters{channel}")
            data[pos_columns] = _affine_trafo(par, loc)

            if not inplace:
                # copied previously, now return
                return data
        elif isinstance(data, roi.PathROI):
            t = mpl.transforms.Affine2D(
                getattr(self, f"parameters{channel}"))
            return roi.PathROI(t.transform_path(data.path),
                               buffer=data.buffer,
                               no_image=(data.image_mask is None))
        else:
            parms = getattr(self, f"parameters{2 if channel == 1 else 1}")

            @helper.pipeline
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

    def test(self, ax: Optional[Sequence] = None):
        """Test validity of the correction parameters

        This plots the affine transformation functions and the coordinates of
        the pairs that were matched in the channels. If everything went well,
        the dots (i. e. pair coordinates) should lie on the line
        (transformation function).

        Parameters
        ----------
        ax
            Two matplotib axes instances for plotting. If `None`, allocate new
            axes using :py:func:`matplotlib.pyplot.subplots`.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            grid_kw = dict(width_ratios=(1, 2))
            fig, ax = plt.subplots(1, 2, gridspec_kw=grid_kw)
        else:
            fig = None

        c1 = self.pairs[self.channel_names[0]]
        c1_corr = self(c1, channel=1, columns=self.columns)
        diff1 = (c1_corr[self.columns["coords"]] -
                 self.pairs[self.channel_names[1]][self.columns["coords"]])
        diff1 = np.sqrt(np.sum(diff1**2, axis=1))

        c2 = self.pairs[self.channel_names[1]]

        ax[0].hist(diff1, bins=20)
        ax[0].set_title("Error")
        ax[0].set_xlabel("distance")
        ax[0].set_ylabel("# data points")

        x_col, y_col = self.columns["coords"]
        ax[1].scatter(c1_corr[x_col], c1_corr[y_col], marker="x", color="blue")
        ax[1].scatter(c2[x_col], c2[y_col], marker="+", color="red")
        ax[1].set_aspect(1, adjustable="datalim")
        ax[1].set_title("Overlay")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")

        if fig is not None:
            # Figure was created here, so it is safe to do this
            fig.tight_layout()

    def save(self, file: Union[BinaryIO, str, Path], fmt: str = "npz",
             key: Tuple[str, str] = ("chromatic_param1", "chromatic_param2")):
        """Save transformation parameters to file

        Parameters
        ----------
        file
            File name or an open file-like object to save data to.
        fmt
            Format to save data. Either numpy ("npz") or MATLAB ("mat").
            Defaults to "npz".

        Other parameters
        ----------------
        key
            Names of two transformations in the saved file.
        """
        vardict = {key[0]: self.parameters1, key[1]: self.parameters2}
        if fmt == "npz":
            np.savez(file, **vardict)
        elif fmt == "mat":
            scipy.io.savemat(file, vardict)
        else:
            raise ValueError("Unknown format: {}".format(fmt))

    @classmethod
    def load(cls, file: Union[BinaryIO, str, Path], fmt: str = "npz",
             key: Tuple[str, str] = ("chromatic_param1", "chromatic_param2")
             ) -> "Registrator":
        """Read paramaters from a file and construct a `Registrator`

        Parameters
        ----------
        file
            File name or an open file-like object to load data from.
        fmt
            Format to save data. Either numpy ("npz") or MATLAB ("mat") or
            `determine_shiftstretch`'s wrp ("wrp"). Defaults to "npz".

        Returns
        -------
        A class instance with the parameters read from the file.

        Other parameters
        ----------------
        key
            Name of the variables in the saved file (does not apply to "wrp").
            Defaults to ("chromatic_param1", "chromatic_param2").
        """
        corr = cls(None, None)
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

    @staticmethod
    def _calc_bases(coords: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Create orthogonal coordinate system for each bead

        First basis vector is from the bead to the furthest of the n_neighbors
        neighbors, second is to the second-furthest minus the projection onto
        the first basis, and so on (Gram-Schmidt process).

        Parameters
        ----------
        coords
            Row-wise coordinates of the beads
        idx
            Row-wise indices of the nearest neighbors, sorted ascendingly by
            distance (i.e., the j-th entry in the i-th row is the index of
            the j-th nearest neighbor of bead i).

        Returns
        -------
        3D array where [i, :, :] yields the matrix of basis column vectors for
        the i-th bead.
        """
        coords = np.asarray(coords, dtype=float)
        n_dim = coords.shape[1]
        # bases = []
        bases = np.empty((len(coords), n_dim, n_dim))
        # Squared lengths of basis vectors
        base_len_sq = []
        for n in range(n_dim):
            # vector from bead to n-th furthest neighbor
            vec = coords[idx[:, -(n+1)]] - coords
            for i, bl in enumerate(base_len_sq):
                # subtract projection onto previously calculated
                # axes
                b = bases[..., i]
                vec -= (np.sum(vec * b, axis=1) / bl)[:, None] * b
            bases[:, :, n] = vec
            base_len_sq.append(np.sum(vec * vec, axis=1))
        return bases

    @staticmethod
    def _calc_local_coords(coords: np.ndarray, n_neighbors: int) -> np.ndarray:
        """Calculate coordinates of neighboring beads in local coordinates

        Create a coordinate system for each bead using :py:meth:`_calc_bases`
        and compute the coordinates of the neighboring beads.

        Parameters
        ----------
        coords
            Row-wise coordinates of the beads
        n_neighbors
            Number of neighboring beads to consider

        Returns
        -------
        3D array where [i, :, :] yields the matrix of coordinate column vectors
        of the neighbors of the i-th bead.
        """
        kd = scipy.spatial.cKDTree(coords)
        # No need to return nearest neighbor (k=1) since this is
        # the bead itself. Counting starts at 1.
        dist, idx = kd.query(coords, k=range(2, n_neighbors + 2))

        # Solve linear equation system to calculate coordinates in
        # the new basis described by axes.
        # Left hand side is the coordinate transform from new basis
        # to cartesian. First index is bead number, last to indices
        # give the transformation matrix.
        bases = __class__._calc_bases(coords, idx)
        # Right hand side are vectors from bead (first index) to
        # its neighbors
        rhs = coords[idx] - coords[:, None, :]
        # Swap last and second-to last indices to produce row
        # vectors
        rhs = np.moveaxis(rhs, -2, -1)
        # Solve system
        local_coords = np.linalg.solve(bases, rhs)
        return local_coords

    @staticmethod
    def _signatures_from_local_coords(local_coords: np.ndarray,
                                      triu: np.ndarray) -> np.ndarray:
        """Extract bead signatures from local coordinates

        Due to the construction of the basis vectors, the furthest neighbor
        will always have coordinates [1., 0., …, .0], the will have
        [x, 1., 0, …, 0.], and so on. Remove these zeros and ones values,
        i.e., get the values from the upper triangle of the flipped matrix as
        the signature.

        Parameters
        ----------
        local_coords
            3D array where [i, :, :] yields the matrix of coordinate column
            vectors of the neighbors of the i-th bead (see
            :py:meth:`_calc_local_coords`).
        triu
            Indices of the upper triangular matrix shifted by one. Generate
            using ``np.triu_indices(n=n_dim, m=n_neighbors, k=1)``. This is
            passed as an argmuent so it can be created once and reused.

        Returns
        -------
        2D array where [i, :] yields the signature of the i-th bead.
        """
        return local_coords[..., ::-1][:, triu[0], triu[1]]

    @staticmethod
    def _pairs_from_signatures(coords: Sequence[np.ndarray],
                               signatures: Sequence[np.ndarray],
                               ambiguity_factor: float) -> Tuple[np.ndarray]:
        """Find pairs from signatures

        Find the nearest neighbors in signature space

        Parameters
        ----------
        coords
            Matrix of bead coordinate row vectors for each channel
        signatures
            Matrix of row-wise bead signatures for each channel
        ambiguity_factor
            Accept only matches where the distance (in signature space) between
            the beads of the second best match is at least ``ambiguity_factor``
            times as large as the distance for the best match.

        Returns
        -------
        Matrices of matched bead coordinate row vectors for each channel. The
        i-th row of the first matrix matches the i-th row of the second.
        """
        unambiguous = []
        matches = []
        # Find pairs and check for ambiguity both ways. Otherwise there will
        # be problems if there are two candidates in one channel for one in the
        # other depending on which is channel 1 and which is channel 2
        for src, dest in (0, 1), (1, 0):
            kd = scipy.spatial.cKDTree(signatures[src])
            match_dist, match_idx = kd.query(signatures[dest], k=2)
            u = match_dist[:, 1] > ambiguity_factor * match_dist[:, 0]
            unambiguous.append(u)
            matches.append(match_idx)

        unamb_twoway = unambiguous[0] & unambiguous[1][matches[0][:, 0]]
        return (coords[0][matches[0][unamb_twoway, 0]],
                coords[1][unamb_twoway])

    def find_pairs(self, n_neighbors: int = 3, ambiguity_factor: float = 5.0):
        """Match features of `feat1` with features of `feat2`

        Find the geomtric signature for each feature in each channel. Those
        with best matching signatures are taken to be the same feature.

        Parameters
        ----------
        n_neighbors
            Number of neighboring beads to consider for signature calculation.
        ambiguity_factor
            Accept only matches where the distance (with respect to the
            geometric signature) between the beads of the second best match is
            at least ``ambiguity_factor`` times as large as the distance for
            the best match.
        """
        pairs = []
        n_dim = len(self.columns["coords"])
        # Indices of triangular matrix, used below
        triu = np.triu_indices(n=n_dim, m=n_neighbors, k=1)
        # At least n_dim neighbors are needed to define local coordinates
        n_neighbors = max(n_neighbors, n_dim)

        for f1, f2 in zip(self.feat1, self.feat2):
            if all(self.columns["time"] in f.columns for f in (f1, f2)):
                # If there is a "frame" column, split data according to
                # frame number since pairs can only be in the same frame
                data = ([f[f[self.columns["time"]] == i] for f in (f1, f2)]
                        for i in f1[self.columns["time"]].unique())
            else:
                # If there is no "frame" column, just take everything
                data = ((f1, f2),)

            for frame_data in data:
                if any(len(f) < n_neighbors + 1 for f in frame_data):
                    # Not enough neighbors
                    continue

                signatures = []
                coordinates = []
                for f in frame_data:
                    # Profile says that the code below takes half the
                    # time compared to f[sel.columns["coords"]].to_numpy()
                    coords = np.array([f[c].to_numpy()
                                       for c in self.columns["coords"]]).T
                    coords = coords[np.all(np.isfinite(coords), axis=1)]
                    lc = self._calc_local_coords(coords, n_neighbors)
                    s = self._signatures_from_local_coords(lc, triu)
                    signatures.append(s)
                    coordinates.append(coords)

                # TODO: flip_axes
                p = self._pairs_from_signatures(coordinates, signatures,
                                                ambiguity_factor)
                pairs.append(np.hstack(p))
        pair_cols = [self.channel_names, self.columns["coords"]]
        self.pairs = pd.DataFrame(
            np.vstack(pairs), columns=pd.MultiIndex.from_product(pair_cols))

    def fit_parameters(self, max_error: float = 1.0):
        """Determine parameters for the affine transformation

        An affine transformation is used to map x and y coordinates of `feat1`
        to to those of `feat2`. This requires :py:attr:`pairs` to be set, e.g.
        by running :py:meth:`find_pairs`.
        The result is saved as :py:attr:`parameters1` (transform of channel 1
        coordinates to channel 2) and :py:attr:`parameters2` (transform of
        channel 2 coordinates to channel 1) attributes.

        :py:attr:`parameters1` is calculated by determining the affine
        transformation between the :py:attr:`pairs` entries using a RANSAC
        algorithm. :py:attr:`parameters2` is its inverse. Therefore,
        results may be slightly different depending on what is channel1 and
        what is channel2.

        Parameters
        ----------
        max_error
            Maximum error (i.e., distance between transformed position from
            channel 1 and matched position in channel 2) to consider a feature
            pair not an outlier.
        """
        ch1_name = self.channel_names[0]
        ch2_name = self.channel_names[1]

        loc1 = self.pairs[ch1_name][self.columns["coords"]]
        loc2 = self.pairs[ch2_name][self.columns["coords"]]
        n_dim = loc1.shape[1]

        self.parameters1[:n_dim, :], good_pairs = cv2.estimateAffine2D(
            loc1.to_numpy(dtype=np.float32), loc2.to_numpy(dtype=np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=max_error)
        self.parameters1[n_dim, :n_dim] = 0.
        self.parameters1[n_dim, n_dim] = 1.

        self.parameters2 = np.linalg.inv(self.parameters1)
        self.pairs = self.pairs[np.squeeze(good_pairs.astype(bool))]

    @classmethod
    def to_yaml(cls, dumper, data):
        """Dump as YAML

        Pass this as the `representer` parameter to your
        :py:class:`yaml.Dumper` subclass's `add_representer` method.
        """
        m = (("channel_names", list(data.channel_names)),
             ("parameters1", data.parameters1),
             ("parameters2", data.parameters2))
        return dumper.represent_mapping(cls.yaml_tag, m)

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct from YAML

        Pass this as the `constructor` parameter to your
        :py:class:`yaml.Loader` subclass's `add_constructor` method.
        """
        m = loader.construct_mapping(node)
        ret = cls()
        with suppress(KeyError):
            ret.channel_names = m["channel_names"]
        ret.parameters1 = m["parameters1"]
        ret.parameters2 = m["parameters2"]
        return ret

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return False
        return (np.allclose(self.parameters1, other.parameters1) and
                np.allclose(self.parameters2, other.parameters2))


with suppress(ImportError):
    from ..io import yaml
    yaml.register_yaml_class(Registrator)
