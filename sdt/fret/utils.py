"""Module containing tools related to images from FRET experiments"""
from collections import defaultdict
import math

import numpy as np
import pandas as pd

from .. import config


class FrameSelector:
    """Select images of a certain excitation type from an image series

    E.g. if employing alternating excitation, this can be used to select only
    the donor or only the acceptor frames.

    Examples
    --------

    >>> # Sequence of 6 "images"
    >>> img_seq = numpy.array([numpy.full((3, 3), i) for i in range(8)])
    >>> sel = FrameSelector("odda")
    >>> sel(img_seq, "o")
    array([[[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
    <BLANKLINE>
           [[4, 4, 4],
            [4, 4, 4],
            [4, 4, 4]]])
    >>> sel(img_seq, "d")
    array([[[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]],
    <BLANKLINE>
           [[2, 2, 2],
            [2, 2, 2],
            [2, 2, 2]],
    <BLANKLINE>
           [[5, 5, 5],
            [5, 5, 5],
            [5, 5, 5]],
    <BLANKLINE>
           [[6, 6, 6],
            [6, 6, 6],
            [6, 6, 6]]])
    """
    def __init__(self, excitation_seq):
        """Parameters
        ----------
        excitation_seq : str or list-like of characters
            Excitation sequence. Typically, "d" would stand for donor, "a" for
            acceptor.

            One needs only specify the shortest sequence that is repeated,
            i. e. "ddddaddddadddda" is the same as "dddda".
        """
        self.excitation_seq = np.array(list(excitation_seq))

    @property
    def excitation_seq(self):
        """numpy.ndarray of dtype("<U1") describing the excitation sequence.
        Typically, "d" would stand for donor, "a" for
        acceptor.

        One needs only specify the shortest sequence that is repeated,
        i. e. "ddddaddddadddda" is the same as "dddda".
        """
        return self._exc_seq

    @property
    def excitation_frames(self):
        """dict mapping the excitation types in :py:attr:`excitation_seq` to
        the corresponding frame numbers (modulo the length of
        py:attr:`excitation_seq`).
        """
        return self._exc_frames

    @excitation_seq.setter
    def excitation_seq(self, v):
        self._exc_seq = np.array(list(v))
        self._exc_frames = defaultdict(list,
                                       {k: np.nonzero(self._exc_seq == k)[0]
                                        for k in np.unique(self._exc_seq)})

    def _renumber(self, frame_nos, which, restore):
        """Do the actual frame renumbering

        for :py:meth:`__call__` and :py:meth:`restore_frame_numbers`.

        Parameters
        ----------
        frame_nos : numpy.ndarray
            Array of frame numbers to be mapped
        which : str or iterable of str
            Excitation type(s). This should match something in
            :py:attr:`excitation_seq`.
        restore : bool
            If `False`, renumber the frames (as in :py:meth:`__call__`).
            If `True`, restore original frame numbers (as in
            :py:meth:`restore_frame_numbers`).

        Returns
        -------
        numpy.ndarray
            New frame numbers
        """
        frame_nos = np.asanyarray(frame_nos, dtype=int)
        good_frame_mask = np.isin(self.excitation_seq, list(which))
        max_frame = np.max(frame_nos)
        if restore:
            # f_map_inv needs at least max_frame + 1 entries (the domain of
            # the restoring map is {0, 1, …, max_frame}), meaning
            # good_frame_mask has to be chained (max_frame + 1) / num_true
            # times, where num_true is the number of True entries in
            # good_frame_mask
            num_true = np.sum(good_frame_mask)
            n_repeats = math.ceil((max_frame + 1) / num_true)
        else:
            # f_map needs at least max_frame + 1 entries (the domain of
            # the frame number map is {0, 1, …, max_frame}). f_map_inv's values
            # (which form the codomain of the map) need to go up to at least
            # max_frame.
            n_repeats = math.ceil(max_frame / len(self.excitation_seq))
        # Calculate the restoring map
        f_map_inv = np.nonzero(np.tile(good_frame_mask, n_repeats))[0]
        if restore:
            return f_map_inv[frame_nos]
        # Invert the restoring map
        f_map = np.full(f_map_inv[-1] + 1, -1, dtype=int)
        f_map[f_map_inv] = np.arange(f_map_inv.size)
        return f_map[frame_nos]

    @config.set_columns
    def __call__(self, data, which="d", renumber=False, columns={}):
        """Get only data corresponding to a certain excitation type

        Parameters
        ----------
        data : pandas.DataFrame or list-like of numpy.ndarrays
            Localization data or sequence of images
        which : str or iterable of str, optional
            Excitation type(s). This should match something in
            :py:attr:`excitation_seq`. Defaults to "d".
        renumber : bool, optional
            Renumber frames so that only frames for excitation types
            corresponding to `which` are counted.

            After selecting only data corresponding to an excitation type (e.g.
            "a") in a :py:class:`pandas.DataFrame`, all other types are
            missing and frame number are not consecutive any more. This can be
            a problem for tracking or diffusion analysis. Setting
            ``renumber=True`` will work around this.

            Only applicable if `data` is a :py:class:`pandas.DataFrame`.
            Defaults to `False`.

        Returns
        -------
        type(data) or slicerator.Slicerator
            Selected frames. If `data` is a DataFrame, return only lines
            corresponding to excitation types given by `which`. If
            ``renumber=True``, a modified copy of `data` is returned.

            If `data` is an image sequence, first an attempt is made at
            indexing the images  directly. If `data` supports this (e.g. if it
            is a :py:class:`numpy.ndarray`), the indexed version is returned.
            Otherwise, `img_seq` is converted to
            :py:class:`sdt.helper.Slicerator`, indexed, and returned.

        Other parameters
        ----------------
        columns : dict, optional
            Override default column names in case `data` is a
            :py:class:`pandas.DataFrame`. The only relevant name is `time`.
            That means, if the DataFrame has frame number columns "frame2",
            set ``columns={"time": "frame2"}``.
        """
        good_frame_mask = np.isin(self.excitation_seq, list(which))
        good_frames = np.nonzero(good_frame_mask)[0]

        if isinstance(data, pd.DataFrame):
            frames = data[columns["time"]] % len(self.excitation_seq)
            ret = data[frames.isin(good_frames)]

            if renumber and not ret.empty:
                ret = ret.copy()
                ret[columns["time"]] = self._renumber(
                    ret[columns["time"]].to_numpy(), which, False)
            return ret

        # Deal with image data
        idx = np.arange(len(data))
        sel_idx = idx[np.isin(idx % len(self.excitation_seq), good_frames)]

        try:
            return data[sel_idx]
        except TypeError:
            # This can happen if e.g. data is a list
            from ..helper import Slicerator
            s = Slicerator(data)
            return s[sel_idx]

    @config.set_columns
    def restore_frame_numbers(self, data, which="d", columns={}):
        """Undo frame renumbering from :py:meth:`__call__`

        `data` is modified in place for this purpose.

        Parameters
        ----------
        data : pandas.DataFrame
            Localization data
        which : str or iterable of str, optional
            Excitation type(s). This should match something in
            :py:attr:`excitation_seq`. Defaults to "d".

        Other parameters
        ----------------
        columns : dict, optional
            Override default column names in case `data` is a
            :py:class:`pandas.DataFrame`. The only relevant name is `time`.
            That means, if the DataFrame has frame number columns "frame2",
            set ``columns={"time": "frame2"}``.
        """
        if not data.empty:
            data[columns["time"]] = self._renumber(
                data[columns["time"]].to_numpy(), which, True)
