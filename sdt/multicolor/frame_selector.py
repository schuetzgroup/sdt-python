# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import re
import string
from typing import Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd

from .. import config, helper


class _FlexMul:
    """Helper class to evaluate "?" in flexible sequences

    In excitation sequences containing multiplication with "?", the "?" is
    replaced by a number so that the excitation sequence length matches
    a given number of frames.
    """
    n_flex_frames: int
    """Number of frames to be filled by repeating sequence part"""

    def __init__(self, n_flex_frames: int):
        """Parameters
        ----------
        n_flex_frames
            Set :py:attr:`n_flex_frames` attribute
        """
        self.n_flex_frames = n_flex_frames

    def __mul__(self, rep_seq: str) -> str:
        """Perform expansion

        `rep_seq` will be repeated until the length of the result is equal to
        :py:attr:`n_flex_frames`.

        Paramters
        ---------
        rep_seq
            Sequence part to be repeated

        Returns
        -------
        Expanded `rep_seq`
        """
        reps, rem = divmod(self.n_flex_frames, len(rep_seq))
        if rem:
            raise ValueError("Number of flexible frames is not divisible by "
                             "length of repeated sequence.")
        return rep_seq * reps

    def __rmul__(self, rep_seq: str) -> str:
        """Same as :py:meth:`__mul__`"""
        return self.__mul__(rep_seq)


class FrameSelector:
    """Select images and datapoints of a certain excitation type

    For instance, if employing alternating excitation for a FRET experiment,
    this can be used to select only the donor or only the acceptor frames.
    For details on how the excitation sequence is specified, see
    :py:attr:`excitation_seq`.

    Examples
    --------
    >>> # Sequence of 8 "images"
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

    A flexible sequence specification uses ``?`` as a multiplicator. This
    causes repetition of a subsequence such that the excitation sequence
    length matches the image sequence length. In the following example,
    ``"da"`` is repeated three times.

    >>> sel = FrameSelector("o + da * ? + c")
    >>> sel(img_seq, "d")
    array([[[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]],
    <BLANKLINE>
           [[3, 3, 3],
            [3, 3, 3],
            [3, 3, 3]],
    <BLANKLINE>
           [[5, 5, 5],
            [5, 5, 5],
            [5, 5, 5]]])
    """
    excitation_seq: str
    """Excitation sequence. Use different letters to describe different
    excitation types, e.g., "d" for donor excitation, "a" for acceptor,
    etc.

    One needs only specify the shortest sequence that is repeated,
    i. e. ``"ddddaddddadddda"`` is equivalent to ``"dddda"``. It is
    possible to specifiy repetition via the multiplication operator.
    ``"c + da * 3 + c"`` is equivalent to ``"cdadadac"``. Also, flexible
    sequences can be specified using ``?`` as a multiplicator. In ``"c +
    da * ? + c"``, ``"da"`` is repeated an appropriate number of times
    to generate a sequence of given length. This length is either derived
    from the image sequence the class instance is applied to or given
    explicitly (see :py:meth:`__call__`).
    """

    _rm_whitespace_trans = str.maketrans("", "", string.whitespace)
    """To be used with ``str.translate`` to remove whitespace"""
    _letter_re = re.compile(r"([A-Za-z]+)")
    """Regular expression grouping upper- and lowercase letters"""

    def __init__(self, excitation_seq: str):
        """Parameters
        ----------
        excitation_seq
            Set :py:attr:`excitation_seq`
        """
        self.excitation_seq = excitation_seq

    def _eval_simple(self, seq: str, glob: Mapping = {}, loc: Mapping = {}
                     ) -> str:
        """Evaluate non-flexible sequence

        Parameters
        seq
            Sequence to evaluate
        glob
            Global variables to pass to :py:func:`eval`
        loc
            Local variables to pass to :py:func:`eval`

        Returns
        -------
        Evaluated sequence string
        """
        re_seq = self._letter_re.sub(r'"\1"', seq)
        return eval(re_seq, glob, loc)

    def eval_seq(self, n_frames: Optional[int] = None) -> np.ndarray:
        """Evaluate excitation sequence

        Parameters
        ----------
        n_frames
            Number of frames. This needs to be given if sequence is flexible,
            i.e., contains multiplication by ``?``. In this case, ``?``
            evaluates to a number such that the total sequence length is
            `n_frames`. If `n_frames` is equal to -1, ``?`` is replaced by
            1. This is useful for testing and finding e.g. the first occurence
            of a frame without depending on actual data.

        Returns
        -------
        Array of characters representing evaluated sequence
        """
        seq = self.excitation_seq.translate(self._rm_whitespace_trans)
        q_pos = seq.find("?")
        if q_pos < 0:
            eseq = self._eval_simple(seq)
            return np.fromiter(eseq, "U1", len(eseq))
        if n_frames is None:
            raise ValueError("`n_frames` must be given for sequences "
                             "containing '?'")

        pre_pos = seq.rfind("+", 0, q_pos)
        post_pos = seq.find("+", q_pos)

        n_fixed = 0
        if pre_pos >= 0:
            n_fixed += len(self._eval_simple(seq[:pre_pos]))
        if post_pos >= 0:
            n_fixed += len(self._eval_simple(seq[post_pos+1:]))

        if n_frames < 0:
            mul = 1
        else:
            mul = _FlexMul(n_frames - n_fixed)
        eseq = self._eval_simple(seq.replace("?", "_", 1), loc={"_": mul})
        return np.fromiter(eseq, "U1", len(eseq))

    @staticmethod
    def _renumber(eval_seq: np.ndarray, frame_nos: np.ndarray,
                  which: Union[str, Iterable[str]], restore: bool
                  ) -> np.ndarray:
        """Do the actual frame renumbering

        for :py:meth:`__call__` and :py:meth:`restore_frame_numbers`.

        Parameters
        ----------
        eval_seq
            Evaluated excitation sequence (array of characters).
            See :py:meth:`eval_seq`.
        frame_nos
            Array of frame numbers to be mapped
        which
            Excitation type(s), one or multiple characters out of
            :py:attr:`excitation_seq`.
        restore
            If `False`, renumber the frames (as in :py:meth:`__call__`).
            If `True`, restore original frame numbers (as in
            :py:meth:`restore_frame_numbers`).

        Returns
        -------
        New frame numbers
        """
        frame_nos = np.asanyarray(frame_nos, dtype=int)
        good_frame_mask = np.isin(eval_seq, list(which))
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
            n_repeats = math.ceil((max_frame + 1) / len(eval_seq))
        # Calculate the restoring map
        f_map_inv = np.nonzero(np.tile(good_frame_mask, n_repeats))[0]
        if restore:
            return f_map_inv[frame_nos]
        # Invert the restoring map
        f_map = np.full(f_map_inv[-1] + 1, -1, dtype=int)
        f_map[f_map_inv] = np.arange(f_map_inv.size)
        return f_map[frame_nos]

    @config.set_columns
    def __call__(self, data: Union[pd.DataFrame, Iterable[np.ndarray]],
                 which: Union[str, Iterable[str]], renumber: bool = False,
                 n_frames: Optional[int] = None, columns: Mapping = {}
                 ) -> Union[pd.DataFrame, np.ndarray, helper.Slicerator]:
        """Get only data corresponding to a certain excitation type

        Parameters
        ----------
        data
            Localization data or sequence of images
        which
            Excitation type(s), one or multiple characters out of
            :py:attr:`excitation_seq`.
        renumber
            Renumber frames so that only frames for excitation types
            corresponding to `which` are counted.

            After selecting only data corresponding to an excitation type (e.g.
            "a") in a :py:class:`pandas.DataFrame`, all other types are
            missing and frame number are not consecutive any more. This can be
            a problem for tracking or diffusion analysis. Setting
            ``renumber=True`` will work around this.

            Only applicable if `data` is a :py:class:`pandas.DataFrame`.
        n_frames
            Number of frames. This needs to be given if sequence is flexible,
            i.e., contains multiplication by ``?``. In this case, ``?``
            evaluates to a number such that the total sequence length is
            `n_frames`.

        Returns
        -------
        Selected frames. If `data` is a DataFrame, return only lines
        corresponding to excitation types given by `which`. If
        ``renumber=True``, a modified copy of `data` is returned.

        If `data` is an image sequence, first an attempt is made at
        indexing the images  directly. If `data` supports this (e.g. if it
        is a :py:class:`numpy.ndarray`), the indexed version is returned.
        Otherwise, `img_seq` is converted to
        :py:class:`helper.Slicerator`, indexed, and returned.

        Other parameters
        ----------------
        columns
            Override default column names in case `data` is a
            :py:class:`pandas.DataFrame`. The only relevant name is `time`.
            That means, if the DataFrame has frame number columns "frame2",
            set ``columns={"time": "frame2"}``.
        """
        if not isinstance(data, pd.DataFrame) and n_frames is None:
            # Use length of image sequence
            n_frames = len(data)
        eval_seq = self.eval_seq(n_frames)

        good_frame_mask = np.isin(eval_seq, list(which))
        good_frames = np.nonzero(good_frame_mask)[0]

        if isinstance(data, pd.DataFrame):
            frames = data[columns["time"]] % len(eval_seq)
            ret = data[frames.isin(good_frames)]

            if renumber and not ret.empty:
                ret = ret.copy()
                ret[columns["time"]] = self._renumber(
                    eval_seq, ret[columns["time"]].to_numpy(), which, False)
            return ret

        # Deal with image data
        idx = np.arange(len(data))
        sel_idx = idx[np.isin(idx % len(eval_seq), good_frames)]

        try:
            return data[sel_idx]
        except TypeError:
            # This can happen if e.g. data is a list
            s = helper.Slicerator(data)
            return s[sel_idx]

    @config.set_columns
    def restore_frame_numbers(self, data: pd.DataFrame,
                              which: Union[str, Iterable[str]],
                              n_frames: Optional[int] = None,
                              columns: Mapping = {}):
        """Undo frame renumbering from :py:meth:`__call__`

        `data` is modified in place for this purpose.

        Parameters
        ----------
        data
            Localization data
        which
            Excitation type(s), one or multiple characters out of
            :py:attr:`excitation_seq`.
        n_frames
            Number of frames. This needs to be given if sequence is flexible,
            i.e., contains multiplication by ``?``. In this case, ``?``
            evaluates to a number such that the total sequence length is
            `n_frames`.

        Other parameters
        ----------------
        columns
            Override default column names in case `data` is a
            :py:class:`pandas.DataFrame`. The only relevant name is `time`.
            That means, if the DataFrame has frame number columns "frame2",
            set ``columns={"time": "frame2"}``.
        """
        if not data.empty:
            eval_seq = self.eval_seq(n_frames)
            data[columns["time"]] = self._renumber(
                eval_seq, data[columns["time"]].to_numpy(), which, True)
