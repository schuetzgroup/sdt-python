# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import re
import string
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.interpolate

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
    >>> sel.select(img_seq, "o")
    array([[[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
    <BLANKLINE>
           [[4, 4, 4],
            [4, 4, 4],
            [4, 4, 4]]])
    >>> sel.select(img_seq, "d")
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
    >>> sel.select(img_seq, "d")
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
    explicitly (see :py:meth:`select`).

    If the sequence is empty, the FrameSelector does nothing. E.g.,
    :py:meth:`select` and :py:meth:`renumber_frames` just return their
    argument unaltered.
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
        if not re_seq:
            return ""
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
    def _find_mask(eval_seq: np.ndarray, frame_nos: np.ndarray,
                   which: Union[str, Iterable[str]]) -> np.ndarray:
        """Find frames of certain type in list of frames, return boolean mask

        Parameters
        ----------
        eval_seq
            Evaluated excitation sequence (array of characters).
            See :py:meth:`eval_seq`.
        frame_nos
            Array of frame numbers from which frames should be selected
        which
            Excitation type(s), one or multiple characters out of
            :py:attr:`excitation_seq`.

        Returns
        -------
        Boolean array indicating whether elements of `frame_nos` are of an
        excitation type given by `which`.
        """
        good_frame_mask = np.isin(eval_seq, list(which))
        good_frames = np.nonzero(good_frame_mask)[0]
        mask = np.isin(frame_nos % len(eval_seq), good_frames)
        return mask

    @staticmethod
    def _find_numbers(eval_seq: np.ndarray, frame_nos: np.ndarray,
                      which: Union[str, Iterable[str]]) -> np.ndarray:
        """Find frames of certain type in list of frames, return numbers

        Parameters
        ----------
        eval_seq
            Evaluated excitation sequence (array of characters).
            See :py:meth:`eval_seq`.
        frame_nos
            Array of frame numbers from which frames should be selected
        which
            Excitation type(s), one or multiple characters out of
            :py:attr:`excitation_seq`.

        Returns
        -------
        Elements of `frame_nos` that are of an excitation type given by
        `which`.
        """
        return frame_nos[__class__._find_mask(eval_seq, frame_nos, which)]

    @staticmethod
    def _get_subseq(data: Sequence, index: Sequence[int]) -> Sequence:
        """Get a subsequence of `data`

        Parameters
        ----------
        data
            Sequence to get subsequence from
        index
            Indices of `data` present in subsequence

        Returns
        -------
        If possible, directly index `data` with `index` (works e.g. for numpy
        arrays). If not, create a :py:class:`helper.Slicerator` instance and
        use `index` on that.
        """
        try:
            return data[index]
        except TypeError:
            # This can happen if e.g. data is a list
            s = helper.Slicerator(data)
            return s[index]

    @staticmethod
    def _renumber(eval_seq: np.ndarray, frame_nos: np.ndarray,
                  which: Union[str, Iterable[str]], restore: bool
                  ) -> np.ndarray:
        """Do the actual frame renumbering

        for :py:meth:`select`, :py:meth:`renumber_frames`, and
        :py:meth:`restore_frame_numbers`. If ``restore=False``, any frames that
        don't belong to the excitation type given by `which` are mapped to -1.

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
            If `False`, renumber the frames (as in :py:meth:`select`).
            If `True`, restore original frame numbers (as in
            :py:meth:`restore_frame_numbers`).

        Returns
        -------
        New frame numbers
        """
        if len(eval_seq) == 0 or len(frame_nos) == 0:
            return frame_nos
        frame_nos = np.asanyarray(frame_nos, dtype=int)
        good_frame_mask = np.isin(eval_seq, list(which))
        n_good = np.sum(good_frame_mask)
        max_frame = np.max(frame_nos)
        if restore:
            if not n_good:
                # None of `which` are in excitation seq. There is nothing we
                # can do.
                raise ValueError(f"excitation sequence '{''.join(eval_seq)}' "
                                 f"does not contain types '{''.join(which)}'")
            # f_map_inv needs at least max_frame + 1 entries (the domain of
            # the restoring map is {0, 1, …, max_frame}), meaning
            # good_frame_mask has to be chained (max_frame + 1) / n_good
            # times, where n_good is the number of True entries in
            # good_frame_mask
            n_repeats = math.ceil((max_frame + 1) / n_good)
        else:
            if not n_good:
                # All are invalid and thus mapped to -1
                return np.full(frame_nos.size, -1, dtype=int)
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

    def renumber_frames(self, frame_nos: np.ndarray,
                        which: Union[str, Iterable[str]],
                        restore: bool = False, n_frames: Optional[int] = None
                        ) -> np.ndarray:
        """Renumber a sequence of frame numbers

        The numbers can be with respect to the original image sequence. In this
        case, set ``restore=False`` and this function will return frame numbers
        with respect to an image sequence which was returned by
        :py:meth:`select`.  Any frames that don't belong to the excitation type
        given by `which` are mapped to -1.
        If ``restore=True`` is set, this works the opposite other way.

        Parameters
        ----------
        frame_nos
            Array of frame numbers to be mapped
        which
            Excitation type(s), one or multiple characters out of
            :py:attr:`excitation_seq`.
        restore
            If `False`, renumber the frames (as in :py:meth:`select`).
            If `True`, restore original frame numbers (as in
            :py:meth:`restore_frame_numbers`).

        Returns
        -------
        New frame numbers
        """
        eval_seq = self.eval_seq(n_frames)
        return self._renumber(eval_seq, frame_nos, which, restore)

    def find_other_frames(self, data: Union[int, Sequence],
                          which: Union[str, Iterable[str]],
                          other: Union[str, Iterable[str]],
                          how: str = "nearest-up") -> Sequence:
        """For given excitation type, find frames of other type

        Typically, this is used to find the closest frame of a certain type,
        e.g., find the next acceptor excitation frame for donor excitation
        frames in FRET ALEX experiments.

        Examples
        --------
        >>> fs = FrameSelector("dddda")
        >>> d_img = fs.select(image_sequence, "d")
        >>> a_img = fs.find_other_frames(image_sequence, "d", "a)
        >>> for di, ai in zip(d_img, a_img):
        ...    # `ai` is the frame of type "a" closest to `di`
        ...    pass

        Parameters
        ----------
        data
            If this is a sequence, return a subsequence of the same length as
            ``select(data, which)``. Each entry of the returned sequence is
            the corresponding entry to what ``select(data, which)`` returns.
            Passing an int is equivalent to using ``numpy.arange(data)``.
        which
            Excitation types for which to find other excitation type's frames
        other
            Other frames' excitation type(s)
        how
            How to do the interpolation. "nearest-up" will return the nearest
            frame. In case of a tie, the “other” frame after the current one
            is used. "nearest" will return the “other” frame before the current
            one in case of a tie. "previous” and "next" return the “other”
            frame before and after the current one, respectively.

        Returns
        -------
        Subsequence of `data` with entries being the “other” entries to
        ``select(data, which)``.
        """
        if isinstance(data, int):
            n_frames = data
            data = None
        else:
            n_frames = len(data)

        eval_seq = self.eval_seq(n_frames)
        idx = np.arange(n_frames)
        source_idx = self._find_numbers(eval_seq, idx, which)
        target_idx = self._find_numbers(eval_seq, idx, other)
        if len(target_idx) == 0:
            raise ValueError("excitation sequence does not contain types any "
                             "of the types in `other`.")
        if len(target_idx) == 1:
            # scipy.interpolate.interp1d needs at least two points, so deal
            # with this case here
            return np.full(len(source_idx), target_idx[0], dtype=int)
        interp_idx = scipy.interpolate.interp1d(
            target_idx, target_idx, how, bounds_error=False,
            fill_value=(target_idx[0], target_idx[-1])
            )(source_idx)
        interp_idx = interp_idx.astype(int)

        if data is not None:
            return self._get_subseq(data, interp_idx)
        return interp_idx

    @config.set_columns
    def select(self, data: Union[pd.DataFrame, Iterable[np.ndarray]],
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
            `n_frames`. If `data` is an image sequence, `n_frames` can be
            inferred from ``len(data)``.

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
        if not self.excitation_seq:
            return data
        if not isinstance(data, pd.DataFrame) and n_frames is None:
            # Use length of image sequence
            n_frames = len(data)
        eval_seq = self.eval_seq(n_frames)

        if isinstance(data, pd.DataFrame):
            ret = data[
                self._find_mask(
                    eval_seq, data[columns["time"]].to_numpy(), which)]
            if renumber and not ret.empty:
                ret = ret.copy()
                ret[columns["time"]] = self._renumber(
                    eval_seq, ret[columns["time"]].to_numpy(), which, False)
            return ret

        # Deal with image data
        sel_idx = self._find_numbers(eval_seq, np.arange(len(data)), which)
        return self._get_subseq(data, sel_idx)

    @config.set_columns
    def restore_frame_numbers(self, data: pd.DataFrame,
                              which: Union[str, Iterable[str]],
                              n_frames: Optional[int] = None,
                              columns: Mapping = {}):
        """Undo frame renumbering from :py:meth:`select`

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
        if not self.excitation_seq or data.empty:
            return
        data[columns["time"]] = self.renumber_frames(
            data[columns["time"]].to_numpy(), which, True, n_frames)

    def __eq__(self, other: "FrameSelector"):
        return self.excitation_seq == other.excitation_seq
