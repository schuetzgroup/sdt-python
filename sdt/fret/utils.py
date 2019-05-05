"""Module containing tools related to images from FRET experiments"""
from collections import defaultdict

import numpy as np


class FretImageSelector:
    """Select images of a certain excitation type from an image series

    E.g. if employing alternating excitation, this can be used to select only
    the donor or only the acceptor frames.

    Examples
    --------

    >>> # Sequence of 6 "images"
    >>> img_seq = numpy.array([numpy.full((3, 3), i) for i in range(8)])
    >>> sel = FretImageSelector("odda")
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

    def __call__(self, img_seq, type="d"):
        """Get only images of a certain excitation type from an image series

        Parameters
        ----------
        img_seq : list-like of numpy.ndarrays
            Series of images
        type : str
            Excitation type. This should match something in
            :py:attr:`excitation_seq`.

        Returns
        -------
        type(img_seq) or slicerator.Slicerator
            Selected images. First an attempt is made at indexing the images
            directly. If `img_seq` supports this (e.g. if it is a
            :py:class:`numpy.ndarray`), the indexed version is returned.
            Otherwise, `img_seq` is converted to
            :py:class:`slicerator.Slicerator`, indexed and returned.
        """
        idx = np.arange(len(img_seq))
        sel_idx = idx[np.isin(idx % len(self.excitation_seq),
                              self.excitation_frames[type])]

        try:
            return img_seq[sel_idx]
        except TypeError:
            # This can happen if e.g. data is a list
            from slicerator import Slicerator
            s = Slicerator(img_seq)
            return s[sel_idx]
