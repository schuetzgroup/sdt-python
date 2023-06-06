# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import collections
import contextlib
import copy
import math
from pathlib import Path
from typing import Dict, IO, Mapping, Optional, Sequence, Union, overload

import numpy as np

from . import yaml


class Image(np.ndarray):
    """`ndarray` with :py:attr:`frame_no` attribute"""
    frame_no: int
    """Original frame number (before slicing the sequnece)"""

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(*args, **kwargs)
        obj.frame_no = -1
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.frame_no = getattr(obj, "frame_no", -1)


class ImageSequence:
    """Sliceable, lazy-loading interface to multi-image files

    Single images can be retrieved by index, while substacks can be created
    by slicing and fancy indexing using lists/arrays of indices or boolean
    indices. Creating substacks does not load data into memory, allowing for
    dealing with containing many images.

    Examples
    --------

    Load 3rd frame:

    >>> with ImageSequence("some_file.tif") as stack:
    ...     img = stack[3]

    Use fancy indexing to create substacks:

    >>> stack = ImageSequence("some_file.tif").open()
    >>> len(stack)
    30
    >>> substack1 = stack[1::2]  # Slice, will not load any data
    >>> len(substack2)
    15
    >>> np.all(substack2[1] == stack[3])  # Actually load data using int index
    True
    >>> substack2 = stack[[3, 5]]  # Create lazy substack using list of indices
    >>> substack3 = stack[[True, False] * len(stack) // 2]  # or boolean index
    >>> seq.close()
    """
    uri: Union[str, Path, bytes, IO]
    """File or file location or data to read from."""
    reader_args: Mapping
    """Keyword arguments passed to :py:func:`imageio.v3.imopen` when opening
    file.
    """
    _slicerator_flag = True  # Make it work with slicerator

    @property
    def is_slice(self) -> bool:
        """Whether this instance is the result of slicing another instance"""
        return self._is_slice

    def __init__(self, uri: Union[str, Path, bytes, IO], **kwargs):
        """Parameters
        ----------
        uri
            File or file location or data to read from.
        format
            File format. Use `None` for automatic detection.
        **kwargs
            Keyword arguments passed to :py:func:`imageio.v3.imopen` when
            opening the file.
        """
        self.uri = uri
        self.reader_args = kwargs
        self._reader = None
        self._indices = None
        self._is_slice = False
        self._is_tiff = False
        self._len = 0
        self._closed = True

    def open(self) -> "ImageSequence":
        """Open the file

        Returns
        -------
        self
        """
        if self._is_slice:
            raise RuntimeError("Cannot open sliced sequence.")
        if not self.closed:
            raise IOError(f"{self.uri} already open.")

        import imageio.v3
        from imageio.plugins.tifffile_v3 import TifffilePlugin

        self._reader = imageio.v3.imopen(self.uri, "r", **self.reader_args)
        self._is_tiff = isinstance(self._reader, TifffilePlugin)

        if self._is_tiff:
            self._len = self._reader.properties(index=..., page=...).n_images
        else:
            self._len = self._reader.properties(index=...).n_images

        self._closed = False
        return self

    def close(self):
        """Close the file"""
        if self._is_slice:
            raise RuntimeError("Cannot close sliced sequence.")
        self._len = 0
        self._closed = True
        self._reader.close()

    @overload
    def _resolve_index(self, t: int) -> int: ...

    def _resolve_index(self, t: Union[slice, Sequence[int], Sequence[bool]]
                       ) -> np.ndarray:
        """Convert index of potentially sliced stack to original index

        Parameters
        ----------
        t
            Index/indices w.r.t. sliced object

        Returns
        -------
        “Original” index/indeces suitable for retrieving images from file
        """
        # Use Iterable as Sequence does not imply numpy.ndarray
        if isinstance(t, (slice, collections.abc.Iterable)):
            if not math.isfinite(len(self)):
                raise IndexError(
                    "slicing impossible for sequences of unknown length")
        if isinstance(t, slice):
            t = np.arange(*t.indices(len(self)))
        if isinstance(t, collections.abc.Iterable):
            t = np.asarray(t)
            if np.issubdtype(t.dtype, np.bool_):
                if len(t) != len(self):
                    raise IndexError(
                        "boolean index did not match; stack length is "
                        f"{len(self)} but corresponding boolean length is "
                        f"{len(t)}")
                t = np.nonzero(t)[0]
            else:
                t[t < 0] += len(self)
            oob = np.nonzero((t < 0) | (t > len(self) - 1))[0]
            if oob.size:
                raise IndexError(
                    f"index {oob[0]} is out of bounds for stack of length "
                    f"{len(self)}")
        else:
            # Treat scalar t separately as this is much faster
            if t < 0:
                t += len(self)
            if t < 0 or t > len(self) - 1:
                raise IndexError(
                    f"index {t} is out of bounds for stack of length "
                    f"{len(self)}")
        if self._indices is None:
            return t
        return self._indices[t]

    def _parse_yaml_description(self, meta: Mapping):
        """Try to parse `description` metadata entry with YAML parser

        Parameters
        ----------
        meta
            Metadata dictionary. If parsing is successful, "description" entry
            is removed and parsing result is added.
        """
        with contextlib.suppress(Exception):
            yaml_md = yaml.safe_load(meta["description"])
            # YAML could be anything: plain string, list, …
            if isinstance(yaml_md, dict):
                meta.pop("description")
                meta.update(yaml_md)

    def _get_single_frame(self, real_t: int, **kwargs) -> Image:
        """Get a single frame and set extra metadata

        Parameters
        ----------
        real_t
            Real frame index (i.e., w.r.t original file)
        **kwargs
            Additional keyword arguments to pass to the imageio plugin's
            ``read()`` method.

        Returns
        -------
        Image data.
        """
        if self._is_tiff:
            ret = self._reader.read(index=..., page=real_t, **kwargs)
        else:
            ret = self._reader.read(index=real_t, **kwargs)
        ret = ret.view(Image)
        ret.frame_no = real_t
        return ret

    def get_data(self, t: int, **kwargs) -> Image:
        """Get a single frame

        Parameters
        ----------
        t
            Frame number
        **kwargs
            Additional keyword arguments to pass to the imageio plugin's
            ``read()`` method.

        Returns
        -------
        Image data. This has a `frame_no` attribute holding the original frame
        number.
        """
        return self._get_single_frame(self._resolve_index(t), **kwargs)

    def get_metadata(self, t: Optional[int] = None) -> Dict:
        """Get metadata for a frame

        If ``t`` is not given, return the global metadata.

        Parameters
        ----------
        t
            Frame number

        Returns
        -------
        Metadata dictionary. A `"frame_no"` entry with the original frame
        number (i.e., before slicing the sequence) is also added.
        """
        real_t = None if t is None else self._resolve_index(t)
        if self._is_tiff:
            ret = self._reader.metadata(index=..., page=real_t)
        else:
            ret = self._reader.metadata(index=real_t)
        self._parse_yaml_description(ret)
        if real_t is not None:
            ret["frame_no"] = real_t
        return ret

    def get_meta_data(self, t: Optional[int] = None) -> Dict:
        """Alias for :py:func:`get_metadata`"""
        return self.get_metadata(t)

    @overload
    def __getitem__(self, t: int) -> Image: ...

    def __getitem__(self, t: Union[slice, Sequence[int], Sequence[bool]]
                    ) -> "ImageSequence":
        """Implement indexing and lazy slicing

        Parameters
        ----------
        t
            Frame number(s)

        Returns
        -------
        If t is a single index, return the corresponding image data. This has a
        `frame_no` attribute holding the original frame number.
        Otherwise, return a copy of ``self`` describing a substack.
        """
        t = self._resolve_index(t)
        if isinstance(t, np.ndarray):
            ret = copy.copy(self)
            ret._indices = t
            ret._is_slice = True
            ret._len = len(t)
            return ret
        # Assume t is a number
        return self._get_single_frame(t)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_trace):
        self.close()

    def __len__(self):
        return self._len

    @property
    def closed(self) -> bool:
        """True if the file is currently closed."""
        return self._closed
