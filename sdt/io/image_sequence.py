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
import imageio.v3

with contextlib.suppress(ImportError):
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

    def __array_wrap__(self, array, context=None):
        # This way numpy functions such as np.min() return a scalar, not a
        # zero-dimensional array.
        # See https://stackoverflow.com/a/19720866
        if array.ndim == 0:
            return array[()]
        return super().__array_wrap__(array, context)


def _parse_yaml_description(meta: Mapping):
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


class BaseImageSequence:
    """Base class for :py:class:`ImageSequence` and :py:class:`MultiImageSequence`"""

    _slicerator_flag = True  # Make it work with slicerator

    def __init__(self):
        self._indices = None
        self._len = 0
        self._is_slice = False
        self._closed = True

    def open(self) -> "BaseImageSequence":
        """Open the file

        Returns
        -------
        self
        """
        return self

    def close(self):
        """Close the file"""
        pass

    @overload
    def _resolve_index(self, t: int) -> int:
        ...

    def _resolve_index(
        self, t: Union[slice, Sequence[int], Sequence[bool]]
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
                raise IndexError("slicing impossible for sequences of unknown length")
        if isinstance(t, slice):
            t = np.arange(*t.indices(len(self)))
        if isinstance(t, collections.abc.Iterable):
            t = np.asarray(t)
            if np.issubdtype(t.dtype, np.bool_):
                if len(t) != len(self):
                    raise IndexError(
                        f"boolean index did not match; stack length is {len(self)} "
                        f"but corresponding boolean length is {len(t)}"
                    )
                t = np.nonzero(t)[0]
            else:
                t[t < 0] += len(self)
            oob = np.nonzero((t < 0) | (t > len(self) - 1))[0]
            if oob.size:
                raise IndexError(
                    f"index {oob[0]} is out of bounds for stack of length {len(self)}"
                )
        else:
            # Treat scalar t separately as this is much faster
            if t < 0:
                t += len(self)
            if t < 0 or t > len(self) - 1:
                raise IndexError(
                    f"index {t} is out of bounds for stack of length {len(self)}"
                )
        if self._indices is None:
            return t
        return self._indices[t]

    def _load_single_frame(self, real_t: int, **kwargs) -> np.ndarray:
        """Load a single frame

        Implement this in a subclass.

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
        return NotImplementedError("implement in subclass")

    def _finalize_frame(self, data: np.ndarray, real_t: int) -> Image:
        """Finalize pixel data array before returning

        Cast to :py:class:`Image`, add original frame number.

        Parameters
        ----------
        real_t
            Real frame index (i.e., w.r.t original file)
        data
            Image array

        Returns
        -------
        Image data.
        """
        ret = data.view(Image)
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
        real_t = int(self._resolve_index(t))
        ret = self._load_single_frame(real_t, **kwargs)
        return self._finalize_frame(ret, real_t)

    @overload
    def __getitem__(self, t: int) -> Image:
        ...

    def __getitem__(
        self, t: Union[slice, Sequence[int], Sequence[bool]]
    ) -> "BaseImageSequence":
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
            return ret
        # Assume t is a number
        t = int(t)
        ret = self._load_single_frame(t)
        return self._finalize_frame(ret, t)

    def _load_metadata(self, real_t: Optional[int]) -> Dict:
        """Get metadata for a frame

        If ``t`` is not given, return the global metadata. Implement in subclass.

        Parameters
        ----------
        real_t
            Real frame index (i.e., w.r.t original file)

        Returns
        -------
        Metadata dictionary.
        """
        raise NotImplementedError("implement in subclass")

    def _finalize_metadata(self, data: Dict, real_t: Optional[int]) -> Dict:
        """Finalize metadata dict before returning

        Tries to parse YAML in ``"description"`` tag, adds original frame number as
        ``"frame_no"`` tag.

        Parameters
        ----------
        data
            Metadata dict
        real_t
            Real frame index (i.e., w.r.t original file)

        Returns
        -------
        Finalized metadata dictionary.
        """
        _parse_yaml_description(data)
        if real_t is not None:
            data["frame_no"] = real_t
        return data

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
        real_t = None if t is None else int(self._resolve_index(t))
        ret = self._load_metadata(real_t)
        return self._finalize_metadata(ret, real_t)

    def get_meta_data(self, t: Optional[int] = None) -> Dict:
        """Alias for :py:meth:`get_metadata`"""
        return self.get_metadata(t)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_trace):
        self.close()

    def __len__(self):
        if self._indices is None:
            return self._len
        return len(self._indices)

    @property
    def is_slice(self) -> bool:
        """Whether this instance is the result of slicing another instance and therefore
        cannot be opened or closed.
        """
        return self._is_slice

    @property
    def closed(self) -> bool:
        """True if the file is currently closed."""
        return self._closed


class ImageSequence(BaseImageSequence):
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
    >>> stack.close()
    """

    uri: Union[str, Path, bytes, IO]
    """File or file location or data to read from."""
    reader_args: Mapping
    """Keyword arguments passed to :py:func:`imageio.v3.imopen` when opening
    file.
    """

    def __init__(self, uri: Union[str, Path, bytes, IO], **kwargs):
        """Parameters
        ----------
        uri
            File or file location or data to read from.
        **kwargs
            Keyword arguments passed to :py:func:`imageio.v3.imopen` when
            opening the file.
        """
        super().__init__()

        self.uri = uri
        self.reader_args = kwargs
        self._reader = None
        self._is_tiff = False

    def open(self) -> "ImageSequence":
        """Open the file

        Returns
        -------
        self
        """
        if self.is_slice:
            raise RuntimeError("Cannot open sliced sequence.")
        if not self.closed:
            raise IOError(f"{self.uri} already open.")

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
        if self.is_slice:
            raise RuntimeError("Cannot close sliced sequence.")
        self._len = 0
        self._closed = True
        self._reader.close()

    def _load_single_frame(self, real_t: int, **kwargs) -> np.ndarray:
        if self._is_tiff:
            return self._reader.read(index=..., page=real_t, **kwargs)
        return self._reader.read(index=real_t, **kwargs)

    def _load_metadata(self, real_t: Optional[int]) -> Dict:
        if self._is_tiff:
            return self._reader.metadata(index=..., page=real_t)
        return self._reader.metadata(index=real_t)


class MultiImageSequence(BaseImageSequence):
    """Sliceable, lazy-loading interface to multiple image files

    Similar to :py:class:`ImageSequence`, but each frame is loaded from a
    different single-image file.

    Examples
    --------

    Load 3rd frame:

    >>> with MultiImageSequence(["f1.tif", "f1.tif", "f3.tif", "f4.tif"]) as stack:
    ...     img = stack[3]

    Use fancy indexing to create substacks:

    >>> stack = MultiImageSequence(["f1.tif", "f1.tif", "f3.tif", "f4.tif"]).open()
    >>> len(stack)
    4
    >>> substack1 = stack[1::2]  # Slice, will not load any data
    >>> len(substack2)
    2
    >>> np.all(substack2[1] == stack[3])  # Actually load data using int index
    True
    >>> substack2 = stack[[2, 3]]  # Create lazy substack using list of indices
    >>> substack3 = stack[[True, False] * len(stack) // 2]  # or boolean index
    """

    uris: Sequence[Union[str, Path, bytes, IO]]
    """Files or file locations or data to read from."""
    reader_args: Mapping
    """Keyword arguments passed to :py:func:`imageio.v3.imread` when reading
    a file.
    """

    def __init__(self, uris: Sequence[Union[str, Path, bytes, IO]], **kwargs):
        """Parameters
        ----------
        uris
            Files or file locations or data to read from.
        **kwargs
            Keyword arguments passed to :py:func:`imageio.v3.imread` when
            reading a file.
        """
        super().__init__()

        self.uris = uris
        self.reader_args = kwargs

    def _load_single_frame(self, real_t: int, **kwargs) -> np.ndarray:
        return imageio.v3.imread(self.uris[real_t], index=0, **kwargs)

    def _load_metadata(self, real_t: Optional[int]) -> Dict:
        return imageio.v3.immeta(self.uris[real_t or 0], index=0, **self.reader_args)

    def __len__(self):
        if self._indices is None:
            return len(self.uris)
        return len(self._indices)
