# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from PySide2 import QtCore, QtQml, QtQuick
import numpy as np
import pims

from .item_models import DictListModel


class ImageSelectorModule(QtQuick.QQuickItem):
    """Select an image (sequence) and frame

    The image (sequences) to choose from can be set via the :py:attr:`images`
    property. The image array of the selected frame is exposed via the
    :py:attr:`output` property.
    """
    def __init__(self, parent: Optional[QtCore.QObject] = None):
        """Parameters
        ---------
        parent
            Parent QObject
        """
        super().__init__(parent)
        self._cur_image = None
        self._cur_image_opened = False
        self._output = None
        self._file_list = DictListModel(roles=["display", "images"],
                                        default_role="display")
        self.images = []

    @staticmethod
    def _validate_images(images: Union[List, Dict]) -> List:
        """Turn a dict into a list of (key, value) pairs

        Parameters
        ----------
        images
            Dict mapping name -> sequence or image list

        Returns
        -------
            Image list
        """
        if len(images) == 0:
            return []
        if isinstance(images, dict):
            return list(images.items())
        return images

    @staticmethod
    def _make_file_list(images: Iterable[Union[Tuple[str, Any],
                                               Path, str, Any]]
                        ) -> List[Dict]:
        """Create data list suitable for :py:class:`DictListModel`

        Parameters
        ----------
        images
            If an entry is a tuple, the first tuple item is interpreted as a
            name to display and the second as image sequence data.
            :py:class:`str` and :py:class:`Path` entries will be interpreted
            as file names, any other types as image sequence data.


        Returns
        -------
        Each list entry has a ``"display"`` key which corresponds to the
        string displayed in the image sequence selection widget and an
        ``"images"`` key which contains either a :py:class:`Path` representing
        the image file to read or an image sequence type such as
        :py:class:`pims.FramesSequence` or a 3D :py:class:`numpy.ndarray`.

        The ``"display"`` value is taken directly from the corresponding entry
        in `images` if it was a tuple, derived from the file name or just an
        increasing number.
        """
        if len(images) == 0:
            return []

        n_figures = int(math.log10(len(images)))
        generic_key_pattern = "<{{:0{}}}>".format(n_figures)

        files = []
        for n, img in enumerate(images):
            if isinstance(img, tuple):
                files.append({"display": img[0], "images": img[1]})
                continue
            if isinstance(img, str):
                img = Path(img)
            if isinstance(img, Path):
                files.append(
                    {"display": "{} ({})".format(img.name, str(img.parent)),
                     "images": img})
                continue
            files.append({"display": generic_key_pattern.format(n),
                          "images": img})
        return files

    imagesChanged = QtCore.Signal("QVariantList")
    """:py:attr:`images` was changed."""

    # Use QVariantList for automatic conversion to Python list by PySide2
    @QtCore.Property("QVariantList", notify=imagesChanged)
    def images(self) -> List:
        """Image sequences to choose from

        Entries can be tuples of (name, sequence data), file names as
        :py:class:`str` or :py:class:`pathlib.Path`, or sequence data such as
        3D :py:class:`numpy.ndarray` or :py:class:`pims.FramesSequence`.

        If the property is set with a dict, it will be automatically converted
        to a list of (key, value) tuples.
        """
        return self._images

    @images.setter
    def setImages(self, img: Union[Dict[str, Any],
                                   Iterable[Union[Tuple[str, Any],
                                                  Path, str, Any]]]):
        self._images = self._validate_images(img)
        self._file_list.resetWithData(self._make_file_list(img))
        self.imagesChanged.emit(self._images)

    outputChanged = QtCore.Signal(np.ndarray)
    """:py:attr:`output` has changed."""

    @QtCore.Property(np.ndarray, notify=outputChanged)
    def output(self) -> np.ndarray:
        """Selected frame from selected image sequence"""
        return self._output

    @QtCore.Property(QtCore.QObject, constant=True)
    def _qmlFileList(self) -> DictListModel:
        """Expose the file list model to QML"""
        return self._file_list

    @QtCore.Slot(int)
    def _fileChanged(self, index: int):
        """Callback upon change of currently selected file

        index
            Index of currently selected file w.r.t. to :py:attr:`images`
        """
        if self._cur_image_opened:
            self._cur_image.close()
            self._cur_image_opened = False

        if index < 0:
            # No file selected
            self._cur_image = None
            self._output = None
            self.outputChanged.emit(None)
            return

        img = self.images[index]
        if isinstance(img, tuple):
            # TODO: What if there is a tuple of images instead of
            # (key, value)?
            img = img[1]
        if isinstance(img, np.ndarray) and img.ndim == 2:
            # Single image
            img = img[None, ...]
        elif isinstance(img, (str, Path)):
            # Open…
            img = pims.open(str(img))
            self._cur_image_opened = True

        self._cur_image = img
        self._qmlNFramesChanged.emit(self._qmlNFrames)

    _qmlNFramesChanged = QtCore.Signal(int)

    @QtCore.Property(int, notify=_qmlNFramesChanged)
    def _qmlNFrames(self) -> int:
        """Expose the number of frames of current sequnece to QML"""
        if self._cur_image is None:
            return 0
        return len(self._cur_image)

    @QtCore.Slot(int)
    def _frameChanged(self, index: int):
        """Callback upon change of currently selected frame

        index
            Index of currently selected frame
        """
        self._output = self._cur_image[index]
        self.outputChanged.emit(self._output)


QtQml.qmlRegisterType(ImageSelectorModule, "SdtGui.Impl", 1, 0,
                      "ImageSelectorImpl")
