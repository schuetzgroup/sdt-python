# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import operator
from typing import Any, Callable, Iterable, Mapping, Optional

from PyQt5 import QtCore, QtQml
import numpy as np
import pandas as pd

from .. import loc
from .option_chooser import OptionChooser
from .qml_wrapper import SimpleQtProperty


class LocOptions(OptionChooser):
    """Set localization options and find localizations in an image

    The typical use-case for this is to find the proper options for feature
    localization. When setting options in the GUI, the localization algorithm
    is run on the :py:attr:`input` image. The result is exposed via the
    :py:attr:`locData` property, which can be used e.g. by
    :py:class:`LocDisplay` in :py:class:`ImageDisplay`'s
    ``overlays`` for visual feedback.
    """
    def __init__(self, parent: Optional[QtCore.QObject] = None):
        """Parameters
        ----------
        parent
            Parent QObject
        """
        super().__init__(argProperties=["image", "algorithm", "options"],
                         resultProperties="locData", parent=parent)
        self._options = {}
        self._image = None
        self._algorithm = "daostorm_3d"
        self._locData = None

    # Properties
    image = SimpleQtProperty("QVariant", comp=operator.is_)
    """Image data (ndarray) to find preview localizations, which are exposed
    via the :py:attr:`locData` property.
    """
    algorithm = SimpleQtProperty(str)
    """Localization algorithm to use. Currently ``"daostorm_3d"`` and
    ``"cg"`` are supported. See also :py:mod:`sdt.loc`.
    """
    options = SimpleQtProperty("QVariantMap")
    """Options to the localization algorithm. See
    :py:func:`sdt.loc.daostorm_3d.locate` and :py:func:`sdt.loc.cg.locate`.
    """
    locData = SimpleQtProperty("QVariant", readOnly=True)
    """Result of running the localization algorithm on the :py:attr:`input`
    image with :py:attr:`options`.
    """

    @QtCore.pyqtSlot(result="QVariant")
    def getBatchFunc(self) -> Callable[[Iterable[np.ndarray]], pd.DataFrame]:
        """Get a function for batch localization using current settings

        Returns
        -------
        A version of the currently selected algorithm's ``batch()`` function
        (e.g., :py:func:`loc.daostorm_3d.batch` with options set according
        to the GUI. The only remaining argument is the image sequence.
        """
        func = getattr(loc, self.algorithm).batch
        return functools.partial(func, **self.options)

    @staticmethod
    def workerFunc(image: np.ndarray, algorithm: str,
                   options: Mapping[str, Any]) -> pd.DataFrame:
        """Run localization algorithm

        This is executed in the worker process.

        Paramters
        ---------
        image
            Image to find localizations in.
        algorithm
            Name of the algorithm, i.e., name of the submodule in
            :py:mod:`sdt.loc`.
        options
            Arguments passed to the ``locate`` function of the submodule
            specified by `algorithm`

        Returns
        -------
        Localization data
        """
        if image is None:
            return None
        algo_mod = getattr(loc, algorithm)
        result = algo_mod.locate(image, **options)
        return result


QtQml.qmlRegisterType(LocOptions, "SdtGui.Templates", 0, 2, "LocOptions")
