# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Main window of the peak locator

Contains MainWindow (derived from QMainWindow)
"""
import os
import collections
import contextlib

import yaml
import numpy as np
import pandas as pd

from PyQt5.QtGui import QPolygonF, QCursor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox,
                             QDockWidget, QFileDialog)
from PyQt5.QtCore import (Qt, QSettings, QModelIndex, QPersistentModelIndex,
                          QMetaObject, QPointF, pyqtSlot, pyqtProperty)

from ..widgets import micro_view
from . import locate_options
from . import file_chooser
from .file_chooser import FileListModel
from . import locate_filter
from . import locate_saver
from . import batch_progress
from . import workers
from ....io import ImageSequence, save, load


def yaml_dict_representer(dumper, data):
    """Represent an :py:class:`OrderedDict` using PyYAML

    This is to be passed as
    ``yaml.add_representer(collections.OrderedDict, yaml_dict_representer)``
    """
    return dumper.represent_dict(data.items())


def yaml_qpolygon_representer(dumper, data):
    """Represent a :py:class:`QPolygon` using PyYAML

    This is to be passed as
    ``yaml.add_representer(QPolygonF, yaml_qpolygon_representer)``
    """
    point_list = [[p.x(), p.y()] for p in data]
    return dumper.represent_list(point_list)


yaml.add_representer(collections.OrderedDict, yaml_dict_representer)
yaml.add_representer(QPolygonF, yaml_qpolygon_representer)


def determine_filename(imgname, fmt):
    """Determine name of data file from name of image file

    Depending on the format (``fmt``) parameter, return the name of the data
    file corresponding to the image file ``imgname``.

    Parameters
    ----------
    imgname : str
        Name of the image file
    fmt : {hdf5, particle_tracker, pkc, pks, yaml}
        Data file format

    Returns
    -------
    dirname : str
        directory of the data file
    fname : str
        name of the data file relative to ``dir``

    Examples
    --------
    >>> determine_filename("tests/beads.tiff", "hdf5")
    ('tests', 'beads.h5')
    >>> determine_filename("tests/beads.tiff", "particle_tracker")
    ('tests/Analysis_particle_tracking', 'beads_positions.mat')
    """
    imgdir = os.path.dirname(imgname)
    imgbase = os.path.basename(imgname)
    imgbase_no_ext = os.path.splitext(imgbase)[0]

    if fmt == "particle_tracker":
        dirname = os.path.join(imgdir, "Analysis_particle_tracking")
        fname = imgbase_no_ext + "_positions.mat"
        return dirname, fname

    if fmt == "hdf5":
        ext = "h5"
    elif fmt in ("pks", "pkc", "yaml"):
        ext = fmt
    else:
        raise ValueError("Unsupported format: " + fmt)

    return imgdir, imgbase_no_ext + os.extsep + ext


class MainWindow(QMainWindow):
    """Main window of the locator app"""
    __clsName = "LocatorMainWindow"

    def _tr(self, string):
        """Translate the string using :py:func:`QApplication.translate`"""
        return QApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create viewer widget
        self._viewer = micro_view.MicroViewWidget()
        self._viewer.setObjectName("viewer")
        self._viewer.showLocalizationsChanged.connect(self.setShowPreview)

        # Create dock widgets
        fileChooser = file_chooser.FileChooser()
        fileChooser.selected.connect(self.open)
        self._fileModel = fileChooser.model()
        self._fileModel.rowsRemoved.connect(self._checkFileList)
        self._fileDock = QDockWidget(self.tr("File selection"), self)
        self._fileDock.setObjectName("fileDock")
        self._fileDock.setWidget(fileChooser)

        self._locOptionsWidget = locate_options.Container()
        self._locOptionsWidget.setObjectName("locOptionsWidget")
        self._locOptionsDock = QDockWidget(self.tr("Localization options"),
                                           self)
        self._locOptionsDock.setObjectName("locOptionsDock")
        self._locOptionsDock.setWidget(self._locOptionsWidget)

        self._locFilterWidget = locate_filter.FilterWidget()
        self._locFilterWidget.filterChanged.connect(self._filterLocalizations)
        self._locFilterDock = QDockWidget(self.tr("Localization filter"), self)
        self._locFilterDock.setObjectName("locFilterDock")
        self._locFilterDock.setWidget(self._locFilterWidget)

        locSaveWidget = locate_saver.SaveWidget()
        self._locSaveDock = QDockWidget(self.tr("Save localizations"), self)
        self._locSaveDock.setObjectName("locSaveDock")
        self._locSaveDock.setWidget(locSaveWidget)

        for d in (self._fileDock, self._locOptionsDock, self._locFilterDock,
                  self._locSaveDock):
            d.setFeatures(d.features() & ~QDockWidget.DockWidgetClosable)
            self.addDockWidget(Qt.LeftDockWidgetArea, d)
        self.setDockOptions(self.dockOptions() | QMainWindow.VerticalTabs)

        self.setCentralWidget(self._viewer)

        # set up the preview worker
        self._previewWorker = workers.PreviewWorker(self)
        self._locOptionsWidget.optionsChanged.connect(
            self._makePreviewWorkerWork)
        self._viewer.currentFrameChanged.connect(self._makePreviewWorkerWork)
        self._previewWorker.finished.connect(self._previewFinished)
        self._previewWorker.error.connect(self._previewError)
        self._previewWorker.enabled = True
        self._previewWorker.busyChanged.connect(self._setBusyCursor)

        # set up the batch worker
        self._batchWorker = workers.BatchWorker(self)
        self._batchWorker.fileFinished.connect(self._locateRunnerFinished)
        self._batchWorker.fileError.connect(self._locateRunnerError)

        # batch progress dialog
        self._progressDialog = batch_progress.BatchProgressDialog(self)

        self._batchWorker.fileFinished.connect(
            self._progressDialog.increaseValue)
        self._batchWorker.fileError.connect(
            self._progressDialog.increaseValue)
        self._progressDialog.canceled.connect(self._batchWorker.stop)

        # Some things to keep track of
        self._currentFile = QPersistentModelIndex()
        self._currentLocData = pd.DataFrame(columns=["x", "y"])
        self._roiPolygon = QPolygonF()

        # load settings and restore window geometry
        settings = QSettings("sdt", "locator")
        v = settings.value("MainWindow/geometry")
        if v is not None:
            self.restoreGeometry(v)
        v = settings.value("MainWindow/state")
        if v is not None:
            self.restoreState(v)

        # restore enable/disable preview
        show = settings.value("Viewer/showPreview", True, type=bool)
        self._viewer.showLocalizations = show

        QMetaObject.connectSlotsByName(self)

    @pyqtSlot(QModelIndex)
    @pyqtSlot(str)
    def open(self, file):
        """Open an image sequence

        Open an image sequence from a file and load it into the viewer widget.
        This is a PyQt slot.

        Parameters
        ----------
        file : QModelIndex or str
            Index of the entry in a `FileListModel` or the file name. If
            `file` is a string, add the file to the `FileListModel` and show
            it.
        """
        if isinstance(file, str):
            file = self._fileModel.addItem(file)

        filename = file.data(FileListModel.FileNameRole)
        try:
            ims = ImageSequence(filename).open()
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error opening image"),
                                 self.tr(str(e)))
            ims = None
            file = None

        if isinstance(ims, collections.abc.Iterable) and not len(ims):
            QMessageBox.critical(self, self.tr(""),
                                 self.tr("Empty image"))
            ims = None
            file = None

        self._currentFile = QPersistentModelIndex(file)
        self._viewer.setImageSequence(ims)
        self._viewer.zoomFit()
        # also the options widget needs to know how many frames there are
        self._locOptionsWidget.numFrames = (0 if (ims is None)
                                            else len(ims))
        if file is not None:
            self.setWindowTitle("locator - {}".format(filename))
        else:
            self.setWindowTitle("locator")

    @pyqtSlot(int)
    def on_viewer_frameReadError(self, frameno):
        """Slot getting called when a frame could not be read"""
        QMessageBox.critical(
            self, self.tr("Read Error"),
            self.tr("Could not read frame number {}".format(frameno + 1)))

    @pyqtSlot()
    def _makePreviewWorkerWork(self):
        """Called when something happens that requires a new preview

        e. g. a new frame is displayed..
        """
        if (not self._currentFile.isValid() or
                not self._viewer.showLocalizations):
            return

        cur_method = self._locOptionsWidget.method
        cur_opts = self._locOptionsWidget.options

        file_method = self._currentFile.data(FileListModel.LocMethodRole)
        file_opts = self._currentFile.data(FileListModel.LocOptionsRole)
        file_roi = self._currentFile.data(FileListModel.ROIRole)
        file_frameRange = self._currentFile.data(FileListModel.FrameRangeRole)

        curFrame = self._viewer.getCurrentFrame()
        curFrameNo = self._viewer.currentFrameNumber

        if (file_method == cur_method and file_opts == cur_opts and
                (self._roiPolygon == file_roi or cur_method == "load file") and
                file_frameRange[0] <= curFrameNo < file_frameRange[1]):
            data = self._currentFile.data(FileListModel.LocDataRole)
            data = data[data["frame"] == curFrameNo]

            self._currentLocData = data
            self._locFilterWidget.setVariables(
                data.columns.values.tolist())
            self._filterLocalizations()

            return

        if cur_method == "load file":
            fname = os.path.join(*determine_filename(
                self._currentFile.data(FileListModel.FileNameRole),
                cur_opts["fmt"]))
            # TODO: Move to worker thread
            try:
                data = load(fname)  # sdt.io.load
            except Exception:
                data = pd.DataFrame()

            modelIdx = QModelIndex(self._currentFile)
            self._fileModel.setData(modelIdx, data,
                                    FileListModel.LocDataRole)
            self._fileModel.setData(modelIdx, cur_opts,
                                    FileListModel.LocOptionsRole)
            self._fileModel.setData(modelIdx, cur_method,
                                    FileListModel.LocMethodRole)
            self._fileModel.setData(modelIdx, (-np.inf, np.inf),
                                    FileListModel.FrameRangeRole)
            # call recursively to update viewer
            self._makePreviewWorkerWork()
            return

        self._previewWorker.processImage(curFrame, curFrameNo, cur_opts,
                                         cur_method, self._roiPolygon)

    def closeEvent(self, event):
        """Window is closed, save state"""
        settings = QSettings("sdt", "locator")
        settings.setValue("MainWindow/geometry", self.saveGeometry())
        settings.setValue("MainWindow/state", self.saveState())
        settings.setValue("Viewer/showPreview", self._viewer.showLocalizations)
        super().closeEvent(event)

    @pyqtSlot()
    def _checkFileList(self):
        """If currently previewed file was removed from list, remove preview

        This gets triggered by the file list model's ``rowsRemoved`` signal
        """
        if not self._currentFile.isValid():
            self._locOptionsWidget.numFrames = 0
            self._viewer.setImageSequence(None)

    @pyqtSlot(bool)
    def _setBusyCursor(self, busy):
        if busy:
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        else:
            QApplication.restoreOverrideCursor()

    @pyqtSlot(pd.DataFrame)
    def _previewFinished(self, data):
        """The preview worker finished

        Update the filter widget with the data column names and filter the
        data using ``_filterLocalizations``.
        """
        self._currentLocData = data
        self._locFilterWidget.setVariables(data.columns.values.tolist())
        self._filterLocalizations()

    @pyqtSlot(Exception)
    def _previewError(self, err):
        QMessageBox.critical(
            self, self.tr("Localization error"),
            self.tr("Failed to create preview.\n\n{}").format(err))

    @pyqtSlot()
    def _filterLocalizations(self):
        """Set good/bad localizations in the viewer

        Anything that passes the filter (from the filter widget) and is in the
        ROI polygon is considered a good localization. Anything that does not
        pass the filter and is in the ROI is considered bad. Anything outside
        the ROI is not considered at all.

        Call the viewer's ``setLocalizationData`` accordingly.
        """
        filterFunc = self._locFilterWidget.getFilter()
        good = filterFunc(self._currentLocData)
        self._viewer.setLocalizationData(self._currentLocData[good],
                                         self._currentLocData[~good])

    @pyqtSlot(QPolygonF)
    def on_viewer_roiChanged(self, roi):
        """Update ROI polygon and filter localizations"""
        self._roiPolygon = roi
        self._makePreviewWorkerWork()

    def setShowPreview(self, show):
        if (self._viewer.showLocalizations == show and
                self._previewWorker.enabled == show):
            return

        self._previewWorker.enabled = show
        self._viewer.showLocalizations = show
        self._makePreviewWorkerWork()

    @pyqtProperty(bool, fset=setShowPreview,
                  doc="Show preview of localizations")
    def showPreview(self):
        return self._viewer.showLocalizations

    def _saveMetadata(self, fname):
        """Save metadata to YAML

        This saves currently selected algorithm, options, ROI, and filter to
        the file specified by ``fname``.

        Parameters
        ----------
        fname : str
            Name of the output file
        """
        metadata = collections.OrderedDict()
        metadata["algorithm"] = self._locOptionsWidget.method
        metadata["options"] = self._locOptionsWidget.options
        metadata["roi"] = self._roiPolygon
        metadata["filter"] = self._locFilterWidget.filterString
        with open(fname, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

    @pyqtSlot()
    def on_locOptionsWidget_save(self):
        # save options
        fname, _ = QFileDialog.getSaveFileName(
            self, self.tr("Save file"), "",
            self.tr("YAML data (*.yaml)") + ";;" +
            self.tr("All files (*)"))
        if not fname:
            # cancelled
            return

        try:
            self._saveMetadata(fname)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error writing to file"),
                                 self.tr(str(e)))

    @pyqtSlot()
    def on_locOptionsWidget_load(self):
        # load options
        fname, _ = QFileDialog.getOpenFileName(
            self, self.tr("Save file"), "",
            self.tr("YAML data (*.yaml)") + ";;" +
            self.tr("All files (*)"))
        if not fname:
            # cancelled
            return

        try:
            with open(fname) as f:
                md = yaml.safe_load(f)
            if not isinstance(md, dict):
                raise RuntimeError()
        except Exception:
            QMessageBox.critical(self, self.tr("Error loading settings"),
                                 self.tr("Error reading file."))
            return

        algo = md.get("algorithm")
        if isinstance(algo, str):
            try:
                self._locOptionsWidget.method = algo
            except Exception:
                QMessageBox.critical(self, self.tr("Error loading settings"),
                                     self.tr("Unsupported algorithm"))
                return
            opts = md.get("options")
            self._locOptionsWidget.options = opts

        filt = md.get("filter")
        if isinstance(filt, str):
            self._locFilterWidget.filterString = filt

        roi = md.get("roi")
        with contextlib.suppress(Exception):
            vert = []
            for x, y in roi:
                vert.append(QPointF(x, y))
            r = QPolygonF(vert)
            self._viewer.roi = r
            self._roiPolygon = r

    @pyqtSlot(str)
    def on_locateSaveWidget_locateAndSave(self, format):
        """Locate all features in all files and save the data and metadata"""
        # TODO: check if current localizations are up-to-date
        # only run locate if not
        self._progressDialog.value = 0
        self._progressDialog.maximum = self._fileModel.rowCount()
        self._progressDialog.show()

        optWid = self._locOptionsWidget
        self._batchWorker.processFiles(self._fileModel, optWid.frameRange,
                                       optWid.options, optWid.method,
                                       self._roiPolygon)

    @pyqtSlot(int, pd.DataFrame, dict)
    def _locateRunnerFinished(self, idx, data, options):
        """A LocateRunner finished locating all peaks in a sequence

        Save data and metadata to a file with the same name as the image file,
        except for the extension.

        Parameters
        ----------
        index : QModelIndex
            Index of the file in the file list model
        data : pandas.DataFrame
            Localization data
        options : dict
            Options used for locating peaks
        """
        optsWidget = self._locOptionsWidget
        index = self._fileModel.index(idx)
        self._fileModel.setData(index, data, FileListModel.LocDataRole)
        self._fileModel.setData(index, options, FileListModel.LocOptionsRole)
        self._fileModel.setData(index, optsWidget.method,
                                FileListModel.LocMethodRole)
        self._fileModel.setData(index, self._roiPolygon, FileListModel.ROIRole)
        self._fileModel.setData(index, optsWidget.frameRange,
                                FileListModel.FrameRangeRole)
        saveFormat = self._locSaveDock.widget().getFormat()

        saveFileName = self._fileModel.data(index, FileListModel.FileNameRole)

        metaFileName = os.path.join(*determine_filename(saveFileName, "yaml"))
        fdir, fname = determine_filename(saveFileName, saveFormat)
        saveFileName = os.path.join(fdir, fname)
        os.makedirs(fdir, exist_ok=True)

        filterFunc = self._locFilterWidget.getFilter()
        data = data[filterFunc(data)]

        save(saveFileName, data)  # sdt.data.save
        self._saveMetadata(metaFileName)

    @pyqtSlot(int, Exception)
    def _locateRunnerError(self, idx, e):
        """A LocateRunner encountered an error

        Show an error message box.

        Parameters
        ----------
        index : QModelIndex
            Index (in the file list model) of the file that caused the error
        """
        index = self._fileModel.index(idx)
        QMessageBox.critical(
            self, self.tr("Localization error"),
            self.tr("Failed to locate features in {}\n\n{}".format(
                index.data(file_chooser.FileListModel.FileNameRole), str(e))))
