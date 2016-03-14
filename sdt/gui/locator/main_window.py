"""Main window of the peak locator

Contains MainWindow (derived from QMainWindow), which is launched if
this is called as a script (__main__)
"""
import os
import collections
import types

import yaml
import numpy as np
import pandas as pd
import pims

from qtpy.QtGui import (QIcon, QPolygonF, QCursor)
from qtpy.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog,
                            QToolBar, QMessageBox, QSplitter, QToolBox,
                            QDockWidget, QWidget, QLabel, QProgressDialog)
from qtpy.QtCore import (pyqtSignal, pyqtSlot, Qt, QObject, QSettings,
                         QModelIndex, QPersistentModelIndex, QMetaObject,
                         QPointF)

from ..widgets import micro_view
from . import locate_options
from . import file_chooser
from .file_chooser import FileListModel
from . import locate_filter
from . import locate_saver
from . import batch_progress
from . import workers
from ...data import save, load


def yaml_dict_representer(dumper, data):
    """Represent an OrderedDict using PyYAML

    This is to be passed as
    `yaml.add_representer(collections.OrderedDict, yaml_dict_representer)`
    """
    return dumper.represent_dict(data.items())


yaml.add_representer(collections.OrderedDict, yaml_dict_representer)


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

    def tr(self, string):
        """Translate the string using ``QApplication.translate``"""
        return QApplication.translate(self.__clsName, string)

    def __init__(self, parent=None):
        """Constructor"""
        super().__init__(parent)

        # Create viewer widget
        self._viewer = micro_view.MicroViewWidget()
        self._viewer.setObjectName("viewer")

        # Create dock widgets
        fileChooser = file_chooser.FileChooser()
        fileChooser.selected.connect(self.open)
        self._fileModel = fileChooser.model()
        self._fileModel.rowsRemoved.connect(self._checkFileList)
        self._fileDock = QDockWidget(self.tr("File selection"), self)
        self._fileDock.setObjectName("fileDock")
        self._fileDock.setWidget(fileChooser)

        optionsWidget = locate_options.Container()
        self._locOptionsDock = QDockWidget(self.tr("Localization options"),
                                           self)
        self._locOptionsDock.setObjectName("locOptionsDock")
        self._locOptionsDock.setWidget(optionsWidget)

        filterWidget = locate_filter.FilterWidget()
        filterWidget.filterChanged.connect(self._filterLocalizations)
        self._locFilterDock = QDockWidget(self.tr("Localization filter"), self)
        self._locFilterDock.setObjectName("locFilterDock")
        self._locFilterDock.setWidget(filterWidget)

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
        optionsWidget.optionsChanged.connect(self._makePreviewWorkerWork)
        self._viewer.currentFrameChanged.connect(self._makePreviewWorkerWork)
        self._previewWorker.finished.connect(self._previewFinished)
        self._previewWorker.enabled = True
        self._previewWorker.busyChanged.connect(self._setBusyCursor)

        # set up the batch worker
        self._batchWorker = workers.BatchWorker(self)
        self._batchWorker.fileFinished.connect(self._locateRunnerFinished)
        self._batchWorker.fileError.connect(self._locateRunnerError)

        # batch progress dialog
        self._progressDialog = batch_progress.BatchProgressDialog(self)
        self._progressDialog.setWindowModality(Qt.WindowModal)

        def inc_progress():
            self._progressDialog.value += 1

        self._batchWorker.fileFinished.connect(inc_progress)
        self._batchWorker.fileError.connect(inc_progress)
        self._progressDialog.canceled.connect(self._batchWorker.stop)

        # Some things to keep track of
        self._currentFile = QPersistentModelIndex()
        self._currentLocData = pd.DataFrame()
        self._roiPolygon = QPolygonF()

        # load settings and restore window geometry
        settings = QSettings("sdt", "locator")
        v = settings.value("MainWindow/geometry")
        if v is not None:
            self.restoreGeometry(v)
        v = settings.value("MainWindow/state")
        if v is not None:
            self.restoreState(v)

        QMetaObject.connectSlotsByName(self)

    @pyqtSlot(QModelIndex)
    def open(self, file):
        """Open an image sequence

        Open an image sequence from a file and load it into the viewer widget.
        This is a PyQt slot.

        Parameters
        ----------
        file : QModelIndex
            Index of the entry in a ``FileListModel``
        """
        try:
            ims = pims.open(file.data(FileListModel.FileNameRole))
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error opening image"),
                                 self.tr(str(e)))
            ims = None
            file = None

        if isinstance(ims, collections.Iterable) and not len(ims):
            QMessageBox.critical(self, self.tr(""),
                                 self.tr("Empty image"))
            ims = None
            file = None

        self._currentFile = QPersistentModelIndex(file)
        self._viewer.setImageSequence(ims)
        self._viewer.zoomFit()
        # also the options widget needs to know how many frames there are
        self._locOptionsDock.widget().numFrames = (0 if (ims is None)
                                                   else len(ims))

    # This is emitted to tell the preview worker that there is something to
    # do (i. e. locate peaks in the current frame for preview)
    _workerSignal = pyqtSignal(np.ndarray, dict, types.FunctionType)

    @pyqtSlot(int)
    def on_viewer_frameReadError(self, frameno):
        """Slot getting called when a frame could not be read"""
        QMessageBox.critical(
            self, self.tr("Read Error"),
            self.tr("Could not read frame number {}".format(frameno + 1)))

    @pyqtSlot()
    def _makePreviewWorkerWork(self):
        """Called when something happens that requires a new preview

        E. g. a new frame is displayed. If the preview worker is already
        working, just tell it that there is yet another job to be done.
        Otherwise start the preview worker (by emitting self._workerSignal).
        """
        cur_method = self._locOptionsDock.widget().method
        cur_opts = self._locOptionsDock.widget().options

        if not self._currentFile.isValid():
            return

        file_method = self._currentFile.data(FileListModel.LocMethodRole)
        file_opts = self._currentFile.data(FileListModel.LocOptionsRole)

        curFrame = self._viewer.getCurrentFrame()

        if file_method == cur_method.name and file_opts == cur_opts:
            curFrameNo = self._viewer.currentFrameNumber

            data = self._currentFile.data(FileListModel.LocDataRole)
            data = data[data["frame"] == curFrameNo]

            self._currentLocData = data
            self._locFilterDock.widget().setVariables(
                data.columns.values.tolist())
            self._filterLocalizations()

            return

        if cur_method.name == "load file":
            fname = os.path.join(*determine_filename(
                self._currentFile.data(FileListModel.FileNameRole),
                cur_opts["fmt"]))
            # TODO: Move to worker thread
            try:
                data = load(fname)  # sdt.data.load
            except Exception:
                data = pd.DataFrame()

            modelIdx = QModelIndex(self._currentFile)
            self._fileModel.setData(modelIdx, data,
                                    FileListModel.LocDataRole)
            self._fileModel.setData(modelIdx, cur_opts,
                                    FileListModel.LocOptionsRole)
            self._fileModel.setData(modelIdx, cur_method.name,
                                    FileListModel.LocMethodRole)
            # call recursively to update viewer
            self._makeWorkerWork()
            return

        self._previewWorker.makePreview(curFrame, cur_opts, cur_method.locate)

    def closeEvent(self, event):
        """Window is closed, save state"""
        settings = QSettings("sdt", "locator")
        settings.setValue("MainWindow/geometry", self.saveGeometry())
        settings.setValue("MainWindow/state", self.saveState())
        super().closeEvent(event)

    @pyqtSlot()
    def _checkFileList(self):
        """If currently previewed file was removed from list, remove preview

        This gets triggered by the file list model's ``rowsRemoved`` signal
        """
        if not self._currentFile.isValid():
            self._locOptionsDock.widget().numFrames = 0
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
        self._locFilterDock.widget().setVariables(data.columns.values.tolist())
        self._filterLocalizations()

    def _applyRoi(self, data):
        """Select peaks in ROI

        Return a boolean vector with the same length as data whose entries are
        True or False depending on whether a data point is inside or outside
        the ROI polygon.
        """
        if len(self._roiPolygon) < 2:
            return np.ones((len(data),), dtype=bool)
        return np.apply_along_axis(
            lambda pos: self._roiPolygon.containsPoint(QPointF(*pos),
                                                       Qt.OddEvenFill),
            1, data[["x", "y"]])

    @pyqtSlot()
    def _filterLocalizations(self):
        """Set good/bad localizations in the viewer

        Anything that passes the filter (from the filter widget) and is in the
        ROI polygon is considered a good localization. Anything that does not
        pass the filter and is in the ROI is considered bad. Anything outside
        the ROI is not considered at all.

        Call the viewer's ``setLocalizationData`` accordingly.
        """
        filterFunc = self._locFilterDock.widget().getFilter()
        good = filterFunc(self._currentLocData)
        inRoi = self._applyRoi(self._currentLocData)
        self._viewer.setLocalizationData(self._currentLocData[good & inRoi],
                                         self._currentLocData[~good & inRoi])

    @pyqtSlot(QPolygonF)
    def on_viewer_roiChanged(self, roi):
        """Update ROI polygon and filter localizations"""
        self._roiPolygon = roi
        self._filterLocalizations()

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
        metadata["algorithm"] = \
            self._locOptionsDock.widget().method.name
        metadata["options"] = self._locOptionsDock.widget().options
        metadata["roi"] = [p for p in self._roiPolygon]
        metadata["filter"] = self._locFilterDock.widget().getFilterString()
        with open(fname, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

    @pyqtSlot(str)
    def on_locateSaveWidget_saveOptions(self, fname):
        """Only save metadata, do not locate"""
        try:
            self._saveMetadata(fname)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error writing to file"),
                                 self.tr(str(e)))

    @pyqtSlot(str)
    def on_locateSaveWidget_locateAndSave(self, format):
        """Locate all features in all files and save the data and metadata"""
        # TODO: check if current localizations are up-to-date
        # only run locate if not
        self._progressDialog.value = 0
        self._progressDialog.maximum = self._fileModel.rowCount()
        self._progressDialog.show()

        optWid = self._locOptionsDock.widget()
        self._batchWorker.processFiles(self._fileModel, optWid.frameRange,
                                       optWid.options, optWid.method.batch)

    @pyqtSlot(QModelIndex, pd.DataFrame, dict)
    def _locateRunnerFinished(self, index, data, options):
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
        self._fileModel.setData(index, data, FileListModel.LocDataRole)
        self._fileModel.setData(index, options, FileListModel.LocOptionsRole)
        self._fileModel.setData(index,
                                self._locOptionsDock.widget().method.name,
                                FileListModel.LocMethodRole)
        saveFormat = self._locSaveDock.widget().getFormat()

        saveFileName = self._fileModel.data(index, FileListModel.FileNameRole)

        metaFileName = os.path.join(*determine_filename(saveFileName, "yaml"))
        fdir, fname = determine_filename(saveFileName, saveFormat)
        saveFileName = os.path.join(fdir, fname)
        os.makedirs(fdir, exist_ok=True)

        filterFunc = self._locFilterDock.widget().getFilter()
        inRoi = self._applyRoi(data)
        data = data[filterFunc(data) & inRoi]

        save(saveFileName, data)  # sdt.data.save
        self._saveMetadata(metaFileName)

    @pyqtSlot(QModelIndex)
    def _locateRunnerError(self, index):
        """A LocateRunner encountered an error

        Show an error message box.

        Parameters
        ----------
        index : QModelIndex
            Index (in the file list model) of the file that caused the error
        """
        QMessageBox.critical(
            self, self.tr("Localization error"),
            self.tr("Failed to locate features in {}".format(
                index.data(file_chooser.FileListModel.FileNameRole))))
