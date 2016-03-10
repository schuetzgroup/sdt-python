"""Main window of the peak locator

Contains MainWindow (derived from QMainWindow), which is launched if
this is called as a script (__main__)
"""
import os
import sys
import collections
import types
import queue

import yaml
import numpy as np
import pandas as pd
import pims

import qtpy
from qtpy.QtGui import (QIcon, QPolygonF, QCursor)
from qtpy.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog,
                            QToolBar, QMessageBox, QSplitter, QToolBox,
                            QDockWidget, QWidget, QLabel, QProgressDialog)
from qtpy.QtCore import (pyqtSignal, pyqtSlot, Qt, QDir, QObject, QThread,
                         QSettings, QRunnable, QThreadPool, QModelIndex,
                         QPersistentModelIndex, QMetaObject, QPointF)

from . import micro_view
from . import locate_options
from . import file_chooser
from .file_chooser import FileListModel
from . import locate_filter
from . import locate_saver
from ..data import save, load


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

        # Set up the QThread where the localizations are calculated
        self._workerThread = QThread(self)
        self._workerThread.start()

        # set up the object that computes the preview
        self._previewWorker = PreviewWorker()
        optionsWidget.optionsChanged.connect(self._makePreviewWorkerWork)
        self._viewer.currentFrameChanged.connect(self._makePreviewWorkerWork)
        self._workerSignal.connect(self._previewWorker.locate)
        self._previewWorker.moveToThread(self._workerThread)
        self._previewWorker.locateFinished.connect(self._locateFinished)
        self._workerWorking = False
        self._newWorkerJob = False

        # set up the object that does the batch localization conputing
        self._batchWorker = BatchWorker()
        self._batchWorker.moveToThread(self._workerThread)
        self._batchSignal.connect(self._batchWorker.locate)
        self._batchWorker.finished.connect(self._locateRunnerFinished)
        self._batchWorker.error.connect(self._locateRunnerError)

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

        if curFrame is None:
            return
        if self._workerWorking:
            # The worker is already working; just store the fact that the
            # worker needs to run again immediately after it finishes
            self._newWorkerJob = True
            return

        self._workerSignal.emit(curFrame, cur_opts, cur_method.locate)
        self._workerWorking = True
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

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

    @pyqtSlot(pd.DataFrame)
    def _locateFinished(self, data):
        """The preview worker thread finished

        If new job has come up while working, call ``_makeWorkerWork`` to
        start the new job immediately.

        In any case, update the filter widget with the data column names
        and filter the data using ``_filterLocalizations``.
        """
        QApplication.restoreOverrideCursor()
        self._workerWorking = False
        if self._newWorkerJob:
            # while we were busy, something new has come up; work on that
            self._makePreviewWorkerWork()
            self._newWorkerJob = False

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

    # This is emitted to tell the batch worker that there is something to do
    _batchSignal = pyqtSignal()

    @pyqtSlot(str)
    def on_locateSaveWidget_locateAndSave(self, format):
        """Locate all features in all files and save the and metadata"""
        # TODO: check if current localizations are up-to-date
        # only run locate if not

        # Progress bar
        progDialog = QProgressDialog(
            "Locating features…", "Cancel", 0, self._fileModel.rowCount(),
            self)
        progDialog.setWindowModality(Qt.WindowModal)
        progDialog.setValue(0)
        progDialog.setMinimumDuration(0)

        def increase_progress():
            progDialog.setValue(progDialog.value() + 1)

        # get options
        opts = self._locOptionsDock.widget().options
        frameRange = self._locOptionsDock.widget().frameRange
        batch_func = self._locOptionsDock.widget().method.batch

        # enqueue jobs for _batchWorker to execute in the worker thread
        for i in range(self._fileModel.rowCount()):
            self._batchWorker.queue.put_nowait(
                (self._fileModel.index(i), opts, frameRange, batch_func))

        self._batchWorker.finished.connect(increase_progress)
        self._batchWorker.error.connect(increase_progress)

        # tell the _batchWorker that there is something to do
        self._batchSignal.emit()

        # if cancel was pressed, abort after current worker finishes
        # Connect directly to _batchWorker.clear, since otherwise
        # it would run in the worker thread (after locate is finished...)
        progDialog.canceled.connect(self._batchWorker.clear,
                                    type=Qt.DirectConnection)

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


class PreviewWorker(QObject):
    """Worker object that does peak localizations for the preview

    This is made to be owned by a worker thread. Connect some signal to the
    :py:meth:`locate` slot, which in turn will emit the ``locateFinished``
    signal when done
    """
    def __init__(self, parent=None):
        """Constructor"""
        super().__init__(parent)

    @pyqtSlot(np.ndarray, dict, types.FunctionType)
    def locate(self, img, options, locate_func):
        """Do the preview localization

        Emits the ``locateFinished`` signal when finished.

        Parameters
        ----------
        img : numpy.ndarray
            image data
        options : dict
            keyword arg dict for ``locate_func``
        locate_func : callable
            actual localization function
        """
        # TODO: restrict locating to bounding rect of ROI for performance gain
        ret = locate_func(img, **options)
        self.locateFinished.emit(ret)

    locateFinished = pyqtSignal(pd.DataFrame)


class BatchWorker(QObject):
    """Worker object that does batch peak localizations

    This is made to be owned by a worker thread. Connect some signal to the
    :py:meth:`locate` slot, which in turn will emit the ``finised``
    signal for each processed file.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = queue.Queue()

    @pyqtSlot()
    def locate(self):
        while True:
            try:
                idx, opts, frameRange, batch_func = self.queue.get_nowait()
            except queue.Empty:
                return

            fname = idx.data(FileListModel.FileNameRole)
            frames = pims.open(fname)
            end = frameRange[1] if frameRange[1] >= 0 else len(frames)
            # TODO: restrict locating to bounding rect of ROI for performance
            # gain
            try:
                data = batch_func(frames[frameRange[0]:end], **opts)
            except Exception:
                self.error.emit(idx)
                continue

            self.finished.emit(idx, data, opts)

    @pyqtSlot()
    def clear(self):
        while not self.queue.empty():
            self.queue.get_nowait()

    finished = pyqtSignal(QModelIndex, pd.DataFrame, dict)
    error = pyqtSignal(QModelIndex)


def main():
    """Start a QApplication and show the main window"""
    app = QApplication(sys.argv)
    try:
        w = MainWindow()
    except Exception as e:
        QMessageBox.critical(
            None,
            app.translate("main", "Startup error"),
            app.translate("main", str(e)))
        sys.exit(1)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
