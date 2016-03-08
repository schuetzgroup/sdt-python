"""Main window of the peak locator

Contains MainWindow (derived from QMainWindow), which is launched if
this is called as a script (__main__)
"""
import os
import sys
import collections
import types

import yaml
import numpy as np
import pandas as pd
import pims

import qtpy
from qtpy.QtGui import (QIcon, QPolygonF)
from qtpy.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog,
                            QToolBar, QMessageBox, QSplitter, QToolBox,
                            QDockWidget, QWidget, QLabel, QProgressDialog)
from qtpy.QtCore import (pyqtSignal, pyqtSlot, Qt, QDir, QObject, QThread,
                         QSettings, QRunnable, QThreadPool, QModelIndex,
                         QMetaObject, QPointF)

from . import micro_view
from . import locate_options
from . import file_chooser
from . import locate_filter
from . import locate_saver
from ..data import save


def yaml_dict_representer(dumper, data):
    """Represent an OrderedDict using PyYAML

    This is to be passed as
    `yaml.add_representer(collections.OrderedDict, yaml_dict_representer)`
    """
    return dumper.represent_dict(data.items())


yaml.add_representer(collections.OrderedDict, yaml_dict_representer)


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

        # Set up the QThread where the previews are calculated
        self._previewWorker = PreviewWorker()
        optionsWidget.optionsChanged.connect(self._makeWorkerWork)
        self._viewer.currentFrameChanged.connect(self._makeWorkerWork)
        self._workerSignal.connect(self._previewWorker.locate)
        self._workerThread = QThread(self)
        self._previewWorker.moveToThread(self._workerThread)
        self._workerThread.start()
        self._previewWorker.locateFinished.connect(self._locateFinished)
        self._workerWorking = False
        self._newWorkerJob = False

        # Some things to keep track of
        self._currentFile = None
        self._currentLocData = pd.DataFrame()
        self._roiPolygon = QPolygonF()

        # The heavy lifting (localizing all) is done in this thread pool
        self._workerThreadPool = QThreadPool(self)
        # where it makes sense, the batch functions are already threaded
        self._workerThreadPool.setMaxThreadCount(1)

        # load settings and restore window geometry
        settings = QSettings("sdt", "locator")
        v = settings.value("MainWindow/geometry")
        if v is not None:
            self.restoreGeometry(v)
        v = settings.value("MainWindow/state")
        if v is not None:
            self.restoreState(v)

        QMetaObject.connectSlotsByName(self)

    @pyqtSlot(str)
    def open(self, fname):
        """Open an image sequence

        Open an image sequence from a file and load it into the viewer widget.
        This is a PyQt slot.

        Parameters
        ----------
        fname : str
            Path of the file
        """
        try:
            ims = pims.open(fname)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error opening image"),
                                 self.tr(str(e)))
            ims = None

        if isinstance(ims, collections.Iterable) and not len(ims):
            QMessageBox.critical(self, self.tr(""),
                                 self.tr("Empty image"))
            ims = None

        self._currentFile = None if (ims is None) else fname
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
    def _makeWorkerWork(self):
        """Called when something happens that requires a new preview

        E. g. a new frame is displayed. If the preview worker is already
        working, just tell it that there is yet another job to be done.
        Otherwise start the preview worker (by emitting self._workerSignal).
        """
        curFrame = self._viewer.getCurrentFrame()
        if curFrame is None:
            return
        if self._workerWorking:
            # The worker is already working; just store the fact that the
            # worker needs to run again immediately after it finishes
            self._newWorkerJob = True
            return

        self._workerSignal.emit(
            curFrame, self._locOptionsDock.widget().options,
            self._locOptionsDock.widget().method.locate)
        self._workerWorking = True

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
        if self._currentFile is None:
            return
        if self._currentFile not in self._fileModel.files():
            self._locOptionsDock.widget().numFrames = 0
            self._currentFile = None
            self._viewer.setImageSequence(None)

    @pyqtSlot(pd.DataFrame)
    def _locateFinished(self, data):
        """The preview worker thread finished

        If new job has come up while working, call ``_makeWorkerWork`` to
        start the new job immediately.

        In any case, update the filter widget with the data column names
        and filter the data using ``_filterLocalizations``.
        """
        self._workerWorking = False
        if self._newWorkerJob:
            # while we were busy, something new has come up; work on that
            self._makeWorkerWork()
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

        # for every file, create a LocateRunner (QRunnable)
        for i in range(self._fileModel.rowCount()):
            runner = LocateRunner(self._fileModel.index(i),
                                  self._locOptionsDock.widget().options,
                                  self._locOptionsDock.widget().frameRange,
                                  self._locOptionsDock.widget().method.batch)
            runner.signals.finished.connect(
                lambda: progDialog.setValue(progDialog.value() + 1))
            runner.signals.finished.connect(self._locateRunnerFinished)
            runner.signals.error.connect(
                lambda: progDialog.setValue(progDialog.value() + 1))
            runner.signals.error.connect(self._locateRunnerError)
            self._workerThreadPool.start(runner)

        # if cancel was pressed, abort after current worker finishes
        progDialog.canceled.connect(self._workerThreadPool.clear)

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
        self._fileModel.setData(index, data,
                                file_chooser.FileListModel.LocDataRole)
        self._fileModel.setData(index, options,
                                file_chooser.FileListModel.LocOptionsRole)
        saveFormat = self._locSaveDock.widget().getFormat()

        saveFileName = os.path.splitext(
            self._fileModel.data(
                index, file_chooser.FileListModel.FileNameRole))[0]

        metaFileName = saveFileName + os.extsep + "yaml"

        if saveFormat == "hdf5":
            saveFileName += os.extsep + "h5"
        elif saveFormat == "particle_tracker":
            fname = os.path.basename(saveFileName) + "_positions.mat"
            fdir = os.path.dirname(saveFileName)
            outdir = os.path.join(fdir, "Analysis_particle_tracking")
            os.makedirs(outdir, exist_ok=True)
            saveFileName = os.path.join(outdir, fname)

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


class LocateRunner(QRunnable):
    """Runnable for locating peaks in am image sequence

    Attributes
    ----------
    signals : Signals
        The QObject signals
    """
    class Signals(QObject):
        """Collection of signals for LocateRunner

        PyQt will not allow deriving LocateRunner from QRunnable and QObject,
        so this is the workaround.

        Signals
        -------
        finished : (QModelIndex, pandas.DataFrame, dict)
            Emitted when the localization is finished
        error : QModelIndex
            Emitted when an error occured
        """
        def __init__(self, parent=None):
            super().__init__(parent)

        finished = pyqtSignal(QModelIndex, pd.DataFrame, dict)
        error = pyqtSignal(QModelIndex)

    def __init__(self, index, options, frameRange, batch_func):
        """Constructor

        Parameters
        ----------
        index : QModelIndex
            Used to access information about the file the runner is works on
        options : dict
            Keyword arguments for ``batch_func``
        frame_range : tuple of int
            Range of frames to work on. If frameRange[1] < 0, go until the end
        batch_func : callable
            Batch localization implementation
        """
        super().__init__()
        self._index = index
        self._options = options
        self._batch_func = batch_func
        self._frameRange = frameRange
        self.signals = self.Signals()

    def run(self):
        """Do the work

        Emits ``self.signals.finished`` if it finishes without errors and
        and ``self.signals.error`` if there was an error.
        """
        fname = self._index.data(file_chooser.FileListModel.FileNameRole)
        frames = pims.open(fname)
        end = self._frameRange[1] if self._frameRange[1] >= 0 else len(frames)
        # TODO: restrict locating to bounding rect of ROI for performance gain
        try:
            data = self._batch_func(frames[self._frameRange[0]:end],
                                    **self._options)
        except Exception:
            self.signals.error.emit(self._index)
            return

        self.signals.finished.emit(self._index, data, self._options)

    finished = pyqtSignal(QModelIndex, pd.DataFrame)


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
