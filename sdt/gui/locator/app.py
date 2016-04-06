import traceback

from qtpy.QtCore import QCommandLineParser, QCommandLineOption
from qtpy.QtWidgets import QApplication, QMessageBox

from .main_window import MainWindow


def _tr(string):
    return QApplication.translate("app", string)


def run(argv):
    """Start a QApplication and show the main window

    Parameters
    ----------
    argv : dict
        command line arguments (like ``sys.argv``)

    Returns
    -------
    int
        exit status of the application
    """
    app = QApplication(argv)
    app.setApplicationName("locator")

    cp = QCommandLineParser()
    cp.setApplicationDescription(_tr("Locate fluorescent features in images"))
    cp.addHelpOption()
    previewOption = QCommandLineOption(
        ["p", "preview"], _tr("Show preview"), "true/false")
    cp.addOption(previewOption)
    cp.addPositionalArgument("files", _tr("Files to open, optional"))
    cp.process(app)

    preview = cp.value(previewOption)
    files = cp.positionalArguments()

    try:
        w = MainWindow()
    except Exception as e:
        QMessageBox.critical(
            None,
            app.translate("main", "Startup error"),
            str(e) + "\n\n" + traceback.format_exc())
        return 1

    w.show()

    for f in files:
        w.open(f)

    if preview:
        w.showPreview = preview.lower() in ("t", "1", "true", "yes")

    return app.exec_()
