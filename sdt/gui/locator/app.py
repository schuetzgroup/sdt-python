from qtpy.QtWidgets import QApplication, QMessageBox

from .main_window import MainWindow


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
    try:
        w = MainWindow()
    except Exception as e:
        QMessageBox.critical(
            None,
            app.translate("main", "Startup error"),
            app.translate("main", str(e)))
        return 1
    w.show()
    return app.exec_()
