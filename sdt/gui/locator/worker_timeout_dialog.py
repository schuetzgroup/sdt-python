from qtpy.QtCore import QCoreApplication
from qtpy.QtWidgets import QMessageBox


class WorkerTimeoutDialog(QMessageBox):
    __clsName = "WorkerTimeoutDialog"

    def tr(self, string):
        """Translate the string using `QCoreApplication.translate`"""
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, what, parent=None):
        super().__init__(
            QMessageBox.Warning,
            self.tr("Problem stopping {}").format(what),
            self.tr("It seems like it takes a long time to stop the "
                    "{}. Do you want to forcefully abort or wait"
                    "longer?").format(what),
            QMessageBox.NoButton, parent)

        self.addButton(self.tr("Abort"), QMessageBox.RejectRole)
        self.addButton(self.tr("Wait"), QMessageBox.AcceptRole)
