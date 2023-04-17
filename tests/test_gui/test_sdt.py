# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from PySide6 import QtCore
from sdt import gui


def test_url(qapp):
    c = gui.Component("""
import QtQuick
import SdtGui

Item {
    property url testUrl: "file:///bla/blub/qua"
    property string localFile: Sdt.urlToLocalFile(testUrl)
    property url parentUrl: Sdt.parentUrl(testUrl)
}
""")
    c.create()
    assert c.status_ == gui.Component.Status.Ready
    assert c.testUrl == QtCore.QUrl("file:///bla/blub/qua")
    assert c.localFile == "/bla/blub/qua"
    assert c.parentUrl == QtCore.QUrl("file:///bla/blub")
