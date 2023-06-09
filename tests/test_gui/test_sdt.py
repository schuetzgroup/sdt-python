# SPDX-FileCopyrightText: 2023 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from PyQt5 import QtCore
from sdt import gui


def test_url(qapp):
    c = gui.Component("""
import QtQuick 2.15
import SdtGui 0.2

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


def test_setQObjectParent(qapp):
    c = gui.Component("""
import QtQuick 2.15
import SdtGui 0.2

Item {
    id: root

    Item {
        objectName: "child"

        Item {
            id: grandChild
            objectName: "grandChild"
        }
    }

    Component.onCompleted: { Sdt.setQObjectParent(grandChild, root) }
}
""")
    c.create()
    assert c.status_ == gui.Component.Status.Ready
    inst = c.instance_
    chld = inst.findChild(QtCore.QObject, "child")
    gchld = inst.findChild(QtCore.QObject, "grandChild")
    assert gchld.parentItem() is chld
    assert gchld.parent() is inst
