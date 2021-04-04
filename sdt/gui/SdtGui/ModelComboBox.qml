// SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQml.Models 2.2
import SdtGui 0.1


ComboBox {
    id: root
    
    property var currentModelData
    property var modelDataRole
    property bool selectFirstOnReset: false

    readonly property var _qmlModel: ListProxyModel {}

    function _resetModelData() {
        if (typeof modelDataRole === "string") {
            if (!modelDataRole || currentIndex < 0) {
                currentModelData = null
                return
            }
            currentModelData = _qmlModel.get(currentIndex, modelDataRole)
            return
        }
            
        var d = {}
        for (var r of modelDataRole) {
            if (!modelDataRole || currentIndex < 0)
                d[r] = null
            else
                d[r] = _qmlModel.get(currentIndex, r)
        }
        currentModelData = d
    }

    Connections {
        target: root._qmlModel
        onItemsChanged: {
            if (index > root.currentIndex ||
                    root.currentIndex >= index + count) 
                return
            if (typeof root.modelDataRole === "string" &&
                    (!roles || roles.includes(root.modelDataRole))) {
                root.currentModelData = _qmlModel.get(root.currentIndex,
                                                       root.modelDataRole)
                return
            }
            var modified = false
            for (var r of root.modelDataRole) {
                if (!roles || roles.includes(r)) {
                    root.currentModelData[r] = _qmlModel.get(root.currentIndex, r)
                    modified = true
                }
            }
            if (modified)
                root.currentModelDataChanged()
        }
        onModelReset: {
            if (selectFirstOnReset && model.rowCount() > 0)
                currentIndex = 0
        }
        onRowsInserted: { _rowsInsertedOrRemoved(first) }
        onRowsRemoved: { _rowsInsertedOrRemoved(first) }

        function _rowsInsertedOrRemoved(first) {
            if (first > root.currentIndex)
                return
            if (typeof root.modelDataRole === "string") {
                root.currentModelData = root._qmlModel.get(
                    root.currentIndex, root.modelDataRole)
                return
            }
            var md = {}
            for (var r of root.modelDataRole)
                md[r] = root._qmlModel.get(root.currentIndex, r)
            root.currentModelData = md
        }
    }

    onModelDataRoleChanged: { _resetModelData() }
    onCurrentIndexChanged: { _resetModelData() }
    onModelChanged: { 
        _qmlModel.sourceModel = model
        _resetModelData()
        if (selectFirstOnReset && model && model.rowCount() > 0)
            currentIndex = 0
    }
}
