# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import OrderedDict

from PyQt5.QtCore import (Qt, QAbstractItemModel, QModelIndex,
                          QCoreApplication, pyqtProperty, pyqtSignal)
from PyQt5.QtWidgets import (QStyledItemDelegate, QSpinBox, QDoubleSpinBox,
                             QComboBox, QTreeView, QWidget, QHBoxLayout)


class OptionElement:
    def __init__(self, name, paramName=None, uncheckedValue=None):
        self.name = name
        self.children = []
        self.parent = None
        self.paramName = paramName
        self._value = None
        self._model = None
        self._dict = OrderedDict()
        self._checked = None
        self.uncheckedValue = None

    def value(self):
        return self._value

    def setValue(self, v):
        if self._value == v:
            return False

        self._value = v
        if isinstance(self.paramName, str) and self._checked != Qt.Unchecked:
            self._dict[self.paramName] = v
        m = self.model()
        if isinstance(m, OptionModel):
            idx = m.indexForElement(self, column=1)
            m.dataChanged.emit(idx, idx)
        return True

    def modelRepr(self):
        return self._value

    def checked(self):
        return self._checked

    def setChecked(self, c):
        if self._checked == c:
            return False

        if isinstance(self.paramName, str):
            if c == Qt.Unchecked:
                self._dict[self.paramName] = self.uncheckedValue
            else:
                self._dict[self.paramName] = self._value
        self._checked = c

        m = self.model()
        if isinstance(m, OptionModel):
            idx = m.indexForElement(self, column=1)
            m.dataChanged.emit(idx, idx)

        return True

    def setTreeValuesFromDict(self, d):
        try:
            v = d[self.paramName]
        except KeyError:
            pass
        else:
            if self._checked is not None:
                if v == self.uncheckedValue:
                    self.setChecked(Qt.Unchecked)
                else:
                    self.setChecked(Qt.Checked)
                    self.setValue(v)
            else:
                self.setValue(v)

        for c in self.children:
            c.setTreeValuesFromDict(d)

    def addChild(self, child):
        self.children.append(child)
        child.parent = self
        child._setDict(self._dict)

    def _setDict(self, d):
        self._dict = d
        if isinstance(self.paramName, str):
            if self._checked != Qt.Unchecked:
                self._dict[self.paramName] = self._value
            else:
                self._dict[self.paramName] = self.uncheckedValue

    def model(self):
        if self.parent is None:
            return self._model
        else:
            return self.parent.model()


class NumberOption(OptionElement):
    def __init__(self, name, paramName, min, max, default, decimals=2,
                 uncheckedValue=None):
        super().__init__(name, paramName, uncheckedValue)
        self.min = min
        self.max = max
        self.decimals = decimals

        self.setValue(default)

    def setValue(self, v):
        if (self.min <= v <= self.max):
            return super().setValue(v)
        else:
            return False

    def createEditor(self, parent):
        if isinstance(self._value, int):
            sb = QSpinBox(parent)
        elif isinstance(self._value, float):
            sb = QDoubleSpinBox(parent)
            sb.setDecimals(self.decimals)

        sb.setMinimum(self.min)
        sb.setMaximum(self.max)
        sb.setValue(self._value)
        sb.setFrame(False)
        return sb

    def setEditorData(self, editor):
        editor.setValue(self._value)

    def setModelData(self, editor, model, index):
        v = editor.value()
        if self._value == v:
            return
        model.setData(index, v)


class ChoiceOption(OptionElement):
    def __init__(self, name, paramName, choices, default, uncheckedValue=None):
        super().__init__(name, paramName, uncheckedValue)
        self.choices = choices

        self.setValue(default)

    def setValue(self, v):
        try:
            self._index = self.choices.index(v)
        except ValueError:
            return False
        return super().setValue(v)

    def createEditor(self, parent):
        cb = QComboBox(parent)
        cb.addItems(self.choices)
        return cb

    def setEditorData(self, editor):
        editor.setCurrentIndex(self._index)

    def setModelData(self, editor, model, index):
        i = editor.currentIndex()
        if self._index == i:
            return
        model.setData(index, self.choices[i])


class ChoiceOptionWithSub(ChoiceOption):
    def __init__(self, name, paramName, subParamName, choices, default,
                 uncheckedValue=None):
        self._allChildren = [[] for i in range(len(choices))]
        self.subParamName = subParamName
        self._childDicts = [OrderedDict() for i in range(len(choices))]
        super().__init__(name, paramName, choices, default,
                         uncheckedValue=uncheckedValue)

    def setValue(self, v):
        try:
            self._index = self.choices.index(v)
        except ValueError:
            return False

        self._value = v
        m = self.model()
        newChildren = self._allChildren[self._index]
        if m is not None:
            idx = m.indexForElement(self)
            m.dataChanged.emit(idx, idx)

            m.beginRemoveRows(idx, 0, len(self.children)-1)
            self.children = []
            m.endRemoveRows()
            m.beginInsertRows(idx, 0, len(newChildren)-1)

        self.children = newChildren
        if (isinstance(self.paramName, str) and self._checked != Qt.Unchecked):
            self._dict[self.paramName] = v
        if isinstance(self.subParamName, str):
            self._dict[self.subParamName] = self._childDicts[self._index]

        if m is not None:
            m.endInsertRows()

        return True

    def setTreeValuesFromDict(self, d):
        try:
            v = d[self.paramName]
        except KeyError:
            pass
        else:
            if self._checked is not None:
                if v == self.uncheckedValue:
                    self.setChecked(Qt.Unchecked)
                else:
                    self.setChecked(Qt.Checked)
                    self.setValue(v)
            else:
                self.setValue(v)

        try:
            sd = d[self.subParamName]
        except KeyError:
            pass
        else:
            for c in self.children:
                c.setTreeValuesFromDict(sd)

    def _setDict(self, d):
        self._dict = d
        if isinstance(self.paramName, str):
            if self._checked != Qt.Unchecked:
                self._dict[self.paramName] = self._value
            else:
                self._dict[self.paramName] = self.uncheckedValue
        if isinstance(self.subParamName, str):
            self._dict[self.subParamName] = self._childDicts[self._index]

    def addChild(self, child, choice):
        i = self.choices.index(choice)
        self._allChildren[i].append(child)
        child.parent = self
        child._setDict(self._childDicts[i])


class RangeOption(OptionElement):
    def __init__(self, name, paramName, min, max, default, decimals=2,
                 uncheckedValue=None):
        super().__init__(name, paramName, uncheckedValue)
        self.min = min
        self.max = max
        self.decimals = decimals

        self.setValue(default)

    def setValue(self, v):
        if ((self.min <= v[0] <= self.max) and
                (self.min <= v[1] <= self.max) and
                (v[0] < v[1])):
            return super().setValue(v)
        else:
            return False

    def createEditor(self, parent):
        w = QWidget(parent)
        l = QHBoxLayout()
        l.setSpacing(0)
        l.setContentsMargins(0, 0, 0, 0)
        for v in self._value:
            if isinstance(v, int):
                sb = QSpinBox()
            elif isinstance(v, float):
                sb = QDoubleSpinBox()
                sb.setDecimals(self.decimals)

            sb.setMinimum(self.min)
            sb.setMaximum(self.max)
            sb.setValue(v)
            sb.setFrame(False)

            l.addWidget(sb)

        w.setLayout(l)
        return w

    def setEditorData(self, editor):
        l = editor.layout()
        for i in range(2):
            l.itemAt(i).widget().setValue(self._value[i])

    def setModelData(self, editor, model, index):
        v = []
        l = editor.layout()
        for i in range(2):
            v.append(l.itemAt(i).widget().value())

        if self._value == v:
            return
        model.setData(index, v)

    def modelRepr(self):
        return str(self._value)


class OptionModel(QAbstractItemModel):
    __clsName = "OptionModel"

    def _tr(self, string):
        """Translate the string using :py:func:`QApplication.translate`"""
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self._root = root
        self._root._model = self
        self._headers = [self._tr("Name"), self._tr("Value")]

    def rowCount(self, parent=QModelIndex()):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            return len(self._root.children)

        return len(parent.internalPointer().children)

    def columnCount(self, parent=QModelIndex()):
        return 2

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            # parent is the root item
            pi = self._root
        else:
            pi = parent.internalPointer()

        child = pi.children[row]
        return self.createIndex(row, column, child)

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        ip = index.internalPointer()
        if ip.parent is None:
            return QModelIndex()
        else:
            return self.createIndex(ip.parent.children.index(ip), 0,
                                    ip.parent)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        c = index.column()
        if role in (Qt.DisplayRole, Qt.EditRole):
            ip = index.internalPointer()
            if c == 0:
                return ip.name
            elif c == 1:
                return ip.modelRepr()
        elif role == Qt.UserRole:
            return index.internalPointer()
        elif role == Qt.CheckStateRole and c == 0:
            return index.internalPointer().checked()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._headers[section]

    def flags(self, index):
        if index.column() == 1:
            f = Qt.ItemIsSelectable
            if index.internalPointer().checked() != Qt.Unchecked:
                f |= Qt.ItemIsEditable | Qt.ItemIsEnabled
        else:
            f = (Qt.ItemIsEnabled | Qt.ItemIsSelectable |
                 Qt.ItemIsUserCheckable)
        return f

    def setData(self, index, value, role=Qt.EditRole):
        c = index.column()
        if role == Qt.CheckStateRole and c == 0:
            if index.internalPointer().setChecked(value):
                self.optionsChanged.emit()
                return True
            else:
                return False
        if role != Qt.EditRole or index.column() != 1:
            return False

        ip = index.internalPointer()
        if ip.setValue(value):
            self.optionsChanged.emit()
            return True
        else:
            return False

    def indexForElement(self, element, column=0):
        return self.createIndex(element.parent.children.index(element), column,
                                element)

    optionsChanged = pyqtSignal()

    def setOptions(self, opts):
        self._root.setTreeValuesFromDict(opts)
        self.optionsChanged.emit()

    @pyqtProperty(dict, fset=setOptions,
                  doc="Localization algorithm parameters")
    def options(self):
        return self._root._dict


class OptionDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        if index.column() == 1:
            return index.data(Qt.UserRole).createEditor(parent)
        else:
            return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        index.data(Qt.UserRole).setEditorData(editor)

    def setModelData(self, editor, model, index):
        index.data(Qt.UserRole).setModelData(editor, model, index)


class OptionTreeView(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._optionDelegate = OptionDelegate()
        self.setItemDelegate(self._optionDelegate)

        self.stretchFactors = [2/3, 1/3]

    def sizeHintForColumn(self, column):
        if column < len(self.stretchFactors):
            return round(self.viewport().width() * self.stretchFactors[column])
        else:
            return -1

    def resizeEvent(self, event):
        super().resizeEvent(event)

        for column in range(len(self.stretchFactors)):
            self.resizeColumnToContents(column)
