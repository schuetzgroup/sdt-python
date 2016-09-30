from collections import OrderedDict

from qtpy.QtCore import (Qt, QAbstractItemModel, QModelIndex, QCoreApplication,
                         Signal, Property)
from qtpy.QtWidgets import (QStyledItemDelegate, QSpinBox, QDoubleSpinBox,
                            QComboBox)


class OptionElement:
    def __init__(self, name, paramName=None):
        self.name = name
        self.children = []
        self.parent = None
        self.paramName = paramName
        self._value = None
        self._model = None
        self._dict = OrderedDict()

    def value(self):
        return self._value

    def setValue(self, v):
        return False

    def setTreeValuesFromDict(self, d):
        try:
            v = d[self.paramName]
        except KeyError:
            pass
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
            self._dict[self.paramName] = self._value

    def model(self):
        if self.parent is None:
            return self._model
        else:
            return self.parent.model()


class NumberOption(OptionElement):
    def __init__(self, name, paramName, min, max, default, decimals=2):
        super().__init__(name, paramName)
        self.min = min
        self.max = max
        self.decimals = decimals

        self.setValue(default)

    def setValue(self, v):
        if (self.min <= v <= self.max) and self._value != v:
            self._value = v
            if isinstance(self.paramName, str):
                self._dict[self.paramName] = v
            m = self.model()
            if isinstance(m, QAbstractItemModel):
                idx = m.indexForElement(self, column=1)
                m.dataChanged.emit(idx, idx)
            return True
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
    def __init__(self, name, paramName, choices, default):
        super().__init__(name, paramName)
        self.choices = choices

        self.setValue(default)

    def setValue(self, v):
        try:
            self._index = self.choices.index(v)
        except ValueError:
            return False

        self._value = v
        if isinstance(self.paramName, str):
            self._dict[self.paramName] = v
        return True

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
    def __init__(self, name, paramName, subParamName, choices, default):
        self._allChildren = [[] for i in range(len(choices))]
        self.subParamName = subParamName
        self._childDicts = [OrderedDict() for i in range(len(choices))]
        super().__init__(name, paramName, choices, default)

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
        if isinstance(self.paramName, str):
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
            self._dict[self.paramName] = self._value
        if isinstance(self.subParamName, str):
            self._dict[self.subParamName] = self._childDicts[self._index]

    def addChild(self, child, choice):
        i = self.choices.index(choice)
        self._allChildren[i].append(child)
        child.parent = self
        child._setDict(self._childDicts[i])


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

        if role in (Qt.DisplayRole, Qt.EditRole):
            c = index.column()
            ip = index.internalPointer()
            if c == 0:
                return ip.name
            elif c == 1:
                return ip.value()
        elif role == Qt.UserRole:
            return index.internalPointer()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._headers[section]

    def flags(self, index):
        if index.column() == 1:
            return Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsEnabled
        else:
            return Qt.ItemIsEnabled

    def setData(self, index, value, role=Qt.EditRole):
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

    optionsChanged = Signal()

    def setOptions(self, opts):
        self._root.setTreeValuesFromDict(opts)
        self.optionsChanged.emit()

    @Property(dict, fset=setOptions, doc="Localization algorithm parameters")
    def options(self):
        return self._root._dict


class OptionDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        return index.data(Qt.UserRole).createEditor(parent)

    def setEditorData(self, editor, index):
        index.data(Qt.UserRole).setEditorData(editor)

    def setModelData(self, editor, model, index):
        index.data(Qt.UserRole).setModelData(editor, model, index)
