from collections import OrderedDict

from qtpy.QtCore import (Qt, QAbstractItemModel, QModelIndex, QCoreApplication,
                         Signal, Property)
from qtpy.QtWidgets import (QStyledItemDelegate, QSpinBox, QDoubleSpinBox,
                            QComboBox)


class OptionElement:
    def __init__(self, name, param_name=None):
        self.name = name
        self.children = []
        self.parent = None
        self.param_name = param_name
        self._value = None

    def value(self):
        return self._value

    def setValue(self):
        return False

    def addChild(self, child):
        self.children.append(child)
        child.parent = self

    def optionsDict(self, d=None):
        if d is None:
            d = OrderedDict()
        if isinstance(self.param_name, str):
            d[self.param_name] = self.value()
        for c in self.children:
            c.optionsDict(d)
        return d

    def subtreeFind(self, key, attr="param_name"):
        a = getattr(self, attr, None)
        if a == key:
            return self

        for c in self.children:
            el = c.subtreeFind(key, attr)
            if el is not None:
                return el

        return None


class NumberOption(OptionElement):
    def __init__(self, name, param_name, min, max, default, decimals=2):
        super().__init__(name, param_name)
        self.min = min
        self.max = max
        self.decimals = decimals

        self.setValue(default)

    def setValue(self, v):
        if self.min <= v <= self.max:
            self._value = v
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
        return sb

    def setEditorData(self, editor):
        editor.setValue(self._value)

    def setModelData(self, editor, model, index):
        v = editor.value()
        if self._value == v:
            return
        model.setData(index, v)


class ChoiceOption(OptionElement):
    def __init__(self, name, param_name, choices, default):
        super().__init__(name, param_name)
        self.choices = choices

        self.setValue(default)

    def setValue(self, v):
        try:
            self._index = self.choices.index(v)
            self._value = v
        except ValueError:
            return False
        else:
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


class OptionModel(QAbstractItemModel):
    __clsName = "OptionModel"

    def _tr(self, string):
        """Translate the string using :py:func:`QApplication.translate`"""
        return QCoreApplication.translate(self.__clsName, string)

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self._root = root
        self._dict = root.optionsDict()
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
                return ip._value
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
            if isinstance(ip.param_name, str):
                self._dict[ip.param_name] = value
            self.dataChanged.emit(index, index)
            self.optionsChanged.emit()
            return True
        else:
            return False

    def indexForElement(self, element):
        return self.createIndex(element.parent.children.index(element), 0,
                                element)

    optionsChanged = Signal()

    def setOptions(self, opts):
        changed = False
        for k, v in opts.items():
            el = self._root.subtreeFind(k)
            if el is None or v == el.value():
                continue
            el.setValue(v)
            idx = self.indexForElement(el)
            self.dataChanged.emit(idx, idx)
            changed = True

        if changed:
            self.optionsChanged.emit()

    @Property(dict, fset=setOptions, doc="Localization algorithm parameters")
    def options(self):
        return self._dict


class OptionDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        return index.data(Qt.UserRole).createEditor(parent)

    def setEditorData(self, editor, index):
        index.data(Qt.UserRole).setEditorData(editor)

    def setModelData(self, editor, model, index):
        index.data(Qt.UserRole).setModelData(editor, model, index)
