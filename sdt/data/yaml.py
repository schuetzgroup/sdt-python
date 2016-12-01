"""Support for loading and saving :py:mod:`sdt` data types from/to YAML

This includes the :py:class:`ArrayDumper` and :py:class:`SafeArrayDumper`
YAML dumpers for pretty-printing numpy arrays and their corresponding loaders,
:py:class:`ArrayLoader` and :py:class:`SafeArrayLoader`.

Additionally, there are the :py:class:`Loader`/:py:class:`SafeLoader`/
:py:class:`Dumper`/:py:class:`SafeDumper` subclasses that have
representers/constructors for many types in the :py:mod:`sdt` package.

All of them can be used by passing them as the `Dumper`/`Loader` parameters
to :py:func:`yaml.dump`/:py:func:`yaml.load`.
"""
import collections

import yaml
import numpy as np

from .. import image_tools


class ArrayNode(yaml.Node):
    """YAML node for numpy arrays"""
    pass


class ArrayEvent(yaml.NodeEvent):
    """YAML parser event for numpy arrays"""
    def __init__(self, anchor, tag, value):
        super().__init__(anchor)
        self.tag = tag
        self.value = value
        self.implicit = False


class ArrayDumperBase:
    """Base class implementing a readable representation of numpy arrays

    Uses :py:func:`numpy.array2string` to create a readable representation
    of an array as a nested list.
    """
    array_tag = "!array"

    def represent_numpy_array(self, array):
        return ArrayNode(self.array_tag, array, None, None)

    def serialize_array_node(self, node):
        alias = self.anchors[node]
        if node in self.serialized_nodes:
            self.emit(yaml.AliasEvent(alias))
        else:
            self.serialized_nodes[node] = True
            self.emit(ArrayEvent(alias, node.tag, node.value))

    def expect_array_node(self):
        self.process_anchor("&")
        self.process_tag()
        self.increase_indent()
        t = np.get_printoptions()["threshold"]
        np.set_printoptions(threshold=np.inf)
        s = np.array2string(self.event.value, separator=", ")
        np.set_printoptions(threshold=t)
        for l in s.split("\n"):
            self.write_indent()
            self.write_plain(l)
        self.indent = self.indents.pop()
        self.state = self.states.pop()


class ArrayDumper(yaml.Dumper, ArrayDumperBase):
    """A :py:class:`yaml.Dumper` making numpy array dumps more readable

    This uses :py:func:`numpy.array2string` to create a readable representation
    of an array as a nested list.

    To use this dumper, simply pass it to :py:func:`yaml.dump` as the `Dumper`
    parameter.
    """
    def serialize_node(self, node, parent, index):
        if isinstance(node, ArrayNode):
            self.serialize_array_node(node)
        else:
            super().serialize_node(node, parent, index)

    def expect_node(self, root=False, sequence=False, mapping=False,
                    simple_key=False):
        if isinstance(self.event, ArrayEvent):
            self.expect_array_node()
        else:
            super().expect_node(root, sequence, mapping, simple_key)


class SafeArrayDumper(yaml.SafeDumper, ArrayDumperBase):
    """A :py:class:`yaml.SafeDumper` making numpy array dumps more readable

    This uses :py:func:`numpy.array2string` to create a readable representation
    of an array as a nested list.

    To use this dumper, simply pass it to :py:func:`yaml.dump` as the `Dumper`
    parameter.
    """
    def serialize_node(self, node, parent, index):
        if isinstance(node, ArrayNode):
            self.serialize_array_node(node)
        else:
            super().serialize_node(node, parent, index)

    def expect_node(self, root=False, sequence=False, mapping=False,
                    simple_key=False):
        if isinstance(self.event, ArrayEvent):
            self.expect_array_node()
        else:
            super().expect_node(root, sequence, mapping, simple_key)


ArrayDumper.add_representer(np.ndarray, ArrayDumper.represent_numpy_array)
SafeArrayDumper.add_representer(np.ndarray,
                                SafeArrayDumper.represent_numpy_array)


class ArrayLoaderBase:
    """Base class for loading arrays dumped by :py:class:`ArrayDumperBase`"""
    def construct_numpy_array(self, data):
        a = self.construct_sequence(data, deep=True)
        return np.array(a)


class ArrayLoader(yaml.Loader, ArrayLoaderBase):
    """A :py:class:`yaml.Loader` for loading numpy arrays

    that were dumped using :py:class:`ArrayDumper`.

    To use this dumper, simply pass it to :py:func:`yaml.load` as the `Loader`
    parameter.
    """
    pass


class SafeArrayLoader(yaml.SafeLoader, ArrayLoaderBase):
    """A :py:class:`yaml.SafeLoader` for loading numpy arrays

    that were dumped using :py:class:`SafeArrayDumper`.

    To use this dumper, simply pass it to :py:func:`yaml.load` as the `Loader`
    parameter.
    """
    pass


ArrayLoader.add_constructor(ArrayDumperBase.array_tag,
                            ArrayLoader.construct_numpy_array)
SafeArrayLoader.add_constructor(ArrayDumperBase.array_tag,
                                SafeArrayLoader.construct_numpy_array)


class Dumper(ArrayDumper):
    """A :py:class:`ArrayDumper` with support for many more types

    of the :py:mod:`sdt` package, e. g. :py:class:`image_tools.ROI`.
    """
    pass


class SafeDumper(SafeArrayDumper):
    """A :py:class:`SafeArrayDumper` with support for many more types

    of the :py:mod:`sdt` package, e. g. :py:class:`image_tools.ROI`.
    """
    pass


class Loader(ArrayLoader):
    """A :py:class:`ArrayLoader` with support for many more types

    of the :py:mod:`sdt` package, e. g. :py:class:`image_tools.ROI`.
    """
    pass


class SafeLoader(SafeArrayLoader):
    """A :py:class:`SafeArrayLoader` with support for many more types

    of the :py:mod:`sdt` package, e. g. :py:class:`image_tools.ROI`.
    """
    pass


# slice
def slice_representer(dumper, data):
    d = (("start", data.start), ("stop", data.stop), ("step", data.step))
    return dumper.represent_mapping("!slice", d)


def slice_constructor(loader, data):
    val = loader.construct_mapping(data)
    return slice(val["start"], val["stop"], val["step"])


# dict-like
def dict_representer(dumper, data):
    return dumper.represent_dict(data)


Dumper.add_representer(slice, slice_representer)
SafeDumper.add_representer(slice, slice_representer)
SafeDumper.add_representer(collections.OrderedDict, dict_representer)
Loader.add_constructor("!slice", slice_constructor)
SafeLoader.add_constructor("!slice", slice_constructor)


def _class_representer_factory(cls):
    """Create a representer for class `cls`

    Parameters
    ----------
    cls : class
        Class to be represented using
        :py:meth:`yaml.Dumper.represent_yaml_object`. This class needs to have
        a `yaml_tag` attribute and may optionally have a `flow_style`
        attribute.

    Returns
    -------
    callable
        Function to be passed to `yaml.Dumper.add_representer`
    """
    flow_style = getattr(cls, "yaml_flow_style", None)

    def rep(dumper, data):
        return dumper.represent_yaml_object(cls.yaml_tag, data, cls,
                                            flow_style=flow_style)

    return rep


def _class_constructor_factory(cls):
    """Create a constructor for class `cls`

    Parameters
    ----------
    cls : class
        Class to be constructed using
        :py:meth:`yaml.Dumper.construct_yaml_object`.

    Returns
    -------
    callable
        Function to be passed to `yaml.Dumper.add_constructor`
    """
    def cons(loader, node):
        return loader.construct_yaml_object(node, cls)

    return cons


_yaml_classes = [image_tools.ROI, image_tools.PathROI, image_tools.EllipseROI,
                 image_tools.RectangleROI]
for c in _yaml_classes:
    # Add representers to `Dumper` and `SafeDumper`
    flow_style = getattr(c, "yaml_flow_style", None)
    rep = getattr(c, "to_yaml", None)
    rep = rep if callable(rep) else _class_representer_factory(c)
    Dumper.add_representer(c, rep)
    SafeDumper.add_representer(c, rep)

    # Add constructors to `Loader` and `SafeLoader`
    cons = getattr(c, "from_yaml", None)
    cons = cons if callable(cons) else _class_representer_factory(c)
    Loader.add_constructor(c.yaml_tag, cons)
    SafeLoader.add_constructor(c.yaml_tag, cons)
