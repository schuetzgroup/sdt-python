# Copyright (c) 2015, Daniel B. Allan
# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Based on https://github.com/soft-matter/slicerator

import random
import types
from io import BytesIO
import pickle

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from sdt.helper.slicerator import (Slicerator, Pipeline, pipeline, index_attr,
                                   propagate_attr)


def assert_letters_equal(actual, expected):
    # check if both lengths are equal
    assert len(actual) == len(expected)
    for actual_, expected_ in zip(actual, expected):
        assert actual_ == expected_


def compare_slice_to_list(actual, expected):
    assert_letters_equal(actual, expected)
    indices = list(range(len(actual)))
    for i in indices:
        # test positive indexing
        assert_letters_equal(actual[i], expected[i])
        # test negative indexing
        assert_letters_equal(actual[-i + 1], expected[-i + 1])
    # in reverse order
    for i in indices[::-1]:
        assert_letters_equal(actual[i], expected[i])
        assert_letters_equal(actual[-i + 1], expected[-i + 1])
    # in shuffled order (using a consistent random seed)
    r = random.Random(5)
    r.shuffle(indices)
    for i in indices:
        assert_letters_equal(actual[i], expected[i])
        assert_letters_equal(actual[-i + 1], expected[-i + 1])
    # test list indexing
    some_indices = [r.choice(indices) for _ in range(2)]
    assert_letters_equal([actual[i] for i in some_indices],
                         [expected[i] for i in some_indices])
    # mixing positive and negative indices
    some_indices = [r.choice(indices + [-i-1 for i in indices])
                    for _ in range(2)]
    assert_letters_equal([actual[i] for i in some_indices],
                         [expected[i] for i in some_indices])
    # test slices
    assert_letters_equal(actual[::2], expected[::2])
    assert_letters_equal(actual[1::2], expected[1::2])
    assert_letters_equal(actual[::3], expected[::3])
    assert_letters_equal(actual[1:], expected[1:])
    assert_letters_equal(actual[:], expected[:])
    assert_letters_equal(actual[:-1], expected[:-1])


v = Slicerator(list('abcdefghij'))
n = Slicerator(list(range(10)))


def test_bool_mask():
    mask = np.array([True, False] * 5)
    s = v[mask]
    assert_letters_equal(s, list('acegi'))


def test_slice_of_slice():
    slice1 = v[4:]
    compare_slice_to_list(slice1, list('efghij'))
    slice2 = slice1[-3:]
    compare_slice_to_list(slice2, list('hij'))
    slice1a = v[[3, 4, 5, 6, 7, 8, 9]]
    compare_slice_to_list(slice1a, list('defghij'))
    slice2a = slice1a[::2]
    compare_slice_to_list(slice2a, list('dfhj'))
    slice2b = slice1a[::-1]
    compare_slice_to_list(slice2b, list('jihgfed'))
    slice2c = slice1a[::-2]
    compare_slice_to_list(slice2c, list('jhfd'))
    slice2d = slice1a[:0:-1]
    compare_slice_to_list(slice2d, list('jihgfe'))
    slice2e = slice1a[-1:1:-1]
    compare_slice_to_list(slice2e, list('jihgf'))
    slice2f = slice1a[-2:1:-1]
    compare_slice_to_list(slice2f, list('ihgf'))
    slice2g = slice1a[::-3]
    compare_slice_to_list(slice2g, list('jgd'))
    slice2h = slice1a[[5, 6, 2, -1, 3, 3, 3, 0]]
    compare_slice_to_list(slice2h, list('ijfjgggd'))


def test_slice_of_slice_of_slice():
    slice1 = v[4:]
    compare_slice_to_list(slice1, list('efghij'))
    slice2 = slice1[1:-1]
    compare_slice_to_list(slice2, list('fghi'))
    slice2a = slice1[[2, 3, 4]]
    compare_slice_to_list(slice2a, list('ghi'))
    slice3 = slice2[1::2]
    compare_slice_to_list(slice3, list('gi'))


def test_slice_of_slice_of_slice_of_slice():
    # Take the red pill. It's slices all the way down!
    slice1 = v[4:]
    compare_slice_to_list(slice1, list('efghij'))
    slice2 = slice1[1:-1]
    compare_slice_to_list(slice2, list('fghi'))
    slice3 = slice2[1:]
    compare_slice_to_list(slice3, list('ghi'))
    slice4 = slice3[1:]
    compare_slice_to_list(slice4, list('hi'))

    # Give me another!
    slice1 = v[2:]
    compare_slice_to_list(slice1, list('cdefghij'))
    slice2 = slice1[0::2]
    compare_slice_to_list(slice2, list('cegi'))
    slice3 = slice2[:]
    compare_slice_to_list(slice3, list('cegi'))
    slice4 = slice3[:-1]
    compare_slice_to_list(slice4, list('ceg'))
    slice4a = slice3[::-1]
    compare_slice_to_list(slice4a, list('igec'))


def test_empty_slice():
    # There was a bug in _index_generator with empty `new_indices`, thus test
    assert list(v[:0]) == []
    assert list(v[[]]) == []
    assert list(v[:0][:]) == []
    assert list(v[:0][:][:]) == []


def test_slice_with_generator():
    slice1 = v[1:]
    compare_slice_to_list(slice1, list('bcdefghij'))
    slice2 = slice1[(i for i in range(2, 5))]
    assert_letters_equal(list(slice2), list('def'))
    assert isinstance(slice2, types.GeneratorType)


def test_no_len_raises():
    with pytest.raises(ValueError):
        Slicerator((i for i in range(5)), (i for i in range(5)))


def test_from_func():
    v = Slicerator.from_func(lambda x: 'abcdefghij'[x], length=10)
    compare_slice_to_list(v, list('abcdefghij'))
    compare_slice_to_list(v[1:], list('bcdefghij'))
    compare_slice_to_list(v[1:][:4], list('bcde'))


def _capitalize(letter):
    return letter.upper()


def _capitalize_if_equal(letter, other_letter):
    if letter == other_letter:
        return letter.upper()
    else:
        return letter


def _a_to_z(letter):
    if letter == 'a':
        return 'z'
    else:
        return letter


@pipeline
def append_zero_inplace(list_obj):
    list_obj.append(0)
    return list_obj


def test_inplace_pipeline():
    n_mutable = Slicerator([list([i]) for i in range(10)])
    appended = append_zero_inplace(n_mutable)

    assert appended[5] == [5, 0]  # execute the function
    assert n_mutable[5] == [5]    # check the original


def test_pipeline_simple():
    capitalize = pipeline(_capitalize)
    cap_v = capitalize(v[:1])

    assert_letters_equal([cap_v[0]], [_capitalize(v[0])])


def test_pipeline_propagation():
    capitalize = pipeline(_capitalize)
    cap_v = capitalize(v)

    assert_letters_equal([cap_v[:1][0]], ['A'])
    assert_letters_equal([cap_v[:1][:2][0]], ['A'])


def test_pipeline_nesting():
    capitalize = pipeline(_capitalize)
    a_to_z = pipeline(_a_to_z)
    nested_v = capitalize(a_to_z(v))

    assert_letters_equal([nested_v[0]], ['Z'])
    assert_letters_equal([nested_v[:1][0]], ['Z'])


def _add_one(number):
    return number + 1


def test_pipeline_nesting_numeric():
    add_one = pipeline(_add_one)
    triple_nested = add_one(add_one(add_one(n)))
    assert_letters_equal([triple_nested[0]], [3])
    assert_letters_equal([triple_nested[:1][0]], [3])


def test_repr():
    repr(v)


def test_getattr():
    class MyList(list):
        attr1 = 'hello'
        attr2 = 'hello again'

        @index_attr
        def s(self, i):
            return list('ABCDEFGHIJ')[i]

        def close(self):
            pass

    a = Slicerator(MyList('abcdefghij'), propagate_attrs=['attr1', 's'])
    assert_letters_equal(a, list('abcdefghij'))
    assert hasattr(a, 'attr1')
    assert not hasattr(a, 'attr2')
    assert hasattr(a, 's')
    assert not hasattr(a, 'close')
    assert a.attr1 == 'hello'
    with pytest.raises(AttributeError):
        a[:5].nonexistent_attr

    compare_slice_to_list(list(a.s), list('ABCDEFGHIJ'))
    compare_slice_to_list(list(a[::2].s), list('ACEGI'))
    compare_slice_to_list(list(a[::2][1:].s), list('CEGI'))

    capitalize = pipeline(_capitalize)
    b = capitalize(a)
    assert_letters_equal(b, list('ABCDEFGHIJ'))
    assert hasattr(b, 'attr1')
    assert not hasattr(b, 'attr2')
    assert hasattr(b, 's')
    assert not hasattr(b, 'close')
    assert b.attr1 == 'hello'
    with pytest.raises(AttributeError):
        b[:5].nonexistent_attr

    compare_slice_to_list(list(b.s), list('ABCDEFGHIJ'))
    compare_slice_to_list(list(b[::2].s), list('ACEGI'))
    compare_slice_to_list(list(b[::2][1:].s), list('CEGI'))


def test_getattr_subclass():
    @Slicerator.from_class
    class Dummy(object):
        propagate_attrs = ['attr1']

        def __init__(self):
            self.frame = list('abcdefghij')

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, i):
            return self.frame[i]

        def attr1(self):
            # propagates through slices of Dummy
            return 'sliced'

        @propagate_attr
        def attr2(self):
            # propagates through slices of Dummy and subclasses
            return 'also in subclasses'

        def attr3(self):
            # does not propagate
            return 'only unsliced'

    class SubClass(Dummy):
        propagate_attrs = ['attr4']  # overwrites propagated attrs from Dummy

        def __len__(self):
            return len(self.frame)

        @property
        def attr4(self):
            # propagates through slices of SubClass
            return 'only subclass'

    dummy = Dummy()
    subclass = SubClass()
    assert hasattr(dummy, 'attr1')
    assert hasattr(dummy, 'attr2')
    assert hasattr(dummy, 'attr3')
    assert not hasattr(dummy, 'attr4')

    assert hasattr(dummy[1:], 'attr1')
    assert hasattr(dummy[1:], 'attr2')
    assert not hasattr(dummy[1:], 'attr3')
    assert not hasattr(dummy[1:], 'attr4')

    assert hasattr(dummy[1:][1:], 'attr1')
    assert hasattr(dummy[1:][1:], 'attr2')
    assert not hasattr(dummy[1:][1:], 'attr3')
    assert not hasattr(dummy[1:][1:], 'attr4')

    assert hasattr(subclass, 'attr1')
    assert hasattr(subclass, 'attr2')
    assert hasattr(subclass, 'attr3')
    assert hasattr(subclass, 'attr4')

    assert not hasattr(subclass[1:], 'attr1')
    assert hasattr(subclass[1:], 'attr2')
    assert not hasattr(subclass[1:], 'attr3')
    assert hasattr(subclass[1:], 'attr4')

    assert not hasattr(subclass[1:][1:], 'attr1')
    assert hasattr(subclass[1:][1:], 'attr2')
    assert not hasattr(subclass[1:][1:], 'attr3')
    assert hasattr(subclass[1:][1:], 'attr4')


def test_pipeline_with_args():
    capitalize = pipeline(_capitalize_if_equal)
    cap_a = capitalize(v, 'a')
    cap_b = capitalize(v, 'b')

    assert_letters_equal(cap_a, 'Abcdefghij')
    assert_letters_equal(cap_b, 'aBcdefghij')
    assert_letters_equal([cap_a[0]], ['A'])
    assert_letters_equal([cap_b[0]], ['a'])
    assert_letters_equal([cap_a[0]], ['A'])


def test_composed_pipelines():
    a_to_z = pipeline(_a_to_z)
    capitalize = pipeline(_capitalize_if_equal)

    composed = capitalize(a_to_z(v), 'c')

    assert_letters_equal(composed, 'zbCdefghij')


def test_pipeline_class():
    sli = Slicerator(np.empty((10, 32, 64)))

    @pipeline
    class crop(Pipeline):
        def __init__(self, reader, bbox):
            self.bbox = bbox
            Pipeline.__init__(self, None, reader)

        def _get(self, key):
            bbox = self.bbox
            return self._ancestors[0][key][bbox[0]:bbox[2], bbox[1]:bbox[3]]

        @property
        def frame_shape(self):
            bbox = self.bbox
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    cropped = crop(sli, (5, 5, 10, 20))
    assert_array_equal(cropped[0], sli[0][5:10, 5:20])
    assert_array_equal(cropped.frame_shape, (5, 15))


def test_serialize():
    # dump Slicerator
    stream = BytesIO()
    pickle.dump(v, stream)
    stream.seek(0)
    v2 = pickle.load(stream)
    stream.close()
    compare_slice_to_list(v2, list('abcdefghij'))
    compare_slice_to_list(v2[4:], list('efghij'))
    compare_slice_to_list(v2[4:][:-1], list('efghi'))

    # dump sliced Slicerator
    stream = BytesIO()
    pickle.dump(v[4:], stream)
    stream.seek(0)
    v2 = pickle.load(stream)
    stream.close()
    compare_slice_to_list(v2, list('efghij'))
    compare_slice_to_list(v2[2:], list('ghij'))
    compare_slice_to_list(v2[2:][:-1], list('ghi'))

    # dump sliced sliced Slicerator
    stream = BytesIO()
    pickle.dump(v[4:][:-1], stream)
    stream.seek(0)
    v2 = pickle.load(stream)
    stream.close()
    compare_slice_to_list(v2, list('efghi'))
    compare_slice_to_list(v2[2:], list('ghi'))
    compare_slice_to_list(v2[2:][:-1], list('gh'))

    # test pipeline
    capitalize = pipeline(_capitalize_if_equal)
    stream = BytesIO()
    pickle.dump(capitalize(v, 'a'), stream)
    stream.seek(0)
    v2 = pickle.load(stream)
    stream.close()
    compare_slice_to_list(v2, list('Abcdefghij'))


def test_from_class():
    class Dummy(object):
        """DocString"""
        def __init__(self):
            self.frame = list('abcdefghij')

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, i):
            """Other Docstring"""
            return self.frame[i]  # actual code of get_frame

        def __repr__(self):
            return 'Repr'

    DummySli = Slicerator.from_class(Dummy)
    assert Dummy()[:2] == ['a', 'b']  # Dummy is unaffected

    # class slots propagate
    assert DummySli.__name__ == Dummy.__name__
    assert DummySli.__doc__ == Dummy.__doc__
    assert DummySli.__module__ == Dummy.__module__

    dummy = DummySli()
    assert isinstance(dummy, Dummy)  # still instance of Dummy
    assert repr(dummy) == 'Repr'  # repr propagates

    compare_slice_to_list(dummy, 'abcdefghij')
    compare_slice_to_list(dummy[1:], 'bcdefghij')
    compare_slice_to_list(dummy[1:][2:], 'defghij')

    capitalize = pipeline(_capitalize_if_equal)
    cap_b = capitalize(dummy, 'b')
    assert_letters_equal(cap_b, 'aBcdefghij')


def test_lazy_hasattr():
    # this ensures that the Slicerator init does not evaluate all properties
    class Dummy(object):
        """DocString"""
        def __init__(self):
            self.frame = list('abcdefghij')

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, i):
            """Other Docstring"""
            return self.frame[i]  # actual code of get_frame

        @property
        def forbidden_property(self):
            raise RuntimeError()

    DummySli = Slicerator.from_class(Dummy)  # noqa: F841


def test_pipeline_multi_input():
    @pipeline(ancestor_count=2)
    def sum_offset(p1, p2, o):
        return p1 + p2 + o

    p1 = Slicerator(list(range(10)))
    p2 = Slicerator(list(range(10, 20)))
    o = 100

    res = sum_offset(p1, p2, o)
    assert isinstance(res, Pipeline)
    assert_array_equal(res, list(range(110, 129, 2)))
    assert len(res) == len(p1)

    resi = sum_offset(1, 2, 3)
    assert(isinstance(resi, int))
    assert(resi == 6)

    p3 = Slicerator(list(range(20)))
    with pytest.raises(ValueError):
        sum_offset(p1, p3)


def test_pipeline_propagate_attrs():
    a1 = Slicerator(list(range(10)))
    a1.attr1 = 10
    a2 = Slicerator(list(range(10, 20)))
    a2.attr1 = 20
    a2.attr2 = 30

    p1 = Pipeline(lambda x, y: x + y, a1, a2,
                  propagate_attrs={"attr1", "attr2"}, propagate_how=0)
    assert p1.attr1 == 10
    assert not hasattr(p1, "attr2")

    p2 = Pipeline(lambda x, y: x + y, a1, a2,
                  propagate_attrs={"attr1", "attr2"}, propagate_how=1)
    assert p2.attr1 == 20
    assert p2.attr2 == 30

    p3 = Pipeline(lambda x, y: x + y, a1, a2,
                  propagate_attrs={"attr1", "attr2"}, propagate_how="first")
    assert p3.attr1 == 10
    assert p3.attr2 == 30

    p4 = Pipeline(lambda x, y: x + y, a1, a2,
                  propagate_attrs={"attr1", "attr2"}, propagate_how="last")
    assert p4.attr1 == 20
    assert p4.attr2 == 30

    a1.attr3 = 40
    a1.attr4 = 50
    a1._propagate_attrs = {"attr3"}
    a1.propagate_attrs = {"attr4"}
    p5 = Pipeline(lambda x, y: x + y, a1, a2, propagate_how="first")
    assert p5.attr3 == 40
    assert p5.attr4 == 50
    assert not hasattr(p5, "attr1")
    assert not hasattr(p5, "attr2")
