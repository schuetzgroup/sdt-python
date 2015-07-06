from struct import calcsize, unpack

import numpy as np

from pims import FramesSequence, Frame


class spec(object):
    """SPE file specification data

    Tuples of (offset, datatype), where offset is the offset in the SPE
    file and datatype is a string describing the datatype as used in
    `struct.unpack`()

    `data_start` is the offset of actual image data.

    `dtypes` translates SPE datatypes (0...4) to numpy ones, e. g. dtypes[0]
    is dtype("<f") (which is np.float32).
    """
    datatype = (108, "h")
    xdim = (42, "H")
    ydim = (656, "H")
    numframes = (1446, "i")

    data_start = 4100

    dtypes = [np.dtype("<f"), np.dtype("<i"), np.dtype("<h"),
              np.dtype("<H"), np.dtype("<I")]


class SPEStack(FramesSequence):
    @classmethod
    def class_exts(cls):
        return {"spe"} | super(SPEStack, cls).class_exts()

    def __init__(self, filename, process_func=None, dtype=None,
                 as_grey=False):
        self._filename = filename
        self._file = open(filename, "rb")

        #determine data type
        self._file.seek(spec.datatype[0])
        b = self._file.read(calcsize(spec.datatype[1]))
        d, = unpack("<{}".format(spec.datatype[1]), b)
        self._file_dtype = spec.dtypes[d]

        if dtype is None:
            self._dtype = self._file_dtype
        else:
            self._dtype = dtype

        #movie dimensions
        self._file.seek(spec.xdim[0])
        b = self._file.read(calcsize(spec.xdim[1]))
        self._file.seek(spec.ydim[0])
        b += self._file.read(calcsize(spec.ydim[1]))
        self._file.seek(spec.numframes[0])
        b += self._file.read(calcsize(spec.numframes[1]))
        self._width, self._height, self._len = unpack(
            "<{}{}{}".format(spec.xdim[1], spec.ydim[1], spec.numframes[1]),
            b)

        self._validate_process_func(process_func)
        self._as_grey(as_grey, process_func)

    @property
    def frame_shape(self):
        return self._width, self._height

    def __len__(self):
        return self._len

    def get_frame(self, j):
        self._file.seek(spec.data_start
                        + j*self._width*self._height*self._file_dtype.itemsize)
        data = np.fromfile(self._file, dtype=self._file_dtype,
                           count=self._width*self._height)
        if self._dtype != self._file_dtype:
            data = data.astype(self._dtype)
        return self.process_func(data.reshape(self._height, self._width))

    @property
    def pixel_type(self):
        return self._dtype

    def __repr__(self):
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {w} x {h}
Pixel Datatype: {dtype}""".format(w=self._width,
                                  h=self._height,
                                  count=self._len,
                                  filename=self._filename,
                                  dtype=self._dtype)