# -*- coding: utf-8 -*-

#If we don't do this, python2 will try to import the local pims.py instead
#of the "real" pims module
from __future__ import absolute_import

import numpy as np

from pims import FramesSequence, Frame


class spec(object):
    """SPE file specification data

    Tuples of (offset, datatype, count), where offset is the offset in the SPE
    file and datatype is a string describing the datatype as used in
    `numpy.fromfile`()

    `data_start` is the offset of actual image data.

    `dtypes` translates SPE datatypes (0...4) to numpy ones, e. g. dtypes[0]
    is dtype("<f") (which is np.float32).
    """
    datatype = (108, "<h")
    xdim = (42, "<H")
    ydim = (656, "<H")
    numframes = (1446, "<i")

    metadata = {
        "comments": (200, "<80S", 5),
        "spare4": (742, "<436S"),
        "ControllerVersion": (0, "<h"),
        "swversion": (688, "<16S"),
        "header_version": (1992, "<f"),
        "date": (20, "<10S"),
        "ExperimentTimeLocal": (172, "<7S"),
        "ExperimentTimeUTC": (179, "<7S"),
        "exp_sec": (10, "<f"),
        "DetTemperature": (36, "<f"),
        "DetType": (40, "<h"),
        "XPrePixels": (98, "<h"),
        "XPostPixels": (100, "<h"),
        "YPrePixels": (102, "<h"),
        "YPostPixels": (104, "<h"),
        "ReadoutTime": (672, "<f"),
        "type": (704, "<h"),
        "clkspd_us": (1428, "<f"),
        "ROIs": (1512, np.dtype([("startx", "<H"),
                                 ("endx", "<H"),
                                 ("groupx", "<H"),
                                 ("starty", "<H"),
                                 ("endy", "<H"),
                                 ("groupy", "<H")]), 5),
        "readoutMode": (1480, "<H"),
        "WindowSize": (1482, "<H")
    }
    num_rois = (1510, "<h")

    data_start = 4100

    dtypes = [np.dtype("<f"), np.dtype("<i"), np.dtype("<h"),
              np.dtype("<H"), np.dtype("<I")]
    
    controllers = [
        "new120 (Type II)", "old120 (Type I)", "ST130", "ST121", "ST138",
        "DC131 (PentaMax)", "ST133 (MicroMax/Roper)", "ST135 (GPIB)", "VTCCD",
        "ST116 (GPIB)", "OMA3 (GPIB)", "OMA4"
    ]


class SpeStack(FramesSequence):
    """Read image data from SPE files

    Attributes:
        metadata (dict): Contains additional metadata.
    """
    @classmethod
    def class_exts(cls):
        return {"spe"} | super(SpeStack, cls).class_exts()

    def __init__(self, filename, process_func=None, dtype=None,
                 as_grey=False, char_encoding="ascii"):
        """Create an iterable object that returns image data as numpy arrays

        Args:
            filename (str): Name of the SPE file
            process_func (callable, optional): Takes one image array as its
                sole argument. It is applied to each image. Defaults to None.
            dtype (numpy.dtype, optional): Which data type to convert the
                images too. No conversion if None. Defaults to None.
            as_grey (bool, optional): Convert image to greyscale. Do not use
                in conjunction with process_func. Defaults to False.
            char_encoding (str, optional): Specifies what character encoding
                is used for metatdata strings. Defaults to "ascii".
        """
        self._filename = filename
        self._file = open(filename, "rb")
        self._char_encoding = char_encoding

        #determine data type
        self._file.seek(spec.datatype[0])
        d = np.fromfile(self._file, spec.datatype[1], count=1)
        self._file_dtype = spec.dtypes[d]

        if dtype is None:
            self._dtype = self._file_dtype
        else:
            self._dtype = dtype

        #movie dimensions
        self._file.seek(spec.xdim[0])
        self._width = np.fromfile(self._file, spec.xdim[1], count=1)[0]
        self._file.seek(spec.ydim[0])
        self._height = np.fromfile(self._file, spec.ydim[1], count=1)[0]
        self._file.seek(spec.numframes[0])
        self._len = np.fromfile(self._file, spec.numframes[1], count=1)[0]

        #read additional metadata
        self.metadata = {}
        self._read_metadata()

        #pims-specific stuff
        self._validate_process_func(process_func)
        self._as_grey(as_grey, process_func)

    def _read_metadata(self):
        """Actual reading of the additional metadata

        Metadata gets written to self.metadata. Strings are decoded to
        python strings using the character encoding self._char_encoding
        """
        #Decode each string from the numpy array read by np.fromfile
        #function definition
        decode = np.vectorize(lambda x: x.decode(self._char_encoding))

        for name, sp in spec.metadata.items():
            self._file.seek(sp[0])
            cnt = (1 if len(sp) < 3 else sp[2])
            v = np.fromfile(self._file, dtype=sp[1], count=cnt)
            if v.dtype.kind == "S":
                #silently ignore string decoding failures
                try:
                    v = decode(v)
                except:
                    pass
            if cnt == 1:
                #for convenience, if the array contains only one single entry,
                #return this entry itself.
                v = np.asscalar(v)
            self.metadata[name] = v

        #The number of ROIs is specified in the SPE file. Only return as many
        #ROIs as specified
        self._file.seek(spec.num_rois[0])
        num_rois = np.fromfile(self._file, spec.num_rois[1], count=1)
        num_rois = (1 if num_rois < 1 else num_rois)
        self.metadata["ROIs"] = self.metadata["ROIs"][:num_rois]
        
        #Translate controller names
        t = self.metadata["type"]
        if 1 <= t <= len(spec.controllers):
            self.metadata["type"] = spec.controllers[t - 1]
        else:
            self.metadata.pop("type", None)

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
        return Frame(
            self.process_func(data.reshape(self._height, self._width)),
            frame_no=j)

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