import numpy as np

import OpenImageIO as oiio
from pims import FramesSequence, Frame

class OIIOStack(FramesSequence):

    def __init__(self, filename, process_func=None, dtype=None,
                 as_grey=False):

        self._filename = filename
        self._input = oiio.ImageInput.open(filename)

        #if not given, find out the pixel type
        if dtype is None:
            tmp = np.array(self._input.read_scanline(0, 0))
            self._dtype = tmp.dtype
        else:
            self._dtype = dtype

        #find out how many frames there are
        self._numframes = 0
        while self._input.seek_subimage(self._numframes, 0):
            self._numframes += 1

        self._validate_process_func(process_func)
        self._as_grey(as_grey, process_func)

    def get_frame(self, j):
        if not self._input.seek_subimage(j, 0):
            raise ValueError("Invalid frame number")
        #make a matrix out of the pixel list
        img = np.reshape(self._input.read_image(),
                         (self._input.spec().height, -1) )
        p_img = self.process_func(img.astype(self._dtype))
        return Frame(p_img, frame_no=j)

    @property
    def pixel_type(self):
        return self._dtype

    @property
    def frame_shape(self):
        spec = self._input.spec()
        return spec.width, spec.height

    def __len__(self):
        return self._numframes

    def __repr__(self):
        spec = self._input.spec()
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {w} x {h}
Pixel Datatype: {dtype}""".format(w=spec.width,
                                  h=spec.height,
                                  count=len(self),
                                  filename=self._filename,
                                  dtype=self._dtype)
