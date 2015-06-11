# -*- coding: utf-8 -*-

"""Tools to deal with SPE files as created by the SDT-control software"""

import collections
import configparser
import io
import logging
import os
import pandas as pd

pd.options.mode.chained_assignment = None #Get rid of the warning

try:
    import OpenImageIO as oiio
except:
    oiio = None

_logger = logging.getLogger(__name__)


extra_metadata_begin = "### Begin extra metadata ###"
extra_metadata_end = "### End extra metadata ###"
extra_metadata_version = (1, 0) #(major, minor)
#exclude metadata that is saved otherwise anyways
excluded_metadata = ["DateTime", "ImageDescription"]


class OIIOError(Exception):
    def __init__(self, message=None):
        super().__init__(oiio.geterror() if message is None else message)


def read_attr_metadata(spec):
    """Read metadata from :var:`OpenImageIO.ImageSpec.extra_attribs`

    Non-standard metadata (such as "DateTime") is saved to a
    :class:`collections.OrderedDict`. This is e. g. useful for getting all
    the SDT-specific metadata from an SPE file (provided the SPE plugin version
    >= 1.4 is used).

    :param spec: The :class:`OpenImageIO.ImageSpec` of the SPE file.

    :returns: A :class:`collections.OrderedDict` containing the metadata
    """
    data = collections.OrderedDict()
    for attr in spec.extra_attribs:
        if attr.name not in excluded_metadata:
            data[attr.name] = attr.value

    return data


def read_imagedesc_metadata(ini):
    """Read metadata from ini-file-like string

    Converts the ini-file-like string to a :class:`collections.OrderedDict`.

    :param ini: Either the string itself or a :class:`OpenImageIO.ImageDesc`
    whose "ImageDescription" attribute is the string.

    :returns: A :class:`collections.OrderedDict` containing the metadata. All
    values are strings and may need conversion.
    """
    if isinstance(ini, str):
        inistr = ini
    elif oiio is not None:
        if isinstance(ini, oiio.ImageSpec):
            inistr = ini.get_attribute("ImageDescription")
    else:
        raise TypeError("Expected str or OpenImageIO.ImageSpec")

    if inistr is None:
        #There is no "ImageDescription" attribute, return empty dict
        return collections.OrderedDict()

    start_pos = inistr.find(extra_metadata_begin)
    if start_pos != -1:
        #extra_metadata_begin string was found. Discard anything before that
        #and the string itself
        inistr = inistr[start_pos+len(extra_metadata_begin):]
    end_pos = inistr.find(extra_metadata_end)
    if end_pos != -1:
        #extra_metadata_end string was found. Discard anything after that
        #and the string itself
        inistr = inistr[:end_pos]

    cp = configparser.ConfigParser(dict_type = collections.OrderedDict)
    #do not transform the keys to lowercase
    cp.optionxform = str
    ret = collections.OrderedDict()
    try:
        cp.read_string(inistr)
        ret = collections.OrderedDict(cp["metadata"])
    except:
        pass
    return ret


def metadata_to_ini_string(metadata):
    """Convert the metadata dicts to ini-file type string

    Use this function to convert the :class:`collections.OrderedDict` created
    by :func:`read_spe_metadata` or :func:`read_imagedesc_metadata` into
    ini-file-like strings that can be saved to the ImageDescription of a
    converted file.

    :param metadata: (ordered) dictionary of metadata

    :returns: ini-file like string of metadata
    """
    cp = configparser.ConfigParser(dict_type = collections.OrderedDict)
    #do not transform the keys to lowercase
    cp.optionxform = str
    #create the dict of dicts required by the config parser
    toplevel_dict = collections.OrderedDict()
    toplevel_dict["metadata"] = metadata
    #also save some version information in case something changes in the
    #future
    toplevel_dict["version"] = \
        {"version": "{}.{}".format(extra_metadata_version[0],
                                   extra_metadata_version[1])}
    cp.read_dict(toplevel_dict)
    strio = io.StringIO()
    cp.write(strio)
    return "{}\n{}\n{}".format(extra_metadata_begin,
                               strio.getvalue(),
                               extra_metadata_end)


def convert_file(infile, outfile):
    """Convert infile to outfile

    OpenImageIO will automatically determine the right formats.

    :param infile: Input file name string

    :param outfile: Output file name string
    """
    input = oiio.ImageInput.open(infile)
    output = oiio.ImageOutput.create(outfile)
    if input is None or output is None:
        raise OIIOError()

    #determine number of subimages
    num_sub = 0
    while input.seek_subimage(num_sub, 0):
        num_sub += 1

    if (num_sub > 1 and not output.supports("multiimage") and
        not output.supports("appendsubimage")):
        raise OIIOError("{}: Output format '{}' does not support appending "
            "subimages".format(outfile, output.format_name()))

    subspec = oiio.ImageSpec(input.spec())
    d = read_attr_metadata(subspec)
    ini_str = metadata_to_ini_string(d)
    subspec.attribute("ImageDescription", ini_str)

    if not output.open(outfile, subspec, oiio.Create):
        raise OIIOError()
    if not output.copy_image(input):
        raise OIIOError()

    for i in range(1, num_sub):
        input.seek_subimage(i, 0)
        if not output.open(outfile, input.spec(), oiio.AppendSubimage):
            raise OIIOError()
        if not output.copy_image(input):
            raise OIIOError()

    input.close()
    output.close()


def convert_folder(indir, out_format, *exts, **kwargs):
    """Recursively converts all files in a directory

    The converted files will have the same file names as the original ones
    with their extension replaced by out_format.

    :param indir: Input directory path

    :param out_format: Extension of the output format. Include the dot, e. g.
    ".jpg"

    :param cont: If True, conversion will continue even if an error was
    encountered. False by default.

    :param exts: File extension strings that will be (case-insensitively)
    matched. If a file has an extension in this tuple, it will be converted.
    Include the dot in the extension name, e. g. ".tiff"
    """
    cont = kwargs.pop("cont", False)
    if kwargs:
        bad_kw = ", ".join([ k for k, v in kwargs.items() ])
        raise TypeError("Unknown keywords encountered: {}".format(bad_kw))

    for root, dirs, files in os.walk(indir):
        uc_ext_list = []
        for ext in exts:
            uc_ext_list.append(ext.upper())

        for file in files:
            path, ext = os.path.splitext(file)
            if ext.upper() in uc_ext_list:
                infile = os.path.join(root, file)
                outfile = os.path.join(root, path + out_format)
                try:
                    convert_file(infile, outfile)
                except Exception as e:
                    if cont:
                        _logger.info(str(e))
                    else:
                        raise e


class ROI(object):
    """Region of interest in a picture

    This class represents a region of interest. It as callable. If called with
    an array-like object as parameter, it will return only the region of
    interest as defined by the top_left and bottom_right attributes.

    top_left is a tuple holding the x and y coordinates of the top-left corner
    of the ROI, while bottom_right holds the x and y coordinates of the
    bottom-right corner.

    (0, 0) is the the top-left corner of the image. (width-1, height-1) is the
    bottom-right corner.

    At the moment, this works only for single channel (i. e. grayscale) images.
    """
    def __init__(self, top_left, bottom_right):
        """Initialze the top_left and bottom_right attributes.

        Both top_left and bottom_right are expected to be tuples holding a x
        and a y coordinate.

        (0, 0) is the the top-left corner of the image. (width-1, height-1) is
        the bottom-right corner.
        """
        self.top_left = top_left
        self.bottom_right = bottom_right

    def __call__(self, data, pos_columns=["x", "y"], reset_origin=True):
        """Restrict data to the region of interest.

        Args:
            data: Either a `pandas.DataFrame` containing feature coordinates,
                or an array-like object containing the raw image data.
            pos_columns (list of str): The names of the columns of the x and y
                coordinates of features. This only applies to DataFrame data
                arguments.
            reset_origin (bool): If True, the top-left corner coordinates will
                be subtracted off all feature coordinates, i. e. the top-left
                corner will be the origin.

        Returns:
            If data was a `pandas.DataFrame` only the lines with coordinates
            within the region of interest are returned, otherwise the cropped
            raw image.
        """
        if isinstance(data, pd.DataFrame):
            x = pos_columns[0]
            y = pos_columns[1]
            roi_data = data[(data[x] > self.top_left[0])
                            & (data[x] < self.bottom_right[0])
                            & (data[y] > self.top_left[1])
                            & (data[y] < self.bottom_right[1])]
            if reset_origin:
                roi_data.loc[:, x] -= self.top_left[0]
                roi_data.loc[:, y] -= self.top_left[1]

            return roi_data

        return data[self.top_left[1]:self.bottom_right[1],
                    self.top_left[0]:self.bottom_right[0]]


def roper_kinetic_to_tiff(spe, outfilename, split=True):
    """Split Roper camera "Kinetics" mode images into a TIFF stack
    
    In "Kinetics" mode, the Roper camera places several (sub-)images next to
    each other on the camera chip. This function splits the subimages into
    a TIFF stack.
    
    Args:
        spe (OpenImageIO.ImageInput): Input SPE file
        outfilename (str): Name of the output file. If `split` is True, the
            `format` function is used to enumerate files. E. g. supplying
            "run_{num:03d}.tiff" will result in files named run_000.tiff,
            run_001.tiff, and so on.
        split (bool, optional): Whether each single (super-)image should be
            recorded to a seperate image stack file. Defaults to True.
    """
    spec = spe.spec()
    subpics = spec.get_attribute("subpics")
    
    outspec = oiio.ImageSpec(spec)
    outspec.full_height = outspec.height = round(spec.height/subpics)
    
    i = 0
    while spe.seek_subimage(i, 0):
        if split:
            output = oiio.ImageOutput.create(outfilename.format(num=i))
            if not output.open(outfilename.format(num=i),
                               outspec,
                               oiio.Create):
                raise OIIOError()
        elif i == 0:
            output = oiio.ImageOutput.create(outfilename)
            if not output.open(outfilename, outspec, oiio.Create):
                raise OIIOError()
        else:
            if not output.open(outfilename, outspec, oiio.AppendSubimage):
                raise OIIOError()
        
        data = spe.read_image()
        for j in range(subpics):
            if j != 0:
                if not output.open(outfilename, outspec, oiio.AppendSubimage):
                    raise OIIOError()
                
            #extract subpic
            s = data[j*outspec.image_pixels():(j+1)*outspec.image_pixels()]
            output.write_image(oiio.UNKNOWN, s)
        
        i += 1