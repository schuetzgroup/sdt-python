# -*- coding: utf-8 -*-

"""Tools to deal with SPE files as created by the SDT-control software"""

import collections
import configparser
import io
import logging
import os

try:
    import OpenImageIO as oiio
except:
    oiio = None

_logger = logging.getLogger(__name__)


extra_metadata_begin = "### Begin extra metadata ###"
extra_metadata_end = "### End extra metadata ###"
extra_metadata_version = (1, 0) #(major, minor)
excluded_metadata = ["DateTime"] #metadata that is saved otherwise anyways


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
    cp.read_string(inistr)
    return collections.OrderedDict(cp["metadata"])


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


def convert_folder(indir, ext_list, out_format, cont=False):
    """Recursively converts all files in a directory

    The converted files will have the same file names as the original ones
    with their extension replaced by out_format.

    :param indir: Input directory path

    :param ext_list: A tuple of extensions that will be (case-insensitively)
    matched. If a file has an extension in this tuple, it will be converted.
    Include the dot in the extension name, e. g. [".tiff"]

    :param out_format: Extension of the output format. Include the dot, e. g.
    ".jpg"

    :param cont: If True, conversion will continue even if an error was
    encountered. False by default.
    """
    for root, dirs, files in os.walk(indir):
        uc_ext_list = []
        for ext in ext_list:
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
