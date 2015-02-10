# -*- coding: utf-8 -*-

"""Tools to deal with SPE files as created by the SDT-control software"""

import collections
import configparser
import io
import logging

try:
    import OpenImageIO as oiio
except:
    oiio = None

_logger = logging.getLogger(__name__)

def parse_comments(comments, dict_to_append_to):
    """Parse metadata written into SPE comments by SDT-control

    The SDT-control software writes metadata that is not in the official
    SPE spefication into the comment fields 2 and 5 encoded as a ASCII
    string. User comments (as entered into the "comments" field in the
    SPE-control software) are in comment fields 1 and 3.

    This is a helper function used by :read_spe_metadata:.

    :param comments: The raw comments as read from the SPE file, a list of
    five strings of 80 characters each

    :param dict_to_append_to: A dictionary to which the parsed comments
    are appended as :OrderedDicts:. Each of these :OrderedDicts: contains
    one logical group of metadata, such as information about SDT-control,
    imaging conditions, etc.
    """
    info = dict_to_append_to
    c2 = comments[1]
    c5 = comments[4]
    if c5[70:78] != "COMVER05":
        raise RuntimeError("Metadata has an unknown format. Not all metadata can be read.")
    sdtDict = collections.OrderedDict()
    sdtDict["major version"] = int(c5[66:68])
    sdtDict["minor version"] = int(c5[68:70])
    sdtDict["controller name"] = c5[0:6]
    info["SDT-control"] = sdtDict
    imgDict = collections.OrderedDict()
    imgDict["exposure time"] = float(c2[64:73])/10**6
    imgDict["color code"] = c5[10:14]
    imgDict["detection channels"] = int(c5[15])
    imgDict["background subtraction"] = (c5[14] == "B")
    imgDict["EM active"] = (c5[32] == "E")
    imgDict["EM gain"] = int(c5[28:32])
    imgDict["laser modulation"] = (c5[33] == "A")
    imgDict["pixel size"] = float(c5[25:28])/10
    info["image acquisition"] = imgDict
    seqDict = collections.OrderedDict()
    seqDict["method"] = c5[6:10]
    seqDict["grid"] = float(c2[16:25])/10**6
    seqDict["N macro"] = int(c2[0:4])
    seqDict["delay macro"] = float(c2[10:19])/10**3
    seqDict["N mini"] = int(c2[4:7])
    seqDict["delay mini"] = float(c2[19:28])/10**6
    seqDict["N micro"] = int(c2[7:10])
    seqDict["delay micro"] = float(c2[28:37])/10**6
    seqDict["subpics"] = int(c2[7:10])
    seqDict["shutter delay"] = float(c2[73:79])/10**6
    info["sequence settings"] = seqDict
    tocDict = collections.OrderedDict()
    tocDict["prebleach delay"] = float(c2[37:46])/10**6
    tocDict["bleach time"] = float(c2[46:55])/10**6
    tocDict["recovery time"] = float(c2[55:64])/10**6
    info["TOCCSL settings"] = tocDict
    info["comment"] = comments[0] + comments [2]


def parse_datetime(date_string, time_string):
    """Parse an SDT-control SPE file's date string

    SDT-control writes the date string as "ddmmmyyyy", where dd are two
    digits for the day, mmm the three letter abbreviation of the month
    in German, and yyyy four digits for the year.

    The time string are six digits "hhmmss" for hours, minutes and
    seconds.

    This is a helper function used by :read_spe_metadata:.

    :param date_string: The date string from the SPE file

    :param time_string: The time string from the SPE file

    :returns: An OpenImageIO DateTime attribute compatible string of the
    form "yyyy:mm:dd hh:mm:ss" with digits representing the year, month,
    day, hour, minute, and second, respectively.
    """
    month_dict = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "Mai": "05",
        "Jun": "06", "Jul": "07", "Aug": "08", "Sep": "09", "Okt": "10",
        "Nov": "11", "Dez": "12"
        }

    return "{}:{}:{} {}:{}:{}".format(
        date_string[5:9], month_dict[date_string[2:5]], date_string[0:2],
        time_string[0:2], time_string[2:4], time_string[4:6]
        )


def read_spe_metadata(spec, dict_to_append_to):
    """Read all metadata generated by the SDT-control software

    This also sets the DateTime attribute of the input :ImageSpec:
    argument correctly, since the SPE file contains a weird way of
    saving this data.

    :param spec: :OpenImageIO.ImageSpec: of the SPE file

    :param dict_to_append_to: A dictionary to which the parsed comments
    are appended as :OrderedDict:s. Each of these :OrderedDict:s contains
    one logical group of metadata, such as information about SDT-control,
    imaging conditions, etc.
    """
    data = dict_to_append_to

    comments = [""]*5
    for i in range(5):
        comments[i] = spec.get_attribute("Comment{}".format(i+1))
    try:
        parse_comments(comments, data)
    except:
        _logger.info("Could not parse SPE file comments.")

    data["laser modulation script"] = spec.get_attribute("Spare_4")

    roiDict = collections.OrderedDict()
    roiDict["start x"] = spec.get_attribute("ROI_startx")
    roiDict["end x"] = spec.get_attribute("ROI_endx")
    roiDict["group x"] = spec.get_attribute("ROI_groupx")
    roiDict["start y"] = spec.get_attribute("ROI_starty")
    roiDict["end y"] = spec.get_attribute("ROI_endy")
    roiDict["group y"] = spec.get_attribute("ROI_groupy")
    data["ROI"] = roiDict

    try:
        datetime = parse_datetime(spec.get_attribute("date"),
                                  spec.get_attribute("ExperimentTimeLocal"))
        spec.attribute("DateTime", datetime)
    except:
        _logger.warn("Failed to read date and time from SPE file.")


def read_imagedesc_metadata(ini, dict_to_append_to):
    """Read metadata from ini-file-like string

    Converts the ini-file-like string to a :collections.OrderedDict:
    of :OrderedDicts:.

    :param ini: Either the string itself or a :OpenImageIO.ImageDesc:
    whose "ImageDescription" attribute is the string.

    :param dict_to_append_to: A dictionary to which the parsed comments
    are appended as :OrderedDicts:. Each of these :OrderedDicts: contains
    one logical group of metadata, such as information about SDT-control,
    imaging conditions, etc.
    """
    data = dict_to_append_to

    if isinstance(ini, str):
        inistr = ini
    elif oiio is not None:
        if isinstance(ini, oiio.ImageSpec):
            inistr = ini.get_attribute("ImageDescription")
    else:
        raise TypeError("Expected str or OpenImageIO.ImageSpec")

    config = configparser.ConfigParser(dict_type = collections.OrderedDict)
    config.read_string(inistr)
    #Copy the config to a dict
    for s in config.sections():
        d = collections.OrderedDict()
        for i in config[s]:
            d[i] = config[s][i]
        data[s] = d


def metadata_to_ini_string(metadata):
    """Convert the metadata dicts to ini-file type string

    Use this function to convert the :OrderedDicts: created by
    :read_spe_metadata: or :read_imagedesc_metadata: into ini-file like
    strings that can be saved to the ImageDescription of a converted file.

    :param metadata: (ordered) dictionary of metadata

    :returns: ini-file like string of metadata
    """
    cp = configparser.ConfigParser(dict_type = collections.OrderedDict)
    cp.read_dict(metadata)
    strio = io.StringIO()
    config.write(strio)
    return strio.getvalue()
