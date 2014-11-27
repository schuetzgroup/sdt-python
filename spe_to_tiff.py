# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 11:42:46 2014

@author: lukas
"""

import os
import configparser
import io
import OpenImageIO as oiio

def spe_attributes_to_string(inSpec):
    config = configparser.ConfigParser()
    config.add_section("Comments")
    config["Comments"]["Comment1"] = inSpec.get_attribute("Comment1")
    config["Comments"]["Comment2"] = inSpec.get_attribute("Comment2")
    config["Comments"]["Comment3"] = inSpec.get_attribute("Comment3")
    config["Comments"]["Comment4"] = inSpec.get_attribute("Comment4")
    config["Comments"]["Comment5"] = inSpec.get_attribute("Comment5")
    config.add_section("ROI")
    config["ROI"]["startx"] = str(inSpec.get_attribute("ROI_startx"))
    config["ROI"]["endx"] = str(inSpec.get_attribute("ROI_endx"))
    config["ROI"]["groupx"] = str(inSpec.get_attribute("ROI_groupx"))
    config["ROI"]["starty"] = str(inSpec.get_attribute("ROI_starty"))
    config["ROI"]["endy"] = str(inSpec.get_attribute("ROI_endy"))
    config["ROI"]["groupy"] = str(inSpec.get_attribute("ROI_groupy"))
    config.add_section("Laser modulation")
    config["Laser modulation"]["script"] = inSpec.get_attribute("Spare_4")
    strio = io.StringIO()
    config.write(strio)
    return strio.getvalue()

def convert_file(ifname, ofname):
    oiio.ImageInput.create(ifname, "../spe/spe.imageio/build/")
    input = oiio.ImageInput.open(ifname)

    output = oiio.ImageOutput.create(ofname)

    i = 0
    while input.seek_subimage(i, 0):
        subspec = oiio.ImageSpec(input.spec())
        if i == 0:
            openmode = oiio.Create
            subspec.attribute("ImageDescription", spe_attributes_to_string(input.spec()))
        else:
            openmode = oiio.AppendSubimage

        output.open(ofname, subspec, openmode)
        output.copy_image(input)

        i += 1

    input.close()
    output.close()

def convert_folder(indir):
    for root, dirs, files in os.walk(indir):
        for file in files:
            path, ext = so.path.splitext(file)
            if ext.upper() == "SPE":
                infile = os.path.join(root, file)
                outfile = os.path.join(root, path + ".tiff")
                convert_file(infile, outfile)
