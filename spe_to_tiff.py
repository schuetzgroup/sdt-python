# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 11:42:46 2014

@author: lukas
"""

import os
import sys
import configparser
import io
import collections

import OpenImageIO as oiio

import sdt_spe

plugin_searchpath = os.path.join(sys.prefix, "lib", "oiio-plugins")
oiio.attribute("plugin_searchpath", plugin_searchpath)

def convert_file(ifname, ofname):
    input = oiio.ImageInput.open(ifname)

    output = oiio.ImageOutput.create(ofname)

    i = 0
    while input.seek_subimage(i, 0):
        subspec = oiio.ImageSpec(input.spec())
        if i == 0:
            openmode = oiio.Create
            d = collections.OrderedDict()
            sdt_spe.read_spe_metadata(subspec, d)
            ini_str = sdt_spe.metadata_to_ini_string(d)
            subspec.attribute("ImageDescription", ini_str)
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
