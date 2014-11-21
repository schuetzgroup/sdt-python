# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 11:42:46 2014

@author: lukas
"""

import os
import OpenImageIO as oiio

def convert_file(ifname, ofname):
    oiio.ImageInput.create(ifname, "../spe/spe.imageio/build/")
    input = oiio.ImageInput.open(ifname)

    outSpec = input.spec()
    output = oiio.ImageOutput.create(ofname)

    i = 0
    while input.seek_subimage(i, 0):
        subspec = input.spec()
        if i == 0:
            openmode = oiio.Create
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
