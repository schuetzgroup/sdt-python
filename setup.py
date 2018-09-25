import os
import re

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# read version from sdt._version
vfile = os.path.join("sdt", "_version.py")
vstr = read(vfile)
vre = r"^__version__ = ['\"]([^'\"]*)['\"]"
match = re.search(vre, vstr, re.M)
if match:
    vstr = match.group(1)
else:
    raise RuntimeError("Unable to find version in " + vfile)


setup(
    name="sdt-python",
    version=vstr,
    description="Tools related to fluorescence microscopy",
    author="Lukas Schrangl",
    author_email="lukas.schrangl@tuwien.ac.at",
    # url =
    # license =
    install_requires=["numpy>=1.10",
                      "scipy>0.18",
                      "pandas",
                      "tables",
                      "tifffile>=0.14.0",
                      "pyyaml",
                      "pims>=0.3.0",
                      "matplotlib",
                      "pywavelets>=0.3.0",],
                      #"micro-helpers>=0.7"],
    packages=find_packages(include=["sdt*"]),
    package_data={"": ["*.ui"],
                  "sdt.gui": ["icons/*.svg"]},
    long_description=read("README.rst")
)
