# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import setup, find_packages


setup(
    python_requires=">=3.6",
    install_requires=["numpy>=1.10",
                      "pandas",
                      "tables",
                      "scipy>0.18",
                      "tifffile>=0.14.0",
                      "pyyaml",
                      "pims>=0.3.0",
                      "matplotlib",
                      "pywavelets>=0.3.0", ],
    packages=find_packages(include=["sdt*"]),
    package_data={"": ["*.ui"],
                  "sdt.gui": ["breeze-icons/icons/*/22/*.svg"]},
)
