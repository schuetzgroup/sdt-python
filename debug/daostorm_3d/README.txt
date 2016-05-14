Python scripts in here typically require a version of the Zhung lab software
(https://github.com/ZhuangLab/storm-analysis) that is compatible with Python 3.
The root directory of their software needs to be on the Python path. If the
script wants to use 3D-DAOSTORM specific things, one also needs to create
a symlink to the 3d_daostorm directory called "threed_daostorm", since Python
packages may not start with at number.
