"""GUIs for the Jupyter notebook
=============================

The :py:mod:`sdt.nbui` module contains graphical user interfaces to be used
within Jupyter notebooks.

Currently, there is the :py:class:`Locator` class for setting the parameters
for the :py:mod:`sdt.loc.daostorm_3d` algorithm with visual feedback.


Examples
--------

In one notebook cell, create the GUI. It is recommended to use the
``notebook`` matplotlib backend.

>>> img_files = sorted(glob("*.tif"))
>>> %matplotlib notebook
>>> locator = Locator(img_files)

Now one can play around with the parameters. Once a satisfactory
combination has been found, get the parameters in another notebook cell:

>>> par = locator.get_options()
>>> par
{'radius': 1.0,
 'model': '2d',
 'threshold': 800.0,
 'find_filter': 'Cg',
 'find_filter_opts': {'feature_radius': 3},
 'min_distance': None,
 'size_range': None}

``**par`` can be passed directly to :py:func:`sdt.loc.daostorm_3d.locate`
and :py:func:`sdt.loc.daostorm_3d.batch`:

>>> data = sdt.loc.daostorm_3d.batch(img_files[0], **par)


Programming reference
---------------------
.. autoclass:: Locator
    :members:
"""
from .locator import Locator
