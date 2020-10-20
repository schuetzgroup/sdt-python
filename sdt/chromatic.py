# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import suppress
import warnings

import numpy as np

from . import channel_reg


class Corrector(channel_reg.Registrator):
    """Deprecated. Use `channel_reg.Registrator`."""
    yaml_tag = "!ChromaticCorrector"

    def __init__(self, *args, **kwargs):
        warnings.warn("`chromatic.Corrector` is deprecated. Use "
                      "`channel_reg.Registrator` as a drop-in replacement.",
                      np.VisibleDeprecationWarning)
        super().__init__(*args, **kwargs)

    def to_registrator(self) -> channel_reg.Registrator:
        """Convert to new :py:class:`Registrator` class"""
        reg = channel_reg.Registrator()
        reg.feat1 = self.feat1
        reg.feat2 = self.feat2
        reg.columns = self.columns
        reg.channel_names = self.channel_names
        reg.parameters1 = self.parameters1
        reg.parameters2 = self.parameters2
        return reg


with suppress(ImportError):
    from .io import yaml
    yaml.register_yaml_class(Corrector)
