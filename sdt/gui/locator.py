# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import sys


if __name__ == "__main__":
    from .legacy.locator import app

    ret = app.run(sys.argv)
    sys.exit(ret)
