# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause


if __name__ == "__main__":
    import sys
    from .legacy.locator import app

    ret = app.run(sys.argv)
    sys.exit(ret)
