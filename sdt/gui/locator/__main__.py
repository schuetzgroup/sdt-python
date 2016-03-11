import sys

from . import app

ret = app.run(sys.argv)
sys.exit(ret)
