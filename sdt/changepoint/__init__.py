from ..helper import numba

from .bayes_offline import BayesOfflinePython, BayesOfflineNumba
from .bayes_online import BayesOnlinePython, BayesOnlineNumba


if numba.numba_available:
    BayesOffline = BayesOfflineNumba
    BayesOnline = BayesOnlineNumba
else:
    BayesOffline = BayesOfflinePython
    BayesOnline = BayesOnlinePython
