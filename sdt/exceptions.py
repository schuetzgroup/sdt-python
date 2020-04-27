# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collection of exception classes"""


class NoConvergence(Exception):
    """The result of an iterative algorithm did not converge

    Attributes
    ----------
    last_result
        The result of the last iteration before raising this exception
    """
    def __init__(self, last_result, text="Result did not converge"):
        """Parameters
        ----------
        last_result
            Set the :py:attr:`last_result` attribute.
        text : str, optional
            What to display when converting the exception to a str
        """
        super().__init__(text)
        self.last_result = last_result
