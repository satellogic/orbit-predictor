# MIT License
#
# Copyright (c) 2017 Satellogic SA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Inspired by https://github.com/poliastro/poliastro/blob/86f971c/src/poliastro/twobody/angles.py
# Copyright (c) 2012-2017 Juan Luis Cano Rodríguez, MIT license
"""Angles and anomalies.

"""
from math import sin, cos, tan, atan, sqrt

from orbit_predictor.utils import njit


@njit
def _kepler_equation(E, M, ecc):
    return E - ecc * sin(E) - M


@njit
def _kepler_equation_prime(E, _, ecc):
    return 1 - ecc * cos(E)


@njit
def ta_to_E(ta, ecc):
    """Eccentric anomaly from true anomaly.

    Parameters
    ----------
    ta : float
        True anomaly (rad).
    ecc : float
        Eccentricity.

    Returns
    -------
    E : float
        Eccentric anomaly.

    """
    E = 2 * atan(sqrt((1 - ecc) / (1 + ecc)) * tan(ta / 2))
    return E


@njit
def E_to_ta(E, ecc):
    """True anomaly from eccentric anomaly.

    Parameters
    ----------
    E : float
        Eccentric anomaly (rad).
    ecc : float
        Eccentricity.

    Returns
    -------
    ta : float
        True anomaly (rad).

    """
    ta = 2 * atan(sqrt((1 + ecc) / (1 - ecc)) * tan(E / 2))
    return ta


@njit
def M_to_E(M, ecc):
    """Eccentric anomaly from mean anomaly.

    Parameters
    ----------
    M : float
        Mean anomaly (rad).
    ecc : float
        Eccentricity.

    Returns
    -------
    E : float
        Eccentric anomaly.

    Note
    -----
    Algorithm taken from Vallado 2007, pp. 73.

    """
    E = M
    while True:
        E_new = E + (M - E + ecc * sin(E)) / (1 - ecc * cos(E))
        if (E_new == E) or (abs((E_new - E) / E) < 1e-15):
            break
        else:
            E = E_new

    return E_new


@njit
def E_to_M(E, ecc):
    """Mean anomaly from eccentric anomaly.

    Parameters
    ----------
    E : float
        Eccentric anomaly (rad).
    ecc : float
        Eccentricity.

    Returns
    -------
    M : float
        Mean anomaly (rad).

    """
    M = _kepler_equation(E, 0.0, ecc)
    return M


@njit
def M_to_ta(M, ecc):
    """True anomaly from mean anomaly.

    Parameters
    ----------
    M : float
        Mean anomaly (rad).
    ecc : float
        Eccentricity.

    Returns
    -------
    ta : float
        True anomaly (rad).

    Examples
    --------
    >>> ta = M_to_ta(radians(30.0), 0.06)
    >>> rad2deg(ta)
    33.673284930211658

    """
    E = M_to_E(M, ecc)
    ta = E_to_ta(E, ecc)
    return ta


@njit
def ta_to_M(ta, ecc):
    """Mean anomaly from true anomaly.

    Parameters
    ----------
    ta : float
        True anomaly (rad).
    ecc : float
        Eccentricity.

    Returns
    -------
    M : float
        Mean anomaly (rad).

    """
    E = ta_to_E(ta, ecc)
    M = E_to_M(E, ecc)
    return M
