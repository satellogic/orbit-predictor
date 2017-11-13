# coding: utf-8
# Inspired by https://github.com/poliastro/poliastro/blob/86f971c/src/poliastro/twobody/angles.py
# Copyright (c) 2012-2017 Juan Luis Cano Rodríguez, MIT license
"""Angles and anomalies.

"""
from math import sin, cos, tan, atan, sqrt

from numpy import isclose


def _kepler_equation(E, M, ecc):
    return E - ecc * sin(E) - M


def _kepler_equation_prime(E, _, ecc):
    return 1 - ecc * cos(E)


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
        if isclose(E_new, E, rtol=1e-15):
            break
        else:
            E = E_new

    return E_new


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
