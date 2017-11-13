# coding: utf-8
# Inspired by
# https://github.com/poliastro/poliastro/blob/1d2f3ca/src/poliastro/twobody/classical.py
# https://github.com/poliastro/poliastro/blob/1d2f3ca/src/poliastro/twobody/rv.py
# Copyright (c) 2012-2017 Juan Luis Cano Rodríguez, MIT license
from math import cos, sin, sqrt

import numpy as np
from numpy.linalg import norm

from orbit_predictor.utils import transform


def rv_pqw(k, p, ecc, nu):
    """Returns r and v vectors in perifocal frame.

    """
    position_pqw = np.array([cos(nu), sin(nu), 0]) * p / (1 + ecc * cos(nu))
    velocity_pqw = np.array([-sin(nu), (ecc + cos(nu)), 0]) * sqrt(k / p)

    return position_pqw, velocity_pqw


def coe2rv(k, p, ecc, inc, raan, argp, ta):
    """Converts from classical orbital elements to vectors.

    Parameters
    ----------
    k : float
        Standard gravitational parameter (km^3 / s^2).
    p : float
        Semi-latus rectum or parameter (km).
    ecc : float
        Eccentricity.
    inc : float
        Inclination (rad).
    raan : float
        Longitude of ascending node (rad).
    argp : float
        Argument of perigee (rad).
    ta : float
        True anomaly (rad).

    """
    position_pqw, velocity_pqw = rv_pqw(k, p, ecc, ta)

    position_eci = transform(np.array(position_pqw), 'z', -argp)
    position_eci = transform(position_eci, 'x', -inc)
    position_eci = transform(position_eci, 'z', -raan)

    velocity_eci = transform(np.array(velocity_pqw), 'z', -argp)
    velocity_eci = transform(velocity_eci, 'x', -inc)
    velocity_eci = transform(velocity_eci, 'z', -raan)

    return position_eci, velocity_eci


def rv2coe(k, r, v, tol=1e-8):
    """Converts from vectors to classical orbital elements.

    Parameters
    ----------
    k : float
        Standard gravitational parameter (km^3 / s^2).
    r : ndarray
        Position vector (km).
    v : ndarray
        Velocity vector (km / s).
    tol : float, optional
        Tolerance for eccentricity and inclination checks, default to 1e-8.

    """
    h = np.cross(r, v)
    n = np.cross([0, 0, 1], h) / norm(h)
    e = ((v.dot(v) - k / (norm(r))) * r - r.dot(v) * v) / k
    ecc = norm(e)
    p = h.dot(h) / k
    inc = np.arccos(h[2] / norm(h))

    circular = ecc < tol
    equatorial = abs(inc) < tol

    if equatorial and not circular:
        raan = 0
        argp = np.arctan2(e[1], e[0]) % (2 * np.pi)  # Longitude of periapsis
        ta = (np.arctan2(h.dot(np.cross(e, r)) / norm(h), r.dot(e)) %
              (2 * np.pi))
    elif not equatorial and circular:
        raan = np.arctan2(n[1], n[0]) % (2 * np.pi)
        argp = 0
        # Argument of latitude
        ta = (np.arctan2(r.dot(np.cross(h, n)) / norm(h), r.dot(n)) %
              (2 * np.pi))
    elif equatorial and circular:
        raan = 0
        argp = 0
        ta = np.arctan2(r[1], r[0]) % (2 * np.pi)  # True longitude
    else:
        raan = np.arctan2(n[1], n[0]) % (2 * np.pi)
        argp = (np.arctan2(e.dot(np.cross(h, n)) / norm(h), e.dot(n)) %
                (2 * np.pi))
        ta = (np.arctan2(r.dot(np.cross(h, e)) / norm(h), r.dot(e))
              % (2 * np.pi))

    return p, ecc, inc, raan, argp, ta
