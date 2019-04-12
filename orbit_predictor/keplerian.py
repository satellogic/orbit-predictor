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
# Inspired by
# https://github.com/poliastro/poliastro/blob/1d2f3ca/src/poliastro/twobody/classical.py
# https://github.com/poliastro/poliastro/blob/1d2f3ca/src/poliastro/twobody/rv.py
# Copyright (c) 2012-2017 Juan Luis Cano Rodríguez, MIT license

from math import cos, sin, sqrt

import numpy as np
from numpy.linalg import norm

from orbit_predictor.utils import transform, njit, cross


@njit
def rv_pqw(k, p, ecc, nu):
    """Returns r and v vectors in perifocal frame.

    """
    position_pqw = np.array([cos(nu), sin(nu), 0]) * p / (1 + ecc * cos(nu))
    velocity_pqw = np.array([-sin(nu), (ecc + cos(nu)), 0]) * sqrt(k / p)

    return position_pqw, velocity_pqw


@njit
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

    position_eci = transform(position_pqw, 2, -argp)
    position_eci = transform(position_eci, 0, -inc)
    position_eci = transform(position_eci, 2, -raan)

    velocity_eci = transform(velocity_pqw, 2, -argp)
    velocity_eci = transform(velocity_eci, 0, -inc)
    velocity_eci = transform(velocity_eci, 2, -raan)

    return position_eci, velocity_eci


@njit
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
    h = cross(r, v)
    n = cross([0, 0, 1], h) / norm(h)
    e = ((np.dot(v, v) - k / (norm(r))) * r - np.dot(r, v) * v) / k
    ecc = norm(e)
    p = np.dot(h, h) / k
    inc = np.arccos(h[2] / norm(h))

    circular = ecc < tol
    equatorial = abs(inc) < tol

    if equatorial and not circular:
        raan = 0
        argp = np.arctan2(e[1], e[0]) % (2 * np.pi)  # Longitude of periapsis
        ta = (np.arctan2(np.dot(h, cross(e, r)) / norm(h), np.dot(r, e)) %
              (2 * np.pi))
    elif not equatorial and circular:
        raan = np.arctan2(n[1], n[0]) % (2 * np.pi)
        argp = 0
        # Argument of latitude
        ta = (np.arctan2(np.dot(r, cross(h, n)) / norm(h), np.dot(r, n)) %
              (2 * np.pi))
    elif equatorial and circular:
        raan = 0
        argp = 0
        ta = np.arctan2(r[1], r[0]) % (2 * np.pi)  # True longitude
    else:
        raan = np.arctan2(n[1], n[0]) % (2 * np.pi)
        argp = (np.arctan2(np.dot(e, cross(h, n)) / norm(h), np.dot(e, n)) %
                (2 * np.pi))
        ta = (np.arctan2(np.dot(r, cross(h, e)) / norm(h), np.dot(r, e))
              % (2 * np.pi))

    return p, ecc, inc, raan, argp, ta
