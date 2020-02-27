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

import datetime as dt
from math import radians, sqrt

from orbit_predictor.angles import ta_to_M, M_to_ta
from orbit_predictor.constants import MU_E
from orbit_predictor.keplerian import coe2rv
from orbit_predictor.predictors import TLEPredictor
from orbit_predictor.predictors.base import CartesianPredictor
from orbit_predictor.utils import mean_motion


def kepler(argp, delta_t_sec, ecc, inc, p, raan, sma, ta):
    # Initial mean anomaly
    M_0 = ta_to_M(ta, ecc)

    # Mean motion
    n = sqrt(MU_E / sma ** 3)

    # Propagation
    M = M_0 + n * delta_t_sec

    # New true anomaly
    ta = M_to_ta(M, ecc)

    # Position and velocity vectors
    position_eci, velocity_eci = coe2rv(MU_E, p, ecc, inc, raan, argp, ta)

    return position_eci, velocity_eci


class KeplerianPredictor(CartesianPredictor):
    """Propagator that uses the Keplerian osculating orbital elements.

    We use a naïve propagation algorithm that advances the anomaly the
    corresponding amount depending on the time difference and keeps all
    the rest of the osculating elements. It's robust against singularities
    as long as the starting elements are well specified but only works for
    elliptical orbits (ecc < 1). This limitation is not a problem since the object
    of study are artificial satellites orbiting the Earth.

    """
    def __init__(self, sma, ecc, inc, raan, argp, ta, epoch):
        """Initializes predictor.

        :param sma: Semimajor axis, km
        :param ecc: Eccentricity
        :param inc: Inclination, deg
        :param raan: Right ascension of the ascending node, deg
        :param argp: Argument of perigee, deg
        :param ta: True anomaly, deg
        :param epoch: Epoch, datetime
        """
        if ecc >= 1.0:
            raise NotImplementedError("Parabolic and elliptic orbits "
                                      "are not implemented")
        self._sma = sma
        self._ecc = ecc
        self._inc = inc
        self._raan = raan
        self._argp = argp
        self._ta = ta
        self._epoch = epoch

    @property
    def sate_id(self):
        # Keplerian predictors are not made of actual observations
        return "<custom>"

    @property
    def mean_motion(self):
        """Mean motion, in radians per minute"""
        return mean_motion(self._sma) * 60

    @classmethod
    def from_tle(cls, sate_id, source, date=None):
        """Returns approximate keplerian elements from TLE.

        The conversion between mean elements in the TEME reference
        frame to osculating elements in any standard reference frame
        is not well defined in literature (see Vallado 3rd edition, pp 236 to 240)

        """
        # Get latest TLE, or the one corresponding to a specified date
        if date is None:
            date = dt.datetime.utcnow()

        # Retrieve TLE position at given date as starting point
        pos = TLEPredictor(sate_id, source).get_position(date)

        return cls(*pos.osculating_elements, epoch=date)

    def propagate_eci(self, when_utc=None):
        """Return position and velocity in the given date using ECI coordinate system.

        """
        if when_utc is None:
            when_utc = dt.datetime.utcnow()

        # Orbit parameters
        sma = self._sma
        ecc = self._ecc
        p = sma * (1 - ecc ** 2)
        inc = radians(self._inc)
        raan = radians(self._raan)
        argp = radians(self._argp)
        ta = radians(self._ta)

        delta_t_sec = (when_utc - self._epoch).total_seconds()

        # Propagate
        position_eci, velocity_eci = kepler(argp, delta_t_sec, ecc, inc, p, raan, sma, ta)

        return tuple(position_eci), tuple(velocity_eci)
