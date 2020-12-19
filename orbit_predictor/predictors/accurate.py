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

"""
Accurate Predictor
~~~~~~~~~~~~~~~~~~

Provides a faster an better predictor

IMPORTANT!!
All calculations use radians, instead degrees
Be warned!



Known Issues
~~~~~~~~~~~~
In some cases not studied deeply, (because we don't have enough data)
ascending or descending point are not found and propagation fails.

Code use some hacks to prevent multiple calculations, and function calls are
small as posible.

Some stuff won't be trivial to understand, but comments and fixes are welcome

"""
import datetime as dt
from functools import lru_cache

from sgp4 import ext, model
from sgp4.api import Satrec, SGP4_ERRORS
from sgp4.earth_gravity import wgs84
from sgp4.model import WGS84
from sgp4.propagation import gstime

from orbit_predictor import coordinate_systems
from ..exceptions import PropagationError

from ..utils import reify, jday_from_datetime, unkozai
from .base import CartesianPredictor

# Hack Zone be warned


@lru_cache(maxsize=365)
def jday_day(year, mon, day):
    return (367.0 * year -
            7.0 * (year + ((mon + 9.0) // 12.0)) * 0.25 // 1.0 +
            275.0 * mon // 9.0 +
            day + 1721013.5)


def jday(year, mon, day, hr, minute, sec):
    base = jday_day(year, mon, day)
    return base + ((sec / 60.0 + minute) / 60.0 + hr) / 24.0


ext.jday = jday
model.jday = jday

# finish hack zone


class HighAccuracyTLEPredictor(CartesianPredictor):
    """A pass predictor with high accuracy on estimations"""

    def __init__(self, sate_id, source):
        self._sate_id = sate_id
        self._source = source
        self.tle = self._source.get_tle(self.sate_id, dt.datetime.utcnow())
        self._propagator = self._get_propagator()

    def _get_propagator(self):
        tle_line_1, tle_line_2 = self.tle.lines
        return Satrec.twoline2rv(tle_line_1, tle_line_2, WGS84)

    def __getstate__(self):
        # See https://docs.python.org/3/library/pickle.html#handling-stateful-objects
        state = self.__dict__.copy()
        del state["_propagator"]
        return state

    def __setstate__(self, state):
        # See https://docs.python.org/3/library/pickle.html#handling-stateful-objects
        self.__dict__.update(state)
        self._propagator = self._get_propagator()

    @property
    def sate_id(self):
        return self._sate_id

    @property
    def source(self):
        return self._source

    @reify
    def mean_motion(self):
        """Mean motion, in radians per minute"""
        return unkozai(
            self._propagator.no_kozai, self._propagator.ecco, self._propagator.inclo, wgs84
        )

    @lru_cache(maxsize=3600 * 24 * 7)  # Max cache, a week
    def _propagate_only_position_ecef(self, when_utc):
        """Return position in the given date using ECEF coordinate system."""
        jd, fr = jday_from_datetime(when_utc)
        status, position_eci, _ = self._propagator.sgp4(jd, fr)
        if status != 0:
            raise PropagationError(SGP4_ERRORS[status])

        gmst = gstime(jd + fr)
        return coordinate_systems.eci_to_ecef(position_eci, gmst)

    def propagate_eci(self, when_utc=None):
        if when_utc is None:
            when_utc = dt.datetime.utcnow()

        jd, fr = jday_from_datetime(when_utc)
        status, position_eci, velocity_eci = self._propagator.sgp4(jd, fr)
        if status != 0:
            raise PropagationError(SGP4_ERRORS[status])

        return position_eci, velocity_eci

    def get_only_position(self, when_utc=None):
        """Return a tuple in ECEF coordinate system

        Code is optimized, dont complain too much!
        """
        if when_utc is None:
            when_utc = dt.datetime.utcnow()

        return self._propagate_only_position_ecef(when_utc)
