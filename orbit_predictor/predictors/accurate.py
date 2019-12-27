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
import warnings

from sgp4 import ext, model
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
from sgp4.propagation import _gstime

from orbit_predictor import coordinate_systems

from ..utils import reify, timetuple_from_dt
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

    @reify
    def tle(self):
        return self.source.get_tle(self.sate_id, dt.datetime.utcnow())

    @reify
    def _propagator(self):
        tle_line_1, tle_line_2 = self.tle.lines
        return twoline2rv(tle_line_1, tle_line_2, wgs84)

    @reify
    def propagator(self):
        warnings.warn(
            "The .propagator property will be made private in a future release",
            DeprecationWarning
        )
        return self._propagator

    @reify
    def mean_motion(self):
        return self._propagator.no  # this speed is in radians/minute

    @lru_cache(maxsize=3600 * 24 * 7)  # Max cache, a week
    def _propagate_only_position_ecef(self, timetuple):
        """Return position in the given date using ECEF coordinate system."""
        position_eci, _ = self._propagator.propagate(*timetuple)
        gmst = _gstime(jday(*timetuple))
        return coordinate_systems.eci_to_ecef(position_eci, gmst)

    def propagate_eci(self, when_utc=None):
        if when_utc is None:
            when_utc = dt.datetime.utcnow()

        timetuple = timetuple_from_dt(when_utc)

        position_eci, velocity_eci = self._propagator.propagate(*timetuple)
        if self._propagator.error != 0:
            raise RuntimeError(self._propagator.error_message)

        return position_eci, velocity_eci

    def get_only_position(self, when_utc):
        """Return a tuple in ECEF coordinate system

        Code is optimized, dont complain too much!
        """
        timetuple = timetuple_from_dt(when_utc)
        return self._propagate_only_position_ecef(timetuple)
