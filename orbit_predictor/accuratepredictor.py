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
import datetime
import logging
import math
from datetime import timedelta
from math import degrees

from sgp4 import ext, model
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
from sgp4.propagation import _gstime

from orbit_predictor import coordinate_systems
from orbit_predictor.exceptions import NotReachable, PropagationError
from orbit_predictor.predictors import PredictedPass, TLEPredictor
from orbit_predictor.utils import lru_cache, reify

logger = logging.getLogger(__name__)

ONE_SECOND = datetime.timedelta(seconds=1)

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


def round_datetime(dt):
    return datetime.datetime(*dt.timetuple()[:6])


class HighAccuracyTLEPredictor(TLEPredictor):
    """A pass predictor with high accuracy on estimations"""

    def __init__(self, sate_id, source):
        super(HighAccuracyTLEPredictor, self).__init__(sate_id, source)

    @reify
    def tle(self):
        return self.source.get_tle(self.sate_id, datetime.datetime.utcnow())

    @reify
    def propagator(self):
        tle_line_1, tle_line_2 = self.tle.lines
        return twoline2rv(tle_line_1, tle_line_2, wgs84)

    @reify
    def mean_motion(self):
        return self.propagator.no  # this speed is in radians/minute

    @lru_cache(maxsize=3600 * 24 * 7)  # Max cache, a week
    def _propagate_only_position_ecef(self, timetuple):
        """Return position in the given date using ECEF coordinate system."""
        position_eci, _ = self.propagator.propagate(*timetuple)
        gmst = _gstime(jday(*timetuple))
        return coordinate_systems.eci_to_ecef(position_eci, gmst)

    def _propagate_ecef(self, when_utc):
        """Return position and velocity in the given date using ECEF coordinate system."""
        timetuple = (when_utc.year, when_utc.month, when_utc.day,
                     when_utc.hour, when_utc.minute, when_utc.second)

        position_eci, velocity_eci = self.propagator.propagate(*timetuple)
        gmst = _gstime(jday(*timetuple))
        position_ecef = coordinate_systems.eci_to_ecef(position_eci, gmst)
        velocity_ecef = coordinate_systems.eci_to_ecef(velocity_eci, gmst)
        return (position_ecef, velocity_ecef)

    def get_only_position(self, when_utc):
        """Return a tuple in ECEF coordinate system

        Code is optimized, dont complain too much!
        """
        timetuple = (when_utc.year, when_utc.month, when_utc.day,
                     when_utc.hour, when_utc.minute, when_utc.second)
        return self._propagate_only_position_ecef(timetuple)

    def passes_over(self, location, when_utc, limit_date=None, max_elevation_gt=0, aos_at_dg=0):
        return LocationPredictor(location, self, when_utc, limit_date,
                                 max_elevation_gt, aos_at_dg)

    def get_next_pass(self, location, when_utc=None, max_elevation_gt=5,
                      aos_at_dg=0, limit_date=None):
        """Implements same api as standard predictor"""
        if when_utc is None:
            when_utc = datetime.datetime.utcnow()

        for pass_ in self.passes_over(location, when_utc, limit_date,
                                      max_elevation_gt=max_elevation_gt,
                                      aos_at_dg=aos_at_dg):
                return pass_
        else:
            raise NotReachable('Propagation limit date exceded')


class AccuratePredictedPass:

    def __init__(self, aos, tca, los, max_elevation):
        self.aos = round_datetime(aos) if aos is not None else None
        self.tca = round_datetime(tca)
        self.los = round_datetime(los) if los is not None else None
        self.max_elevation = max_elevation

    @property
    def valid(self):
        return self.max_elevation > 0 and self.aos is not None and self.los is not None

    @reify
    def max_elevation_deg(self):
        return degrees(self.max_elevation)

    @reify
    def duration(self):
        return self.los - self.aos


class LocationPredictor(object):
    """Predicts passes over a given location
    Exposes an iterable interface
    """

    def __init__(self, location, propagator, start_date, limit_date=None,
                 max_elevation_gt=0, aos_at_dg=0):
        self.location = location
        self.propagator = propagator
        self.start_date = start_date
        self.limit_date = limit_date

        self.max_elevation_gt = math.radians(max([max_elevation_gt, aos_at_dg]))
        self.aos_at = math.radians(aos_at_dg)

    def __iter__(self):
        """Returns one pass each time"""
        current_date = self.start_date
        while True:
            if self.is_ascending(current_date):
                # we need a descending point
                ascending_date = current_date
                descending_date = self._find_nearest_descending(ascending_date)
                pass_ = self._refine_pass(ascending_date, descending_date)
                if pass_.valid:
                    yield self._build_predicted_pass(pass_)

                if self.limit_date is not None and current_date > self.limit_date:
                    break

                current_date = pass_.tca + self._orbit_step(0.6)

            else:
                current_date = self._find_nearest_ascending(current_date)

    def _build_predicted_pass(self, accuratepass):
        """Returns a classic predicted pass"""
        tca_position = self.propagator.get_position(accuratepass.tca)

        return PredictedPass(self.location, self.propagator.sate_id,
                             max_elevation_deg=accuratepass.max_elevation_deg,
                             aos=accuratepass.aos,
                             los=accuratepass.los,
                             duration_s=accuratepass.duration.total_seconds(),
                             max_elevation_position=tca_position,
                             max_elevation_date=accuratepass.tca,
                             )

    def _find_nearest_descending(self, ascending_date):
        for candidate in self._sample_points(ascending_date):
            if not self.is_ascending(candidate):
                return candidate
        else:
            logger.error('Could not find a descending pass over %s start date: %s - TLE: %s',
                         self.location, ascending_date, self.propagator.tle)
            raise PropagationError("Can not find an descending phase")

    def _find_nearest_ascending(self, descending_date):
        for candidate in self._sample_points(descending_date):
            if self.is_ascending(candidate):
                return candidate
        else:
            logger.error('Could not find an ascending pass over %s start date: %s - TLE: %s',
                         self.location, descending_date, self.propagator.tle)
            raise PropagationError('Can not find an ascending phase')

    def _sample_points(self, date):
        """Helper method to found ascending or descending phases of elevation"""
        start = date
        end = date + self._orbit_step(0.99)
        mid = self.midpoint(start, end)
        mid_right = self.midpoint(mid, end)
        mid_left = self.midpoint(start, mid)

        return [end, mid, mid_right, mid_left]

    def _refine_pass(self, ascending_date, descending_date):
        tca = self._find_tca(ascending_date, descending_date)
        elevation = self._elevation_at(tca)

        if elevation > self.max_elevation_gt:
            aos = self._find_aos(tca)
            los = self._find_los(tca)
        else:
            aos = los = None

        return AccuratePredictedPass(aos, tca, los, elevation)

    def _find_tca(self, ascending_date, descending_date):
        while not self._precision_reached(ascending_date, descending_date):
            midpoint = self.midpoint(ascending_date, descending_date)
            if self.is_ascending(midpoint):
                ascending_date = midpoint
            else:
                descending_date = midpoint

        return ascending_date

    def _precision_reached(self, start, end):
        return end - start <= ONE_SECOND

    @staticmethod
    def midpoint(start, end):
        """Returns the midpoint between two dates"""
        return start + (end - start) / 2

    def _elevation_at(self, when_utc):
        position = self.propagator.get_only_position(when_utc)
        return self.location.elevation_for(position)

    def is_passing(self, when_utc):
        """Returns a boolean indicating if satellite is actually visible"""
        return bool(self._elevation_at(when_utc))

    def is_ascending(self, when_utc):
        """Check is elevation is ascending or descending on a given point"""
        elevation = self._elevation_at(when_utc)
        next_elevation = self._elevation_at(when_utc + ONE_SECOND)
        return elevation <= next_elevation

    def _orbit_step(self, size):
        """Returns a time step, that will make the satellite advance a given number of orbits"""
        step_in_radians = size * 2 * math.pi
        seconds = (step_in_radians / self.propagator.mean_motion) * 60
        return timedelta(seconds=seconds)

    def _find_aos(self, tca):
        end = tca
        start = tca - self._orbit_step(0.34)  # On third of the orbit
        elevation = self._elevation_at(start)
        assert elevation < 0
        while not self._precision_reached(start, end):
            midpoint = self.midpoint(start, end)
            elevation = self._elevation_at(midpoint)
            if elevation < self.aos_at:
                start = midpoint
            else:
                end = midpoint
        return end

    def _find_los(self, tca):
        start = tca
        end = tca + self._orbit_step(0.34)
        while not self._precision_reached(start, end):
            midpoint = self.midpoint(start, end)
            elevation = self._elevation_at(midpoint)

            if elevation < self.aos_at:
                end = midpoint
            else:
                start = midpoint

        return start
