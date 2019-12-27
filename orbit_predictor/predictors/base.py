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
import logging
import warnings
from collections import namedtuple
from math import pi, acos, degrees, radians

import numpy as np

from orbit_predictor.constants import MU_E
from orbit_predictor.exceptions import NotReachable, PropagationError
from orbit_predictor import coordinate_systems
from orbit_predictor.keplerian import rv2coe
from orbit_predictor.utils import (
    cross_product,
    dot_product,
    reify,
    vector_diff,
    vector_norm,
    gstime_from_datetime
)

logger = logging.getLogger(__name__)

ONE_SECOND = dt.timedelta(seconds=1)


def round_datetime(dt_):
    return dt_


class Position(namedtuple(
        "Position", ['when_utc', 'position_ecef', 'velocity_ecef', 'error_estimate'])):

    @reify
    def position_llh(self):
        """Latitude (deg), longitude (deg), altitude (km)."""
        return coordinate_systems.ecef_to_llh(self.position_ecef)

    @reify
    def osculating_elements(self):
        """Osculating Keplerian orbital elements.

        Semimajor axis (km), eccentricity, inclination (deg),
        right ascension of the ascending node or RAAN (deg),
        argument of perigee (deg), true anomaly (deg).

        """
        gmst = gstime_from_datetime(self.when_utc)
        position_eci = coordinate_systems.ecef_to_eci(self.position_ecef, gmst)
        velocity_eci = coordinate_systems.ecef_to_eci(self.velocity_ecef, gmst)

        # Convert position to Keplerian osculating elements
        p, ecc, inc, raan, argp, ta = rv2coe(
            MU_E, np.array(position_eci), np.array(velocity_eci)
        )
        # Transform to more familiar semimajor axis
        sma = p / (1 - ecc ** 2)

        return sma, ecc, degrees(inc), degrees(raan), degrees(argp), degrees(ta)


class PredictedPass:
    def __init__(self, location, sate_id,
                 max_elevation_deg,
                 aos, los, duration_s,
                 max_elevation_position=None,
                 max_elevation_date=None):
        self.location = location
        self.sate_id = sate_id
        self.max_elevation_position = max_elevation_position
        self.max_elevation_date = max_elevation_date
        self.max_elevation_deg = max_elevation_deg
        self.aos = aos
        self.los = los
        self.duration_s = duration_s

    @property
    def midpoint(self):
        """Returns a datetime of the midpoint of the pass"""
        return self.aos + (self.los - self.aos) / 2

    def __repr__(self):
        return "<PredictedPass {} over {} on {}>".format(self.sate_id, self.location, self.aos)

    def __eq__(self, other):
        return all([issubclass(other.__class__, PredictedPass),
                    self.location == other.location,
                    self.sate_id == other.sate_id,
                    self.max_elevation_position == other.max_elevation_position,
                    self.max_elevation_date == other.max_elevation_date,
                    self.max_elevation_deg == other.max_elevation_deg,
                    self.aos == other.aos,
                    self.los == other.los,
                    self.duration_s == other.duration_s])

    def get_off_nadir_angle(self):
        warnings.warn("This method is deprecated!", DeprecationWarning)
        return self.off_nadir_deg

    @reify
    def off_nadir_deg(self):
        """Computes off-nadir angle calculation

        Given satellite position ``sate_pos``, velocity ``sate_vel``, and
        location ``target`` in a common frame, off-nadir angle ``off_nadir_angle``
        is given by:
            t2b = sate_pos - target
            cos(off_nadir_angle) =     (sate_pos  · t2b)     # Vectorial dot product
                                    _______________________
                                    || sate_pos || || t2b||

        Sign for the rotation is calculated this way

        cross = target ⨯ sate_pos
        sign =   cross · sate_vel
               ____________________
               | cross · sate_vel |
        """
        sate_pos = self.max_elevation_position.position_ecef
        sate_vel = self.max_elevation_position.velocity_ecef
        target = self.location.position_ecef
        t2b = vector_diff(sate_pos, target)
        angle = acos(
            dot_product(sate_pos, t2b) / (vector_norm(sate_pos) * vector_norm(t2b))
        )

        cross = cross_product(target, sate_pos)
        dot = dot_product(cross, sate_vel)
        try:
            sign = dot / abs(dot)
        except ZeroDivisionError:
            sign = 1

        return degrees(angle) * sign


class Predictor:

    def __init__(self, sate_id, source):
        self.sate_id = sate_id
        self.source = source

    def propagate_eci(self, when_utc=None):
        raise NotImplementedError

    def get_position(self, when_utc=None):
        raise NotImplementedError("You have to implement it!")


class CartesianPredictor(Predictor):

    def _propagate_ecef(self, when_utc=None):
        """Return position and velocity in the given date using ECEF coordinate system."""
        if when_utc is None:
            when_utc = dt.datetime.utcnow()

        position_eci, velocity_eci = self.propagate_eci(when_utc)
        gmst = gstime_from_datetime(when_utc)
        position_ecef = coordinate_systems.eci_to_ecef(position_eci, gmst)
        velocity_ecef = coordinate_systems.eci_to_ecef(velocity_eci, gmst)
        return position_ecef, velocity_ecef

    @reify
    def mean_motion(self):
        raise NotImplementedError

    def get_position(self, when_utc=None):
        """Return a Position namedtuple in ECEF coordinate system"""
        if when_utc is None:
            when_utc = dt.datetime.utcnow()

        position_ecef, velocity_ecef = self._propagate_ecef(when_utc)

        return Position(when_utc=when_utc, position_ecef=position_ecef,
                        velocity_ecef=velocity_ecef, error_estimate=None)

    def get_only_position(self, when_utc):
        """Return a tuple in ECEF coordinate system"""
        return self.get_position(when_utc).position_ecef

    def passes_over(self, location, when_utc, limit_date=None, max_elevation_gt=0, aos_at_dg=0):
        return LocationPredictor(location, self, when_utc, limit_date,
                                 max_elevation_gt, aos_at_dg)

    def get_next_pass(self, location, when_utc=None, max_elevation_gt=5,
                      aos_at_dg=0, limit_date=None):
        """Return a PredictedPass instance with the data of the next pass over the given location

        location_llh: point on Earth we want to see from the satellite.
        when_utc: datetime UTC after which the pass is calculated, default to now.
        max_elevation_gt: filter passes with max_elevation under it.
        aos_at_dg: This is if we want to start the pass at a specific elevation.

        The next pass with a LOS strictly after when_utc will be returned,
        possibly the current pass.
        """
        if when_utc is None:
            when_utc = dt.datetime.utcnow()

        for pass_ in self.passes_over(location, when_utc, limit_date,
                                      max_elevation_gt=max_elevation_gt,
                                      aos_at_dg=aos_at_dg):
            return pass_
        else:
            raise NotReachable('Propagation limit date exceeded')


class GPSPredictor(Predictor):
    pass


class LocationPredictor:
    """Predicts passes over a given location
    Exposes an iterable interface
    """

    def __init__(self, location, predictor, start_date, limit_date=None,
                 max_elevation_gt=0, aos_at_dg=0, *, propagator=None):
        if propagator is not None:
            warnings.warn(
                "propagator parameter was renamed to predictor "
                "and will be removed in a future release",
                DeprecationWarning
            )
            predictor = propagator

        self.location = location
        self.predictor = predictor
        self.start_date = start_date
        self.limit_date = limit_date

        self.max_elevation_gt = radians(max([max_elevation_gt, aos_at_dg]))
        self.aos_at = radians(aos_at_dg)

    @property
    def propagator(self):
        warnings.warn(
            "propagator parameter was renamed to predictor "
            "and will be removed in a future release",
            DeprecationWarning
        )
        return self.predictor

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
                    if self.limit_date is not None and pass_.aos > self.limit_date:
                        break
                    yield self._build_predicted_pass(pass_)

                if self.limit_date is not None and current_date > self.limit_date:
                    break

                current_date = pass_.tca + self._orbit_step(0.6)

            else:
                current_date = self._find_nearest_ascending(current_date)

    def _build_predicted_pass(self, accuratepass):
        """Returns a classic predicted pass"""
        tca_position = self.predictor.get_position(accuratepass.tca)

        return PredictedPass(self.location, self.predictor.sate_id,
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
                         self.location, ascending_date, self.predictor.tle)
            raise PropagationError("Can not find an descending phase")

    def _find_nearest_ascending(self, descending_date):
        for candidate in self._sample_points(descending_date):
            if self.is_ascending(candidate):
                return candidate
        else:
            logger.error('Could not find an ascending pass over %s start date: %s - TLE: %s',
                         self.location, descending_date, self.predictor.tle)
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
        # TODO: Allow the precision to change from the outside
        return end - start <= ONE_SECOND

    @staticmethod
    def midpoint(start, end):
        """Returns the midpoint between two dates"""
        return start + (end - start) / 2

    def _elevation_at(self, when_utc):
        position = self.predictor.get_only_position(when_utc)
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
        step_in_radians = size * 2 * pi
        seconds = (step_in_radians / self.predictor.mean_motion) * 60
        return dt.timedelta(seconds=seconds)

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
