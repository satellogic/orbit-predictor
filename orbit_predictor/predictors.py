# -*- coding: utf-8 -*-
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


import datetime
import logging
import warnings
from collections import namedtuple
from math import acos, degrees

from sgp4.earth_gravity import wgs84
from sgp4.ext import jday
from sgp4.io import twoline2rv
from sgp4.propagation import _gstime

from orbit_predictor import coordinate_systems
from orbit_predictor.exceptions import NotReachable
from orbit_predictor.locations import Location
from orbit_predictor.utils import (
    cross_product,
    dot_product,
    reify,
    vector_diff,
    vector_norm
)

logger = logging.getLogger(__name__)


class Position(namedtuple(
        "Position", ['when_utc', 'position_ecef', 'velocity_ecef', 'error_estimate'])):

    @reify
    def position_llh(self):
        return coordinate_systems.ecef_to_llh(self.position_ecef)


class PredictedPass(object):
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
            cos(off_nadir_angle) =     (target  · t2b)     # Vectorial dot product
                                    _____________________
                                    || target || || t2b||

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
            dot_product(target, t2b) / (vector_norm(target) * vector_norm(t2b))
        )

        cross = cross_product(target, sate_pos)
        dot = dot_product(cross, sate_vel)
        try:
            sign = dot / abs(dot)
        except ZeroDivisionError:
            sign = 1

        return degrees(angle) * sign


class Predictor(object):

    def __init__(self, source, sate_id):
        self.source = source
        self.sate_id = sate_id

    def get_position(self, when_utc=None):
        raise NotImplementedError("You have to implement it!")


class TLEPredictor(Predictor):

    def __init__(self, sate_id, source):
        super(TLEPredictor, self).__init__(source, sate_id)
        self._iterations = 0

    def _propagate_eci(self, when_utc=None):
        """Return position and velocity in the given date using ECI coordinate system."""
        tle = self.source.get_tle(self.sate_id, when_utc)
        logger.debug("Propagating using ECI. sate_id: %s, when_utc: %s, tle: %s",
                     self.sate_id, when_utc, tle)
        tle_line_1, tle_line_2 = tle.lines
        sgp4_sate = twoline2rv(tle_line_1, tle_line_2, wgs84)
        timetuple = when_utc.timetuple()[:6]
        position_eci, velocity_eci = sgp4_sate.propagate(*timetuple)
        return (position_eci, velocity_eci)

    def _gstime_from_datetime(self, when_utc):
        timetuple = when_utc.timetuple()[:6]
        return _gstime(jday(*timetuple))

    def _propagate_ecef(self, when_utc=None):
        """Return position and velocity in the given date using ECEF coordinate system."""
        position_eci, velocity_eci = self._propagate_eci(when_utc)
        gmst = self._gstime_from_datetime(when_utc)
        position_ecef = coordinate_systems.eci_to_ecef(position_eci, gmst)
        velocity_ecef = coordinate_systems.eci_to_ecef(velocity_eci, gmst)
        return (position_ecef, velocity_ecef)

    def get_position(self, when_utc=None):
        """Return a Position namedtuple in ECEF coordinate system"""
        if when_utc is None:
            when_utc = datetime.datetime.utcnow()
        position_ecef, velocity_ecef = self._propagate_ecef(when_utc)

        return Position(when_utc=when_utc, position_ecef=position_ecef,
                        velocity_ecef=velocity_ecef, error_estimate=None)

    def get_next_pass(self, location, when_utc=None, max_elevation_gt=5,
                      aos_at_dg=0, limit_date=None):
        """Return a PredictedPass instance with the data of the next pass over the given location

        locattion_llh: point on Earth we want to see from the satellite.
        when_utc: datetime UTC.
        max_elevation_gt: filter passings with max_elevation under it.
        aos_at_dg: This is if we want to start the pass at a specific elevation.
        """
        if when_utc is None:
            when_utc = datetime.datetime.utcnow()
        if max_elevation_gt < aos_at_dg:
            max_elevation_gt = aos_at_dg
        pass_ = self._get_next_pass(location, when_utc, aos_at_dg, limit_date)
        while pass_.max_elevation_deg < max_elevation_gt:
            pass_ = self._get_next_pass(
                location, pass_.los, aos_at_dg, limit_date)  # when_utc is changed!
        return pass_

    def _get_next_pass(self, location, when_utc, aos_at_dg=0, limit_date=None):
        if not isinstance(location, Location):
            raise TypeError("location must be a Location instance")

        pass_ = PredictedPass(location=location, sate_id=self.sate_id, aos=None, los=None,
                              max_elevation_date=None, max_elevation_position=None,
                              max_elevation_deg=0, duration_s=0)

        seconds = 0
        self._iterations = 0
        while True:
            # to optimize the increment in seconds must be inverse proportional to
            # the distance of 0 elevation
            date = when_utc + datetime.timedelta(seconds=seconds)

            if limit_date is not None and date > limit_date:
                raise NotReachable('Propagation limit date exceded')

            elev_pos = self.get_position(date)
            _, elev = location.get_azimuth_elev(elev_pos)
            elev_deg = degrees(elev)

            if elev_deg > pass_.max_elevation_deg:
                pass_.max_elevation_position = elev_pos
                pass_.max_elevation_date = date
                pass_.max_elevation_deg = elev_deg

            if elev_deg > aos_at_dg and pass_.aos is None:
                pass_.aos = date
            if pass_.aos and elev_deg < aos_at_dg:
                pass_.los = date
                pass_.duration_s = (pass_.los - pass_.aos).total_seconds()
                break

            if elev_deg < -2:
                delta_s = abs(elev_deg) * 15 + 10
            else:
                delta_s = 20

            seconds += delta_s
            self._iterations += 1

        return pass_


class GPSPredictor(Predictor):
    pass
