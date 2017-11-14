# -*- coding: utf8 -*-

import datetime
import logging
import warnings
from collections import namedtuple
from math import acos, degrees

from orbit_predictor import coordinate_systems
from orbit_predictor.utils import (
    cross_product,
    dot_product,
    reify,
    vector_diff,
    vector_norm,
    gstime_from_datetime)

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


class CartesianPredictor(Predictor):

    def _propagate_ecef(self, when_utc=None):
        """Return position and velocity in the given date using ECEF coordinate system."""
        position_eci, velocity_eci = self._propagate_eci(when_utc)
        gmst = gstime_from_datetime(when_utc)
        position_ecef = coordinate_systems.eci_to_ecef(position_eci, gmst)
        velocity_ecef = coordinate_systems.eci_to_ecef(velocity_eci, gmst)
        return position_ecef, velocity_ecef

    def get_position(self, when_utc=None):
        """Return a Position namedtuple in ECEF coordinate system"""
        if when_utc is None:
            when_utc = datetime.datetime.utcnow()
        position_ecef, velocity_ecef = self._propagate_ecef(when_utc)

        return Position(when_utc=when_utc, position_ecef=position_ecef,
                        velocity_ecef=velocity_ecef, error_estimate=None)


class GPSPredictor(Predictor):
    pass
