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
from math import pi, degrees

import numpy as np
try:
    from scipy.optimize import brentq, minimize_scalar
except ImportError:
    warnings.warn('scipy module was not found, some features may not work properly.',
                  ImportWarning)

from orbit_predictor.constants import MU_E
from orbit_predictor.exceptions import NotReachable
from orbit_predictor import coordinate_systems
from orbit_predictor.keplerian import rv2coe
from orbit_predictor.utils import (
    angle_between,
    reify,
    vector_norm,
    gstime_from_datetime,
    get_shadow,
    get_sun,
    eclipse_duration,
    get_satellite_minus_penumbra_verticals,
)

from .pass_iterators import (  # noqa: F401
    LocationPredictor,
    PredictedPass,
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

        # NOTE: rv2coe already does % (2 * np.pi)
        # but under some circumstances this might require another pass,
        # see https://github.com/satellogic/orbit-predictor/pull/106#issuecomment-730177598
        return sma, ecc, degrees(inc), degrees(raan), degrees(argp), degrees(ta) % 360


class Predictor:

    @property
    def sate_id(self):
        raise NotImplementedError

    def propagate_eci(self, when_utc=None):
        raise NotImplementedError

    def get_position(self, when_utc=None):
        raise NotImplementedError("You have to implement it!")

    def get_shadow(self, when_utc=None):
        """Gives illumination at given time (2 for illuminated, 1 for penumbra, 0 for umbra)."""
        if when_utc is None:
            when_utc = dt.datetime.utcnow()

        return get_shadow(
            self.get_position(when_utc).position_ecef,
            when_utc
        )

    def get_normal_vector(self, when_utc=None):
        """Gets unitary normal vector (orthogonal to orbital plane) at given time."""
        if when_utc is None:
            when_utc = dt.datetime.utcnow()

        position, velocity = self.propagate_eci(when_utc)
        orbital_plane_normal = np.cross(position, velocity)
        return orbital_plane_normal / vector_norm(orbital_plane_normal)

    def get_beta(self, when_utc=None):
        """Gets angle between orbital plane and Sun direction (beta) at given time, in degrees."""
        if when_utc is None:
            when_utc = dt.datetime.utcnow()

        # Here we calculate the complementary angle of beta,
        # because we use the normal vector of the orbital plane
        beta_comp = angle_between(
            get_sun(when_utc),
            self.get_normal_vector(when_utc)
        )

        # We subtract from 90 degrees to return the real beta angle
        return 90 - beta_comp


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
        """Mean motion, in radians per minute"""
        raise NotImplementedError

    @reify
    def period(self):
        """Orbital period, in minutes"""
        return 2 * pi / self.mean_motion

    def get_position(self, when_utc=None):
        """Return a Position namedtuple in ECEF coordinate system"""
        if when_utc is None:
            when_utc = dt.datetime.utcnow()

        position_ecef, velocity_ecef = self._propagate_ecef(when_utc)

        return Position(when_utc=when_utc, position_ecef=position_ecef,
                        velocity_ecef=velocity_ecef, error_estimate=None)

    def get_only_position(self, when_utc=None):
        """Return a tuple in ECEF coordinate system"""
        return self.get_position(when_utc).position_ecef

    def get_eclipse_duration(self, when_utc=None, tolerance=1e-1):
        """Gets eclipse duration at given time, in minutes"""
        ecc = self.get_position(when_utc).osculating_elements[1]
        if ecc > tolerance:
            raise NotImplementedError("Non circular orbits are not supported")

        beta = self.get_beta(when_utc)
        return eclipse_duration(beta, self.period)

    def passes_over(self, location, when_utc, limit_date=None, max_elevation_gt=0, aos_at_dg=0,
                    location_predictor_class=LocationPredictor, tolerance_s=1.0):
        return location_predictor_class(location, self, when_utc, limit_date,
                                        max_elevation_gt, aos_at_dg, tolerance_s=tolerance_s)

    def get_next_pass(self, location, when_utc=None, max_elevation_gt=5,
                      aos_at_dg=0, limit_date=None,
                      location_predictor_class=LocationPredictor, tolerance_s=1.0):
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
                                      aos_at_dg=aos_at_dg,
                                      location_predictor_class=location_predictor_class,
                                      tolerance_s=tolerance_s):
            return pass_
        else:
            raise NotReachable('Propagation limit date exceeded')

    def eclipses_since(self, when_utc=None, limit_date=None):
        """
        An iterator that yields all eclipses start and end times between
        when_utc and limit_date.

        The next eclipse with a end strictly after when_utc will be returned,
        possibly the current eclipse.
        The last eclipse returned starts before limit_date, but it can end
        strictly after limit_date.
        No circular orbits are not supported, and will raise NotImplementedError.
        """
        def _get_illumination(t):
            my_start = start + dt.timedelta(seconds=t)
            result = get_satellite_minus_penumbra_verticals(
                self.get_only_position(my_start),
                my_start
            )
            return result

        if when_utc is None:
            when_utc = dt.datetime.utcnow()

        orbital_period_s = self.period * 60
        # A third of the orbit period is used as the base window of the search.
        # This window ensures the function get_satellite_minus_penumbra_verticals
        # will not have more than one local minimum (one in the illuminated phase and
        # the other in penumbra).
        base_search_window_s = orbital_period_s / 3
        start = when_utc

        while limit_date is None or start < limit_date:

            # a minimum negative value is aproximatelly the middle point of the eclipse
            minimum_illumination = minimize_scalar(
                _get_illumination,
                bounds=(0, base_search_window_s),
                method="bounded",
                options={"xatol": 1e-2},
            )
            eclipse_center_candidate_delta_s = minimum_illumination.x

            # If found a minimum that is not illuminated, there is an eclipse here
            if _get_illumination(eclipse_center_candidate_delta_s) < 0:
                # Search now both zeros to get the start and end of the eclipse
                # We know that in (0, base_search_window_s) there is a minimum with negative value,
                # and also on the opposite side of the eclipse we expect sunlight,
                # therefore we already have two robust bracketing intervals
                eclipse_start_delta_s = brentq(
                    _get_illumination,
                    eclipse_center_candidate_delta_s - orbital_period_s / 2,
                    eclipse_center_candidate_delta_s,
                    xtol=1e-2,
                    full_output=False,
                )
                eclipse_end_delta_s = brentq(
                    _get_illumination,
                    eclipse_center_candidate_delta_s,
                    eclipse_center_candidate_delta_s + orbital_period_s / 2,
                    xtol=1e-2,
                    full_output=False,
                )
                eclipse_start = start + dt.timedelta(seconds=eclipse_start_delta_s)
                eclipse_end = start + dt.timedelta(seconds=eclipse_end_delta_s)
                yield eclipse_start, eclipse_end
                start = eclipse_end + dt.timedelta(seconds=base_search_window_s)
            else:
                start += dt.timedelta(seconds=base_search_window_s)


class GPSPredictor(Predictor):
    pass
