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

from math import degrees, radians, sqrt, cos, sin
import datetime as dt

try:
    from math import isclose
except ImportError:
    def isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

import numpy as np

from orbit_predictor.constants import OMEGA, MU_E, R_E_KM, J2, OMEGA_E
from orbit_predictor.predictors.keplerian import KeplerianPredictor
from orbit_predictor.angles import ta_to_M, M_to_ta
from orbit_predictor.keplerian import coe2rv
from orbit_predictor.utils import njit, raan_from_ltan, float_to_hms, mean_motion


def is_sun_synchronous(predictor, rtol=1e-3, epoch=None):
    """Check if predictor corresponds to Sun-synchronous orbit within tolerance.

    """
    if epoch is None:
        epoch = dt.datetime.now()

    sma_km, ecc, inc_deg, *_ = predictor.get_position(epoch).osculating_elements
    p = sma_km * (1 - ecc ** 2)
    n = mean_motion(sma_km)

    raan_dot_sec = - 3 * n * R_E_KM ** 2 * J2 / (2 * p ** 2) * cos(radians(inc_deg))

    return isclose(raan_dot_sec, OMEGA, rel_tol=rtol)


def sun_sync_plane_constellation(num_satellites, *,
                                 alt_km=None, ecc=None, inc_deg=None, ltan_h=12, date=None):
    """Creates num_satellites in the same Sun-synchronous plane, uniformly spaced.

    Parameters
    ----------
    num_satellites : int
        Number of satellites.
    alt_km : float, optional
        Altitude, in km.
    ecc : float, optional
        Eccentricity.
    inc_deg : float, optional
        Inclination, in degrees.
    ltan_h : int, optional
        Local Time of the Ascending Node, in hours (default to noon).
    date : datetime.date, optional
        Reference date for the orbit, (default to today).

    """
    for ta_deg in np.linspace(0, 360, num_satellites, endpoint=False):
        yield J2Predictor.sun_synchronous(
            alt_km=alt_km, ecc=ecc, inc_deg=inc_deg, ltan_h=ltan_h, date=date, ta_deg=ta_deg
        )


def repeating_ground_track_sma(orbits, days=1, *, ecc, inc_deg=0, tolerance=1e-8):
    """Computes semimajor axis for repeating ground track orbit.

    Parameters
    ----------
    orbits : int
        Number of orbits in a given period.
    days : int, optional
        Number of days to cover the given orbits, default to 1.
    ecc : float
        Eccentricity.
    inc_deg : float, optional
        Inclination in degrees, default to 0 (equatorial).

    Returns
    -------
    sma : float
        Semimajor axis.

    Notes
    -----
    See Vallado "Fundamentals of Astrodynamics and Applications", 4th ed (2013)
    and Wertz et al. "Space Mission Engineering: The New SMAD" (2011).

    """
    if not (isinstance(orbits, int) and isinstance(days, int)):
        raise ValueError("Number of orbits and number of days must be integer.")

    k = orbits / days
    n = k * OMEGA_E

    while True:
        sma_new = np.cbrt(MU_E * (1 / n) ** 2)
        p = sma_new * (1 - ecc ** 2)
        node_dot = - 3 * n * J2 / 2 * (R_E_KM / p) ** 2 * np.cos(np.radians(inc_deg))
        argp_dot = 3 * n * J2 / 4 * (R_E_KM / p) ** 2 * (4 - 5 * np.sin(np.radians(inc_deg)) ** 2)
        M0_dot = (
            3 * n * J2 / 4 * (R_E_KM / p) ** 2 * np.sqrt(1 - ecc ** 2)
            * (2 - 3 * np.sin(np.radians(inc_deg)) ** 2)
        )
        n = k * (OMEGA_E - node_dot) - (M0_dot + argp_dot)
        sma = np.cbrt(MU_E * (1 / n) ** 2)
        if np.isclose(sma, sma_new, rtol=tolerance):
            break

    return sma


@njit
def pkepler(argp, delta_t_sec, ecc, inc, p, raan, sma, ta):
    """Perturbed Kepler problem (only J2)

    Notes
    -----
    Based on algorithm 64 of Vallado 3rd edition

    """
    # Mean motion
    n = sqrt(MU_E / sma ** 3)

    # Initial mean anomaly
    M_0 = ta_to_M(ta, ecc)

    # Update for perturbations
    delta_raan = (
        - (3 * n * R_E_KM ** 2 * J2) / (2 * p ** 2) *
        cos(inc) * delta_t_sec
    )
    raan = raan + delta_raan

    delta_argp = (
        (3 * n * R_E_KM ** 2 * J2) / (4 * p ** 2) *
        (4 - 5 * sin(inc) ** 2) * delta_t_sec
    )
    argp = argp + delta_argp

    M0_dot = (
        (3 * n * R_E_KM ** 2 * J2) / (4 * p ** 2) *
        (2 - 3 * sin(inc) ** 2) * sqrt(1 - ecc ** 2)
    )
    M_dot = n + M0_dot

    # Propagation
    M = M_0 + M_dot * delta_t_sec

    # New true anomaly
    ta = M_to_ta(M, ecc)

    # Position and velocity vectors
    position_eci, velocity_eci = coe2rv(MU_E, p, ecc, inc, raan, argp, ta)

    return position_eci, velocity_eci


class InvalidOrbitError(Exception):
    pass


class J2Predictor(KeplerianPredictor):
    """Propagator that uses secular variations due to J2.

    """
    @classmethod
    def sun_synchronous(cls, *, alt_km=None, ecc=None, inc_deg=None, ltan_h=12, date=None,
                        ta_deg=0):
        """Creates Sun synchronous predictor instance.

        Parameters
        ----------
        alt_km : float, optional
            Altitude, in km.
        ecc : float, optional
            Eccentricity.
        inc_deg : float, optional
            Inclination, in degrees.
        ltan_h : int, optional
            Local Time of the Ascending Node, in hours (default to noon).
        date : datetime.date, optional
            Reference date for the orbit, (default to today).
        ta_deg : float
            Increment or decrement of true anomaly, will adjust the epoch
            accordingly.

        Notes
        -----
        See Vallado "Fundamentals of Astrodynamics and Applications", 4th ed (2013)
        section 11.4.1.

        """
        if date is None:
            date = dt.datetime.today().date()

        try:
            with np.errstate(invalid="raise"):
                if alt_km is not None and ecc is not None:
                    # Normal case, solve for inclination
                    sma = R_E_KM + alt_km
                    inc_deg = degrees(np.arccos(
                        (-2 * sma ** (7 / 2) * OMEGA * (1 - ecc ** 2) ** 2)
                        / (3 * R_E_KM ** 2 * J2 * np.sqrt(MU_E))
                    ))

                elif alt_km is not None and inc_deg is not None:
                    # Not so normal case, solve for eccentricity
                    sma = R_E_KM + alt_km
                    ecc = np.sqrt(
                        1
                        - np.sqrt(
                            (-3 * R_E_KM ** 2 * J2 * np.sqrt(MU_E) * np.cos(radians(inc_deg)))
                            / (2 * OMEGA * sma ** (7 / 2))
                        )
                    )

                elif ecc is not None and inc_deg is not None:
                    # Rare case, solve for altitude
                    sma = (-np.cos(radians(inc_deg)) * (3 * R_E_KM ** 2 * J2 * np.sqrt(MU_E))
                           / (2 * OMEGA * (1 - ecc ** 2) ** 2)) ** (2 / 7)

                else:
                    raise ValueError(
                        "Exactly two of altitude, eccentricity and inclination must be given"
                    )

        except FloatingPointError as e:
            raise InvalidOrbitError(
                "Cannot find Sun-synchronous orbit with given parameters"
            ) from e

        # TODO: Allow change in time or location
        # Right the epoch is fixed given the LTAN, as well as the sub-satellite point
        epoch = dt.datetime(date.year, date.month, date.day, *float_to_hms(ltan_h))
        raan = raan_from_ltan(epoch, ltan_h)

        return cls(sma, ecc, inc_deg, raan, 0, ta_deg, epoch)

    @classmethod
    def repeating_ground_track(
            cls, *, orbits, days=1, ecc=0.0, inc_deg=0, raan_deg=0, argp_deg=0, ta_deg=0
    ):
        sma = repeating_ground_track_sma(orbits, days, ecc=ecc, inc_deg=inc_deg)

        return cls(sma, ecc, inc_deg, raan_deg, argp_deg, ta_deg, dt.datetime.now())

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
        position_eci, velocity_eci = pkepler(argp, delta_t_sec, ecc, inc, p, raan, sma, ta)

        return tuple(position_eci), tuple(velocity_eci)
