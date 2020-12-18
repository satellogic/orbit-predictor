import datetime as dt
import logging
from math import pi, acos, degrees, radians
import warnings

import numpy as np
try:
    from scipy.signal import find_peaks
    from scipy.optimize import root_scalar, minimize_scalar
except ImportError:
    warnings.warn(
        "scipy module was not found, some features may not work properly",
        ImportWarning,
        stacklevel=2,
    )

from orbit_predictor.exceptions import PropagationError
from orbit_predictor.utils import (
    cross_product,
    dot_product,
    reify,
    vector_diff,
    vector_norm,
    orbital_period,
)

ONE_SECOND = dt.timedelta(seconds=1)

logger = logging.getLogger(__name__)


def round_datetime(dt_):
    return dt_


class BaseLocationPredictor:

    def __init__(self, location, predictor, start_date, limit_date=None,
                 max_elevation_gt=0, aos_at_dg=0, tolerance_s=1.0):
        self.location = location
        self.predictor = predictor
        self.start_date = start_date
        self.limit_date = limit_date

        self.max_elevation_gt = radians(max([max_elevation_gt, aos_at_dg]))
        self.aos_at = radians(aos_at_dg)

        self.tolerance_s = tolerance_s
        self.tolerance = dt.timedelta(seconds=tolerance_s)

    def __iter__(self):
        yield from self.iter_passes()

    def iter_passes(self):
        """Yields passes"""
        raise NotImplementedError


class LocationPredictor(BaseLocationPredictor):
    """Predicts passes over a given location.

    Exposes an iterable interface.
    Notice that this algorithm is not fully exhaustive,
    see https://github.com/satellogic/orbit-predictor/issues/99 for details.

    """

    def iter_passes(self):
        """Returns one pass each time"""
        current_date = self.start_date
        while True:
            if self._is_ascending(current_date):
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
            if not self._is_ascending(candidate):
                return candidate
        else:
            logger.error('Could not find a descending pass over %s start date: %s - TLE: %s',
                         self.location, ascending_date, self.predictor.tle)
            raise PropagationError("Can not find an descending phase")

    def _find_nearest_ascending(self, descending_date):
        for candidate in self._sample_points(descending_date):
            if self._is_ascending(candidate):
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
            if self._is_ascending(midpoint):
                ascending_date = midpoint
            else:
                descending_date = midpoint

        return ascending_date

    def _precision_reached(self, start, end):
        return end - start <= self.tolerance

    @staticmethod
    def midpoint(start, end):
        """Returns the midpoint between two dates"""
        return start + (end - start) / 2

    def _elevation_at(self, when_utc):
        position = self.predictor.get_only_position(when_utc)
        return self.location.elevation_for(position)

    def _is_ascending(self, when_utc):
        """Check is elevation is ascending or descending on a given point"""
        elevation = self._elevation_at(when_utc)
        next_elevation = self._elevation_at(when_utc + self.tolerance)
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


class SmartLocationPredictor(BaseLocationPredictor):
    """Predicts passes over a given location using a different algorithm.

    This uses a sampling interval of 3 minutes,
    which seems like a good compromise for Low-Earth Orbits.
    However, this means that, under certain circumstances, passes
    shorter than this duration could theoretically be missed.

    """

    def iter_passes(self):
        # Explore all values of t every 3 minutes
        t_values = np.arange(0, (self.limit_date - self.start_date).total_seconds(), 180)
        elev_values = np.array([self._elevation(delta_seconds) for delta_seconds in t_values])

        peaks_idx, _ = find_peaks(elev_values)

        for peak_idx in peaks_idx:
            if elev_values[peak_idx] < self.aos_at:
                continue
            else:
                t_approximate_tca = t_values[peak_idx]
                period_s = orbital_period(self.predictor.mean_motion) * 60

                aos, tca, los, max_elevation = self._refine_pass(t_approximate_tca, period_s)

                yield PredictedPass(
                    self.location,
                    self.predictor.sate_id,
                    max_elevation_deg=degrees(max_elevation),
                    aos=aos,
                    los=los,
                    duration_s=(los - aos).total_seconds(),
                    max_elevation_position=self.predictor.get_position(tca),
                    max_elevation_date=tca,
                )

    def _elevation(self, delta_seconds):
        when_utc = self.start_date + dt.timedelta(seconds=delta_seconds)
        position = self.predictor.get_only_position(when_utc)
        return self.location.elevation_for(position)

    def _refine_pass(self, t_approximate_tca, period_s):
        # AOS must be between half the previous period and the approximate TCA
        t_aos = root_scalar(
            lambda t: self._elevation(t) - self.aos_at,
            bracket=(t_approximate_tca - period_s / 2, t_approximate_tca),
            xtol=self.tolerance_s,
            method="brentq",
        ).root
        aos = self.start_date + dt.timedelta(seconds=t_aos)

        # LOS must be between the approximate TCA and half the next period
        t_los = root_scalar(
            lambda t: self._elevation(t) - self.aos_at,
            bracket=(t_approximate_tca, t_approximate_tca + period_s / 2),
            xtol=self.tolerance_s,
            method="brentq",
        ).root
        los = self.start_date + dt.timedelta(seconds=t_los)

        # Find date for maximum elevation between AOS and LOS
        # NOTE: If the tolerance is too loose, wrong results might be returned!
        res_tca = minimize_scalar(
            lambda t: -self._elevation(t),
            bracket=(t_aos, (t_los + t_aos) / 2, t_los),
            method="brent",
        )
        t_tca = res_tca.x
        max_elevation = -res_tca.fun
        tca = self.start_date + dt.timedelta(seconds=t_tca)

        return aos, tca, los, max_elevation


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
