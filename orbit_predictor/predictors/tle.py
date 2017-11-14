import datetime
from math import degrees

from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

from orbit_predictor.exceptions import NotReachable
from orbit_predictor.locations import Location
from orbit_predictor.predictors import PredictedPass
from orbit_predictor.predictors.base import CartesianPredictor, logger


class TLEPredictor(CartesianPredictor):

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
        return position_eci, velocity_eci

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
