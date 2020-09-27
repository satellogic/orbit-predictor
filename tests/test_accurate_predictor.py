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
from unittest import TestCase, mock

import logassert
from hypothesis import example, given, settings
from hypothesis.strategies import floats, tuples, datetimes
import pytest

from orbit_predictor.predictors.base import ONE_SECOND
from orbit_predictor.exceptions import PropagationError
from orbit_predictor.locations import Location, ARG
from orbit_predictor.predictors import TLEPredictor
from orbit_predictor.predictors.pass_iterators import SmartLocationPredictor, LocationPredictor
from orbit_predictor.sources import MemoryTLESource


SATE_ID = '41558U'  # newsat 2
LINES = (
    '1 41558U 16033C   17065.21129769  .00002236  00000-0  88307-4 0  9995',
    '2 41558  97.4729 144.7611 0014207  16.2820 343.8872 15.26500433 42718',
)

BUGSAT_SATE_ID = 'BUGSAT-1'
BUGSAT1_TLE_LINES = (
    "1 40014U 14033E   14294.41438078  .00003468  00000-0  34565-3 0  3930",
    "2 40014  97.9781 190.6418 0032692 299.0467  60.7524 14.91878099 18425")


TRICKY_SAT_ID = "99999U"
TRICKY_SAT_TLE_LINES = (
    "1 99999U 20003C   20266.16335194  .00000680  00000-0  27888-4 0  9992",
    "2 99999  97.3095 330.4275 0013971 101.1830 259.0981 15.27499258 38313"
)


class AccuratePredictorTests(TestCase):

    def setUp(self):
        # Source
        self.db = MemoryTLESource()
        self.start = dt.datetime(2017, 3, 6, 7, 51)
        self.db.add_tle(SATE_ID, LINES, self.start)
        # Predictor
        self.predictor = TLEPredictor(SATE_ID, self.db)
        self.end = self.start + dt.timedelta(days=5)

    def test_predicted_passes_are_equal_between_executions(self):
        location = Location('bad-case-1', 11.937501570612568,
                            -55.35189435098657, 1780.674044538666)
        first_set = list(
            self.predictor.passes_over(location, self.start, self.end))
        second_set = list(
            self.predictor.passes_over(location, self.start + dt.timedelta(seconds=3), self.end)
        )

        # We use delta=ONE_SECOND because
        # that's the hardcoded value for the precision
        self.assertAlmostEqual(first_set[0].aos, second_set[0].aos, delta=ONE_SECOND)
        self.assertAlmostEqual(first_set[0].los, second_set[0].los, delta=ONE_SECOND)

    def test_predicted_passes_have_elevation_positive_and_visible_on_date(self):
        end = self.start + dt.timedelta(days=60)
        for pass_ in self.predictor.passes_over(ARG, self.start, end):
            self.assertGreater(pass_.max_elevation_deg, 0)
            position = self.predictor.get_position(pass_.max_elevation_date)
            ARG.is_visible(position)
            self.assertGreaterEqual(pass_.off_nadir_deg, -90)
            self.assertLessEqual(pass_.off_nadir_deg, 90)

    def test_predicted_passes_off_nadir_angle_works(self):
        start = dt.datetime(2017, 3, 6, 13, 30)
        end = start + dt.timedelta(hours=1)
        location = Location('bad-case-1', 11.937501570612568,
                            -55.35189435098657, 1780.674044538666)

        pass_ = self.predictor.get_next_pass(location, when_utc=start, limit_date=end)
        self.assertGreaterEqual(0, pass_.off_nadir_deg)

    @given(start=datetimes(
               min_value=dt.datetime(2017, 1, 1),
               max_value=dt.datetime(2020, 12, 31),
           ),
           location=tuples(
               floats(min_value=-90, max_value=90),
               floats(min_value=0, max_value=180),
               floats(min_value=-200, max_value=9000)
           ))
    @settings(max_examples=10000, deadline=None)
    @example(start=dt.datetime(2017, 1, 26, 11, 51, 51),
             location=(-37.69358328273305, 153.96875, 0.0))
    def test_pass_is_always_returned(self, start, location):
        location = Location('bad-case-1', *location)
        pass_ = self.predictor.get_next_pass(location, start)
        self.assertGreater(pass_.max_elevation_deg, 0)

    def test_aos_deg_can_be_used_in_get_next_pass(self):
        start = dt.datetime(2017, 3, 6, 13, 30)
        end = start + dt.timedelta(hours=1)
        location = Location('bad-case-1', 11.937501570612568,
                            -55.35189435098657, 1780.674044538666)
        complete_pass = self.predictor.get_next_pass(location, when_utc=start,
                                                     limit_date=end)

        pass_with_aos = self.predictor.get_next_pass(location, when_utc=start,
                                                     limit_date=end,
                                                     aos_at_dg=5)

        self.assertGreater(pass_with_aos.aos, complete_pass.aos)
        self.assertLess(pass_with_aos.aos, complete_pass.max_elevation_date)
        self.assertAlmostEqual(pass_with_aos.max_elevation_date,
                               complete_pass.max_elevation_date,
                               delta=dt.timedelta(seconds=1))

        self.assertGreater(pass_with_aos.los, complete_pass.max_elevation_date)
        self.assertLess(pass_with_aos.los, complete_pass.los)

        position = self.predictor.get_position(pass_with_aos.aos)
        _, elev = location.get_azimuth_elev_deg(position)

        self.assertAlmostEqual(elev, 5, delta=0.1)

        position = self.predictor.get_position(pass_with_aos.los)
        _, elev = location.get_azimuth_elev_deg(position)

        self.assertAlmostEqual(elev, 5, delta=0.1)

    def test_predicted_passes_whit_aos(self):
        end = self.start + dt.timedelta(days=60)
        for pass_ in self.predictor.passes_over(ARG, self.start, end, aos_at_dg=5):
            self.assertGreater(pass_.max_elevation_deg, 5)
            position = self.predictor.get_position(pass_.aos)
            _, elev = ARG.get_azimuth_elev_deg(position)
            self.assertAlmostEqual(elev, 5, delta=0.1)


class AccurateVsGpredictTests(TestCase):

    def setUp(self):
        # Source
        self.db = MemoryTLESource()
        self.db.add_tle(BUGSAT_SATE_ID, BUGSAT1_TLE_LINES, dt.datetime.utcnow())
        # Predictor
        self.predictor = TLEPredictor(BUGSAT_SATE_ID, self.db)

    def test_get_next_pass_with_stk_data(self):
        STK_DATA = """
        ------------------------------------------------------------------------------------------------
         AOS                      TCA                      LOS                      Duration      Max El
        ------------------------------------------------------------------------------------------------
         2014/10/23 01:27:33.224  2014/10/23 01:32:41.074  2014/10/23 01:37:47.944  00:10:14.720   12.76
         2014/10/23 03:01:37.007  2014/10/23 03:07:48.890  2014/10/23 03:14:01.451  00:12:24.000   39.32
         2014/10/23 14:49:34.783  2014/10/23 14:55:44.394  2014/10/23 15:01:51.154  00:12:16.000   41.75
         2014/10/23 16:25:54.939  2014/10/23 16:30:50.152  2014/10/23 16:35:44.984  00:09:50.000   11.45
         2014/10/24 01:35:47.889  2014/10/24 01:41:13.181  2014/10/24 01:46:37.548  00:10:50.000   16.07
         2014/10/24 03:10:23.486  2014/10/24 03:16:27.230  2014/10/24 03:22:31.865  00:12:08.000   30.62
         2014/10/24 14:58:07.378  2014/10/24 15:04:21.721  2014/10/24 15:10:33.546  00:12:26.000   54.83
         2014/10/24 16:34:48.635  2014/10/24 16:39:20.960  2014/10/24 16:43:53.204  00:09:04.000    8.78
         2014/10/25 01:44:05.771  2014/10/25 01:49:45.487  2014/10/25 01:55:24.414  00:11:18.000   20.07
         2014/10/25 03:19:12.611  2014/10/25 03:25:05.674  2014/10/25 03:30:59.815  00:11:47.000   24.09"""  # NOQA

        for line in STK_DATA.splitlines()[4:]:
            line_parts = line.split()
            aos = dt.datetime.strptime(" ".join(line_parts[:2]), '%Y/%m/%d %H:%M:%S.%f')
            max_elevation_date = dt.datetime.strptime(" ".join(line_parts[2:4]),
                                                      '%Y/%m/%d %H:%M:%S.%f')
            los = dt.datetime.strptime(" ".join(line_parts[4:6]), '%Y/%m/%d %H:%M:%S.%f')
            duration = dt.datetime.strptime(line_parts[6], '%H:%M:%S.%f')
            duration_s = dt.timedelta(
                minutes=duration.minute, seconds=duration.second).total_seconds()
            max_elev_deg = float(line_parts[7])

            try:
                date = pass_.los  # NOQA
            except UnboundLocalError:
                date = dt.datetime.strptime(
                    "2014-10-22 20:18:11.921921", '%Y-%m-%d %H:%M:%S.%f')

            pass_ = self.predictor.get_next_pass(ARG, date)
            self.assertAlmostEqual(pass_.aos, aos, delta=ONE_SECOND)
            self.assertAlmostEqual(pass_.los, los, delta=ONE_SECOND)
            self.assertAlmostEqual(pass_.max_elevation_date, max_elevation_date, delta=ONE_SECOND)
            self.assertAlmostEqual(pass_.duration_s, duration_s, delta=2 * 1)
            self.assertAlmostEqual(pass_.max_elevation_deg, max_elev_deg, delta=0.05)


class AccuratePredictorCalculationErrorTests(TestCase):
    """Check that we can learn from calculation errors and provide patches for corner cases"""

    def setUp(self):
        # Source
        self.db = MemoryTLESource()
        self.db.add_tle(BUGSAT_SATE_ID, BUGSAT1_TLE_LINES, dt.datetime.utcnow())
        # Predictor
        self.predictor = TLEPredictor(BUGSAT_SATE_ID, self.db)
        self.is_ascending_mock = self._patch(
            'orbit_predictor.predictors.base.LocationPredictor._is_ascending')
        self.start = dt.datetime(2017, 3, 6, 7, 51)
        logassert.setup(self,  'orbit_predictor.predictors.pass_iterators')

    def _patch(self, *args,  **kwargs):
        patcher = mock.patch(*args, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_ascending_failure(self):
        self.is_ascending_mock.return_value = False
        with self.assertRaises(PropagationError):
            self.predictor.get_next_pass(ARG, self.start,
                                         location_predictor_class=LocationPredictor)

        self.assertLoggedError(str(ARG), str(self.start), *BUGSAT1_TLE_LINES)

    def test_descending_failure(self):
        self.is_ascending_mock.return_value = True
        with self.assertRaises(PropagationError):
            self.predictor.get_next_pass(ARG, self.start,
                                         location_predictor_class=LocationPredictor)

        self.assertLoggedError(str(ARG), str(self.start), *BUGSAT1_TLE_LINES)


class SkippedPassesRegressionTests(TestCase):
    """Check that we do not skip passes"""
    # See https://github.com/satellogic/orbit-predictor/issues/99

    def setUp(self):
        self.db = MemoryTLESource()
        self.db.add_tle(TRICKY_SAT_ID, TRICKY_SAT_TLE_LINES, dt.datetime.now())
        self.predictor = TLEPredictor(TRICKY_SAT_ID, self.db)

    @pytest.mark.xfail(reason="Legacy LocationPredictor skips some passes")
    def test_pass_is_not_skipped_old(self):
        loc = Location(
            name="loc",
            latitude_deg=-15.137152171507697,
            longitude_deg=-0.4276612055384211,
            elevation_m=1.665102900005877e-05,
        )

        PASS_DATE = dt.datetime(2020, 9, 25, 9, 2, 6)
        LIMIT_DATE = dt.datetime(2020, 9, 25, 10, 36, 0)

        predicted_passes = list(self.predictor.passes_over(
            loc,
            when_utc=PASS_DATE,
            limit_date=LIMIT_DATE,
            aos_at_dg=0, max_elevation_gt=0,
            location_predictor_class=LocationPredictor,
        ))

        assert predicted_passes

    def test_pass_is_not_skipped_smart(self):
        loc = Location(
            name="loc",
            latitude_deg=-15.137152171507697,
            longitude_deg=-0.4276612055384211,
            elevation_m=1.665102900005877e-05,
        )

        PASS_DATE = dt.datetime(2020, 9, 25, 9, 2, 6)
        LIMIT_DATE = dt.datetime(2020, 9, 25, 10, 36, 0)

        predicted_passes = list(self.predictor.passes_over(
            loc,
            when_utc=PASS_DATE,
            limit_date=LIMIT_DATE,
            aos_at_dg=0, max_elevation_gt=0,
            location_predictor_class=SmartLocationPredictor,
        ))

        assert predicted_passes
