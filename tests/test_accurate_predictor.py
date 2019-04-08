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

import time
from datetime import datetime, timedelta
from unittest import TestCase

import logassert
from hypothesis import example, given, settings
from hypothesis.strategies import floats, tuples, datetimes

from orbit_predictor.accuratepredictor import (
    ONE_SECOND,
    HighAccuracyTLEPredictor
)
from orbit_predictor.exceptions import PropagationError
from orbit_predictor.locations import Location, ARG, EUROPA1
from orbit_predictor.predictors import TLEPredictor
from orbit_predictor.sources import MemoryTLESource

try:
    from unittest import mock  # py3
except ImportError:
    import mock  # py2


SATE_ID = '41558U'  # newsat 2
LINES = (
    '1 41558U 16033C   17065.21129769  .00002236  00000-0  88307-4 0  9995',
    '2 41558  97.4729 144.7611 0014207  16.2820 343.8872 15.26500433 42718',
)

BUGSAT_SATE_ID = 'BUGSAT-1'
BUGSAT1_TLE_LINES = (
    "1 40014U 14033E   14294.41438078  .00003468  00000-0  34565-3 0  3930",
    "2 40014  97.9781 190.6418 0032692 299.0467  60.7524 14.91878099 18425")


class AccuratePredictorTests(TestCase):

    def setUp(self):
        # Source
        self.db = MemoryTLESource()
        self.start = datetime(2017, 3, 6, 7, 51)
        self.db.add_tle(SATE_ID, LINES, self.start)
        # Predictor
        self.predictor = HighAccuracyTLEPredictor(SATE_ID, self.db)
        self.end = self.start + timedelta(days=5)

    def test_predicted_passes_are_equal_between_executions(self):
        location = Location('bad-case-1', 11.937501570612568,
                            -55.35189435098657, 1780.674044538666)
        first_set = list(
            self.predictor.passes_over(location, self.start, self.end))
        second_set = list(
            self.predictor.passes_over(location, self.start + timedelta(seconds=3), self.end)
        )

        # We use delta=ONE_SECOND because
        # that's the hardcoded value for the precision
        self.assertAlmostEqual(first_set[0].aos, second_set[0].aos, delta=ONE_SECOND)
        self.assertAlmostEqual(first_set[0].los, second_set[0].los, delta=ONE_SECOND)

    def test_predicted_passes_have_elevation_positive_and_visible_on_date(self):
        end = self.start + timedelta(days=60)
        for pass_ in self.predictor.passes_over(ARG, self.start, end):
            self.assertGreater(pass_.max_elevation_deg, 0)
            position = self.predictor.get_position(pass_.max_elevation_date)
            ARG.is_visible(position)
            self.assertGreaterEqual(pass_.off_nadir_deg, -90)
            self.assertLessEqual(pass_.off_nadir_deg, 90)

    def test_predicted_passes_off_nadir_angle_works(self):
        start = datetime(2017, 3, 6, 13, 30)
        end = start + timedelta(hours=1)
        location = Location('bad-case-1', 11.937501570612568,
                            -55.35189435098657, 1780.674044538666)

        pass_ = self.predictor.get_next_pass(location, when_utc=start, limit_date=end)
        self.assertGreaterEqual(0, pass_.off_nadir_deg)

    @given(start=datetimes(
               min_value=datetime(2017, 1, 1),
               max_value=datetime(2020, 12, 31),
           ),
           location=tuples(
               floats(min_value=-90, max_value=90),
               floats(min_value=0, max_value=180),
               floats(min_value=-200, max_value=9000)
           ))
    @settings(max_examples=10000, deadline=None)
    @example(start=datetime(2017, 1, 26, 11, 51, 51),
             location=(-37.69358328273305, 153.96875, 0.0))
    def test_pass_is_always_returned(self, start, location):
        location = Location('bad-case-1', *location)
        pass_ = self.predictor.get_next_pass(location, start)
        self.assertGreater(pass_.max_elevation_deg, 0)

    def test_aos_deg_can_be_used_in_get_next_pass(self):
        start = datetime(2017, 3, 6, 13, 30)
        end = start + timedelta(hours=1)
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
                               delta=timedelta(seconds=1))

        self.assertGreater(pass_with_aos.los, complete_pass.max_elevation_date)
        self.assertLess(pass_with_aos.los, complete_pass.los)

        position = self.predictor.get_position(pass_with_aos.aos)
        _, elev = location.get_azimuth_elev_deg(position)

        self.assertAlmostEqual(elev, 5, delta=0.1)

        position = self.predictor.get_position(pass_with_aos.los)
        _, elev = location.get_azimuth_elev_deg(position)

        self.assertAlmostEqual(elev, 5, delta=0.1)

    def test_predicted_passes_whit_aos(self):
        end = self.start + timedelta(days=60)
        for pass_ in self.predictor.passes_over(ARG, self.start, end, aos_at_dg=5):
            self.assertGreater(pass_.max_elevation_deg, 5)
            position = self.predictor.get_position(pass_.aos)
            _, elev = ARG.get_azimuth_elev_deg(position)
            self.assertAlmostEqual(elev, 5, delta=0.1)


class AccurateVsGpredictTests(TestCase):

    def setUp(self):
        # Source
        self.db = MemoryTLESource()
        self.db.add_tle(BUGSAT_SATE_ID, BUGSAT1_TLE_LINES, datetime.now())
        # Predictor
        self.predictor = HighAccuracyTLEPredictor(BUGSAT_SATE_ID, self.db)

    def test_get_next_pass_with_gpredict_data(self):
        GPREDICT_DATA = """
        -------------------------------------------------------------------------------------------------
         AOS                  TCA                  LOS                  Duration  Max El  AOS Az  LOS Az
        -------------------------------------------------------------------------------------------------
         2014/10/23 01:27:09  2014/10/23 01:33:03  2014/10/23 01:38:57  00:11:47   25.85   40.28  177.59
         2014/10/23 03:02:44  2014/10/23 03:08:31  2014/10/23 03:14:17  00:11:32   20.55  341.35  209.65
         2014/10/23 14:48:23  2014/10/23 14:54:39  2014/10/23 15:00:55  00:12:31   75.31  166.30  350.27
         2014/10/23 16:25:19  2014/10/23 16:29:32  2014/10/23 16:33:46  00:08:27    7.14  200.60  287.00
         2014/10/24 01:35:34  2014/10/24 01:41:37  2014/10/24 01:47:39  00:12:05   32.20   34.97  180.38
         2014/10/24 03:11:40  2014/10/24 03:17:11  2014/10/24 03:22:42  00:11:02   16.30  335.44  213.21
         2014/10/24 14:57:00  2014/10/24 15:03:16  2014/10/24 15:09:32  00:12:32   84.30  169.06  345.11
         2014/10/24 16:34:18  2014/10/24 16:38:02  2014/10/24 16:41:45  00:07:27    5.09  205.18  279.57
         2014/10/25 01:44:01  2014/10/25 01:50:11  2014/10/25 01:56:20  00:12:19   40.61   29.75  183.12
         2014/10/25 03:20:39  2014/10/25 03:25:51  2014/10/25 03:31:04  00:10:25   12.78  329.21  217.10"""  # NOQA

        for line in GPREDICT_DATA.splitlines()[4:]:
            line_parts = line.split()
            aos = datetime.strptime(" ".join(line_parts[:2]), '%Y/%m/%d %H:%M:%S')
            max_elevation_date = datetime.strptime(" ".join(line_parts[2:4]),
                                                   '%Y/%m/%d %H:%M:%S')
            los = datetime.strptime(" ".join(line_parts[4:6]), '%Y/%m/%d %H:%M:%S')
            duration = datetime.strptime(line_parts[6], '%H:%M:%S')
            duration_s = timedelta(
                minutes=duration.minute, seconds=duration.second).total_seconds()
            max_elev_deg = float(line_parts[7])

            try:
                date = pass_.los  # NOQA
            except UnboundLocalError:
                date = datetime.strptime(
                    "2014-10-22 20:18:11.921921", '%Y-%m-%d %H:%M:%S.%f')

            pass_ = self.predictor.get_next_pass(ARG, date)
            self.assertAlmostEqual(pass_.aos, aos, delta=ONE_SECOND)
            self.assertAlmostEqual(pass_.los, los, delta=ONE_SECOND)
            self.assertAlmostEqual(pass_.max_elevation_date, max_elevation_date, delta=ONE_SECOND)
            self.assertAlmostEqual(pass_.duration_s, duration_s, delta=1)
            self.assertAlmostEqual(pass_.max_elevation_deg, max_elev_deg, delta=0.05)


class AccuratePredictorCalculationErrorTests(TestCase):
    """Check that we can learn from calculation errors and provide patches for corner cases"""

    def setUp(self):
        # Source
        self.db = MemoryTLESource()
        self.db.add_tle(BUGSAT_SATE_ID, BUGSAT1_TLE_LINES, datetime.now())
        # Predictor
        self.predictor = HighAccuracyTLEPredictor(BUGSAT_SATE_ID, self.db)
        self.is_ascending_mock = self._patch(
            'orbit_predictor.predictors.base.LocationPredictor.is_ascending')
        self.start = datetime(2017, 3, 6, 7, 51)
        logassert.setup(self,  'orbit_predictor.predictors.base')

    def _patch(self, *args,  **kwargs):
        patcher = mock.patch(*args, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_ascending_failure(self):
        self.is_ascending_mock.return_value = False
        with self.assertRaises(PropagationError):
            self.predictor.get_next_pass(ARG, self.start)

        self.assertLoggedError(str(ARG), str(self.start), *BUGSAT1_TLE_LINES)

    def test_descending_failure(self):
        self.is_ascending_mock.return_value = True
        with self.assertRaises(PropagationError):
            self.predictor.get_next_pass(ARG, self.start)

        self.assertLoggedError(str(ARG), str(self.start), *BUGSAT1_TLE_LINES)
