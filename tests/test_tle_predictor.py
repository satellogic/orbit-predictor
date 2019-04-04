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
import unittest

from orbit_predictor.coordinate_systems import llh_to_ecef
from orbit_predictor.locations import Location, tortu1
from orbit_predictor.predictors import (
    NotReachable,
    Position,
    PredictedPass,
    TLEPredictor
)
from orbit_predictor.sources import MemoryTLESource

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch  # Python2


SATE_ID = 'BUGSAT-1'
BUGSAT1_TLE_LINES = (
    "1 40014U 14033E   14294.41438078  .00003468  00000-0  34565-3 0  3930",
    "2 40014  97.9781 190.6418 0032692 299.0467  60.7524 14.91878099 18425")


class TLEPredictorTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Source
        cls.db = MemoryTLESource()
        cls.db.add_tle(SATE_ID, BUGSAT1_TLE_LINES, datetime.datetime.now())
        # Predictor
        cls.predictor = TLEPredictor(SATE_ID, cls.db)

    def test_predicted_pass_eq(self):
        aos = datetime.datetime.utcnow()
        max_elevation_date = datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
        los = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
        max_elevation_position = Position(
            when_utc=max_elevation_date,
            position_ecef=(1, 1, 1),
            velocity_ecef=(1, 1, 1),
            error_estimate=0)

        p1 = PredictedPass(
            sate_id=1, location=tortu1,
            aos=aos, los=los, duration_s=600,
            max_elevation_date=max_elevation_date,
            max_elevation_position=max_elevation_position,
            max_elevation_deg=10)
        p2 = PredictedPass(
            sate_id=1, location=tortu1,
            aos=aos, los=los, duration_s=600,
            max_elevation_date=max_elevation_date,
            max_elevation_position=max_elevation_position,
            max_elevation_deg=10)

        self.assertEqual(p1, p2)
        self.assertEqual(p2, p1)

    def test_predicted_pass_no_eq(self):
        aos = datetime.datetime.utcnow()
        max_elevation_date = datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
        los = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
        max_elevation_position = Position(
            when_utc=max_elevation_date,
            position_ecef=(1, 1, 1),
            velocity_ecef=(1, 1, 1),
            error_estimate=0)

        p1 = PredictedPass(
            sate_id=1, location=tortu1,
            aos=aos, los=los, duration_s=600,
            max_elevation_date=max_elevation_date,
            max_elevation_position=max_elevation_position,
            max_elevation_deg=10)
        p2 = PredictedPass(
            sate_id=1, location=tortu1,
            aos=aos, los=los, duration_s=600,
            max_elevation_date=max_elevation_date,
            max_elevation_position=max_elevation_position,
            max_elevation_deg=50)

        self.assertNotEqual(p1, p2)
        self.assertNotEqual(p2, p1)

    def test_predicted_pass_eq_subclass(self):

        class SubPredictedPass(PredictedPass):
            pass

        aos = datetime.datetime.utcnow()
        max_elevation_date = datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
        los = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
        max_elevation_position = Position(
            when_utc=max_elevation_date,
            position_ecef=(1, 1, 1),
            velocity_ecef=(1, 1, 1),
            error_estimate=0)

        p1 = PredictedPass(
            sate_id=1, location=tortu1,
            aos=aos, los=los, duration_s=600,
            max_elevation_date=max_elevation_date,
            max_elevation_position=max_elevation_position,
            max_elevation_deg=10)
        p2 = SubPredictedPass(
            sate_id=1, location=tortu1,
            aos=aos, los=los, duration_s=600,
            max_elevation_date=max_elevation_date,
            max_elevation_position=max_elevation_position,
            max_elevation_deg=10)

        self.assertEqual(p1, p2)
        self.assertEqual(p2, p1)

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
            aos = datetime.datetime.strptime(" ".join(line_parts[:2]), '%Y/%m/%d %H:%M:%S')
            max_elevation_date = datetime.datetime.strptime(" ".join(line_parts[2:4]),
                                                            '%Y/%m/%d %H:%M:%S')
            los = datetime.datetime.strptime(" ".join(line_parts[4:6]), '%Y/%m/%d %H:%M:%S')
            duration = datetime.datetime.strptime(line_parts[6], '%H:%M:%S')
            duration_s = datetime.timedelta(
                minutes=duration.minute, seconds=duration.second).total_seconds()
            max_elev_deg = float(line_parts[7])

            try:
                date = pass_.los  # NOQA
            except UnboundLocalError:
                date = datetime.datetime.strptime(
                    "2014-10-22 20:18:11.921921", '%Y-%m-%d %H:%M:%S.%f')
            pass_ = self.predictor.get_next_pass(tortu1, date)
            self.assertAlmostEqual((pass_.aos - aos).total_seconds(), 0, delta=25)
            self.assertAlmostEqual((pass_.max_elevation_date - max_elevation_date).total_seconds(),
                                   0, delta=25)
            self.assertAlmostEqual((pass_.los - los).total_seconds(), 0, delta=25)
            self.assertAlmostEqual(pass_.max_elevation_deg, max_elev_deg, delta=1)
            self.assertAlmostEqual(pass_.duration_s, duration_s, delta=10)

    def test_get_next_pass(self):
        date = datetime.datetime.strptime("2014-10-22 20:18:11.921921", '%Y-%m-%d %H:%M:%S.%f')
        pass_ = self.predictor.get_next_pass(tortu1, date, max_elevation_gt=15)
        for i in range(20):
            pass_ = self.predictor.get_next_pass(tortu1, pass_.los, max_elevation_gt=15)
            self.assertGreaterEqual(pass_.max_elevation_deg, 15)

    def test_get_next_pass_with_limit_exception(self):
        date = datetime.datetime.strptime("2014-10-22 20:18:11.921921", '%Y-%m-%d %H:%M:%S.%f')
        pass_ = self.predictor.get_next_pass(tortu1, date, max_elevation_gt=15)
        with self.assertRaises(NotReachable):
            self.predictor.get_next_pass(tortu1, date, max_elevation_gt=15,
                                         limit_date=pass_.aos - datetime.timedelta(minutes=1))

    def test_get_next_pass_with_limit(self):
        date = datetime.datetime.strptime("2014-10-22 20:18:11.921921", '%Y-%m-%d %H:%M:%S.%f')
        pass_ = self.predictor.get_next_pass(tortu1, date, max_elevation_gt=15)
        new_pass = self.predictor.get_next_pass(
            tortu1, date, max_elevation_gt=15,
            limit_date=pass_.los + datetime.timedelta(seconds=1))
        self.assertEqual(pass_, new_pass)

    def test_get_next_pass_while_passing(self):
        date = datetime.datetime.strptime("2014/10/23 01:32:09", '%Y/%m/%d %H:%M:%S')
        pass_ = self.predictor.get_next_pass(tortu1, date)
        self.assertEqual(pass_.aos, date)
        self.assertTrue(date < pass_.los)

        position = self.predictor.get_position(date)
        self.assertTrue(tortu1.is_visible(position))

    def test_grater_than_deg(self):
        date = datetime.datetime.strptime("2014/10/23 01:25:09", '%Y/%m/%d %H:%M:%S')
        pass5 = self.predictor.get_next_pass(tortu1, date, aos_at_dg=5)
        pass10 = self.predictor.get_next_pass(tortu1, date, aos_at_dg=10)
        pass15 = self.predictor.get_next_pass(tortu1, date, aos_at_dg=15)
        self.assertTrue(pass5.aos < pass10.aos < pass15.aos)
        self.assertTrue(pass5.los > pass10.los > pass15.los)
        self.assertTrue(pass5.max_elevation_deg == pass10.max_elevation_deg)

    @patch("orbit_predictor.predictors.TLEPredictor._propagate_ecef")
    def test_get_position(self, mocked_propagate):
        mocked_propagate.return_value = ('foo', 'bar')
        when_utc = datetime.datetime.utcnow()
        position = self.predictor.get_position(when_utc)

        self.assertIsInstance(position, Position)
        self.assertEqual(when_utc, position.when_utc)
        self.assertIsNone(position.error_estimate)
        self.assertEqual(position.position_ecef, 'foo')
        self.assertEqual(position.velocity_ecef, 'bar')

    def test_off_nadir_computable_and_reasonable(self):
        date = datetime.datetime.strptime("2014-10-22 20:18:11.921921", '%Y-%m-%d %H:%M:%S.%f')
        pass_ = self.predictor.get_next_pass(tortu1, date)
        self.assertLessEqual(abs(pass_.off_nadir_deg), 90)


class OffNadirAngleTests(unittest.TestCase):
    def setUp(self):
        self.location = Location("A random location", latitude_deg=0, longitude_deg=0,
                                 elevation_m=0)

    def test_off_nadir_satellite_exactly_over(self):
        position_ecef = llh_to_ecef(0, 0, 500 * 1000)  # A satellite exactyl over the point
        velocity_ecef = (0, 0, 1)
        max_elevation_position = Position(None, position_ecef, velocity_ecef, None)

        pass_ = PredictedPass(sate_id=1, location=self.location,
                              aos=None, los=None, duration_s=None,
                              max_elevation_position=max_elevation_position,
                              max_elevation_deg=None)

        self.assertAlmostEqual(pass_.off_nadir_deg, 0, delta=0.02)

    def test_off_nadir_satellite_passing_left_means_positive_sign(self):
        position_ecef = llh_to_ecef(0, -10, 500 * 1000)  # A satellite exactyl over the point
        velocity_ecef = (0, 0, -1)
        max_elevation_position = Position(None, position_ecef, velocity_ecef, None)

        pass_ = PredictedPass(sate_id=1, location=self.location,
                              aos=None, los=None, duration_s=None,
                              max_elevation_position=max_elevation_position,
                              max_elevation_deg=None)

        self.assertGreaterEqual(pass_.off_nadir_deg, 0)

    def test_off_nadir_satellite_passing_right_means_negative_sign(self):
        position_ecef = llh_to_ecef(0, 10, 500 * 1000)  # A satellite exactyl over the point
        velocity_ecef = (0, 0, -1)
        max_elevation_position = Position(None, position_ecef, velocity_ecef, None)

        pass_ = PredictedPass(sate_id=1, location=self.location,
                              aos=None, los=None, duration_s=None,
                              max_elevation_position=max_elevation_position,
                              max_elevation_deg=None)

        self.assertLessEqual(pass_.off_nadir_deg, 0)
