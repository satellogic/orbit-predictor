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
import unittest
from math import degrees

from orbit_predictor.locations import Location, ARG
from orbit_predictor.predictors import TLEPredictor
from orbit_predictor.sources import MemoryTLESource

SATE_ID = 'BUGSAT-1'
BUGSAT1_TLE_LINES = ("1 40014U 14033E   14294.41438078  .00003468  00000-0  34565-3 0  3930",
                     "2 40014  97.9781 190.6418 0032692 299.0467  60.7524 14.91878099 18425")


class LocationTestCase(unittest.TestCase):

    def setUp(self):
        # Source
        self.db = MemoryTLESource()
        self.db.add_tle(SATE_ID, BUGSAT1_TLE_LINES, dt.datetime.utcnow())
        # Predictor
        self.predictor = TLEPredictor(SATE_ID, self.db)
        date = dt.datetime.strptime("2014-10-22 20:18:11.921921", '%Y-%m-%d %H:%M:%S.%f')
        self.next_pass = self.predictor.get_next_pass(ARG, when_utc=date)

    def test_compare_eq(self):
        l1 = Location(latitude_deg=1, longitude_deg=2, elevation_m=3, name="location1")
        l2 = Location(latitude_deg=1, longitude_deg=2, elevation_m=3, name="location1")

        self.assertEqual(l1, l2)
        self.assertEqual(l2, l1)

    def test_compare_no_eq(self):
        l1 = Location(latitude_deg=1, longitude_deg=2, elevation_m=3, name="location_other")
        l2 = Location(latitude_deg=1, longitude_deg=2, elevation_m=3, name="location1")

        self.assertNotEqual(l1, l2)
        self.assertNotEqual(l2, l1)

    def test_compare_eq_subclass(self):

        class SubLocation(Location):
            pass

        l1 = Location(latitude_deg=1, longitude_deg=2, elevation_m=3, name="location1")
        l2 = SubLocation(latitude_deg=1, longitude_deg=2, elevation_m=3, name="location1")

        self.assertEqual(l1, l2)
        self.assertEqual(l2, l1)

    def test_get_azimuth_elev(self):
        date = dt.datetime.strptime("2014-10-21 22:47:29.147740", '%Y-%m-%d %H:%M:%S.%f')
        azimuth, elevation = ARG.get_azimuth_elev(self.predictor.get_position(date))

        self.assertAlmostEqual(degrees(azimuth), 249.7, delta=0.1)
        self.assertAlmostEqual(degrees(elevation), -52.1, delta=0.1)

    def test_get_azimuth_elev_deg(self):
        date = dt.datetime.strptime("2014-10-21 22:47:29.147740", '%Y-%m-%d %H:%M:%S.%f')
        azimuth, elevation = ARG.get_azimuth_elev_deg(self.predictor.get_position(date))

        self.assertAlmostEqual(azimuth, 249.7, delta=0.1)
        self.assertAlmostEqual(elevation, -52.1, delta=0.1)

    def test_is_visible(self):
        position = self.predictor.get_position(self.next_pass.aos)
        self.assertTrue(ARG.is_visible(position))

    def test_no_visible(self):
        position = self.predictor.get_position(self.next_pass.los + dt.timedelta(minutes=10))
        self.assertFalse(ARG.is_visible(position))

    def test_is_visible_with_deg(self):
        position = self.predictor.get_position(self.next_pass.aos + dt.timedelta(minutes=4))
        # 21 deg
        self.assertTrue(ARG.is_visible(position, elevation=4))

    def test_no_visible_with_deg(self):
        position = self.predictor.get_position(self.next_pass.aos + dt.timedelta(minutes=4))
        # 21 deg
        self.assertFalse(ARG.is_visible(position, elevation=30))

    def test_doppler_factor(self):
        date = dt.datetime.strptime("2014-10-21 23:06:11.132438", '%Y-%m-%d %H:%M:%S.%f')
        position = self.predictor.get_position(date)
        doppler_factor = ARG.doppler_factor(position)

        self.assertAlmostEqual((2 - doppler_factor)*437.445e6, 437.445632e6, delta=100)
