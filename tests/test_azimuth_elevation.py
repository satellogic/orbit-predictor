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

import unittest
import datetime as dt

from orbit_predictor.utils import sun_azimuth_elevation


class AzimuthElevationTests(unittest.TestCase):
    """Ensures calculations are ok, comparing against:
        http://www.esrl.noaa.gov/gmd/grad/solcalc/
        all dates are UTC
    """
    def setUp(self):
        self.coords = (0, 0)
        self.base_date = dt.datetime(2016, 9, 8)

    def test_return_value(self):
        obj = sun_azimuth_elevation(*self.coords, when=self.base_date)
        self.assertTrue(hasattr(obj, 'azimuth'))
        self.assertTrue(hasattr(obj, 'elevation'))

    def test_8(self):
        date = self.base_date + dt.timedelta(hours=8)
        azimuth, elevation = sun_azimuth_elevation(*self.coords, when=date)
        self.assertAlmostEqual(azimuth, 83.64, delta=0.5)
        self.assertAlmostEqual(elevation, 30.67, delta=0.5)

    def test_9(self):
        date = self.base_date + dt.timedelta(hours=9)
        azimuth, elevation = sun_azimuth_elevation(*self.coords, when=date)
        self.assertAlmostEqual(azimuth, 82.21, delta=0.5)
        self.assertAlmostEqual(elevation, 45.36, delta=0.5)

    def test_10(self):
        date = self.base_date + dt.timedelta(hours=10)
        azimuth, elevation = sun_azimuth_elevation(*self.coords, when=date)
        self.assertAlmostEqual(azimuth, 79, delta=0.5)
        self.assertAlmostEqual(elevation, 60.16, delta=0.5)

    def test_11(self):
        date = self.base_date + dt.timedelta(hours=11)
        azimuth, elevation = sun_azimuth_elevation(*self.coords, when=date)
        self.assertAlmostEqual(azimuth, 69.05, delta=0.5)
        self.assertAlmostEqual(elevation, 74.64, delta=0.5)

    def test_12(self):
        date = self.base_date + dt.timedelta(hours=12)
        azimuth, elevation = sun_azimuth_elevation(*self.coords, when=date)
        self.assertAlmostEqual(azimuth, 353.54, delta=0.5)
        self.assertAlmostEqual(elevation, 84.55, delta=0.5)

    def test_13(self):
        date = self.base_date + dt.timedelta(hours=13)
        azimuth, elevation = sun_azimuth_elevation(*self.coords, when=date)
        self.assertAlmostEqual(azimuth, 289.36, delta=0.5)
        self.assertAlmostEqual(elevation, 73.5, delta=0.5)

    def test_14(self):
        date = self.base_date + dt.timedelta(hours=14)
        azimuth, elevation = sun_azimuth_elevation(*self.coords, when=date)
        self.assertAlmostEqual(azimuth, 280.49, delta=0.5)
        self.assertAlmostEqual(elevation, 58.96, delta=0.5)

    def test_15(self):
        date = self.base_date + dt.timedelta(hours=15)
        azimuth, elevation = sun_azimuth_elevation(*self.coords, when=date)
        self.assertAlmostEqual(azimuth, 277.49, delta=0.5)
        self.assertAlmostEqual(elevation, 44.14, delta=0.5)

    def test_16(self):
        date = self.base_date + dt.timedelta(hours=16)
        azimuth, elevation = sun_azimuth_elevation(*self.coords, when=date)
        self.assertAlmostEqual(azimuth, 276.14, delta=0.5)
        self.assertAlmostEqual(elevation, 29.26, delta=0.5)

    def test_19(self):
        date = self.base_date + dt.timedelta(hours=19)
        azimuth, elevation = sun_azimuth_elevation(*self.coords, when=date)
        self.assertAlmostEqual(azimuth, 275.51, delta=0.5)
        self.assertAlmostEqual(elevation, -15.55, delta=0.5)

    def test_default_when(self):
        sun_azimuth_elevation(0, 0)
