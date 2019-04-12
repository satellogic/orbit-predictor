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
import time
import unittest

from orbit_predictor.utils import sun_azimuth_elevation

buenos_aires_llh = (-34.628284, -58.436045)
buenos_aires_timezone = -3
north_pole_llh = (84.0, 0.0)
greenwich_timezone = 0
tokio_llh = (35.6833333, 139.7666666)
tokio_timezone = 9


class ElevationTestCase(unittest.TestCase):

    def assertElevation(self, date, expected):
        lat, lon = self.location
        _, actual = sun_azimuth_elevation(lat, lon, date)
        self.assertAlmostEqual(actual, expected, delta=4)


class TestOnBuenosAires(ElevationTestCase):

    def setUp(self):
        super(self.__class__, self).setUp()

        self.january_10th = Date.from_utc('10/1/2008', buenos_aires_timezone)
        self.march_10th = Date.from_utc('10/3/2008', buenos_aires_timezone)
        self.june_10th = Date.from_utc('10/6/2008', buenos_aires_timezone)
        self.september_10th = Date.from_utc('10/9/2008', buenos_aires_timezone)

        self.location = buenos_aires_llh

    def test_q1(self):
        self.assertElevation(self.january_10th.noon(), 71.58688)
        self.assertElevation(self.january_10th.dawn(), 12.10156)
        self.assertElevation(self.january_10th.dust(), 12.10156)

    def test_q2(self):
        self.assertElevation(self.march_10th.noon(), 55.78998)
        self.assertElevation(self.march_10th.dawn(), 1.39351)
        self.assertElevation(self.march_10th.dust(), 2.87432)

    def test_q3(self):
        self.assertElevation(self.june_10th.noon(), 30.93056)
        self.assertElevation(self.june_10th.dawn(), -11.55022)
        self.assertElevation(self.june_10th.dust(), -14.17221)

    def test_q4(self):
        self.assertElevation(self.september_10th.noon(), 48.96952)
        self.assertElevation(self.september_10th.dawn(), -0.75196)
        self.assertElevation(self.september_10th.dust(), -4.5362)


class TestOnNorthPole(ElevationTestCase):

    def setUp(self):
        super(self.__class__, self).setUp()

        self.january_10th = Date.from_utc('10/1/2008', greenwich_timezone)
        self.march_10th = Date.from_utc('10/3/2008', greenwich_timezone)
        self.june_10th = Date.from_utc('10/6/2008', greenwich_timezone)
        self.september_10th = Date.from_utc('10/9/2008', greenwich_timezone)

        self.location = north_pole_llh

    def test_q1(self):
        self.assertElevation(self.january_10th.noon(), -16.00948)
        self.assertElevation(self.january_10th.dawn(), -20.55059)
        self.assertElevation(self.january_10th.dust(), -23.20414)

    def test_q2(self):
        self.assertElevation(self.march_10th.noon(), 2.14842)
        self.assertElevation(self.march_10th.dawn(), -2.6164)
        self.assertElevation(self.march_10th.dust(), -5.00546)

    def test_q3(self):
        self.assertElevation(self.june_10th.noon(), 29.06153)
        self.assertElevation(self.june_10th.dawn(), 24.4841)
        self.assertElevation(self.june_10th.dust(), 21.39807)

    def test_q4(self):
        self.assertElevation(self.september_10th.noon(), 10.68454)
        self.assertElevation(self.september_10th.dawn(), 6.36817)
        self.assertElevation(self.september_10th.dust(), 2.91842)


class TestOnTokio(ElevationTestCase):

    def setUp(self):
        super(self.__class__, self).setUp()

        self.january_10th = Date.from_utc('10/1/2008', tokio_timezone)
        self.march_10th = Date.from_utc('10/3/2008', tokio_timezone)
        self.june_10th = Date.from_utc('10/6/2008', tokio_timezone)
        self.september_10th = Date.from_utc('10/9/2008', tokio_timezone)

        self.location = tokio_llh

    def test_q1(self):
        self.assertElevation(self.january_10th.noon(), 32.18769)
        self.assertElevation(self.january_10th.dawn(), 0.75589)
        self.assertElevation(self.january_10th.dust(), -26.7901)

    def test_q2(self):
        self.assertElevation(self.march_10th.noon(), 50.27033)
        self.assertElevation(self.march_10th.dawn(), 11.41627)
        self.assertElevation(self.march_10th.dust(), -16.22019)

    def test_q3(self):
        self.assertElevation(self.june_10th.noon(), 76.65314)
        self.assertElevation(self.june_10th.dawn(), 28.86539)
        self.assertElevation(self.june_10th.dust(), -1.48132)

    def test_q4(self):
        self.assertElevation(self.september_10th.noon(), 58.7265)
        self.assertElevation(self.september_10th.dawn(), 19.47538)
        self.assertElevation(self.september_10th.dust(), -13.66267)


class Date:
    def __init__(self, dt_, tz):
        self.date = dt_ - dt.timedelta(hours=tz)

    def noon(self):
        return self.date + dt.timedelta(hours=12)

    def dawn(self):
        return self.date + dt.timedelta(hours=7)

    def dust(self):
        return self.date + dt.timedelta(hours=19)

    @classmethod
    def from_utc(cls, time_string, tz):
        t = time.strptime(time_string, '%d/%m/%Y')
        return cls(dt.datetime(t.tm_year, t.tm_mon, t.tm_mday, tzinfo=UTC), tz)


# NOTE: These classes seem necessary to allow the dates to be introduced as UTC and then
#       localized so that when input to get_sun_azimuth_elevation they are interpreted
#       correctly as the original UTC...
#       It's horrible, and if you know of a better way to do this please let me know.
class UTCTimezone(dt.tzinfo):
    ZERO = dt.timedelta(0)

    def utcoffset(self, dt_):
        return UTCTimezone.ZERO

    def dst(self, dt_):
        return UTCTimezone.ZERO

    def tzname(self, dt_):
        return 'UTC'


UTC = UTCTimezone()


class LocalTimezone(dt.tzinfo):
    STDOFFSET = dt.timedelta(seconds=-time.timezone)
    if time.daylight:
        DSTOFFSET = dt.timedelta(seconds=-time.altzone)
    else:
        DSTOFFSET = STDOFFSET
    DSTDIFF = DSTOFFSET - STDOFFSET

    def utcoffset(self, dt_):
        if self._isdst(dt_):
            return LocalTimezone.DSTOFFSET
        else:
            return LocalTimezone.STDOFFSET

    def dst(self, dt_):
        if self._isdst(dt_):
            return LocalTimezone.DSTDIFF
        else:
            return UTCTimezone.ZERO

    def tzname(self, dt_):
        return time.tzname[self._isdst(dt_)]

    def _isdst(self, dt_):
        tt = (dt_.year, dt_.month, dt_.day,
              dt_.hour, dt_.minute, dt_.second,
              dt_.weekday(), 0, 0)
        stamp = time.mktime(tt)
        tt = time.localtime(stamp)
        return tt.tm_isdst > 0


Local = LocalTimezone()
