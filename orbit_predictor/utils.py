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

import functools
from collections import namedtuple
from datetime import datetime
from math import asin, atan2, cos, degrees, floor, radians, sin, sqrt

try:
    from functools import lru_cache
except ImportError:
    class lru_cache(object):
        """dummy function for python 2"""
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, f):
            return f

# This function was ported from its Matlab equivalent here:
# http://www.mathworks.com/matlabcentral/fileexchange/23051-vectorized-solar-azimuth-and-elevation-estimation

DECEMBER_31TH_1999_MIDNIGHT_JD = 2451543.5


def compose(*functions):
    """Performs function composition with variadic arguments"""
    return functools.reduce(lambda f, g: lambda *args: f(g(*args)),
                            functions,
                            lambda x: x)


cos_d = compose(cos, radians)
sin_d = compose(sin, radians)
atan2_d = compose(degrees, atan2)
asin_d = compose(degrees, asin)


AzimuthElevation = namedtuple('AzimuthElevation', 'azimuth elevation')


def euclidean_distance(*components):
    """Returns the norm of a vector"""
    return sqrt(sum(c**2 for c in components))


def dot_product(a, b):
    """Computes dot product between two vectors writen as tuples or lists"""
    return sum(ai * bj for ai, bj in zip(a, b))


def vector_diff(a, b):
    """Computes difference between two vectors"""
    return tuple((ai - bi) for ai, bi in zip(a, b))


def cross_product(a, b):
    """Computes cross product between two vectors"""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )


def vector_norm(a):
    """Returns the norm of a vector"""
    return euclidean_distance(*a)


def sun_azimuth_elevation(latitude_deg, longitude_deg, when=None):
    """
    Return (azimuth, elevation) of the Sun at ground point

    :param latitude_deg: a float number representing latitude on degrees
    :param longitude_deg: a float number representing longitude on degrees
    :param when: a ``datetime.datetime`` object in utc, if not provided,
        utcnow() is used
    :returns: an ``AzimuthElevation`` namedtuple
    """
    if when is None:
        when = datetime.utcnow()

    utc_time_tuple = when.timetuple()
    jd = juliandate(utc_time_tuple)
    date = jd - DECEMBER_31TH_1999_MIDNIGHT_JD

    w = 282.9404 + 4.70935e-5 * date    # longitude of perihelion degrees
    eccentricity = 0.016709 - 1.151e-9 * date      # eccentricity
    M = (356.0470 + 0.9856002585 * date) % 360    # mean anomaly degrees
    L = w + M                        # Sun's mean longitude degrees
    oblecl = 23.4393 - 3.563e-7 * date  # Sun's obliquity of the ecliptic

    # auxiliary angle
    auxiliary_angle = M + degrees(eccentricity * sin_d(M) * (1 + eccentricity * cos_d(M)))

    # rectangular coordinates in the plane of the ecliptic (x axis toward perhilion)
    x = cos_d(auxiliary_angle) - eccentricity
    y = sin_d(auxiliary_angle) * sqrt(1 - eccentricity**2)

    # find the distance and true anomaly
    r = euclidean_distance(x, y)
    v = atan2_d(y, x)

    # find the longitude of the sun
    sun_lon = v + w

    # compute the ecliptic rectangular coordinates
    xeclip = r * cos_d(sun_lon)
    yeclip = r * sin_d(sun_lon)
    zeclip = 0.0

    # rotate these coordinates to equitorial rectangular coordinates
    xequat = xeclip
    yequat = yeclip * cos_d(oblecl) + zeclip * sin_d(oblecl)
    zequat = yeclip * sin_d(23.4406) + zeclip * cos_d(oblecl)

    # convert equatorial rectangular coordinates to RA and Decl:
    r = euclidean_distance(xequat, yequat, zequat)
    RA = atan2_d(yequat, xequat)
    delta = asin_d(zequat/r)

    # Following the RA DEC to Az Alt conversion sequence explained here:
    # http://www.stargazing.net/kepler/altaz.html

    sidereal = sidereal_time(utc_time_tuple, longitude_deg, L)

    # Replace RA with hour angle HA
    HA = sidereal * 15 - RA

    # convert to rectangular coordinate system
    x = cos_d(HA) * cos_d(delta)
    y = sin_d(HA) * cos_d(delta)
    z = sin_d(delta)

    # rotate this along an axis going east-west.
    xhor = x * cos_d(90 - latitude_deg) - z * sin_d(90 - latitude_deg)
    yhor = y
    zhor = x * sin_d(90 - latitude_deg) + z * cos_d(90 - latitude_deg)

    # Find the h and AZ
    azimuth = atan2_d(yhor, xhor) + 180
    elevation = asin_d(zhor)

    return AzimuthElevation(azimuth, elevation)


def juliandate(utc_tuple):
    year, month, day, hour, minute, sec = utc_tuple[:6]
    if month <= 2:
        year -= 1
        month += 12

    return (floor(365.25*(year + 4716.0)) + floor(30.6001*(month+1.0)) + 2.0 -
            floor(year / 100.0) + floor(floor(year / 100.0) / 4.0) + day - 1524.5 +
            (hour + minute / 60.0 + sec / 3600.0) / 24.0)


def sidereal_time(utc_tuple, local_lon, sun_lon):
    # Find the J2000 value
    # J2000 = jd - 2451545.0;
    UTH = utc_tuple.tm_hour + utc_tuple.tm_min / 60.0 + utc_tuple.tm_sec / 3600.0

    # Calculate local siderial time
    GMST0 = ((sun_lon + 180) % 360) / 15
    return GMST0 + UTH + local_lon / 15


class reify(object):
    """
    Use as a class method decorator.  It operates almost exactly like the
    Python ``@property`` decorator, but it puts the result of the method it
    decorates into the instance dict after the first call, effectively
    replacing the function it decorates with an instance variable.  It is, in
    Python parlance, a non-data descriptor.

    Taken from: http://docs.pylonsproject.org/projects/pyramid/en/latest/api/decorator.html
    """

    def __init__(self, wrapped):
        self.wrapped = wrapped
        functools.update_wrapper(self, wrapped)

    def __get__(self, inst, objtype=None):
        if inst is None:
            return self
        val = self.wrapped(inst)
        setattr(inst, self.wrapped.__name__, val)
        return val
