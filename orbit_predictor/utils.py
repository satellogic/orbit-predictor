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
import datetime as dt
from math import asin, atan2, cos, degrees, floor, radians, sin, sqrt, tan, modf

import numpy as np
from sgp4.ext import jday
from sgp4.propagation import _gstime

from .constants import AU, R_E_KM, MU_E
from .coordinate_systems import eci_to_radec, ecef_to_eci

# Inspired in https://github.com/poliastro/poliastro/blob/88edda8/src/poliastro/jit.py
try:
    from numba import njit
except ImportError:
    import inspect

    def njit(first=None, *args, **kwargs):
        """Identity JIT, returns unchanged function."""
        def _jit(f):
            return f

        if inspect.isfunction(first):
            return first
        else:
            return _jit

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


def angle_between(a, b):
    """
    Computes angle between two vectors in degrees.

    Notes
    -----
    Naïve algorithm, see https://scicomp.stackexchange.com/q/27689/782.

    """
    return degrees(np.arccos(dot_product(a, b) / (vector_norm(a) * vector_norm(b))))


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


@njit
def cross(a, b):
    """Computes cross product between two vectors"""
    # np.cross is not supported in numba nopython mode, see
    # https://github.com/numba/numba/issues/2978
    return np.array((
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ))


# Inspired by https://github.com/poliastro/poliastro/blob/aaa1bb2/poliastro/util.py
# and https://github.com/poliastro/poliastro/blob/06ef6ba/poliastro/util.py
# Copyright (c) 2012-2017 Juan Luis Cano Rodríguez, MIT license
@njit
def rotate(vec, ax, angle):
    """Rotates the coordinate system around axis x, y or z a CCW angle.

    Parameters
    ----------
    vec : ndarray
        Dimension 3 vector.
    ax : int
        Axis to be rotated.
    angle : float
        Angle of rotation (rad).

    Notes
    -----
    This performs a so-called active or alibi transformation: rotates the
    vector while the coordinate system remains unchanged. To do the opposite
    operation (passive or alias transformation) call the function as
    `rotate(vec, ax, -angle)` or use the convenience function `transform`,
    see `[1]_`.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Rotation_matrix#Ambiguities

    """
    assert vec.shape == (3,)

    rot = np.eye(3)
    if ax == 0:
        sl = slice(1, 3, 1)
    elif ax == 1:
        sl = slice(0, 3, 2)
    elif ax == 2:
        sl = slice(0, 2, 1)
    else:
        raise ValueError("Invalid axis: must be one of 0, 1 or 2")

    rot[sl, sl] = np.array((
        (cos(angle), -sin(angle)),
        (sin(angle), cos(angle))
    ))

    if ax == 1:
        return np.dot(rot.T, vec.astype(rot.dtype))
    else:
        return np.dot(rot, vec.astype(rot.dtype))


@njit
def transform(vec, ax, angle):
    """Rotates a coordinate system around axis a positive right-handed angle.

    Notes
    -----
    This is a convenience function, equivalent to `rotate(vec, ax, -angle)`.
    Refer to the documentation of that function for further information.

    """
    return rotate(vec, ax, -angle)


def raan_from_ltan(when, ltan=12.0):
    sun_eci = get_sun(when)

    # convert equatorial rectangular coordinates to RA and Decl:
    RA, _, _ = eci_to_radec(sun_eci)
    RA = degrees(RA)

    # Idea from
    # https://www.mathworks.com/matlabcentral/fileexchange/39085-mean-local-time-of-the-ascending-node
    raan = (RA + 15.0 * (ltan - 12.0)) % 360
    return raan


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
        when = dt.datetime.utcnow()

    utc_time_tuple = when.timetuple()
    jd = juliandate(timetuple_from_dt(when))
    date = jd - DECEMBER_31TH_1999_MIDNIGHT_JD

    w, M, L, eccentricity, oblecl = _sun_mean_ecliptic_elements(date)
    sun_eci = _sun_eci(w, M, L, eccentricity, oblecl)

    # convert equatorial rectangular coordinates to RA and Decl:
    RA, DEC, r = eci_to_radec(sun_eci)

    RA = degrees(RA)
    DEC = degrees(DEC)

    # Following the RA DEC to Az Alt conversion sequence explained here:
    # http://www.stargazing.net/kepler/altaz.html

    sidereal = sidereal_time(utc_time_tuple, longitude_deg, L)

    # Replace RA with hour angle HA
    HA = sidereal * 15 - RA

    # convert to rectangular coordinate system
    x = cos_d(HA) * cos_d(DEC)
    y = sin_d(HA) * cos_d(DEC)
    z = sin_d(DEC)

    # rotate this along an axis going east-west.
    xhor = x * cos_d(90 - latitude_deg) - z * sin_d(90 - latitude_deg)
    yhor = y
    zhor = x * sin_d(90 - latitude_deg) + z * cos_d(90 - latitude_deg)

    # Find the h and AZ
    azimuth = atan2_d(yhor, xhor) + 180
    elevation = asin_d(zhor)

    return AzimuthElevation(azimuth, elevation)


def _sun_mean_ecliptic_elements(t_ut1):
    w = 282.9404 + 4.70935e-5 * t_ut1    # longitude of perihelion degrees
    eccentricity = 0.016709 - 1.151e-9 * t_ut1      # eccentricity
    M = (356.0470 + 0.9856002585 * t_ut1) % 360    # mean anomaly degrees
    L = w + M                        # Sun's mean longitude degrees
    oblecl = 23.4393 - 3.563e-7 * t_ut1  # Sun's obliquity of the ecliptic

    return w, M, L, eccentricity, oblecl


def _sun_eci(w, M, L, eccentricity, oblecl):
    # auxiliary angle
    auxiliary_angle = M + degrees(eccentricity * sin_d(M) * (1 + eccentricity * cos_d(M)))

    # rectangular coordinates in the plane of the ecliptic (x axis toward perihelion)
    x = cos_d(auxiliary_angle) - eccentricity
    y = sin_d(auxiliary_angle) * sqrt(1 - eccentricity**2)

    # find the distance and true anomaly
    r = euclidean_distance(x, y)
    v = atan2_d(y, x)

    # find the true longitude of the sun
    sun_lon = v + w

    # compute the ecliptic rectangular coordinates
    xeclip = r * cos_d(sun_lon)
    yeclip = r * sin_d(sun_lon)
    zeclip = 0.0

    # rotate these coordinates to equatorial rectangular coordinates
    xequat = xeclip
    yequat = yeclip * cos_d(oblecl) + zeclip * sin_d(oblecl)
    zequat = yeclip * sin_d(23.4406) + zeclip * cos_d(oblecl)

    return [xequat, yequat, zequat]


def get_sun(when):
    """
    Returns inertial position of the Sun, in au.
    """
    jd = juliandate(timetuple_from_dt(when))
    date = jd - DECEMBER_31TH_1999_MIDNIGHT_JD

    w, M, L, eccentricity, oblecl = _sun_mean_ecliptic_elements(date)
    sun_eci = _sun_eci(w, M, L, eccentricity, oblecl)

    return np.array(sun_eci)


def get_shadow(r, when_utc):
    """
    Gives illumination of Earth satellite (2 for illuminated, 1 for penumbra, 0 for umbra).

    Parameters
    ----------
    r : numpy.ndarray or list
        ECEF vector pointing to the satellite in km.
    when_utc : datetime.datetime
        Time of calculation.

    """
    gmst = gstime_from_datetime(when_utc)
    r_sun = get_sun(when_utc) * AU

    return shadow(r_sun, ecef_to_eci(r, gmst))


def shadow(r_sun, r, r_p=R_E_KM):
    """
    Gives illumination of Earth satellite (2 for illuminated, 1 for penumbra, 0 for umbra).

    Parameters
    ----------
    r_sun : numpy.ndarray or list
        Vector pointing to the Sun in km.
    r : numpy.ndarray or list
        Vector pointing to the satellite in km.
    r_p : float, optional
        Radius of the planet, default to Earth WGS84.

    Notes
    -----
    Algorithm 34 from Vallado, section 5.3.

    """
    alpha_umb = radians(0.264121687)
    alpha_pen = radians(0.269007205)

    shadow_result = 2

    if dot_product(r_sun, r) < 0:
        angle = angle_between(-r_sun, r)
        sat_horiz = vector_norm(r) * cos_d(angle)
        sat_vert = vector_norm(r) * sin_d(angle)
        x = r_p / sin(alpha_pen)
        pen_vert = tan(alpha_pen) * (x + sat_horiz)

        if sat_vert <= pen_vert:
            y = r_p / sin(alpha_umb)
            umb_vert = tan(alpha_umb) * (y - sat_horiz)

            if sat_vert <= umb_vert:
                shadow_result = 0
            else:
                shadow_result = 1

    return shadow_result


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

    # Calculate local sidereal time
    GMST0 = ((sun_lon + 180) % 360) / 15
    return GMST0 + UTH + local_lon / 15


def gstime_from_datetime(when_utc):
    timetuple = timetuple_from_dt(when_utc)
    return _gstime(jday(*timetuple))


def float_to_hms(hour):
    rem, hour = modf(hour)
    rem, minute = modf(rem * 60)
    rem, second = modf(rem * 60)

    return int(hour), int(minute), int(second), int(rem * 1e6)


def timetuple_from_dt(when_utc):
    timetuple = (when_utc.year, when_utc.month, when_utc.day,
                 when_utc.hour, when_utc.minute, when_utc.second + when_utc.microsecond * 1e-6)
    return timetuple


def mean_motion(sma_km):
    return sqrt(MU_E / sma_km ** 3)  # rad / s


class reify:
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
