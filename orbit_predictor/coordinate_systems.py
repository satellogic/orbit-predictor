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


from math import asin, atan, atan2, cos, degrees, pi, radians, sin, sqrt


def _euclidean_distance(*components):
    # TODO: Remove code duplication with utils
    return sqrt(sum(c**2 for c in components))


def llh_to_ecef(lat_deg, lon_deg, h_km):
    """
    Latitude is geodetic, height is above ellipsoid. Output is in km.
    Formula from http://mathforum.org/library/drmath/view/51832.html
    """
    f = 1 / 298.257224
    a = 6378.137
    lat_rad = radians(lat_deg)
    lon_rad = radians(lon_deg)
    cos_lat = cos(lat_rad)
    sin_lat = sin(lat_rad)
    C = 1 / sqrt(cos_lat ** 2 + (1 - f)**2 * sin_lat ** 2)
    S = (1 - f) ** 2 * C
    k1 = a * C + h_km
    return (k1 * cos_lat * cos(lon_rad), k1 * cos_lat * sin(lon_rad), (a * S + h_km) * sin_lat)


# TODO: Same transformation as llh_to_ecef
def geodetic_to_ecef(lat, lon, height_km):
    a = 6378.137
    b = 6356.7523142
    f = (a - b) / a
    e2 = ((2 * f) - (f * f))
    normal = a / sqrt(1. - (e2 * (sin(lat) * sin(lat))))
    x = (normal + height_km) * cos(lat) * cos(lon)
    y = (normal + height_km) * cos(lat) * sin(lon)
    z = ((normal * (1. - e2)) + height_km) * sin(lat)
    return x, y, z


def ecef_to_llh(ecef_km):
    # WGS-84 ellipsoid parameters */
    a = 6378.1370
    b = 6356.752314

    p = sqrt(ecef_km[0] ** 2 + ecef_km[1] ** 2)
    thet = atan(ecef_km[2] * a / (p * b))
    esq = 1.0 - (b / a) ** 2
    epsq = (a / b) ** 2 - 1.0

    lat = atan((ecef_km[2] + epsq * b * sin(thet) ** 3) / (p - esq * a * cos(thet) ** 3))
    lon = atan2(ecef_km[1], ecef_km[0])
    n = a * a / sqrt(a * a * cos(lat) ** 2 + b ** 2 * sin(lat) ** 2)
    h = p / cos(lat) - n

    lat = degrees(lat)
    lon = degrees(lon)
    return lat, lon, h


def eci_to_ecef(eci_coords, gmst):
    # ccar.colorado.edu/ASEN5070/handouts/coordsys.doc
    #
    # [X] [C -S 0][X]
    # [Y] = [S C 0][Y]
    # [Z]eci [0 0 1][Z]ecef
    #
    #
    # Inverse:
    # [X] [C S 0][X]
    # [Y] = [-S C 0][Y]
    # [Z]ecef [0 0 1][Z]e
    sin_gmst = sin(gmst)
    cos_gmst = cos(gmst)
    eci_x, eci_y, eci_z = eci_coords
    x = (eci_x * cos_gmst) + (eci_y * sin_gmst)
    y = (eci_x * (-sin_gmst)) + (eci_y * cos_gmst)
    z = eci_z
    return x, y, z


def ecef_to_eci(eci_coords, gmst):
    # ccar.colorado.edu/ASEN5070/handouts/coordsys.doc
    #
    # [X] [C -S 0][X]
    # [Y] = [S C 0][Y]
    # [Z]eci [0 0 1][Z]ecef
    #
    #
    # Inverse:
    # [X] [C S 0][X]
    # [Y] = [-S C 0][Y]
    # [Z]ecef [0 0 1][Z]e
    x = (eci_coords[0] * cos(gmst)) - (eci_coords[1] * sin(gmst))
    y = (eci_coords[0] * (sin(gmst))) + (eci_coords[1] * cos(gmst))
    z = eci_coords[2]
    return x, y, z


def eci_to_radec(eci_coords):
    xequat, yequat, zequat = eci_coords

    # convert equatorial rectangular coordinates to RA and Decl:
    r = _euclidean_distance(xequat, yequat, zequat)
    RA = atan2(yequat, xequat)
    DEC = asin(zequat/r)

    return RA, DEC, r


def radec_to_eci(radec_coords):
    raise NotImplementedError


def horizon_to_az_elev(top_s, top_e, top_z):
    range_sat = sqrt((top_s * top_s) + (top_e * top_e) + (top_z * top_z))
    elevation = asin(top_z / range_sat)
    azimuth = atan2(-top_e, top_s) + pi
    return azimuth, elevation


def to_horizon(observer_pos_lat_rad, observer_pos_long_rad, observer_pos_ecef, object_coords_ecef):
    # http://www.celestrak.com/columns/v02n02/
    # TS Kelso's method, except I'm using ECF frame
    # and he uses ECI.

    rx = object_coords_ecef[0] - observer_pos_ecef[0]
    ry = object_coords_ecef[1] - observer_pos_ecef[1]
    rz = object_coords_ecef[2] - observer_pos_ecef[2]

    sin_observer_lat = sin(observer_pos_lat_rad)
    sin_observer_long = sin(observer_pos_long_rad)
    cos_observer_lat = cos(observer_pos_lat_rad)
    cos_observer_long = cos(observer_pos_long_rad)

    top_s = ((sin_observer_lat * cos_observer_long * rx) +
             (sin_observer_lat * sin_observer_long * ry) -
             (cos_observer_lat * rz))
    top_e = -sin_observer_long * rx + cos_observer_long * ry
    top_z = ((cos_observer_lat * cos_observer_long * rx) +
             (cos_observer_lat * sin_observer_long * ry) +
             (sin_observer_lat * rz))

    return top_s, top_e, top_z


def deg_to_dms(deg):
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return [d, m, sd]
