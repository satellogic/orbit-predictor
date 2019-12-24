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
import importlib
from math import asin, cos, degrees, radians, sin, sqrt
import os

from orbit_predictor.constants import LIGHT_SPEED_KMS
from orbit_predictor import coordinate_systems
from orbit_predictor.utils import reify, sun_azimuth_elevation


class Location:
    def __init__(self, name, latitude_deg, longitude_deg, elevation_m):
        """Location.

        Parameters
        ----------
        latitude_deg : float
            Latitude in degrees.
        longitude_deg : float
            Longitude in degrees.
        elevation_m : float
            Elevation in meters.

        """
        self.name = name
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.elevation_m = elevation_m
        self.position_ecef = coordinate_systems.geodetic_to_ecef(
            radians(latitude_deg),
            radians(longitude_deg),
            elevation_m / 1000.)
        self.position_llh = latitude_deg, longitude_deg, elevation_m

    def __eq__(self, other):
        return all([issubclass(other.__class__, Location),
                    self.name == other.name,
                    self.latitude_deg == other.latitude_deg,
                    self.longitude_deg == other.longitude_deg,
                    self.elevation_m == other.elevation_m])

    def __repr__(self):
        return "<Location {}>".format(self.name)

    def __str__(self):
        return self.name

    @reify
    def latitude_rad(self):
        return radians(self.latitude_deg)

    @reify
    def longitude_rad(self):
        return radians(self.longitude_deg)

    @reify
    def _cached_elevation_calculation_data(self):
        sin_lat, sin_long = sin(self.latitude_rad), sin(self.longitude_rad)
        cos_lat, cos_long = cos(self.latitude_rad), cos(self.longitude_rad)
        return (cos_lat * cos_long,
                cos_lat * sin_long,
                sin_lat)

    def sun_elevation_on_earth(self, when_utc=None):
        """Return Sun elevation on Earth of location at when_utc."""
        if when_utc is None:
            when_utc = dt.datetime.utcnow()
        _, elevation = sun_azimuth_elevation(self.latitude_deg, self.longitude_deg, when_utc)
        return elevation

    def elevation_for(self, position):
        """Returns elevation to given position in radians

        calculation is made inline to have better performance
        """
        observer_pos_ecef = self.position_ecef
        object_coords_ecef = position

        rx = object_coords_ecef[0] - observer_pos_ecef[0]
        ry = object_coords_ecef[1] - observer_pos_ecef[1]
        rz = object_coords_ecef[2] - observer_pos_ecef[2]

        a, b, c = self._cached_elevation_calculation_data

        top_z = (a * rx) + (b * ry) + (c * rz)

        range_sat = sqrt((rx * rx) + (ry * ry) + (rz * rz))

        return asin(top_z / range_sat)

    def get_azimuth_elev(self, position):
        """Return azimuth and elevation of position_ecef from the current Location instance."""

        top = coordinate_systems.to_horizon(self.latitude_rad, self.longitude_rad,
                                            self.position_ecef, position.position_ecef)

        return coordinate_systems.horizon_to_az_elev(*top)

    def get_azimuth_elev_deg(self, position):
        """Idem that get_azimuth_elev() but using degrees."""
        az, el = self.get_azimuth_elev(position)
        return degrees(az), degrees(el)

    def is_visible(self, position, elevation=0):
        """Return True if the Satellite if visible from the current instance."""
        _, elev_deg = self.get_azimuth_elev_deg(position)
        return elev_deg >= elevation

    def slant_range_km(self, position_ecef):
        """Distance to the satellite in straight line"""
        pos = position_ecef
        loc = self.position_ecef
        return sqrt((pos[0]-loc[0])**2 + (pos[1]-loc[1])**2 + (pos[2]-loc[2])**2)

    def slant_range_velocity_kms(self, position):
        """Velocity the satellite from location's point of view"""
        pos = position.position_ecef
        vel = position.velocity_ecef

        current_range = self.slant_range_km(pos)
        next_pos = (pos[0]+vel[0], pos[1]+vel[1], pos[2]+vel[2])
        next_range = self.slant_range_km(next_pos)

        return next_range - current_range

    def doppler_factor(self, position):
        """Doppler effect factor relative to 1"""
        range_rate = self.slant_range_velocity_kms(position)
        return 1. + (range_rate / LIGHT_SPEED_KMS)


# A hardcoded list of locations. Some of them are satellite groundstations or HAM
AFRICA1 = Location(
    "AFRICA1", latitude_deg=-4.2937711, longitude_deg=15.4493049, elevation_m=266.00)
AFRICA2 = Location(
    "AFRICA2", latitude_deg=-19.9243839, longitude_deg=23.439418, elevation_m=939.12)
AFRICA3 = Location(
    "AFRICA3", latitude_deg=-26.0317764, longitude_deg=28.254681, elevation_m=1617.62)
AFRICA4 = Location(
    "AFRICA4", latitude_deg=0.3979327, longitude_deg=32.5021788, elevation_m=1165.38)
AFRICA5 = Location(
    "AFRICA5", latitude_deg=-1.2960418, longitude_deg=36.9340893, elevation_m=1599.74)
AMERICA1 = Location(
    "AMERICA1", latitude_deg=40.6599903, longitude_deg=-74.1713736, elevation_m=10.46)
AMERICA10 = Location(
    "AMERICA10", latitude_deg=34.0863943, longitude_deg=-118.0329261, elevation_m=88.67)
AMERICA11 = Location(
    "AMERICA11", latitude_deg=37.9916467, longitude_deg=-122.0559013, elevation_m=6.53)
AMERICA2 = Location(
    "AMERICA2", latitude_deg=38.9054965, longitude_deg=-77.0230685, elevation_m=25.25)
AMERICA3 = Location(
    "AMERICA3", latitude_deg=33.7800684, longitude_deg=-84.5208486, elevation_m=245.74)
AMERICA4 = Location(
    "AMERICA4", latitude_deg=29.9414947, longitude_deg=-90.0633866, elevation_m=3.64)
AMERICA5 = Location(
    "AMERICA5", latitude_deg=29.9865571, longitude_deg=-95.3423456, elevation_m=29.35)
AMERICA6 = Location(
    "AMERICA6", latitude_deg=19.4361691, longitude_deg=-99.0719249, elevation_m=2224.95)
AMERICA7 = Location(
    "AMERICA7", latitude_deg=20.5216683, longitude_deg=-103.310728, elevation_m=1530.03)
AMERICA8 = Location(
    "AMERICA8", latitude_deg=35.0401972, longitude_deg=-106.6090026, elevation_m=1619.52)
AMERICA9 = Location(
    "AMERICA9", latitude_deg=33.6928889, longitude_deg=-112.078808, elevation_m=450.69)
ARG = Location("ARG", latitude_deg=-31.2884, longitude_deg=-64.2032868, elevation_m=492.96)
ASIA1 = Location("ASIA1", latitude_deg=32.0092853, longitude_deg=34.8945777, elevation_m=38.03)
ASIA10 = Location("ASIA10", latitude_deg=23.8450823, longitude_deg=90.4016501, elevation_m=11.71)
ASIA11 = Location("ASIA11", latitude_deg=16.9069935, longitude_deg=96.1342117, elevation_m=24.81)
ASIA12 = Location("ASIA12", latitude_deg=13.9125333, longitude_deg=100.6068365, elevation_m=6.22)
ASIA13 = Location("ASIA13", latitude_deg=21.2137962, longitude_deg=105.805638, elevation_m=12.45)
ASIA14 = Location("ASIA14", latitude_deg=23.3924882, longitude_deg=113.2990193, elevation_m=5.97)
ASIA15 = Location("ASIA15", latitude_deg=31.7392443, longitude_deg=118.8733768, elevation_m=13.34)
ASIA16 = Location("ASIA16", latitude_deg=37.5602464, longitude_deg=126.7909134, elevation_m=20.35)
ASIA17 = Location("ASIA17", latitude_deg=33.5846715, longitude_deg=130.4510901, elevation_m=5.67)
ASIA18 = Location("ASIA18", latitude_deg=34.7854932, longitude_deg=135.4384313, elevation_m=10.01)
ASIA19 = Location("ASIA19", latitude_deg=35.7120096, longitude_deg=139.4033569, elevation_m=91.00)
ASIA2 = Location("ASIA2", latitude_deg=24.5536664, longitude_deg=39.7053953, elevation_m=638.85)
ASIA3 = Location("ASIA3", latitude_deg=33.2621459, longitude_deg=44.234124, elevation_m=36.05)
ASIA4 = Location("ASIA4", latitude_deg=35.6883245, longitude_deg=51.3143664, elevation_m=1185.27)
ASIA5 = Location("ASIA5", latitude_deg=36.2320987, longitude_deg=59.6430435, elevation_m=996.20)
ASIA6 = Location("ASIA6", latitude_deg=31.628871, longitude_deg=65.7371749, elevation_m=1014.35)
ASIA7 = Location("ASIA7", latitude_deg=33.6187486, longitude_deg=73.0960301, elevation_m=502.00)
ASIA8 = Location("ASIA8", latitude_deg=28.5543983, longitude_deg=77.086455, elevation_m=226.08)
ASIA9 = Location("ASIA9", latitude_deg=12.950055, longitude_deg=77.66856, elevation_m=889.07)
AUSTRALIA1 = Location(
    "AUSTRALIA1", latitude_deg=-31.9170947, longitude_deg=115.970206, elevation_m=12.73)
AUSTRALIA2 = Location(
    "AUSTRALIA2", latitude_deg=-17.8184141, longitude_deg=122.2364966, elevation_m=28.67)
AUSTRALIA5 = Location(
    "AUSTRALIA5", latitude_deg=-34.7794086, longitude_deg=138.6370729, elevation_m=22.43)
AUSTRALIA6 = Location(
    "AUSTRALIA6", latitude_deg=-36.7103328, longitude_deg=144.3303179, elevation_m=197.96)
AUSTRALIA7 = Location(
    "AUSTRALIA7", latitude_deg=-34.0096484, longitude_deg=150.6926073, elevation_m=74.77)
BA1 = Location("BA1", latitude_deg=-34.5561944, longitude_deg=-58.41368, elevation_m=7.02)
CHILE = Location("CHILE", latitude_deg=-33.3631552, longitude_deg=-70.7904123, elevation_m=477.00)
EASTER_ISLAND = Location(
    "EASTER_ISLAND", latitude_deg=-27.0578009, longitude_deg=-109.3817317, elevation_m=61.69)
EUROPA1 = Location("EUROPA1", latitude_deg=41.2486859, longitude_deg=-8.6813677, elevation_m=56.44)
EUROPA10 = Location("EUROPA10", latitude_deg=45.7274069, longitude_deg=65.37, elevation_m=122.98)
EUROPA11 = Location(
    "EUROPA11", latitude_deg=45.4452575, longitude_deg=9.2767394, elevation_m=106.02)
EUROPA12 = Location(
    "EUROPA12", latitude_deg=48.3534778, longitude_deg=11.7864782, elevation_m=447.39)
EUROPA13 = Location(
    "EUROPA13", latitude_deg=42.4310529, longitude_deg=14.1828016, elevation_m=9.81)
EUROPA14 = Location(
    "EUROPA14", latitude_deg=41.1150865, longitude_deg=16.8624173, elevation_m=13.89)
EUROPA15 = Location("EUROPA15", latitude_deg=37.9364517, longitude_deg=23.94452, elevation_m=80.00)
EUROPA16 = Location(
    "EUROPA16", latitude_deg=38.2925088, longitude_deg=27.1556125, elevation_m=119.68)
EUROPA17 = Location(
    "EUROPA17", latitude_deg=35.1544144, longitude_deg=33.3585865, elevation_m=172.25)
EUROPA3 = Location("EUROPA3", latitude_deg=37.4189722, longitude_deg=-5.8929429, elevation_m=27.52)
EUROPA5 = Location(
    "EUROPA5", latitude_deg=40.4915238, longitude_deg=-3.5677712, elevation_m=597.39)
EUROPA7 = Location("EUROPA7", latitude_deg=39.4892396, longitude_deg=-0.4819177, elevation_m=60.64)
EUROPA9 = Location("EUROPA9", latitude_deg=49.0067717, longitude_deg=2.5529958, elevation_m=102.37)
MADAGASCAR1 = Location(
    "MADAGASCAR1", latitude_deg=15.4967687, longitude_deg=44.2171958, elevation_m=2186.00)
MADAGASCAR2 = Location(
    "MADAGASCAR2", latitude_deg=-18.7825536, longitude_deg=47.4800904, elevation_m=1260.62)
NZ1 = Location("NZ1", latitude_deg=-44.7149065, longitude_deg=169.2468643, elevation_m=339.58)
NZ2 = Location("NZ2", latitude_deg=-36.5886632, longitude_deg=174.8717244, elevation_m=0.00)
RIO = Location("RIO", latitude_deg=-22.910590, longitude_deg=-43.188958, elevation_m=16.92)
USA = Location("USA", latitude_deg=40.24, longitude_deg=-101.9, elevation_m=1100)
australia = Location('australia', latitude_deg=-25.1, longitude_deg=134.5, elevation_m=290)
brazil = Location("brazil", latitude_deg=-11.2, longitude_deg=-54.66, elevation_m=310)
blq_leafline = Location('blq_leafline', latitude_deg=45.59, longitude_deg=9.361, elevation_m=194)
central_america = Location(
    "central_america", latitude_deg=11.17, longitude_deg=-87.23, elevation_m=310)
central_argentina = Location(
    'central_argentina', latitude_deg=-35.75, longitude_deg=-63.9, elevation_m=133)
china = Location('china', latitude_deg=35.4, longitude_deg=110, elevation_m=1000)
eastern_russia = Location('eastern_russia', latitude_deg=66, longitude_deg=145, elevation_m=650)
france = Location('france', latitude_deg=46.4, longitude_deg=2.75, elevation_m=300)
germany = Location("ALEMANIA", latitude_deg=52.515083, longitude_deg=13.323723, elevation_m=30)
india = Location('india', latitude_deg=23.5, longitude_deg=78.5, elevation_m=550)
moscu = Location('moscu', latitude_deg=55.7, longitude_deg=37.5, elevation_m=137)
niger = Location('niger', latitude_deg=20, longitude_deg=12.5, elevation_m=430)
riogrande = Location("RIOGRANDE", latitude_deg=-53.8, longitude_deg=-67.75, elevation_m=30)


def extend_from_module(module, vars):
    mod = importlib.import_module(module)
    vars.update(mod.__dict__)


# Load custom locations, if the variable is specified
if os.getenv("ORBIT_PREDICTOR_CUSTOM_LOCATIONS"):
    extend_from_module(os.environ["ORBIT_PREDICTOR_CUSTOM_LOCATIONS"], locals())
