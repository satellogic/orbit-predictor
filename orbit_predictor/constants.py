from math import pi

from sgp4.earth_gravity import wgs84


AU = 149597870.700  # km

OMEGA = 2 * pi / (86400 * 365.2421897)  # rad / s
MU_E = wgs84.mu  # km3 / s2
R_E_KM = wgs84.radiusearthkm  # km
J2 = wgs84.j2
OMEGA_E = 7.292115e-5  # rad / s
