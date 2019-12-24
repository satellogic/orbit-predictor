from math import pi

from sgp4.earth_gravity import wgs84


AU = 149597870.700  # km

LIGHT_SPEED_KMS = 299792.458  # km / s

OMEGA = 2 * pi / (86400 * 365.2421897)  # rad / s
MU_E = wgs84.mu  # km3 / s2
R_E_KM = wgs84.radiusearthkm  # km
F_E = 1 / 298.257223560
J2 = wgs84.j2
OMEGA_E = 7.292115e-5  # rad / s
