import datetime as dt

import numpy as np
import pytest

from orbit_predictor.utils import angle_between, get_sun, get_shadow


# Data obtained from Astropy using the JPL ephemerides
# coords = get_body("sun", Time(when_utc)).represent_as(CartesianRepresentation).xyz.to("au").T.value
@pytest.mark.parametrize("when_utc,expected_eci", [
    [dt.datetime(2000, 1, 1, 12), np.array([0.17705013, -0.88744275, -0.38474906])],
    [dt.datetime(2009, 6, 1, 18, 30), np.array([0.32589889, 0.88109849, 0.38197646])],
    [dt.datetime(2019, 11, 25, 18, 46, 0), np.array([-0.449363, -0.80638653, -0.34956405])],
    [dt.datetime(2025, 12, 1, 12), np.array([-0.35042293, -0.84565374, -0.36657211])],
])
def test_get_sun_matches_expected_result_within_precision(when_utc, expected_eci):
    eci = get_sun(when_utc)

    assert angle_between(eci, expected_eci) < 1.0  # Claimed precision
    assert angle_between(eci, expected_eci) < 0.5  # Actual precision


# Data obtained from GMAT
@pytest.mark.parametrize("when_utc,r_ecef,expected_shadow", [
    [dt.datetime(2000, 1, 1, 12, 9, 0), np.array([1272.929355, 6984.992047, 1299.821897]), 2],
    [dt.datetime(2000, 1, 1, 12, 30, 0), np.array([-7298.548961, 500.322464, 639.443822]), 0],
])
def test_get_shadow_matches_expected_result(when_utc, r_ecef, expected_shadow):
    shadow = get_shadow(r_ecef, when_utc)

    assert shadow == expected_shadow


# Data obtained from GMAT
# Testing the penumbra is much harder, because it only lasts a few seconds
# and the uncertainty in the Sun position is even larger than the angle difference
# between umbra and penumbra
@pytest.mark.xfail
@pytest.mark.parametrize("when_utc,r_ecef", [
    [dt.datetime(2000, 1, 1, 12, 10, 5), np.array([-2779.471958, 6565.365892, 1625.185914])],
    [dt.datetime(2000, 1, 1, 12, 10, 15), np.array([-2842.327184, 6539.439097, 1625.522584])],
])
def test_get_shadow_gives_penumbra(when_utc, r_ecef):
    shadow = get_shadow(r_ecef, when_utc)

    assert shadow == 1
