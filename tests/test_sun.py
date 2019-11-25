import datetime as dt

import numpy as np
import pytest

from orbit_predictor.utils import angle_between, get_sun


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
