import datetime as dt

from hypothesis import given
from hypothesis.strategies import datetimes
import pytest

from orbit_predictor.predictors.keplerian import KeplerianPredictor
from orbit_predictor.predictors.numerical import J2Predictor
from orbit_predictor.utils import eclipse_duration


@pytest.fixture()
def elliptical_orbit():
    return KeplerianPredictor(
        7000, 0.2, 0, 0, 0, 0, dt.datetime(2020, 2, 26, 0, 0, 0)
    )


@pytest.fixture()
def non_sun_synchronous():
    return J2Predictor(
        7000, 0, 42, 0, 0, 0, dt.datetime(2020, 2, 26, 0, 0, 0)
    )


@given(datetimes())
def test_get_beta_always_between_m_90_and_90(non_sun_synchronous, when_utc):
    assert -90 <= non_sun_synchronous.get_beta(when_utc) <= 90


def test_get_eclipse_duration_fails_for_eccentric_orbits(elliptical_orbit):
    with pytest.raises(NotImplementedError) as excinfo:
        elliptical_orbit.get_eclipse_duration()

    assert "Non circular orbits are not supported" in excinfo.exconly()


def test_get_eclipse_duration_changes_for_non_sun_synchronous_satellite(non_sun_synchronous):
    dt1 = dt.datetime(2020, 2, 1, 0, 0, 0)
    dt2 = dt.datetime(2020, 2, 15, 0, 0, 0)

    maximum_eclipse_duration = eclipse_duration(0, non_sun_synchronous.period)

    eclipse_duration1 = non_sun_synchronous.get_eclipse_duration(dt1)
    eclipse_duration2 = non_sun_synchronous.get_eclipse_duration(dt2)

    assert eclipse_duration1 <= maximum_eclipse_duration
    assert eclipse_duration2 <= maximum_eclipse_duration
    assert abs(eclipse_duration1 - eclipse_duration2) > 1  # minutes
