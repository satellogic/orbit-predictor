import datetime as dt

from hypothesis import given
from hypothesis.strategies import datetimes, composite, floats
import pytest

from orbit_predictor.constants import R_E_KM
from orbit_predictor.predictors.keplerian import KeplerianPredictor
from orbit_predictor.predictors.numerical import J2Predictor
from orbit_predictor.utils import eclipse_duration, get_shadow
from orbit_predictor.sources import get_predictor_from_tle_lines


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


@composite
def equatorial_orbits(
    draw,
    sma=floats(R_E_KM, 42000),
    ecc=floats(0, 1, exclude_max=True),
    argp=floats(0, 360),
    ta=floats(0, 360),
    epoch=datetimes(),
):
    return KeplerianPredictor(
        draw(sma), draw(ecc), 0, 0, draw(argp), draw(ta), draw(epoch),
    )


@pytest.fixture()
def sate_predictors():
    # First tle: The nominal case: LEO orbit sun-sync
    # Second tle: Beta angle with no-eclipse months
    predictors = {
        'nominal_newsat':
            ('1 45018U 20003C   20059.45818850  .00000954  00000-0  38528-4 0  9996',
             '2 45018  97.3318 127.6031 0014006 111.9460 248.3269 15.27107050  6765'),
        'rare_ltan6':
            ('1 37673U 11024A   20058.76186510  .00000058  00000-0  16774-4 0  9995',
             '2 37673  98.0102  67.8905 0000719  69.0278 104.2216 14.72969019468543'),
    }
    return {sate_name: get_predictor_from_tle_lines(tle)
            for sate_name, tle in predictors.items()}


@given(predictor=equatorial_orbits(), when_utc=datetimes())
def test_get_normal_vector_zero_inclination_always_z_aligned(predictor, when_utc):
    normal_vector = predictor.get_normal_vector(when_utc)

    assert normal_vector[0] == normal_vector[1] == 0
    assert normal_vector[2] == 1


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


@pytest.mark.parametrize('sate_dates_list', [[
    ['nominal_newsat', dt.datetime(2021, 9, 4), dt.datetime(2021, 9, 5)],
    ['rare_ltan6', dt.datetime(2021, 8, 5), dt.datetime(2021, 8, 6)],
]])
def test_eclipses_since_is_consistent_with_get_shadow(sate_predictors, sate_dates_list):
    for sate_name, start, limit in sate_dates_list:
        predictor = sate_predictors[sate_name]
        for ecl_start, ecl_end in predictor.eclipses_since(start, limit):
            one_second = dt.timedelta(seconds=1)

            pre_eclipse_start = ecl_start - one_second
            post_eclipse_start = ecl_start + one_second
            pre_eclipse_end = ecl_end - one_second
            post_eclipse_end = ecl_end + one_second

            for time_to_check, must_be_illuminated in [
                (pre_eclipse_start, True),
                (post_eclipse_start, False),
                (pre_eclipse_end, False),
                (post_eclipse_end, True),
            ]:
                pos = predictor.get_only_position(time_to_check)
                assert (get_shadow(pos, time_to_check) == 2) == must_be_illuminated


@pytest.mark.parametrize('sate_dates_list', [[
    ['nominal_newsat', dt.datetime(2021, 9, 4), dt.datetime(2021, 9, 4, 12, 0, 0)],
    ['rare_ltan6', dt.datetime(2021, 8, 5), dt.datetime(2021, 8, 5, 12, 0, 0)],
]])
def test_eclipses_since_finds_all_eclipses_in_a_few_orbits(sate_predictors, sate_dates_list):
    for sate_name, start, limit in sate_dates_list:
        predictor = sate_predictors[sate_name]
        eclipses = list(predictor.eclipses_since(start, limit))
        total_duration_m = int((limit - start).total_seconds() / 60)
        for mins in range(total_duration_m):
            current_time = start + dt.timedelta(minutes=mins)
            pos = predictor.get_only_position(current_time)
            if get_shadow(pos, current_time) != 2:
                assert any(ecl_start <= current_time <= ecl_end
                           for ecl_start, ecl_end in eclipses)
