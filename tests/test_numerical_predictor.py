import datetime as dt
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest

from orbit_predictor.locations import ARG
from orbit_predictor.predictors.numerical import (
    J2Predictor, InvalidOrbitError, R_E_KM, is_sun_synchronous
)


class J2PredictorTests(TestCase):
    def setUp(self):
        # Converted to classical orbital elements
        sma = 6780
        ecc = 0.001
        inc = 28.5
        raan = 67.0
        argp = 355.0
        ta = 250.0

        self.epoch = dt.datetime(2000, 1, 1, 12, 0)

        self.predictor = J2Predictor(sma, ecc, inc, raan, argp, ta, self.epoch)

    def test_propagate_eci(self):
        # Data from GMAT
        expected_position = np.array([2085.9287615146, -6009.5713894563, -2357.3802307070])
        expected_velocity = np.array([6.4787522759177, 3.2366136616580, -2.5063420188165])

        when_utc = self.epoch + dt.timedelta(hours=3)

        position_eci, velocity_eci = self.predictor.propagate_eci(when_utc)

        assert_allclose(position_eci, expected_position, rtol=1e-2)
        assert_allclose(velocity_eci, expected_velocity, rtol=1e-2)

    def test_get_next_pass(self):
        pass_ = self.predictor.get_next_pass(ARG)

        assert pass_.sate_id == "<custom>"


class SunSynchronousTests(TestCase):
    def test_invalid_parameters_raises_error(self):
        self.assertRaises(
            InvalidOrbitError, J2Predictor.sun_synchronous, alt_km=400, inc_deg=90)
        self.assertRaises(
            InvalidOrbitError, J2Predictor.sun_synchronous, alt_km=10000, ecc=0)

    def test_sun_sync_from_altitude_and_eccentricity(self):
        # Vallado 3rd edition, example 11-2
        expected_inc = 98.6

        pred = J2Predictor.sun_synchronous(alt_km=800, ecc=0)
        self.assertAlmostEqual(pred.get_position().osculating_elements[2], expected_inc, places=2)

    def test_sun_sync_from_altitude_and_inclination(self):
        # Hardcoded from our implementation
        expected_ecc = 0.14546153131334466

        pred = J2Predictor.sun_synchronous(alt_km=475, inc_deg=97)
        self.assertAlmostEqual(pred.get_position().osculating_elements[1], expected_ecc, places=14)

    def test_sun_sync_from_eccentricity_and_inclination(self):
        # Vallado 3rd edition, example 11-2
        expected_sma = 7346.846

        pred = J2Predictor.sun_synchronous(ecc=0.2, inc_deg=98.6)
        self.assertAlmostEqual(pred.get_position().osculating_elements[0], expected_sma, places=1)

    def test_sun_sync_delta_true_anomaly_has_expected_anomaly_and_epoch(self):
        date = dt.datetime.today().date()
        ltan_h = 12
        expected_ref_epoch = dt.datetime(date.year, date.month, date.day, 12)

        for expected_ta_deg in [-30, 0, 30]:
            pred = J2Predictor.sun_synchronous(
                alt_km=800, ecc=0, date=date, ltan_h=ltan_h, ta_deg=expected_ta_deg
            )

            ta_deg = pred.get_position(expected_ref_epoch).osculating_elements[5]
            self.assertAlmostEqual(ta_deg, expected_ta_deg % 360, places=12)

    def test_sun_sync_delta_true_anomaly_non_circular(self):
        date = dt.datetime.today().date()
        ltan_h = 12
        expected_ref_epoch = dt.datetime(date.year, date.month, date.day, 12)

        for expected_ta_deg in [-30, 30]:
            pred = J2Predictor.sun_synchronous(
                alt_km=475, ecc=0.1455, date=date, ltan_h=ltan_h, ta_deg=expected_ta_deg
            )

            ta_deg = pred.get_position(expected_ref_epoch).osculating_elements[5]
            self.assertAlmostEqual(ta_deg, expected_ta_deg % 360, places=12)


# Test data from Wertz et al. "Space Mission Engineering: The New SMAD" (2011), table 9-13
@pytest.mark.parametrize("orbits,days,inc_deg,expected_h", [
    (14, 1, 28, 817.14),
    (43, 3, 28, 701.34),
    (29, 2, 28, 645.06),
    (59, 4, 28, 562.55),
    (74, 5, 28, 546.31),
    (15, 1, 28, 482.25),
])
def test_repeated_groundtrack_sma(orbits, days, inc_deg, expected_h):
    pred = J2Predictor.repeating_ground_track(orbits=orbits, days=days, ecc=0.0, inc_deg=inc_deg)

    assert_almost_equal(pred.get_position().osculating_elements[0] - R_E_KM, expected_h, decimal=0)


def test_is_sun_sync_returns_false_for_non_sun_sync_orbit():
    pred1 = J2Predictor(7000, 0, 0, 0, 0, 0, dt.datetime.now())

    assert not is_sun_synchronous(pred1)


def test_is_sun_sync_detects_almost_sun_sync_orbit():
    pred2 = J2Predictor(R_E_KM + 460, 0.001, 97.4, 0, 0, 0, dt.datetime.now())

    assert not is_sun_synchronous(pred2)
    assert is_sun_synchronous(pred2, rtol=1e-1)


def test_is_sun_sync_returns_true_for_sun_sync_orbit():
    pred1 = J2Predictor.sun_synchronous(alt_km=500, ecc=0)
    pred2 = J2Predictor.sun_synchronous(alt_km=500, inc_deg=97)
    pred3 = J2Predictor.sun_synchronous(ecc=0, inc_deg=97)

    assert is_sun_synchronous(pred1)
    assert is_sun_synchronous(pred2)
    assert is_sun_synchronous(pred3)
