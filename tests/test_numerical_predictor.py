import datetime as dt
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose

from orbit_predictor.locations import ARG
from orbit_predictor.predictors.numerical import J2Predictor, InvalidOrbitError


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

        position_eci, velocity_eci = self.predictor._propagate_eci(when_utc)

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
        self.assertAlmostEqual(pred._inc, expected_inc, places=2)

    def test_sun_sync_from_altitude_and_inclination(self):
        # Hardcoded from our implementation
        expected_ecc = 0.14546153131334466

        pred = J2Predictor.sun_synchronous(alt_km=475, inc_deg=97)
        self.assertAlmostEqual(pred._ecc, expected_ecc, places=16)

    def test_sun_sync_from_eccentricity_and_inclination(self):
        # Vallado 3rd edition, example 11-2
        expected_sma = 7346.846

        pred = J2Predictor.sun_synchronous(ecc=0.2, inc_deg=98.6)
        self.assertAlmostEqual(pred._sma, expected_sma, places=1)

    def test_sun_sync_delta_true_anomaly_has_expected_anomaly_and_epoch(self):
        date = dt.datetime.today().date()
        ltan_h = 12
        expected_ref_epoch = dt.datetime(date.year, date.month, date.day, 12, tzinfo=dt.timezone.utc)

        for delta_ta_deg in [-30, 0, 30]:
            pred = J2Predictor.sun_synchronous(
                alt_km=800, ecc=0, date=date, ltan_h=ltan_h, delta_ta_deg=delta_ta_deg
            )

            expected_epoch = expected_ref_epoch + dt.timedelta(
                minutes=np.radians(delta_ta_deg) / pred.mean_motion
            )

            self.assertEqual(pred._ta, delta_ta_deg)
            self.assertEqual(pred._epoch, expected_epoch)
