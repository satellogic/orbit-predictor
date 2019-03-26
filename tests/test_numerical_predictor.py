from datetime import datetime, timedelta

from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose

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

        self.epoch = datetime(2000, 1, 1, 12, 0)

        self.predictor = J2Predictor(sma, ecc, inc, raan, argp, ta, self.epoch)

    def test_propagate_eci(self):
        # Data from GMAT
        expected_position = np.array([2085.9287615146, -6009.5713894563, -2357.3802307070])
        expected_velocity = np.array([6.4787522759177, 3.2366136616580, -2.5063420188165])

        when_utc = self.epoch + timedelta(hours=3)

        position_eci, velocity_eci = self.predictor._propagate_eci(when_utc)

        assert_allclose(position_eci, expected_position, rtol=1e-2)
        assert_allclose(velocity_eci, expected_velocity, rtol=1e-2)


class SunsynchronousTests(TestCase):
    def test_invalid_parameters_raises_error(self):
        self.assertRaises(
            InvalidOrbitError, J2Predictor.sun_synchronous, alt=400, inc=90)
        self.assertRaises(
            InvalidOrbitError, J2Predictor.sun_synchronous, alt=10000, ecc=0)
