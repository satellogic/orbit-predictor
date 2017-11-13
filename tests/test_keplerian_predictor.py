from unittest import TestCase

from datetime import datetime, timedelta

import numpy as np
from numpy.testing import assert_allclose

from orbit_predictor.predictors import TLEPredictor
from orbit_predictor.predictors.keplerian import KeplerianPredictor
from orbit_predictor.sources import MemoryTLESource


class KeplerianPredictorTests(TestCase):
    def setUp(self):
        # Data from Vallado, example 2.4
        # Converted to classical orbital elements
        sma = 7200.478692389954
        ecc = 0.00810123424807035
        inc = 98.59998936154028
        raan = 319.7043176816153
        argp = 70.87958362589532
        ta = 0.004121614893481961

        self.epoch = datetime(2000, 1, 1, 12, 0)

        self.predictor = KeplerianPredictor(sma, ecc, inc, raan, argp, ta, self.epoch)

    def test_initial_position(self):
        expected_position = np.array([1131.340, -2282.343, 6672.423])
        expected_velocity = np.array([-5.64305, 4.30333, 2.42879])

        position_eci, velocity_eci = self.predictor._propagate_eci(self.epoch)

        assert_allclose(position_eci, expected_position, rtol=1e-7)
        assert_allclose(velocity_eci, expected_velocity, rtol=1e-6)

    def test_propagate_eci(self):
        expected_position = np.array([-4219.7527, 4363.0292, -3958.7666])
        expected_velocity = np.array([3.689866, -1.916735, -6.112511])

        when_utc = self.epoch + timedelta(minutes=40)

        position_eci, velocity_eci = self.predictor._propagate_eci(when_utc)

        assert_allclose(position_eci, expected_position, rtol=1e-5)
        assert_allclose(velocity_eci, expected_velocity, rtol=1e-5)


class KeplerianPredictorOneDayTests(TestCase):
    def setUp(self):
        # Converted to classical orbital elements
        sma = 6780
        ecc = 0.001
        inc = 28.5
        raan = 67.0
        argp = 355.0
        ta = 250.0

        self.epoch = datetime(2000, 1, 1, 12, 0)

        self.predictor = KeplerianPredictor(sma, ecc, inc, raan, argp, ta, self.epoch)

    def test_initial_position(self):
        expected_position = np.array([3852.57404763, -4749.1872318, -2933.02952967])
        expected_velocity = np.array([5.33068317, 5.28723659, -1.54255441])

        position_eci, velocity_eci = self.predictor._propagate_eci(self.epoch)

        assert_allclose(position_eci, expected_position, rtol=1e-7)
        assert_allclose(velocity_eci, expected_velocity, rtol=1e-6)

    def test_propagate_eci(self):
        expected_position = np.array([-5154.02724044, 3011.19175291, 3214.77198183])
        expected_velocity = np.array([-3.67652279, -6.71613987, 0.41267465])

        when_utc = self.epoch + timedelta(hours=23.999999999)

        position_eci, velocity_eci = self.predictor._propagate_eci(when_utc)

        assert_allclose(position_eci, expected_position, rtol=1e-2)
        assert_allclose(velocity_eci, expected_velocity, rtol=1e-2)

    def test_propagate_one_day(self):
        expected_position = np.array([-5154.02724044, 3011.19175291, 3214.77198183])
        expected_velocity = np.array([-3.67652279, -6.71613987, 0.41267465])

        when_utc = self.epoch + timedelta(hours=24)

        position_eci, velocity_eci = self.predictor._propagate_eci(when_utc)

        assert_allclose(position_eci, expected_position, rtol=1e-2)
        assert_allclose(velocity_eci, expected_velocity, rtol=1e-2)


class TLEConversionTests(TestCase):

    SATE_ID = '41558U'  # newsat 1
    LINES = (
        '1 41558U 16033C   17065.21129769  .00002236  00000-0  88307-4 0  9995',
        '2 41558  97.4729 144.7611 0014207  16.2820 343.8872 15.26500433 42718',
    )

    def test_from_tle_returns_same_initial_conditions_on_epoch(self):
        start = datetime(2017, 3, 6, 7, 51)
        db = MemoryTLESource()
        db.add_tle(self.SATE_ID, self.LINES, start)

        keplerian_predictor = KeplerianPredictor.from_tle(self.SATE_ID, db, start)
        tle_predictor = TLEPredictor(self.SATE_ID, db)

        epoch = keplerian_predictor._epoch

        pos_keplerian = keplerian_predictor.get_position(epoch)
        pos_tle = tle_predictor.get_position(epoch)

        assert_allclose(pos_keplerian.position_ecef, pos_tle.position_ecef, rtol=1e-11)
        assert_allclose(pos_keplerian.velocity_ecef, pos_tle.velocity_ecef, rtol=1e-13)
