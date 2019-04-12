# Inspired by
# Copyright (c) 2012-2017 Juan Luis Cano Rodríguez, MIT license
from unittest import TestCase

from math import radians
import numpy as np
from numpy.testing import assert_allclose

from sgp4.earth_gravity import wgs84

from orbit_predictor.keplerian import coe2rv, rv2coe


class COE2RVTests(TestCase):
    def test_convert_coe_to_rv(self):
        # Data from Vallado, example 2.6
        p = 11067.790
        ecc = 0.83285
        inc = radians(87.87)
        raan = radians(227.89)
        argp = radians(53.38)
        ta = radians(92.335)

        expected_position = [6525.344, 6861.535, 6449.125]
        expected_velocity = [4.902276, 5.533124, -1.975709]

        position, velocity = coe2rv(wgs84.mu, p, ecc, inc, raan, argp, ta)

        assert_allclose(position, expected_position, rtol=1e-5)
        assert_allclose(velocity, expected_velocity, rtol=1e-5)


class RV2COETests(TestCase):
    def test_convert_rv_to_coe(self):
        # Data from Vallado, example 2.5
        position = np.array([6524.384, 6862.875, 6448.296])
        velocity = np.array([4.901327, 5.533756, -1.976341])

        expected_p = 11067.79
        expected_ecc = 0.832853
        expected_inc = radians(87.870)
        expected_raan = radians(227.89)
        expected_argp = radians(53.38)
        expected_ta = radians(92.335)

        p, ecc, inc, raan, argp, ta = rv2coe(wgs84.mu, position, velocity)

        self.assertAlmostEqual(p, expected_p, places=0)
        self.assertAlmostEqual(ecc, expected_ecc, places=4)
        self.assertAlmostEqual(inc, expected_inc, places=4)
        self.assertAlmostEqual(raan, expected_raan, places=3)
        self.assertAlmostEqual(argp, expected_argp, places=3)
        self.assertAlmostEqual(ta, expected_ta, places=5)
