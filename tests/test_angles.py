# Inspired by
# https://github.com/poliastro/poliastro/blob/86f971c/src/poliastro/twobody/tests/test_angles.py
# Copyright (c) 2012-2017 Juan Luis Cano Rodríguez, MIT license

from unittest import TestCase

from math import radians, degrees

import numpy as np
from numpy.testing import assert_allclose

from orbit_predictor import angles
from orbit_predictor.utils import rotate


class AnglesTests(TestCase):
    def test_true_to_eccentric(self):
        # Data from NASA-TR-R-158
        data = [
            # ecc, E (deg), ta(deg)
            (0.0, 0.0, 0.0),
            (0.05, 10.52321, 11.05994),
            (0.10, 54.67466, 59.49810),
            (0.35, 142.27123, 153.32411),
            (0.61, 161.87359, 171.02189)
        ]
        for row in data:
            ecc, expected_E, ta = row

            E = angles.ta_to_E(radians(ta), ecc)

            self.assertAlmostEqual(degrees(E), expected_E, places=4)

    def test_mean_to_true(self):
        # Data from Schlesinger & Udick, 1912
        data = [
            # ecc, M (deg), ta (deg)
            (0.0, 0.0, 0.0),
            (0.05, 10.0, 11.06),
            (0.06, 30.0, 33.67),
            (0.04, 120.0, 123.87),
            (0.14, 65.0, 80.50),
            (0.19, 21.0, 30.94),
            (0.35, 65.0, 105.71),
            (0.48, 180.0, 180.0),
            (0.75, 125.0, 167.57)
        ]
        for row in data:
            ecc, M, expected_ta = row

            ta = angles.M_to_ta(radians(M), ecc)

            self.assertAlmostEqual(degrees(ta), expected_ta, places=2)

    def test_true_to_mean(self):
        # Data from Schlesinger & Udick, 1912
        data = [
            # ecc, M (deg), ta (deg)
            (0.0, 0.0, 0.0),
            (0.05, 10.0, 11.06),
            (0.06, 30.0, 33.67),
            (0.04, 120.0, 123.87),
            (0.14, 65.0, 80.50),
            (0.19, 21.0, 30.94),
            (0.35, 65.0, 105.71),
            (0.48, 180.0, 180.0),
            (0.75, 125.0, 167.57)
        ]
        for row in data:
            ecc, expected_M, ta = row

            M = angles.ta_to_M(radians(ta), ecc)

            self.assertAlmostEqual(degrees(M), expected_M, places=1)


class RotateTests(TestCase):
    def test_rotate_simple(self):
        vec = np.array([1, 0, 0])

        assert_allclose(rotate(vec, 0, np.radians(90)), np.array([1, 0, 0]), atol=1e-16)
        assert_allclose(rotate(vec, 1, np.radians(90)), np.array([0, 0, -1]), atol=1e-16)
        assert_allclose(rotate(vec, 2, np.radians(90)), np.array([0, 1, 0]), atol=1e-16)

    def test_rotate_raises_error(self):
        vec_unused = np.ones(3)
        self.assertRaises(ValueError, rotate, vec_unused, 3, 0)
