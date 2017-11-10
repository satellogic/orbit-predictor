# MIT License
#
# Copyright (c) 2017 Satellogic SA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import unittest

from orbit_predictor import coordinate_systems as coords

buenos_aires_llh = (-34.628284, -58.436045, 0.0)
lujan_llh = (-34.561207, -59.114542, 0.0)
buenos_aires_lujan_distance = 64.0
north_pole_llh = (84.0, 0.0, 0.0)
sixty_four_km_away_from_north_pole_llh = (83.53837, -3.230163, 0.0)


class TestLLHToECEF(unittest.TestCase):

    def test_buenos_aires(self):
        ecef = coords.llh_to_ecef(buenos_aires_llh[0], buenos_aires_llh[1], .025)
        self.assertAlmostEqual(ecef[0], 2750.19, delta=.2)
        self.assertAlmostEqual(ecef[1], -4476.679, delta=.2)
        self.assertAlmostEqual(ecef[2], -3604.011, delta=.2)

    def test_buenos_aires_high(self):
        ecef = coords.llh_to_ecef(buenos_aires_llh[0], buenos_aires_llh[1], 400.0)
        self.assertAlmostEqual(ecef[0], 2922.48, delta=.2)
        self.assertAlmostEqual(ecef[1], -4757.126, delta=.2)
        self.assertAlmostEqual(ecef[2], -3831.311, delta=.2)

    def test_lisbon(self):
        ecef = coords.llh_to_ecef(38.716666, 9.16666, .002)
        self.assertAlmostEqual(ecef[0], 4919.4248, delta=.2)
        self.assertAlmostEqual(ecef[1], 793.8355, delta=.2)
        self.assertAlmostEqual(ecef[2], 3967.8253, delta=.2)

    def test_lisbon_high(self):
        ecef = coords.llh_to_ecef(38.716666, 9.16666, 450)
        self.assertAlmostEqual(ecef[0], 5266.051, delta=.2)
        self.assertAlmostEqual(ecef[1], 849.7698, delta=.2)
        self.assertAlmostEqual(ecef[2], 4249.2854, delta=.2)

    def test_north_pole(self):
        ecef = coords.llh_to_ecef(89.99, -70, .025)
        self.assertAlmostEqual(ecef[0], .382, delta=.2)
        self.assertAlmostEqual(ecef[1], -1.0495, delta=.2)
        self.assertAlmostEqual(ecef[2], 6356.7772, delta=.2)

    def test_south_pole(self):
        ecef = coords.llh_to_ecef(-89.99, -70, .025)
        self.assertAlmostEqual(ecef[0], .382, delta=.2)
        self.assertAlmostEqual(ecef[1], -1.0495, delta=.2)
        self.assertAlmostEqual(ecef[2], -6356.7772, delta=.2)


class TestECEFToLLH(unittest.TestCase):

    def test_buenos_aires(self):
        llh = coords.ecef_to_llh((2750.19, -4476.679, -3604.011))
        self.assertAlmostEqual(llh[0], buenos_aires_llh[0], delta=.2)
        self.assertAlmostEqual(llh[1], buenos_aires_llh[1], delta=.2)
        self.assertAlmostEqual(llh[2], 0, delta=.2)

    def test_buenos_aires_high(self):
        llh = coords.ecef_to_llh((2922.48, -4757.126, -3831.311))
        self.assertAlmostEqual(llh[0], buenos_aires_llh[0], delta=.2)
        self.assertAlmostEqual(llh[1], buenos_aires_llh[1], delta=.2)
        self.assertAlmostEqual(llh[2], 400, delta=.2)

    def test_lisbon(self):
        llh = coords.ecef_to_llh((4919.4248, 793.8355, 3967.8253))
        self.assertAlmostEqual(llh[0], 38.716666, delta=.2)
        self.assertAlmostEqual(llh[1], 9.16666, delta=.2)
        self.assertAlmostEqual(llh[2], .002, delta=.2)

    def test_lisbon_high(self):
        llh = coords.ecef_to_llh((5266.051, 849.7698, 4249.2854))
        self.assertAlmostEqual(llh[0], 38.716666, delta=.2)
        self.assertAlmostEqual(llh[1], 9.16666, delta=.2)
        self.assertAlmostEqual(llh[2], 450, delta=.2)

    def test_north_pole(self):
        llh = coords.ecef_to_llh((.382, -1.0495, 6356.7772))
        self.assertAlmostEqual(llh[0], 89.99, delta=.2)
        self.assertAlmostEqual(llh[1], -70, delta=.2)
        self.assertAlmostEqual(llh[2], .025, delta=.2)

    def test_south_pole(self):
        llh = coords.ecef_to_llh((.382, -1.0495, -6356.7772))
        self.assertAlmostEqual(llh[0], -89.99, delta=.2)
        self.assertAlmostEqual(llh[1], -70, delta=.2)
        self.assertAlmostEqual(llh[2], .025, delta=.2)
