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

from orbit_predictor import coordinate_systems
from orbit_predictor.predictors import Position


class PositionTestCase(unittest.TestCase):
    def test_position_convertion(self):
        position_ecef = (2750.19, -4476.679, -3604.011)

        p = Position(None, position_ecef, None, None)
        ecef_est = coordinate_systems.llh_to_ecef(*p.position_llh)

        self.assertAlmostEqual(position_ecef[0], ecef_est[0], 3)
        self.assertAlmostEqual(position_ecef[1], ecef_est[1], 3)
        self.assertAlmostEqual(position_ecef[2], ecef_est[2], 3)
