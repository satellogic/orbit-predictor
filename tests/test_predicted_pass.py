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
from datetime import datetime, timedelta

from orbit_predictor.locations import ARG
from orbit_predictor.predictors import PredictedPass


class PredictedPassTests(unittest.TestCase):

    def test_midpoint(self):
        aos = datetime.utcnow()
        max_elevation_date = aos + timedelta(minutes=5)
        los = aos + timedelta(minutes=10)
        pass_ = PredictedPass(sate_id=1,
                              location=ARG,
                              aos=aos,
                              los=los,
                              duration_s=600,
                              max_elevation_date=max_elevation_date,
                              max_elevation_position=None,
                              max_elevation_deg=10)

        self.assertEqual(pass_.midpoint, aos + timedelta(minutes=5))
