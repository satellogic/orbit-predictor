# -*- coding: utf-8 -*-
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

from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

from orbit_predictor.predictors.base import CartesianPredictor, logger


class TLEPredictor(CartesianPredictor):

    def __init__(self, sate_id, source):
        super(TLEPredictor, self).__init__(source, sate_id)
        self._iterations = 0

    def _propagate_eci(self, when_utc=None):
        """Return position and velocity in the given date using ECI coordinate system."""
        tle = self.source.get_tle(self.sate_id, when_utc)
        logger.debug("Propagating using ECI. sate_id: %s, when_utc: %s, tle: %s",
                     self.sate_id, when_utc, tle)
        tle_line_1, tle_line_2 = tle.lines
        sgp4_sate = twoline2rv(tle_line_1, tle_line_2, wgs84)
        timelist = list(when_utc.timetuple()[:6])
        timelist[5] = timelist[5] + when_utc.microsecond * 1e-6
        position_eci, velocity_eci = sgp4_sate.propagate(*timelist)
        return position_eci, velocity_eci
