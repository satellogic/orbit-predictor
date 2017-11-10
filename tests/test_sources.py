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

import datetime
import os
import shutil
import tempfile
import unittest

from orbit_predictor import sources
from orbit_predictor.accuratepredictor import HighAccuracyTLEPredictor
from orbit_predictor.predictors import TLEPredictor

try:
    from unittest.mock import Mock, patch
except ImportError:
    from mock import Mock, patch  # Python2

try:
    from urllib import parse as urlparse
except ImportError:
    import urlparse  # Python2


SATE_ID = "AAUSAT-II"
SAMPLE_TLE = ("1 32788U 08021F   15227.82608814  .00001480  00000-0  15110-3 0  9997",
              "2 32788  97.6474 275.2739 0011863 204.9398 155.1249 14.92031413395491")

SAMPLE_TLE2 = ("1 32791U 08021J   15228.17298173  .00001340  00000-0  14806-3 0  9999",
               "2 32791  97.6462 271.6584 0012961 215.4867 144.5490 14.88966377395242")


class TestMemoryTLESource(unittest.TestCase):
    def setUp(self):
        self.db = sources.MemoryTLESource()

    def test_add_tle(self):
        self.db.add_tle(SATE_ID, SAMPLE_TLE, datetime.datetime.now())
        tle = self.db._get_tle(SATE_ID, datetime.datetime.now())
        self.assertEqual(tle, SAMPLE_TLE)

    def test_add_tle_twice(self):
        self.db.add_tle(SATE_ID, SAMPLE_TLE, datetime.datetime.now())
        self.db.add_tle(SATE_ID, SAMPLE_TLE2, datetime.datetime.now())
        tle = self.db._get_tle(SATE_ID, datetime.datetime.now())
        self.assertEqual(tle, SAMPLE_TLE2)

    def test_add_tle_two_id(self):
        self.db.add_tle(SATE_ID, SAMPLE_TLE, datetime.datetime.now())
        self.db.add_tle("fake_id", SAMPLE_TLE2, datetime.datetime.now())
        tle = self.db._get_tle(SATE_ID, datetime.datetime.now())
        self.assertEqual(tle, SAMPLE_TLE)

    def test_empty(self):
        with self.assertRaises(LookupError):
            self.db._get_tle(SATE_ID, datetime.datetime.now())

    # this methods are from TLESource()
    def test_get(self):
        date = datetime.datetime.now()
        self.db.add_tle(SATE_ID, SAMPLE_TLE, date)
        tle = self.db.get_tle(SATE_ID, date)
        self.assertEqual(tle.lines, SAMPLE_TLE)
        self.assertEqual(tle.sate_id, SATE_ID)
        self.assertEqual(tle.date, date)

    def test_get_predictor(self):
        predictor = self.db.get_predictor(SATE_ID)

        self.assertIsInstance(predictor, TLEPredictor)
        self.assertEqual(predictor.sate_id, SATE_ID)
        self.assertEqual(predictor.source, self.db)

    def test_get_predictor_precise(self):
        predictor = self.db.get_predictor(SATE_ID, precise=True)
        self.assertIsInstance(predictor, HighAccuracyTLEPredictor)
        self.assertEqual(predictor.sate_id, SATE_ID)
        self.assertEqual(predictor.source, self.db)


class TestEtcTLESource(unittest.TestCase):
    def setUp(self):
        self.dirname = tempfile.mkdtemp()
        self.filename = os.path.join(self.dirname, "tle_file")

        with open(self.filename, "w") as fd:
            fd.write(SATE_ID + "\n")
            for l in SAMPLE_TLE:
                fd.write(l + "\n")

    def test_add_tle(self):
        db = sources.EtcTLESource(self.filename)

        db.add_tle(SATE_ID, SAMPLE_TLE2, datetime.datetime.now())
        tle = db._get_tle(SATE_ID, datetime.datetime.now())
        self.assertEqual(tle, SAMPLE_TLE2)

    def test_read_tle(self):
        db = sources.EtcTLESource(self.filename)

        tle = db._get_tle(SATE_ID, datetime.datetime.now())
        self.assertEqual(tle, SAMPLE_TLE)

    def test_wrong_sate(self):
        db = sources.EtcTLESource(self.filename)

        with self.assertRaises(LookupError):
            db._get_tle("fake_id", datetime.datetime.now())

    def tearDown(self):
        shutil.rmtree(self.dirname)


class TestWSTLESource(unittest.TestCase):
    def setUp(self):
        self.mock_json = {
            "date": "2015-01-15T08:56:33Z",
            "lines": [
                "1 40014U 14033E   15014.37260739  .00003376  00000-0  33051-3 0  6461",
                "2 40014  97.9772 276.7067 0034296  17.9121 342.3159 14.92570778 31092"
            ]
        }

        self.expected_lines = (
            "1 40014U 14033E   15014.37260739  .00003376  00000-0  33051-3 0  6461",
            "2 40014  97.9772 276.7067 0034296  17.9121 342.3159 14.92570778 31092")

        self.headers = {'user-agent': 'orbit-predictor', 'Accept': 'application/json'}

    @patch("requests.get")
    def test_get_tle(self, mocked_requests):
        expected_url = urlparse.urlparse(
            "http://test.none/api/tle/closest/?date=2015-01-01&satellite_number=40014U")
        expected_qs = urlparse.parse_qs(expected_url.query)

        mocked_response = Mock()
        mocked_response.ok = True
        mocked_response.json.return_value = self.mock_json
        mocked_requests.return_value = mocked_response

        source = sources.WSTLESource(url="http://test.none/")
        tle = source._get_tle('40014U', datetime.datetime(2015, 1, 1))

        call_args = mocked_requests.call_args
        url = urlparse.urlparse(call_args[0][0])
        url_qs = urlparse.parse_qs(url.query)

        self.assertEqual(url.path, expected_url.path)
        self.assertEqual(url_qs, expected_qs)
        self.assertEqual(call_args[1], {'headers': self.headers})
        self.assertEqual(tle, self.expected_lines)

    @patch("requests.get")
    def test_get_last_update(self, mocked_requests):
        url = "http://test.none/api/tle/last/?satellite_number=40014U"
        mocked_response = Mock()
        mocked_response.ok = True
        mocked_response.json.return_value = self.mock_json
        mocked_requests.return_value = mocked_response

        source = sources.WSTLESource(url="http://test.none/")
        tle = source.get_last_update('40014U')

        mocked_requests.assert_called_with(url, headers=self.headers)
        self.assertEqual(tle, self.expected_lines)
