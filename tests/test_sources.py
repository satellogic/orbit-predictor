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
from unittest.mock import Mock, patch

import datetime as dt
import os
import shutil
import tempfile
from requests.exceptions import RequestException
from urllib import parse as urlparse

from orbit_predictor import sources
from orbit_predictor.predictors import TLEPredictor


SATE_ID = "AAUSAT-II"
SAMPLE_TLE = ("1 32788U 08021F   15227.82608814  .00001480  00000-0  15110-3 0  9997",
              "2 32788  97.6474 275.2739 0011863 204.9398 155.1249 14.92031413395491")

SAMPLE_TLE2 = ("1 32791U 08021J   15228.17298173  .00001340  00000-0  14806-3 0  9999",
               "2 32791  97.6462 271.6584 0012961 215.4867 144.5490 14.88966377395242")


class TestMemoryTLESource(unittest.TestCase):
    def setUp(self):
        self.db = sources.MemoryTLESource()

    def test_add_tle(self):
        self.db.add_tle(SATE_ID, SAMPLE_TLE, dt.datetime.utcnow())
        tle = self.db._get_tle(SATE_ID, dt.datetime.utcnow())
        self.assertEqual(tle, SAMPLE_TLE)

    def test_add_tle_twice(self):
        self.db.add_tle(SATE_ID, SAMPLE_TLE, dt.datetime.utcnow())
        self.db.add_tle(SATE_ID, SAMPLE_TLE2, dt.datetime.utcnow())
        tle = self.db._get_tle(SATE_ID, dt.datetime.utcnow())
        self.assertEqual(tle, SAMPLE_TLE2)

    def test_add_tle_two_id(self):
        self.db.add_tle(SATE_ID, SAMPLE_TLE, dt.datetime.utcnow())
        self.db.add_tle("fake_id", SAMPLE_TLE2, dt.datetime.utcnow())
        tle = self.db._get_tle(SATE_ID, dt.datetime.utcnow())
        self.assertEqual(tle, SAMPLE_TLE)

    def test_empty(self):
        with self.assertRaises(LookupError):
            self.db._get_tle(SATE_ID, dt.datetime.utcnow())

    # this methods are from TLESource()
    def test_get(self):
        date = dt.datetime.utcnow()
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

        db.add_tle(SATE_ID, SAMPLE_TLE2, dt.datetime.utcnow())
        tle = db._get_tle(SATE_ID, dt.datetime.utcnow())
        self.assertEqual(tle, SAMPLE_TLE2)

    def test_read_tle(self):
        db = sources.EtcTLESource(self.filename)

        tle = db._get_tle(SATE_ID, dt.datetime.utcnow())
        self.assertEqual(tle, SAMPLE_TLE)

    def test_wrong_sate(self):
        db = sources.EtcTLESource(self.filename)

        with self.assertRaises(LookupError):
            db._get_tle("fake_id", dt.datetime.utcnow())

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
        tle = source._get_tle('40014U', dt.datetime(2015, 1, 1))

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


class TestNoradTLESource(unittest.TestCase):
    def setUp(self):
        self.mock_txt = b'GOKTURK 1A              \r\n1 41875U 16073A   17332.47105147  .00000094  00000-0  27523-4 0  9993\r\n2 41875  98.1351 225.1796 0001337  70.3616 289.7724 14.62796168 52313\r\nRESOURCESAT-2A          \r\n1 41877U 16074A   17332.46918040  .00000036  00000-0  36250-4 0  9992\r\n2 41877  98.6936  45.7184 0001162 115.7445 244.3855 14.21645136 50623\r\nCARTOSAT-2D             \r\n1 41948U 17008A   17331.89174699  .00000681  00000-0  35585-4 0  9995\r\n2 41948  97.4741  31.1262 0007151  20.5044 339.6476 15.19208270 43374\r\nSENTINEL-2B             \r\n1 42063U 17013A   17332.41017229  .00000031  00000-0  28432-4 0  9996\r\n2 42063  98.5661  44.9680 0001235  82.8511 277.2813 14.30817061 38090\r\nZHUHAI-1 02 (CAS 4B)    \r\n1 42759U 17034B   17332.40391186  .00000468  00000-0  42601-4 0  9995\r\n2 42759  43.0173 105.3798 0009262 292.4612 164.2517 15.09117618 25119\r\nNUSAT-3 (MILANESAT)     \r\n1 42760U 17034C   17332.49069753  .00000991  00000-0  75325-4 0  9998\r\n2 42760  43.0161 105.1955 0008405 298.8282 137.7431 15.08929193 25132\r\nZHUHAI-1 01 (CAS 4A)    \r\n1 42761U 17034D   17332.46417949  .00000494  00000-0  44138-4 0  9996\r\n2 42761  43.0169 104.9484 0009568 293.2627 175.8187 15.09193142 25115\r\nCARTOSAT-2E             \r\n1 42767U 17036C   17331.77194191  .00000774  00000-0  39908-4 0  9998\r\n2 42767  97.4296  29.1430 0010292  57.9583 302.2650 15.19266595 23928\r\nSENTINEL-5P             \r\n1 42969U 17064A   17331.88357473 -.00000147  00000-0 -49140-4 0  9994\r\n2 42969  98.7157 269.4842 0000887  63.2664 296.8591 14.19555818  6445\r\nSKYSAT-C11              \r\n1 42987U 17068A   17332.38352858  .00000890  00000-0  48450-4 0  9992\r\n2 42987  97.3501  85.7549 0022336 116.8959 243.4561 15.16735466  4160\r\nSKYSAT-C10              \r\n1 42988U 17068B   17332.38095293  .00000905  00000-0  48925-4 0  9994\r\n2 42988  97.3500  85.7580 0022294 119.1231 241.2238 15.16949729  4163\r\nSKYSAT-C9               \r\n1 42989U 17068C   17332.37848268  .00000930  00000-0  49865-4 0  9994\r\n2 42989  97.3500  85.7610 0022012 121.4467 238.8923 15.17178527  4167\r\nSKYSAT-C8               \r\n1 42990U 17068D   17332.37403557  .00000950  00000-0  50313-4 0  9996\r\n2 42990  97.3500  85.7673 0021907 124.3608 235.9703 15.17567328  4164\r\nSKYSAT-C7               \r\n1 42991U 17068E   17332.37135448  .00000970  00000-0  51078-4 0  9999\r\n2 42991  97.3501  85.7706 0022397 126.9371 233.3919 15.17734630  4165\r\nSKYSAT-C6               \r\n1 42992U 17068F   17332.36747368  .00000986  00000-0  51507-4 0  9993\r\n2 42992  97.3498  85.7743 0022634 128.8620 231.4639 15.17985098  4168\r\n' # NOQA
        self.expected_lines = (
            '1 42760U 17034C   17332.49069753  .00000991  00000-0  75325-4 0  9998',
            '2 42760  43.0161 105.1955 0008405 298.8282 137.7431 15.08929193 25132'
        )

        self.headers = {'user-agent': 'orbit-predictor', 'Accept': 'text/plain'}

    @patch("requests.get")
    def test_load_tle_from_url(self, mocked_requests):

        mocked_response = Mock()
        mocked_response.ok = True
        mocked_response.content = self.mock_txt
        mocked_requests.return_value = mocked_response

        source = sources.NoradTLESource.from_url(url="http://test.none/")
        tle = source._get_tle('NUSAT-3', None)

        call_args = mocked_requests.call_args

        self.assertEqual(call_args[1], {'headers': self.headers})
        self.assertEqual(tle, self.expected_lines)

    @patch("requests.get")
    def test_load_tle_from_url_exception(self, mocked_requests):

        mocked_requests.side_effect = RequestException()
        self.assertRaises(RequestException,
                          sources.NoradTLESource.from_url, None)

    def test_load_tle_from_file(self):
        test_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "assets", "resource.txt")

        source = sources.NoradTLESource.from_file(filename=test_file)
        tle = source._get_tle('NUSAT-3', None)

        self.assertEqual(tle, self.expected_lines)

    def test_get_tle_not_found(self):
        test_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "assets", "resource.txt")

        source = sources.NoradTLESource.from_file(filename=test_file)

        self.assertRaises(LookupError,
                          source._get_tle, 'INEXISTENT-SAT', None)

    def test_get_predictor_from_lines(self):
        BUGSAT1_TLE_LINES = (
            "1 40014U 14033E   14294.41438078  .00003468  00000-0  34565-3 0  3930",
            "2 40014  97.9781 190.6418 0032692 299.0467  60.7524 14.91878099 18425")

        predictor = sources.get_predictor_from_tle_lines(BUGSAT1_TLE_LINES)
        position = predictor.get_position(dt.datetime(2019, 1, 1))
        self.assertEqual(
            position.position_ecef, (-5280.795613274576, -3977.487633239489, -2061.43227648734))
