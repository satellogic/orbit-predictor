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

import logging
from collections import defaultdict, namedtuple

import requests
from urllib import parse as urlparse
from urllib.parse import urlencode

from sgp4.api import Satrec

from orbit_predictor.predictors import TLEPredictor
from orbit_predictor.utils import datetime_from_jday

logger = logging.getLogger(__name__)

TLE = namedtuple('TLE',
                 ['sate_id', 'lines', 'date'])


class GPSSource:
    def get_position_ecef(self, sate_id, when_utc):
        raise NotImplementedError("You have to implement it.")


class TLESource:

    def add_tle(self, sate_id, tle, epoch):
        raise NotImplementedError("You have to implement it.")

    def _get_tle(self, sate_id, date):
        raise NotImplementedError("You have to implement it.")

    def get_tle(self, sate_id, date):
        logger.debug("searching a TLE for %s, date: %s", sate_id, date)
        lines = self._get_tle(sate_id, date)
        return TLE(sate_id=sate_id, date=date, lines=lines)

    def get_predictor(self, sate_id):
        """Return a Predictor instance using the current storage."""
        return TLEPredictor(sate_id, self)


class MemoryTLESource(TLESource):
    def __init__(self):
        self.tles = defaultdict(set)

    def add_tle(self, sate_id, tle, epoch):
        self.tles[sate_id].add((epoch, tle))

    def _get_tle(self, sate_id, date):
        candidates = self.tles[sate_id]

        winner = None
        winner_dt = float("inf")

        for epoch, candidate in candidates:
            c_dt = abs((epoch - date).total_seconds())
            if c_dt < winner_dt:
                winner = candidate
                winner_dt = c_dt

        if winner is None:
            raise LookupError("no tles in storage")

        return winner


class EtcTLESource(TLESource):
    def __init__(self, filename="/etc/latest_tle"):
        self.filename = filename

    def add_tle(self, sate_id, tle, epoch):
        with open(self.filename, "w") as fd:
            fd.write(sate_id + "\n")
            for line in tle:
                fd.write(line + "\n")

    def _get_tle(self, sate_id, date):
        with open(self.filename) as fd:
            data = fd.read()
            lines = data.split("\n")
            if not lines[0] == sate_id:
                raise LookupError("Stored satellite id not found")
            return tuple(lines[1:3])


class WSTLESource(TLESource):

    def __init__(self, url):
        self.url = url
        self.cache = MemoryTLESource()

    def add_tle(self, *args):
        raise ValueError("You can't add TLEs. The service has his own update task.")

    def _get_tle(self, sate_id, date):
        # first lookup on cache
        try:
            lines_from_cache = self.cache._get_tle(sate_id, date)
        except LookupError:
            pass
        else:
            return lines_from_cache

        lines = self.get_tle_for_date(sate_id, date)
        # save on cache
        self.cache.add_tle(sate_id, lines, date)
        return lines

    def get_last_update(self, sate_id):
        return self._fetch_tle("api/tle/last/", sate_id)

    def get_tle_for_date(self, sate_id, date):
        return self._fetch_tle("api/tle/closest/", sate_id, date)

    def _fetch_tle(self, path, sate_id, date=None):
        url = urlparse.urljoin(self.url, path)
        url = urlparse.urlparse(url)
        qargs = {'satellite_number': sate_id}
        if date is not None:
            date_str = date.strftime("%Y-%m-%dT%H:%M:%S")
            qargs['date'] = date_str

        query_string = urlencode(qargs)
        url = urlparse.urlunsplit((url.scheme, url.netloc, url.path, query_string, url.fragment))
        headers = {'user-agent': 'orbit-predictor', 'Accept': 'application/json'}
        try:
            response = requests.get(url, headers=headers)
        except requests.exceptions.RequestException as error:
            logger.error("Exception requesting TLE: %s", error)
            raise
        if response.ok and 'lines' in response.json():
            lines = tuple(response.json()['lines'])
            return lines
        else:
            raise ValueError("Error requesting TLE: %s", response.text)


class NoradTLESource(TLESource):
    """
    This source is intended to be used with norad-like multi-line files
    eg. https://www.celestrak.com/NORAD/elements/resource.txt
    """
    def __init__(self, content):
        self.content = content

    @classmethod
    def from_url(cls, url):
        headers = {'user-agent': 'orbit-predictor', 'Accept': 'text/plain'}
        try:
            response = requests.get(url, headers=headers)
        except requests.exceptions.RequestException as error:
            logger.error("Exception requesting TLE: %s", error)
            raise
        lines = response.content.decode("UTF-8").splitlines()
        return cls(lines)

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
        return cls(lines)

    def _get_tle(self, sate_id, date):
        content = iter(self.content)
        for sate, line_1, line_2 in zip(content, content, content):
            if sate_id in sate:
                return tuple([line_1, line_2])

        raise LookupError("Couldn't find it. Wrong file?")


def get_predictor_from_tle_lines(tle_lines):
    db = MemoryTLESource()
    sgp4_sat = Satrec.twoline2rv(tle_lines[0], tle_lines[1])
    db.add_tle(
        sgp4_sat.satnum,
        tuple(tle_lines),
        datetime_from_jday(sgp4_sat.jdsatepoch, sgp4_sat.jdsatepochF),
    )
    predictor = TLEPredictor(sgp4_sat.satnum, db)
    return predictor
