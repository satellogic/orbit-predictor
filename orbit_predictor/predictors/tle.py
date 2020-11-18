import datetime as dt

from .accurate import HighAccuracyTLEPredictor
from ..utils import reify

# Backwards compatibility
TLEPredictor = HighAccuracyTLEPredictor


class PersistedTLEPredictor(TLEPredictor):
    def __init__(self, sate_id, source):
        self._sate_id = sate_id
        self._source = source
        self._tle = self._source.get_tle(self.sate_id, dt.datetime.utcnow())

    @reify
    def tle(self):
        return self._tle
