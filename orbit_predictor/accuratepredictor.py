# For backwards compatibility
import warnings

from .predictors.base import ONE_SECOND
from .predictors.accurate import HighAccuracyTLEPredictor

warnings.warn(
    "Use `from orbit_predictor.predictors import TLEPredictor` instead, "
    "this module will be removed in the future",
    FutureWarning,
)


__all__ = ["HighAccuracyTLEPredictor", "ONE_SECOND"]
