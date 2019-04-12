# For backwards compatibility
import warnings

from .predictors.accurate import HighAccuracyTLEPredictor, ONE_SECOND

warnings.warn(
    "Use `from orbit_predictor.predictors import TLEPredictor` instead, "
    "this module will be removed in the future",
    FutureWarning,
)


__all__ = ["HighAccuracyTLEPredictor", "ONE_SECOND"]
