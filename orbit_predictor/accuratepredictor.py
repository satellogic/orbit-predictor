# For backwards compatibility
import warnings

from .predictors.accurate import HighAccuracyTLEPredictor, ONE_SECOND

warnings.warn(
    "Use `from orbit_predictor.predictors import HighAccuracyTLEPredictor` instead",
    FutureWarning,
)


__all__ = ["HighAccuracyTLEPredictor", "ONE_SECOND"]
