# For backwards compatibility
import warnings
warnings.warn(
    "Use `from orbit_predictor.predictors import HighAccuracyTLEPredictor` instead",
    FutureWarning,
)

from .predictors.accurate import HighAccuracyTLEPredictor, ONE_SECOND
