"""Stage 4 macro traffic dynamics and origin-destination analytics."""

from .config import AnalyticsConfig, CpiThresholds
from .cpi import classify_cpi, corridor_metric, space_mean_speed_kph
from .engine import MacroAnalyticsEngine
from .od_matrix import ODMatrixAccumulator, TripSessionizer
from .spatial import H3SpatialIndex
from .types import (
    AnalyticsSighting,
    CongestionLevel,
    CorridorMetric,
    CorridorTraversal,
    HexBinMetric,
    ODMatrixEntry,
    ODMatrixResult,
    TripSession,
)

__all__ = [
    "AnalyticsConfig", "CpiThresholds", "AnalyticsSighting", "HexBinMetric", "TripSession",
    "ODMatrixEntry", "ODMatrixResult", "CorridorTraversal", "CorridorMetric", "CongestionLevel",
    "H3SpatialIndex", "TripSessionizer", "ODMatrixAccumulator", "space_mean_speed_kph",
    "classify_cpi", "corridor_metric", "MacroAnalyticsEngine",
]
