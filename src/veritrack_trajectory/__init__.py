"""VeriTrack Stage 3 -- spatio-temporal trajectory reconstruction and anomaly engine.

Smart India Hackathon PS 26127 (Bharat Electronics Limited).

Turns discrete camera sightings into continuous, map-matched journeys, and
flags the kinematic and visual impossibilities that indicate a cloned or
swapped registration plate.

Typical use::

    from veritrack_trajectory import TrajectoryEngine, build_synthetic_grid

    graph = build_synthetic_grid(6, 6, spacing_m=500.0)
    engine = TrajectoryEngine(graph)
    trajectory = engine.reconstruct(sightings)
    geojson = trajectory.to_geojson()

Stage 3 consumes pseudonymised rows from the Stage 2 ``sightings`` hypertable.
It never needs a cleartext registration number: same-day linkage is exactly
what the rotating HMAC salt preserves, so reconstruction works on pseudonyms
alone.
"""

from __future__ import annotations

from .anomaly import AnomalyDetector
from .config import (
    DEFAULT_CONFIG,
    AnomalyConfig,
    FusionConfig,
    GraphConfig,
    TrajectoryConfig,
    ViterbiConfig,
    load_config,
)
from .engine import EngineStats, TrajectoryEngine
from .fusion import (
    ConfidenceFusion,
    cosine_similarity,
    damerau_levenshtein,
    kinematic_similarity,
    text_similarity,
    visual_similarity,
)
from .graph import EdgeCandidate, RoadNetworkGraph, build_synthetic_grid
from .types import (
    AnomalyRecord,
    AnomalySeverity,
    AnomalyType,
    CameraSighting,
    GeoPoint,
    MatchScore,
    ReconstructedTrajectory,
    RoadSegment,
    TrajectoryWaypoint,
)
from .viterbi import MapMatchResult, ViterbiMapMatcher, emission_logprob, transition_logprob

__version__ = "3.0.0"

__all__ = [
    "__version__",
    # config
    "TrajectoryConfig", "GraphConfig", "FusionConfig", "ViterbiConfig", "AnomalyConfig",
    "load_config", "DEFAULT_CONFIG",
    # types
    "GeoPoint", "CameraSighting", "RoadSegment", "TrajectoryWaypoint",
    "AnomalyRecord", "AnomalyType", "AnomalySeverity", "MatchScore",
    "ReconstructedTrajectory",
    # graph
    "RoadNetworkGraph", "EdgeCandidate", "build_synthetic_grid",
    # fusion
    "ConfidenceFusion", "damerau_levenshtein", "text_similarity",
    "cosine_similarity", "visual_similarity", "kinematic_similarity",
    # viterbi
    "ViterbiMapMatcher", "MapMatchResult", "emission_logprob", "transition_logprob",
    # anomaly + engine
    "AnomalyDetector", "TrajectoryEngine", "EngineStats",
]
