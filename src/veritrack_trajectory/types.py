"""Typed domain models for Stage 3 trajectory reconstruction.

These are the objects that travel between the graph, the fusion scorer, the
Viterbi matcher and the orchestrator. They are frozen: a trajectory is a
statement about what happened, and nothing downstream should be able to edit
one after the engine has produced it.

Note what a :class:`CameraSighting` carries and what it does not. It holds the
plate *pseudonym* alongside an optional cleartext plate, because Stage 3 runs
against the Stage 2 ``sightings`` hypertable, whose rows are pseudonymised. The
cleartext field is populated only for a warrant-scoped query against
``hotlist_hits``. Trajectory reconstruction itself works perfectly on
pseudonyms -- same-day linkage is exactly what the rotating salt preserves --
so the default path never needs a registration number at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "AnomalyType",
    "AnomalySeverity",
    "GeoPoint",
    "CameraSighting",
    "RoadSegment",
    "TrajectoryWaypoint",
    "AnomalyRecord",
    "MatchScore",
    "ReconstructedTrajectory",
    "haversine_m",
    "bearing_deg",
    "angular_difference_deg",
]

EARTH_RADIUS_M = 6_371_008.8


# =====================================================================
# Geodesy helpers
# =====================================================================


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres.

    Used as the A* heuristic for network distance. It is admissible because a
    road segment is never shorter than the straight line between its endpoints,
    so A* with this heuristic cannot return a suboptimal path.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi * 0.5) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda * 0.5) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, in [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return math.degrees(math.atan2(y, x)) % 360.0


def angular_difference_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in [0, 180].

    Wrapping matters: a vehicle heading 359 degrees and an edge heading 1
    degree differ by 2 degrees, not 358. Getting this wrong makes the emission
    model reject correct candidates at the north crossing.
    """
    return abs((a - b + 180.0) % 360.0 - 180.0)


class AnomalyType(str, Enum):
    """The anomaly conditions Stage 3 can raise."""

    CLONED_PLATE = "CLONED_PLATE"
    SWAPPED_PLATE = "SWAPPED_PLATE"
    UNREACHABLE_TRANSITION = "UNREACHABLE_TRANSITION"


class AnomalySeverity(int, Enum):
    """Maps onto the Stage 5 dispatch ladder."""

    ADVISORY = 1
    ELEVATED = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True, slots=True)
class GeoPoint:
    """A WGS84 position."""

    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"latitude out of range: {self.latitude}")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"longitude out of range: {self.longitude}")

    def distance_to(self, other: "GeoPoint") -> float:
        return haversine_m(self.latitude, self.longitude, other.latitude, other.longitude)

    def bearing_to(self, other: "GeoPoint") -> float:
        return bearing_deg(self.latitude, self.longitude, other.latitude, other.longitude)

    def as_lon_lat(self) -> Tuple[float, float]:
        """GeoJSON ordering: longitude first. The most common source of silent bugs."""
        return (self.longitude, self.latitude)


@dataclass(frozen=True, slots=True)
class CameraSighting:
    """One vehicle pass, as read back from the Stage 2 store."""

    sighting_id: str
    camera_id: str
    timestamp_utc: datetime
    location: GeoPoint

    #: The pseudonymised plate. This is the join key in normal operation.
    plate_pseudonym: str = ""
    #: Cleartext, populated only on the warrant-scoped path.
    plate_text: Optional[str] = None

    plate_sequence_confidence: float = 0.0
    reid_embedding: Tuple[float, ...] = ()
    vehicle_class: str = "unknown"
    travel_heading_azimuth: Optional[float] = None
    vehicle_speed_kmh: Optional[float] = None
    corridor_id: Optional[str] = None
    text_entropy: Optional[float] = None
    repair_cost_nats: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        if not 0.0 <= self.plate_sequence_confidence <= 1.0:
            raise ValueError("plate_sequence_confidence must lie in [0, 1]")
        if self.reid_embedding and len(self.reid_embedding) != 128:
            raise ValueError(
                f"reid_embedding must hold 128 components, got {len(self.reid_embedding)}"
            )

    @property
    def epoch_s(self) -> float:
        return self.timestamp_utc.timestamp()

    @property
    def identity_key(self) -> str:
        """Whatever identifies the vehicle in the current query scope."""
        return self.plate_text or self.plate_pseudonym or self.sighting_id

    def seconds_to(self, other: "CameraSighting") -> float:
        return other.epoch_s - self.epoch_s


@dataclass(frozen=True, slots=True)
class RoadSegment:
    """A directed road segment: one edge of the network multigraph."""

    segment_id: str
    from_node: str
    to_node: str
    from_point: GeoPoint
    to_point: GeoPoint
    length_m: float
    speed_limit_kmh: float
    heading_deg: float
    street_name: str = ""
    #: Disambiguates parallel edges between the same node pair (a road and the
    #: flyover above it), which is why the graph is a MultiDiGraph.
    edge_key: int = 0
    free_flow_time_sec: Optional[float] = None
    corridor_id: Optional[str] = None
    is_oneway: bool = True

    def __post_init__(self) -> None:
        if self.length_m <= 0.0:
            raise ValueError(f"segment {self.segment_id} has non-positive length")
        if self.speed_limit_kmh <= 0.0:
            raise ValueError(f"segment {self.segment_id} has non-positive speed limit")
        if self.free_flow_time_sec is None:
            object.__setattr__(
                self, "free_flow_time_sec", self.length_m / (self.speed_limit_kmh / 3.6)
            )

    @property
    def free_flow_speed_ms(self) -> float:
        return self.speed_limit_kmh / 3.6

    @property
    def endpoints(self) -> Tuple[str, str, int]:
        return (self.from_node, self.to_node, self.edge_key)

    def midpoint(self) -> GeoPoint:
        return GeoPoint(
            latitude=(self.from_point.latitude + self.to_point.latitude) * 0.5,
            longitude=(self.from_point.longitude + self.to_point.longitude) * 0.5,
        )

    def as_coordinates(self) -> List[Tuple[float, float]]:
        return [self.from_point.as_lon_lat(), self.to_point.as_lon_lat()]


@dataclass(frozen=True, slots=True)
class TrajectoryWaypoint:
    """One matched point on the reconstructed path.

    ``is_observed`` distinguishes a real camera sighting from a segment the
    Viterbi matcher interpolated to fill a blind spot. Downstream consumers
    must be able to tell evidence from inference, so this flag is never
    optional.
    """

    sequence_index: int
    location: GeoPoint
    segment_id: str
    street_name: str
    is_observed: bool
    timestamp_utc: Optional[datetime] = None
    sighting_id: Optional[str] = None
    camera_id: Optional[str] = None
    emission_logprob: Optional[float] = None
    cumulative_distance_m: float = 0.0
    heading_deg: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "latitude": self.location.latitude,
            "longitude": self.location.longitude,
            "segment_id": self.segment_id,
            "street_name": self.street_name,
            "is_observed": self.is_observed,
            "timestamp_utc": self.timestamp_utc.isoformat() if self.timestamp_utc else None,
            "sighting_id": self.sighting_id,
            "camera_id": self.camera_id,
            "emission_logprob": self.emission_logprob,
            "cumulative_distance_m": round(self.cumulative_distance_m, 2),
            "heading_deg": self.heading_deg,
        }


@dataclass(frozen=True, slots=True)
class MatchScore:
    """A fused pairwise match, with its components exposed.

    The components are kept rather than collapsed into the total because an
    analyst reviewing a flagged trajectory needs to know *why* two sightings
    were linked -- a 0.61 driven by text is a different claim from a 0.61
    driven by appearance.
    """

    total: float
    s_text: float
    s_vis: float
    s_kin: float
    w_text: float
    w_vis: float
    w_kin: float
    gate: float
    network_distance_m: float
    delta_t_s: float
    network_speed_kmh: float
    kinematically_feasible: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": round(self.total, 6),
            "components": {
                "s_text": round(self.s_text, 6),
                "s_vis": round(self.s_vis, 6),
                "s_kin": round(self.s_kin, 6),
            },
            "weights": {
                "w_text": round(self.w_text, 6),
                "w_vis": round(self.w_vis, 6),
                "w_kin": round(self.w_kin, 6),
            },
            "gate": round(self.gate, 6),
            "network_distance_m": round(self.network_distance_m, 2),
            "delta_t_s": round(self.delta_t_s, 3),
            "network_speed_kmh": round(self.network_speed_kmh, 2),
            "kinematically_feasible": self.kinematically_feasible,
        }


@dataclass(frozen=True, slots=True)
class AnomalyRecord:
    """A detected anomaly, carrying the evidence that produced it."""

    anomaly_type: AnomalyType
    severity: AnomalySeverity
    identity_key: str
    from_sighting_id: str
    to_sighting_id: str
    from_camera_id: str
    to_camera_id: str
    detected_at: datetime
    description: str
    network_distance_m: float = 0.0
    delta_t_s: float = 0.0
    implied_speed_kmh: float = 0.0
    reid_similarity: Optional[float] = None
    text_similarity: Optional[float] = None
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type.value,
            "severity": int(self.severity),
            "identity_key": self.identity_key,
            "from_sighting_id": self.from_sighting_id,
            "to_sighting_id": self.to_sighting_id,
            "from_camera_id": self.from_camera_id,
            "to_camera_id": self.to_camera_id,
            "detected_at": self.detected_at.isoformat(),
            "description": self.description,
            "network_distance_m": round(self.network_distance_m, 2),
            "delta_t_s": round(self.delta_t_s, 3),
            "implied_speed_kmh": round(self.implied_speed_kmh, 2),
            "reid_similarity": (
                round(self.reid_similarity, 6) if self.reid_similarity is not None else None
            ),
            "text_similarity": (
                round(self.text_similarity, 6) if self.text_similarity is not None else None
            ),
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class ReconstructedTrajectory:
    """The Stage 3 output artefact."""

    trajectory_id: str
    identity_key: str
    waypoints: Tuple[TrajectoryWaypoint, ...]
    segments: Tuple[RoadSegment, ...]
    anomalies: Tuple[AnomalyRecord, ...]
    match_scores: Tuple[MatchScore, ...]

    total_distance_km: float
    transit_time_s: float
    average_speed_kmh: float
    observed_sighting_count: int
    interpolated_segment_count: int

    started_at: datetime
    ended_at: datetime
    viterbi_logprob: float = 0.0
    is_contiguous: bool = True
    notes: Tuple[str, ...] = ()

    @property
    def transit_time_minutes(self) -> float:
        return self.transit_time_s / 60.0

    @property
    def has_anomalies(self) -> bool:
        return bool(self.anomalies)

    @property
    def max_severity(self) -> Optional[AnomalySeverity]:
        if not self.anomalies:
            return None
        return max(anomaly.severity for anomaly in self.anomalies)

    @property
    def street_names(self) -> List[str]:
        """Ordered street names with consecutive duplicates collapsed."""
        names: List[str] = []
        for segment in self.segments:
            name = segment.street_name or segment.segment_id
            if not names or names[-1] != name:
                names.append(name)
        return names

    def to_geojson(self) -> Dict[str, Any]:
        """A GeoJSON FeatureCollection: the path plus one point per waypoint.

        The LineString is built from the matched *segments*, not from the raw
        camera coordinates, which is the whole point of map matching: the path
        follows the roadway rather than cutting across blocks between poles.
        """
        coordinates: List[Tuple[float, float]] = []
        for segment in self.segments:
            if not coordinates:
                coordinates.append(segment.from_point.as_lon_lat())
            elif coordinates[-1] != segment.from_point.as_lon_lat():
                coordinates.append(segment.from_point.as_lon_lat())
            coordinates.append(segment.to_point.as_lon_lat())

        if len(coordinates) < 2:
            # A GeoJSON LineString needs two positions. With a single matched
            # segment or none, fall back to the observed waypoints so the
            # feature is still valid rather than malformed.
            coordinates = [
                waypoint.location.as_lon_lat() for waypoint in self.waypoints
            ][:2]
            while len(coordinates) < 2 and self.waypoints:
                coordinates.append(coordinates[0])

        features: List[Dict[str, Any]] = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {
                    "trajectory_id": self.trajectory_id,
                    "identity_key": self.identity_key,
                    "total_distance_km": round(self.total_distance_km, 4),
                    "transit_time_s": round(self.transit_time_s, 2),
                    "average_speed_kmh": round(self.average_speed_kmh, 2),
                    "street_names": self.street_names,
                    "observed_sighting_count": self.observed_sighting_count,
                    "interpolated_segment_count": self.interpolated_segment_count,
                    "is_contiguous": self.is_contiguous,
                    "anomaly_types": [anomaly.anomaly_type.value for anomaly in self.anomalies],
                    "max_severity": int(self.max_severity) if self.max_severity else None,
                    "started_at": self.started_at.isoformat(),
                    "ended_at": self.ended_at.isoformat(),
                    "viterbi_logprob": round(self.viterbi_logprob, 6),
                },
            }
        ]
        for waypoint in self.waypoints:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": list(waypoint.location.as_lon_lat()),
                    },
                    "properties": waypoint.to_dict(),
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "identity_key": self.identity_key,
            "total_distance_km": round(self.total_distance_km, 4),
            "transit_time_s": round(self.transit_time_s, 2),
            "transit_time_minutes": round(self.transit_time_minutes, 3),
            "average_speed_kmh": round(self.average_speed_kmh, 2),
            "observed_sighting_count": self.observed_sighting_count,
            "interpolated_segment_count": self.interpolated_segment_count,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "is_contiguous": self.is_contiguous,
            "viterbi_logprob": round(self.viterbi_logprob, 6),
            "street_names": self.street_names,
            "waypoints": [waypoint.to_dict() for waypoint in self.waypoints],
            "anomalies": [anomaly.to_dict() for anomaly in self.anomalies],
            "match_scores": [score.to_dict() for score in self.match_scores],
            "notes": list(self.notes),
            "geojson": self.to_geojson(),
        }
