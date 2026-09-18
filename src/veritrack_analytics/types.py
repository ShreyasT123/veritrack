"""Typed, immutable domain values for Stage 4 analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Tuple

__all__ = [
    "AnalyticsSighting", "HexBinMetric", "TripSession", "ODMatrixEntry", "ODMatrixResult",
    "CorridorTraversal", "CorridorMetric", "CongestionLevel",
]


def _aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class AnalyticsSighting:
    """A geolocated, pseudonym-safe Stage 2 sighting used by Stage 4."""

    vehicle_id: str
    camera_id: str
    observed_at: datetime
    latitude: float
    longitude: float
    is_perimeter_exit: bool = False
    corridor_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.vehicle_id or not self.camera_id:
            raise ValueError("vehicle_id and camera_id must be non-empty")
        _aware(self.observed_at, "observed_at")
        if not -90.0 <= self.latitude <= 90.0 or not -180.0 <= self.longitude <= 180.0:
            raise ValueError("sighting coordinates are outside WGS84 bounds")


@dataclass(frozen=True, slots=True)
class HexBinMetric:
    h3_index: str
    resolution: int
    count: int
    window_start: datetime
    window_end: datetime
    centroid_latitude: float
    centroid_longitude: float

    def __post_init__(self) -> None:
        if not self.h3_index or self.count < 0:
            raise ValueError("hex metric needs a non-empty index and non-negative count")
        _aware(self.window_start, "window_start")
        _aware(self.window_end, "window_end")


@dataclass(frozen=True, slots=True)
class TripSession:
    vehicle_id: str
    sightings: Tuple[AnalyticsSighting, ...]
    completed: bool
    termination_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.vehicle_id or not self.sightings:
            raise ValueError("a trip session needs an identity and at least one sighting")
        if any(s.vehicle_id != self.vehicle_id for s in self.sightings):
            raise ValueError("all session sightings must share the vehicle_id")
        if any(a.observed_at > b.observed_at for a, b in zip(self.sightings, self.sightings[1:])):
            raise ValueError("session sightings must be chronological")
        if self.completed != (self.termination_reason is not None):
            raise ValueError("completed sessions require a termination reason, and vice versa")

    @property
    def origin(self) -> AnalyticsSighting:
        return self.sightings[0]

    @property
    def destination(self) -> AnalyticsSighting:
        return self.sightings[-1]

    @property
    def duration_seconds(self) -> float:
        return (self.destination.observed_at - self.origin.observed_at).total_seconds()


@dataclass(frozen=True, slots=True)
class ODMatrixEntry:
    origin_h3: str
    destination_h3: str
    volume: int
    probability: float
    mean_duration_seconds: float
    origin_centroid: Tuple[float, float]
    destination_centroid: Tuple[float, float]

    def __post_init__(self) -> None:
        if not self.origin_h3 or not self.destination_h3 or self.volume < 1:
            raise ValueError("O-D entries need valid H3 cells and positive volume")
        if not 0.0 <= self.probability <= 1.0 or self.mean_duration_seconds < 0.0:
            raise ValueError("invalid O-D probability or duration")


@dataclass(frozen=True, slots=True)
class ODMatrixResult:
    window_start: datetime
    window_end: datetime
    entries: Tuple[ODMatrixEntry, ...]
    completed_trips: int

    def __post_init__(self) -> None:
        _aware(self.window_start, "window_start")
        _aware(self.window_end, "window_end")
        if self.completed_trips < 0:
            raise ValueError("completed_trips cannot be negative")


class CongestionLevel(str, Enum):
    NOMINAL = "NOMINAL"
    ADAPTIVE_SIGNAL = "ADAPTIVE_SIGNAL"
    BOTTLENECK_WARNING = "BOTTLENECK_WARNING"
    CRITICAL_CONGESTION = "CRITICAL_CONGESTION"


@dataclass(frozen=True, slots=True)
class CorridorTraversal:
    corridor_id: str
    vehicle_id: str
    entered_at: datetime
    exited_at: datetime
    length_km: float

    def __post_init__(self) -> None:
        if not self.corridor_id or not self.vehicle_id or self.length_km <= 0.0:
            raise ValueError("traversal requires ids and a positive length")
        _aware(self.entered_at, "entered_at")
        _aware(self.exited_at, "exited_at")
        if self.exited_at <= self.entered_at:
            raise ValueError("traversal exit must occur after entry")

    @property
    def travel_seconds(self) -> float:
        return (self.exited_at - self.entered_at).total_seconds()


@dataclass(frozen=True, slots=True)
class CorridorMetric:
    corridor_id: str
    window_start: datetime
    window_end: datetime
    traversals: int
    space_mean_speed_kph: float
    free_flow_speed_kph: float
    cpi: float
    level: CongestionLevel
    incoming_volume_vph: float
    outflow_capacity_vph: Optional[float]
    wave_detected: bool
    estimated_queue_vehicles: float

    def __post_init__(self) -> None:
        _aware(self.window_start, "window_start")
        _aware(self.window_end, "window_end")
        if self.traversals < 0 or min(self.space_mean_speed_kph, self.free_flow_speed_kph, self.cpi) < 0.0:
            raise ValueError("corridor metrics cannot be negative")
