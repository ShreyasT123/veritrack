"""In-memory real-time orchestrator for Stage 4 macro analytics."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import AnalyticsConfig
from .cpi import corridor_metric
from .od_matrix import ODMatrixAccumulator, TripSessionizer
from .spatial import H3SpatialIndex
from .types import AnalyticsSighting, CorridorMetric, CorridorTraversal, HexBinMetric, ODMatrixResult

__all__ = ["MacroAnalyticsEngine"]


class MacroAnalyticsEngine:
    """Accept Stage 2 sightings and expose live density, O-D and CPI queries.

    This class is deliberately deterministic and in-memory. A service process
    may replay TimescaleDB rows at startup, while a production stream consumer
    calls ``ingest_sighting`` as rows arrive. Retention bounds memory without
    changing results inside configured query windows.
    """

    def __init__(self, config: AnalyticsConfig = AnalyticsConfig()) -> None:
        self._config = config
        self._spatial = H3SpatialIndex()
        self._sessionizer = TripSessionizer(timedelta(minutes=config.dwell_timeout_minutes))
        self._od = ODMatrixAccumulator(self._spatial, config.h3_res_od)
        self._sightings: List[AnalyticsSighting] = []
        self._traversals: DefaultDict[str, List[CorridorTraversal]] = defaultdict(list)
        self._last_corridor: Dict[str, AnalyticsSighting] = {}

    def ingest_sighting(self, sighting: AnalyticsSighting) -> None:
        """Ingest one geolocated sighting and update all applicable accumulators."""
        for session in self._sessionizer.ingest(sighting):
            self._od.add(session)
        self._sightings.append(sighting)
        previous = self._last_corridor.get(sighting.vehicle_id)
        if (
            previous is not None
            and previous.corridor_id is not None
            and previous.corridor_id == sighting.corridor_id
            and sighting.corridor_id in self._config.corridor_lengths_km
            and sighting.observed_at > previous.observed_at
        ):
            self.ingest_traversal(CorridorTraversal(
                corridor_id=sighting.corridor_id,
                vehicle_id=sighting.vehicle_id,
                entered_at=previous.observed_at,
                exited_at=sighting.observed_at,
                length_km=self._config.corridor_lengths_km[sighting.corridor_id],
            ))
        self._last_corridor[sighting.vehicle_id] = sighting
        self._prune(sighting.observed_at)

    def ingest_traversal(self, traversal: CorridorTraversal) -> None:
        """Add an externally measured corridor traversal, e.g. detector passage pair."""
        self._traversals[traversal.corridor_id].append(traversal)
        self._prune(traversal.exited_at)

    def ingest_stage2_row(self, row: Mapping[str, Any], camera_coordinates: Mapping[str, Tuple[float, float]]) -> None:
        """Adapt a Stage 2 row using a camera-id-to-WGS84 registry.

        Accepted identity fields are ``plate_pseudonym``, ``plate_hash``, and
        ``vehicle_id``. Timestamp accepts aware ``datetime``, ISO-8601 text,
        or milliseconds since Unix epoch.
        """
        camera_id = str(row["camera_id"])
        if camera_id not in camera_coordinates:
            raise KeyError(f"camera {camera_id!r} has no registered coordinates")
        identity = next((row.get(key) for key in ("plate_pseudonym", "plate_hash", "vehicle_id") if row.get(key)), None)
        if identity is None:
            raise KeyError("Stage 2 row needs plate_pseudonym, plate_hash, or vehicle_id")
        raw_time = row.get("observed_at", row.get("ts_first"))
        observed_at = self._parse_time(raw_time)
        latitude, longitude = camera_coordinates[camera_id]
        self.ingest_sighting(AnalyticsSighting(
            vehicle_id=str(identity), camera_id=camera_id, observed_at=observed_at,
            latitude=latitude, longitude=longitude,
            is_perimeter_exit=bool(row.get("is_perimeter_exit", False)),
            corridor_id=str(row["corridor_id"]) if row.get("corridor_id") is not None else None,
        ))

    def get_density_heatmap(self, res: int = 8, time_window_min: int = 15, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        if time_window_min < 1:
            raise ValueError("time_window_min must be positive")
        if now is None:
            now = self._latest_time()
        self._validate_aware(now)
        start = now - timedelta(minutes=time_window_min)
        counts: DefaultDict[str, int] = defaultdict(int)
        for sighting in self._sightings:
            if start <= sighting.observed_at <= now:
                counts[self._spatial.cell(sighting.latitude, sighting.longitude, res)] += 1
        features = []
        for cell, count in sorted(counts.items()):
            lat, lon = self._spatial.centroid(cell)
            metric = HexBinMetric(cell, res, count, start, now, lat, lon)
            features.append(self._spatial.polygon_feature(cell, {
                "count": metric.count, "window_start": metric.window_start.isoformat(),
                "window_end": metric.window_end.isoformat(), "centroid": [lon, lat],
            }))
        return self._spatial.feature_collection(features)

    def get_corridor_cpi_report(self, *, now: Optional[datetime] = None) -> List[CorridorMetric]:
        now = now or self._latest_time()
        self._validate_aware(now)
        start = now - timedelta(minutes=self._config.cpi_window_minutes)
        reports: List[CorridorMetric] = []
        for corridor_id, free_flow in self._config.corridor_free_flow_kph.items():
            samples = [sample for sample in self._traversals.get(corridor_id, []) if start <= sample.exited_at <= now]
            reports.append(corridor_metric(
                corridor_id, samples, free_flow_speed_kph=free_flow, window_start=start, window_end=now,
                thresholds=self._config.cpi_thresholds,
                outflow_capacity_vph=self._config.corridor_capacity_vph.get(corridor_id),
            ))
        return sorted(reports, key=lambda report: report.corridor_id)

    def get_od_flow_matrix(self, time_window_hours: float = 1.0, *, now: Optional[datetime] = None) -> ODMatrixResult:
        if time_window_hours <= 0.0:
            raise ValueError("time_window_hours must be positive")
        now = now or self._latest_time()
        self._validate_aware(now)
        return self._od.result(now, timedelta(hours=time_window_hours))

    def close_stale_sessions(self, now: datetime) -> None:
        self._validate_aware(now)
        for session in self._sessionizer.close_stale(now):
            self._od.add(session)
        self._prune(now)

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=self._config.retention_hours)
        self._sightings = [item for item in self._sightings if item.observed_at >= cutoff]
        for corridor_id in list(self._traversals):
            self._traversals[corridor_id] = [item for item in self._traversals[corridor_id] if item.exited_at >= cutoff]
        self._od.prune_before(cutoff)

    def _latest_time(self) -> datetime:
        candidates = [item.observed_at for item in self._sightings]
        candidates.extend(item.exited_at for samples in self._traversals.values() for item in samples)
        return max(candidates) if candidates else datetime.now(timezone.utc)

    @staticmethod
    def _parse_time(value: Any) -> datetime:
        if isinstance(value, datetime):
            result = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result = datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
        elif isinstance(value, str):
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            raise ValueError("Stage 2 timestamp must be datetime, ISO-8601 string, or epoch milliseconds")
        MacroAnalyticsEngine._validate_aware(result)
        return result

    @staticmethod
    def _validate_aware(value: datetime) -> None:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
