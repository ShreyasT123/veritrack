"""Trip sessionization and windowed origin-destination aggregation."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import DefaultDict, Dict, Iterable, List, Tuple

from .spatial import H3SpatialIndex
from .types import AnalyticsSighting, ODMatrixEntry, ODMatrixResult, TripSession

__all__ = ["TripSessionizer", "ODMatrixAccumulator"]


class TripSessionizer:
    """Stateful per-vehicle sessionizer for chronologically ingested sightings."""

    def __init__(self, dwell_timeout: timedelta) -> None:
        if dwell_timeout <= timedelta(0):
            raise ValueError("dwell_timeout must be positive")
        self._dwell_timeout = dwell_timeout
        self._open: Dict[str, List[AnalyticsSighting]] = {}

    def ingest(self, sighting: AnalyticsSighting) -> Tuple[TripSession, ...]:
        """Add one sighting and return sessions completed by this event."""
        active = self._open.get(sighting.vehicle_id)
        completed: List[TripSession] = []
        if active is not None:
            previous = active[-1]
            if sighting.observed_at < previous.observed_at:
                raise ValueError("sightings for each vehicle must be ingested chronologically")
            if sighting.observed_at - previous.observed_at > self._dwell_timeout:
                completed.append(TripSession(sighting.vehicle_id, tuple(active), True, "dwell_timeout"))
                active = None

        if active is None:
            active = []
            self._open[sighting.vehicle_id] = active
        active.append(sighting)
        if sighting.is_perimeter_exit:
            completed.append(TripSession(sighting.vehicle_id, tuple(active), True, "perimeter_exit"))
            del self._open[sighting.vehicle_id]
        return tuple(completed)

    def close_stale(self, now: datetime) -> Tuple[TripSession, ...]:
        """Close sessions idle longer than the dwell threshold at ``now``."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        completed: List[TripSession] = []
        for vehicle_id, sightings in list(self._open.items()):
            if now - sightings[-1].observed_at > self._dwell_timeout:
                completed.append(TripSession(vehicle_id, tuple(sightings), True, "dwell_timeout"))
                del self._open[vehicle_id]
        return tuple(completed)

    @classmethod
    def sessionize(cls, sightings: Iterable[AnalyticsSighting], dwell_timeout: timedelta) -> Tuple[TripSession, ...]:
        """Pure batch helper; returns completed and currently-open sessions."""
        grouped: DefaultDict[str, List[AnalyticsSighting]] = defaultdict(list)
        for sighting in sightings:
            grouped[sighting.vehicle_id].append(sighting)
        result: List[TripSession] = []
        for vehicle_id, events in grouped.items():
            events.sort(key=lambda item: item.observed_at)
            current: List[AnalyticsSighting] = []
            for event in events:
                if current and event.observed_at - current[-1].observed_at > dwell_timeout:
                    result.append(TripSession(vehicle_id, tuple(current), True, "dwell_timeout"))
                    current = []
                current.append(event)
                if event.is_perimeter_exit:
                    result.append(TripSession(vehicle_id, tuple(current), True, "perimeter_exit"))
                    current = []
            if current:
                result.append(TripSession(vehicle_id, tuple(current), False, None))
        return tuple(result)


class ODMatrixAccumulator:
    """Retains completed trips and derives O-D flows for arbitrary windows."""

    def __init__(self, spatial: H3SpatialIndex, resolution: int) -> None:
        self._spatial = spatial
        self._resolution = resolution
        self._records: List[Tuple[TripSession, str, str]] = []

    def add(self, session: TripSession) -> None:
        if not session.completed:
            return
        origin = self._spatial.cell(session.origin.latitude, session.origin.longitude, self._resolution)
        destination = self._spatial.cell(session.destination.latitude, session.destination.longitude, self._resolution)
        self._records.append((session, origin, destination))

    def prune_before(self, cutoff: datetime) -> None:
        self._records = [record for record in self._records if record[0].destination.observed_at >= cutoff]

    def result(self, window_end: datetime, window: timedelta) -> ODMatrixResult:
        if window <= timedelta(0):
            raise ValueError("window must be positive")
        window_start = window_end - window
        records = [r for r in self._records if window_start <= r[0].destination.observed_at <= window_end]
        aggregate: DefaultDict[Tuple[str, str], List[float]] = defaultdict(lambda: [0.0, 0.0])
        totals: DefaultDict[str, int] = defaultdict(int)
        for session, origin, destination in records:
            aggregate[(origin, destination)][0] += 1.0
            aggregate[(origin, destination)][1] += session.duration_seconds
            totals[origin] += 1

        entries: List[ODMatrixEntry] = []
        for (origin, destination), (count, duration_total) in aggregate.items():
            volume = int(count)
            entries.append(ODMatrixEntry(
                origin_h3=origin,
                destination_h3=destination,
                volume=volume,
                probability=volume / totals[origin],
                mean_duration_seconds=duration_total / volume,
                origin_centroid=self._spatial.centroid(origin),
                destination_centroid=self._spatial.centroid(destination),
            ))
        entries.sort(key=lambda entry: (-entry.volume, entry.origin_h3, entry.destination_h3))
        return ODMatrixResult(window_start, window_end, tuple(entries), len(records))
