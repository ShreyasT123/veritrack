"""Trajectory reconstruction orchestrator for Stage 3.

Ties the four components together into one call:

.. code-block:: text

    sightings ──► temporal ordering and journey splitting
                      │
                      ▼
              pairwise fusion scoring  (dynamic w_text / w_vis / w_kin)
                      │
                      ▼
              anomaly conditions A, B, C
                      │
                      ▼
              Viterbi map-matching  (candidate pruning -> trellis -> backtrack)
                      │
                      ▼
              ReconstructedTrajectory + GeoJSON

A note on what "distance" means here. The reported ``total_distance_km`` is the
length of the **matched road path**, not the sum of straight lines between
camera poles. Those differ substantially in a real street grid -- a vehicle
going two blocks east and two blocks north travels 4 blocks, while the
straight-line distance is about 2.83. Reporting the latter would understate
every journey and corrupt the Stage 4 corridor statistics that build on it.

On data sources: the engine consumes :class:`CameraSighting` objects and does
not import Stage 2. A caller supplies rows from the ``sightings`` hypertable
(pseudonymised, which is all reconstruction needs -- same-day linkage is exactly
what the rotating salt preserves) or from ``hotlist_hits`` on a warrant-scoped
query. Keeping the dependency inverted means Stage 3 is testable without a
database and reusable for offline replay.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .anomaly import AnomalyDetector
from .config import TrajectoryConfig
from .fusion import ConfidenceFusion
from .graph import RoadNetworkGraph
from .types import (
    AnomalyRecord,
    CameraSighting,
    GeoPoint,
    MatchScore,
    ReconstructedTrajectory,
    RoadSegment,
    TrajectoryWaypoint,
)
from .viterbi import MapMatchResult, ViterbiMapMatcher

__all__ = ["TrajectoryEngine", "EngineStats"]


class EngineStats:
    """Cumulative counters, for the ops dashboard."""

    __slots__ = ("trajectories", "sightings", "anomalies", "unreachable_pairs", "split_journeys")

    def __init__(self) -> None:
        self.trajectories = 0
        self.sightings = 0
        self.anomalies = 0
        self.unreachable_pairs = 0
        self.split_journeys = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "trajectories": self.trajectories,
            "sightings": self.sightings,
            "anomalies": self.anomalies,
            "unreachable_pairs": self.unreachable_pairs,
            "split_journeys": self.split_journeys,
        }


class TrajectoryEngine:
    """End-to-end spatio-temporal trajectory reconstruction."""

    __slots__ = ("_graph", "_config", "_fusion", "_matcher", "_detector", "_stats")

    def __init__(
        self,
        graph: RoadNetworkGraph,
        config: Optional[TrajectoryConfig] = None,
        *,
        fusion: Optional[ConfidenceFusion] = None,
        matcher: Optional[ViterbiMapMatcher] = None,
        detector: Optional[AnomalyDetector] = None,
    ) -> None:
        self._graph = graph
        self._config = config if config is not None else TrajectoryConfig()
        self._fusion = fusion if fusion is not None else ConfidenceFusion(self._config)
        self._matcher = (
            matcher if matcher is not None else ViterbiMapMatcher(graph, self._config)
        )
        self._detector = (
            detector if detector is not None else AnomalyDetector(graph, self._config, self._fusion)
        )
        self._stats = EngineStats()

    # -- accessors -----------------------------------------------------

    @property
    def graph(self) -> RoadNetworkGraph:
        return self._graph

    @property
    def config(self) -> TrajectoryConfig:
        return self._config

    @property
    def fusion(self) -> ConfidenceFusion:
        return self._fusion

    @property
    def matcher(self) -> ViterbiMapMatcher:
        return self._matcher

    @property
    def detector(self) -> AnomalyDetector:
        return self._detector

    def stats(self) -> Dict[str, int]:
        return self._stats.to_dict()

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _trajectory_id(identity_key: str, sightings: Sequence[CameraSighting]) -> str:
        """Deterministic id, so a re-run produces the same key.

        Same rationale as Stage 1's ``pass_id``: an idempotent identifier lets
        the downstream store deduplicate on the primary key instead of needing
        a coordination protocol.
        """
        digest = hashlib.blake2b(
            "|".join(
                [identity_key]
                + [f"{s.sighting_id}:{s.epoch_s:.3f}" for s in sightings]
            ).encode("utf-8"),
            digest_size=12,
        )
        return digest.hexdigest()

    def _network_distance_between(
        self, left: CameraSighting, right: CameraSighting
    ) -> Tuple[float, float]:
        """(network distance in metres, corridor free-flow speed in km/h)."""
        from_node, _ = self._graph.nearest_node(left.location)
        to_node, _ = self._graph.nearest_node(right.location)
        distance = self._graph.network_distance(from_node, to_node)
        free_flow = self._graph.corridor_free_flow_kmh(
            from_node, to_node, fallback=self._config.fusion.default_free_flow_kmh
        )
        return distance, free_flow

    def split_journeys(
        self, sightings: Sequence[CameraSighting]
    ) -> List[List[CameraSighting]]:
        """Split a sighting stream into separate journeys on long idle gaps.

        A vehicle seen at 09:00 and again at 19:00 did not spend ten hours in
        transit; it parked. Interpolating a road path across that gap would
        invent a journey that never happened and pollute every corridor
        statistic downstream.
        """
        if not sightings:
            return []
        ordered = sorted(sightings, key=lambda sighting: sighting.epoch_s)
        journeys: List[List[CameraSighting]] = [[ordered[0]]]
        for previous, current in zip(ordered, ordered[1:]):
            if previous.seconds_to(current) > self._config.max_journey_gap_s:
                journeys.append([current])
                self._stats.split_journeys += 1
            else:
                journeys[-1].append(current)
        return journeys

    def score_pairs(self, sightings: Sequence[CameraSighting]) -> List[MatchScore]:
        """Fuse every consecutive pair in a journey."""
        scores: List[MatchScore] = []
        ordered = sorted(sightings, key=lambda sighting: sighting.epoch_s)
        for left, right in zip(ordered, ordered[1:]):
            distance, free_flow = self._network_distance_between(left, right)
            scores.append(
                self._fusion.score(
                    left, right, network_distance_m=distance, free_flow_kmh=free_flow
                )
            )
        return scores

    # -- main entry point ----------------------------------------------

    def reconstruct(
        self,
        sightings: Sequence[CameraSighting],
        *,
        identity_key: Optional[str] = None,
    ) -> Optional[ReconstructedTrajectory]:
        """Reconstruct one journey from a time-ordered sighting sequence.

        Returns ``None`` only when there is nothing to reconstruct (no
        sightings). A single sighting still yields a degenerate trajectory with
        zero distance, because "this vehicle was seen here once" is a real
        answer that a caller may need to render.
        """
        if not sightings:
            return None

        ordered = sorted(sightings, key=lambda sighting: sighting.epoch_s)
        resolved_identity = identity_key or ordered[0].identity_key
        self._stats.sightings += len(ordered)

        match_scores = self.score_pairs(ordered)
        anomalies = self._detector.analyse_sequence(ordered)
        self._stats.anomalies += len(anomalies)
        self._stats.unreachable_pairs += sum(
            1 for score in match_scores if not math.isfinite(score.network_distance_m)
        )

        match_result: MapMatchResult = self._matcher.match(ordered)

        # Projection-to-projection, not the sum of whole matched segments:
        # the vehicle was never observed traversing the tail of the first
        # segment or the head of the last.
        total_distance_m = match_result.travelled_distance_m
        transit_time_s = ordered[-1].epoch_s - ordered[0].epoch_s

        # Average speed is path length over elapsed time. Deliberately *not*
        # the mean of the per-leg speeds: legs have different durations, and an
        # unweighted mean would let a 10-second hop count as much as a
        # 10-minute cruise.
        if transit_time_s > 0.0 and total_distance_m > 0.0:
            average_speed_kmh = (total_distance_m / transit_time_s) * 3.6
        else:
            average_speed_kmh = 0.0

        notes = list(match_result.notes)
        weak = [
            score for score in match_scores if score.total < self._config.min_match_score
        ]
        if weak:
            notes.append(
                f"{len(weak)} of {len(match_scores)} consecutive pairs scored below the "
                f"{self._config.min_match_score:.2f} match threshold; identity continuity "
                f"across this journey is not established on fusion evidence alone"
            )

        waypoints = match_result.waypoints
        if not waypoints:
            # Candidate pruning found nothing anywhere -- every camera sits
            # outside the network extract. Fall back to the raw observations so
            # the caller still receives the sightings rather than an empty hull.
            waypoints = [
                TrajectoryWaypoint(
                    sequence_index=index,
                    location=sighting.location,
                    segment_id="",
                    street_name="",
                    is_observed=True,
                    timestamp_utc=sighting.timestamp_utc,
                    sighting_id=sighting.sighting_id,
                    camera_id=sighting.camera_id,
                )
                for index, sighting in enumerate(ordered)
            ]
            notes.append("no map match available; waypoints are raw camera positions")

        self._stats.trajectories += 1
        return ReconstructedTrajectory(
            trajectory_id=self._trajectory_id(resolved_identity, ordered),
            identity_key=resolved_identity,
            waypoints=tuple(waypoints),
            segments=tuple(match_result.path_segments),
            anomalies=tuple(anomalies),
            match_scores=tuple(match_scores),
            total_distance_km=total_distance_m / 1000.0,
            transit_time_s=transit_time_s,
            average_speed_kmh=average_speed_kmh,
            observed_sighting_count=len(ordered),
            interpolated_segment_count=match_result.interpolated_segment_count,
            started_at=ordered[0].timestamp_utc,
            ended_at=ordered[-1].timestamp_utc,
            viterbi_logprob=match_result.total_logprob,
            is_contiguous=match_result.is_contiguous,
            notes=tuple(notes),
        )

    def reconstruct_all(
        self,
        sightings: Sequence[CameraSighting],
        *,
        identity_key: Optional[str] = None,
    ) -> List[ReconstructedTrajectory]:
        """Split into journeys on idle gaps, then reconstruct each separately."""
        trajectories: List[ReconstructedTrajectory] = []
        for journey in self.split_journeys(sightings):
            trajectory = self.reconstruct(journey, identity_key=identity_key)
            if trajectory is not None:
                trajectories.append(trajectory)
        return trajectories

    def reconstruct_by_identity(
        self, sightings: Iterable[CameraSighting]
    ) -> Dict[str, List[ReconstructedTrajectory]]:
        """Group a mixed stream by vehicle identity and reconstruct each.

        This is the batch entry point: hand it an hour of the ``sightings``
        hypertable and it returns every vehicle's journeys, keyed by pseudonym.
        """
        grouped: Dict[str, List[CameraSighting]] = {}
        for sighting in sightings:
            grouped.setdefault(sighting.identity_key, []).append(sighting)
        return {
            identity: self.reconstruct_all(group, identity_key=identity)
            for identity, group in grouped.items()
        }

    # -- serialisation -------------------------------------------------

    @staticmethod
    def to_geojson(trajectories: Sequence[ReconstructedTrajectory]) -> Dict[str, Any]:
        """Merge several trajectories into one FeatureCollection for a map layer."""
        features: List[Dict[str, Any]] = []
        for trajectory in trajectories:
            features.extend(trajectory.to_geojson()["features"])
        return {"type": "FeatureCollection", "features": features}

    @staticmethod
    def from_stage2_rows(
        rows: Iterable[Any],
        *,
        embedding_field: str = "reid_embedding",
    ) -> List[CameraSighting]:
        """Adapt Stage 2 ``sightings`` rows into :class:`CameraSighting`.

        Accepts mappings or objects with matching attributes, so an asyncpg
        ``Record``, a dict from a JSON export and a dataclass all work without
        the caller reshaping anything.
        """

        def get(row: Any, key: str, default: Any = None) -> Any:
            if isinstance(row, dict):
                return row.get(key, default)
            try:
                if hasattr(row, "keys") and key in row.keys():
                    return row[key]
            except (TypeError, AttributeError):
                pass
            return getattr(row, key, default)

        sightings: List[CameraSighting] = []
        for row in rows:
            timestamp = get(row, "timestamp_utc")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            if timestamp is None:
                raise ValueError("Stage 2 row is missing timestamp_utc")
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            latitude = get(row, "latitude")
            longitude = get(row, "longitude")
            if latitude is None or longitude is None:
                raise ValueError(
                    "Stage 2 row has no camera coordinates; join against the "
                    "`cameras` table before reconstruction"
                )

            embedding = get(row, embedding_field) or ()
            sightings.append(
                CameraSighting(
                    sighting_id=str(get(row, "pass_id", "")),
                    camera_id=str(get(row, "camera_id", "")),
                    timestamp_utc=timestamp,
                    location=GeoPoint(latitude=float(latitude), longitude=float(longitude)),
                    plate_pseudonym=str(get(row, "plate_pseudonym", "")),
                    plate_text=get(row, "plate_number"),
                    plate_sequence_confidence=float(
                        get(row, "plate_sequence_confidence", 0.0) or 0.0
                    ),
                    reid_embedding=tuple(float(value) for value in embedding),
                    vehicle_class=str(get(row, "vehicle_class", "unknown")),
                    travel_heading_azimuth=get(row, "travel_heading_azimuth"),
                    vehicle_speed_kmh=get(row, "vehicle_speed_kmh"),
                    corridor_id=get(row, "corridor_id"),
                    text_entropy=get(row, "text_entropy"),
                    repair_cost_nats=float(get(row, "repair_cost_nats", 0.0) or 0.0),
                )
            )
        return sightings
