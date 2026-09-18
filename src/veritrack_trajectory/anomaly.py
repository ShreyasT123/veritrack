"""Kinematic anomaly and cloned-plate detection for Stage 3.

Three independent conditions, each answering a different question about a pair
of consecutive sightings that claim to be the same vehicle.

``Condition A -- CLONED_PLATE``
    The implied network-distance speed exceeds the physical ceiling. One
    vehicle did not cover that distance in that time, so two vehicles are
    wearing the same registration number.

``Condition B -- SWAPPED_PLATE``
    The plate strings match exactly, but the Re-ID embeddings do not. The
    number is the same; the vehicle is not. This is the signature of a plate
    physically moved between vehicles, and it is invisible to any system that
    only reads text.

``Condition C -- UNREACHABLE_TRANSITION``
    No directed path exists between the two sightings at all -- a creek with no
    bridge, a one-way system that cannot be traversed in that order. Distinct
    from A: A says "too fast", C says "impossible at any speed".

Why the guards matter more than the thresholds
----------------------------------------------
Every one of these flags can put a real person under suspicion, so the detector
refuses to raise one on weak evidence. Three guards apply throughout:

* **Minimum elapsed time.** Two cameras firing 0.4 s apart produce a speed
  dominated by timestamp quantisation, not by motion. Dividing by a near-zero
  ``delta_t`` manufactures spectacular velocities out of clock jitter.
* **Minimum network distance.** Over a few metres the speed estimate is
  dominated by pole-survey error rather than by travel.
* **Minimum OCR confidence.** A cloned-plate accusation resting on a
  0.3-confidence read is far more likely to be a misread than a crime. The
  correct output there is "uncertain", not "cloned".

Severity is graded by margin rather than fixed, because a vehicle at 145 km/h
is a borderline case worth a look and one at 400 km/h is arithmetic proof.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from .config import AnomalyConfig, TrajectoryConfig
from .fusion import ConfidenceFusion, cosine_similarity, text_similarity
from .graph import RoadNetworkGraph
from .types import (
    AnomalyRecord,
    AnomalySeverity,
    AnomalyType,
    CameraSighting,
)

__all__ = ["AnomalyDetector"]


class AnomalyDetector:
    """Applies conditions A, B and C to consecutive sighting pairs."""

    __slots__ = ("_graph", "_config", "_anomaly", "_fusion")

    def __init__(
        self,
        graph: RoadNetworkGraph,
        config: Optional[TrajectoryConfig] = None,
        fusion: Optional[ConfidenceFusion] = None,
    ) -> None:
        self._graph = graph
        self._config = config if config is not None else TrajectoryConfig()
        self._anomaly: AnomalyConfig = self._config.anomaly
        self._fusion = fusion if fusion is not None else ConfidenceFusion(self._config)

    @property
    def config(self) -> TrajectoryConfig:
        return self._config

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _resolve_nodes(self, left: CameraSighting, right: CameraSighting) -> Tuple[str, str]:
        """Map two camera locations onto their nearest graph nodes."""
        from_node, _ = self._graph.nearest_node(left.location)
        to_node, _ = self._graph.nearest_node(right.location)
        return from_node, to_node

    def _pair_confidence(self, left: CameraSighting, right: CameraSighting) -> float:
        return min(left.plate_sequence_confidence, right.plate_sequence_confidence)

    @staticmethod
    def _severity_for_speed(speed_kmh: float, threshold_kmh: float) -> AnomalySeverity:
        """Grade by how far past the ceiling the implied speed sits.

        A 5% overshoot could still be survey error compounding with clock skew.
        A 3x overshoot cannot be anything but two vehicles.
        """
        if not math.isfinite(speed_kmh):
            return AnomalySeverity.CRITICAL
        ratio = speed_kmh / max(threshold_kmh, 1e-6)
        if ratio >= 3.0:
            return AnomalySeverity.CRITICAL
        if ratio >= 1.8:
            return AnomalySeverity.HIGH
        if ratio >= 1.25:
            return AnomalySeverity.ELEVATED
        return AnomalySeverity.ADVISORY

    @staticmethod
    def _severity_for_divergence(similarity: float, threshold: float) -> AnomalySeverity:
        """Grade by how far below the Re-ID threshold the pair sits."""
        if similarity <= 0.0:
            return AnomalySeverity.CRITICAL
        margin = (threshold - similarity) / max(threshold, 1e-6)
        if margin >= 0.75:
            return AnomalySeverity.CRITICAL
        if margin >= 0.45:
            return AnomalySeverity.HIGH
        return AnomalySeverity.ELEVATED

    # -----------------------------------------------------------------
    # Conditions
    # -----------------------------------------------------------------

    def check_teleportation(
        self,
        left: CameraSighting,
        right: CameraSighting,
        *,
        network_distance_m: Optional[float] = None,
    ) -> Optional[AnomalyRecord]:
        """Condition A: spatio-temporal teleportation -> CLONED_PLATE."""
        delta_t_s = left.seconds_to(right)
        if delta_t_s < self._anomaly.min_delta_t_s:
            return None

        if network_distance_m is None:
            from_node, to_node = self._resolve_nodes(left, right)
            network_distance_m = self._graph.network_distance(from_node, to_node)

        if not math.isfinite(network_distance_m):
            # Unreachable is Condition C's business, not A's. Reporting both
            # for the same pair would double-count one physical event.
            return None
        if network_distance_m < self._anomaly.min_network_distance_m:
            return None

        speed_kmh = (network_distance_m / delta_t_s) * 3.6
        if speed_kmh <= self._anomaly.max_velocity_kmh:
            return None

        confidence = self._pair_confidence(left, right)
        if confidence < self._anomaly.min_confidence_for_flag:
            return None

        reid = (
            cosine_similarity(left.reid_embedding, right.reid_embedding)
            if left.reid_embedding and right.reid_embedding
            else None
        )
        return AnomalyRecord(
            anomaly_type=AnomalyType.CLONED_PLATE,
            severity=self._severity_for_speed(speed_kmh, self._anomaly.max_velocity_kmh),
            identity_key=left.identity_key,
            from_sighting_id=left.sighting_id,
            to_sighting_id=right.sighting_id,
            from_camera_id=left.camera_id,
            to_camera_id=right.camera_id,
            detected_at=datetime.now(timezone.utc),
            description=(
                f"Implied network speed of {speed_kmh:.1f} km/h over "
                f"{network_distance_m / 1000.0:.2f} km in {delta_t_s:.0f} s exceeds the "
                f"{self._anomaly.max_velocity_kmh:.0f} km/h physical ceiling; a single "
                f"vehicle cannot have produced both sightings."
            ),
            network_distance_m=network_distance_m,
            delta_t_s=delta_t_s,
            implied_speed_kmh=speed_kmh,
            reid_similarity=reid,
            confidence=confidence,
            evidence={
                "condition": "A",
                "threshold_kmh": self._anomaly.max_velocity_kmh,
                "overshoot_ratio": round(speed_kmh / self._anomaly.max_velocity_kmh, 3),
                "from_camera": left.camera_id,
                "to_camera": right.camera_id,
            },
        )

    def check_visual_divergence(
        self, left: CameraSighting, right: CameraSighting
    ) -> Optional[AnomalyRecord]:
        """Condition B: identical text, divergent appearance -> SWAPPED_PLATE."""
        if not left.reid_embedding or not right.reid_embedding:
            # Without both embeddings there is no visual evidence, and the
            # absence of evidence is not evidence of a swap.
            return None

        left_text = left.plate_text or left.plate_pseudonym
        right_text = right.plate_text or right.plate_pseudonym
        if not left_text or not right_text:
            return None

        s_text = text_similarity(left_text, right_text)
        if self._anomaly.swapped_plate_requires_exact_text and s_text < 1.0:
            # The whole premise of Condition B is "same number, different
            # vehicle". A near-miss on the text has an innocent explanation --
            # one of the two reads is simply wrong -- so it is not a swap.
            return None

        s_vis = cosine_similarity(left.reid_embedding, right.reid_embedding)
        if s_vis >= self._anomaly.min_reid_similarity:
            return None

        confidence = self._pair_confidence(left, right)
        if confidence < self._anomaly.min_confidence_for_flag:
            return None

        delta_t_s = left.seconds_to(right)
        return AnomalyRecord(
            anomaly_type=AnomalyType.SWAPPED_PLATE,
            severity=self._severity_for_divergence(s_vis, self._anomaly.min_reid_similarity),
            identity_key=left.identity_key,
            from_sighting_id=left.sighting_id,
            to_sighting_id=right.sighting_id,
            from_camera_id=left.camera_id,
            to_camera_id=right.camera_id,
            detected_at=datetime.now(timezone.utc),
            description=(
                f"Registration matches exactly but Re-ID cosine similarity is "
                f"{s_vis:.3f}, below the {self._anomaly.min_reid_similarity:.2f} "
                f"divergence threshold; the plate appears to have moved between "
                f"two physically different vehicles."
            ),
            delta_t_s=delta_t_s,
            reid_similarity=s_vis,
            text_similarity=s_text,
            confidence=confidence,
            evidence={
                "condition": "B",
                "reid_threshold": self._anomaly.min_reid_similarity,
                "embedding_dim": len(left.reid_embedding),
                "from_vehicle_class": left.vehicle_class,
                "to_vehicle_class": right.vehicle_class,
                # A vehicle-class change is strong corroboration: a plate that
                # moves from a two-wheeler to a truck is not an embedding
                # artefact.
                "vehicle_class_changed": left.vehicle_class != right.vehicle_class,
            },
        )

    def check_topological_impossibility(
        self, left: CameraSighting, right: CameraSighting
    ) -> Optional[AnomalyRecord]:
        """Condition C: no directed route exists -> UNREACHABLE_TRANSITION."""
        delta_t_s = left.seconds_to(right)
        if delta_t_s < self._anomaly.min_delta_t_s:
            return None

        from_node, to_node = self._resolve_nodes(left, right)
        if from_node == to_node:
            return None

        distance = self._graph.network_distance(from_node, to_node)
        if math.isfinite(distance):
            return None

        confidence = self._pair_confidence(left, right)
        if confidence < self._anomaly.min_confidence_for_flag:
            return None

        reid = (
            cosine_similarity(left.reid_embedding, right.reid_embedding)
            if left.reid_embedding and right.reid_embedding
            else None
        )
        return AnomalyRecord(
            anomaly_type=AnomalyType.UNREACHABLE_TRANSITION,
            severity=AnomalySeverity.HIGH,
            identity_key=left.identity_key,
            from_sighting_id=left.sighting_id,
            to_sighting_id=right.sighting_id,
            from_camera_id=left.camera_id,
            to_camera_id=right.camera_id,
            detected_at=datetime.now(timezone.utc),
            description=(
                f"No directed route exists from {left.camera_id} to {right.camera_id} "
                f"within the {self._graph.config.max_search_distance_m / 1000.0:.0f} km "
                f"search horizon; the transition is topologically impossible at any speed."
            ),
            network_distance_m=math.inf,
            delta_t_s=delta_t_s,
            implied_speed_kmh=math.inf,
            reid_similarity=reid,
            confidence=confidence,
            evidence={
                "condition": "C",
                "from_node": from_node,
                "to_node": to_node,
                "search_horizon_m": self._graph.config.max_search_distance_m,
                # Distinguishes "genuinely disconnected" from "farther than we
                # were willing to search", which is an important operational
                # difference when tuning the horizon.
                "straight_line_m": round(left.location.distance_to(right.location), 2),
            },
        )

    # -----------------------------------------------------------------
    # Sequence-level entry point
    # -----------------------------------------------------------------

    def analyse_pair(
        self,
        left: CameraSighting,
        right: CameraSighting,
        *,
        network_distance_m: Optional[float] = None,
    ) -> List[AnomalyRecord]:
        """Apply all three conditions to one consecutive pair.

        Conditions A and C are mutually exclusive by construction (A requires a
        finite distance, C requires an infinite one), but B is independent of
        both: a plate can be simultaneously swapped and teleporting, and each
        is separately actionable.
        """
        anomalies: List[AnomalyRecord] = []

        unreachable = self.check_topological_impossibility(left, right)
        if unreachable is not None:
            anomalies.append(unreachable)
        else:
            teleport = self.check_teleportation(
                left, right, network_distance_m=network_distance_m
            )
            if teleport is not None:
                anomalies.append(teleport)

        swapped = self.check_visual_divergence(left, right)
        if swapped is not None:
            anomalies.append(swapped)

        return anomalies

    def analyse_sequence(self, sightings: Sequence[CameraSighting]) -> List[AnomalyRecord]:
        """Apply the conditions across a time-ordered sighting sequence.

        Consecutive pairs only. Testing every pair would be O(n^2) and would
        also double-report: if A->B teleports, so does A->C for most C, and
        flooding an analyst with the same physical event restated twenty ways
        is how a detection system gets ignored.
        """
        if len(sightings) < 2:
            return []
        ordered = sorted(sightings, key=lambda sighting: sighting.epoch_s)
        anomalies: List[AnomalyRecord] = []
        for left, right in zip(ordered, ordered[1:]):
            anomalies.extend(self.analyse_pair(left, right))
        return anomalies

    def summarise(self, anomalies: Sequence[AnomalyRecord]) -> Dict[str, object]:
        """Counts and worst-case severity, for dashboards and Stage 5 routing."""
        counts: Dict[str, int] = {}
        for anomaly in anomalies:
            counts[anomaly.anomaly_type.value] = counts.get(anomaly.anomaly_type.value, 0) + 1
        return {
            "total": len(anomalies),
            "by_type": counts,
            "max_severity": (
                int(max(anomaly.severity for anomaly in anomalies)) if anomalies else None
            ),
            "identity_keys": sorted({anomaly.identity_key for anomaly in anomalies}),
        }
