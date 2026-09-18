"""Configuration for the VeriTrack Stage 3 trajectory reconstruction engine.

Frozen dataclasses rather than Pydantic, for the same reason Stage 1 uses them:
this configuration is read inside a Viterbi inner loop that runs
``O(T x |C|^2)`` times per trajectory, and the values come from a trusted
operator file rather than from the network. Pydantic earns its cost at the
Stage 2 ingest boundary, where the input is hostile; it does not earn it here.

Every default is stated with the physical reasoning behind it, because these
are the numbers a reviewer will challenge first.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Final, Mapping

__all__ = [
    "TrajectoryConfig",
    "FusionConfig",
    "ViterbiConfig",
    "AnomalyConfig",
    "GraphConfig",
    "load_config",
    "DEFAULT_CONFIG",
]

#: Earth mean radius in metres (WGS84 authalic). Used by every haversine call.
EARTH_RADIUS_M: Final[float] = 6_371_008.8


@dataclass(frozen=True, slots=True)
class GraphConfig:
    """Road network representation and shortest-path behaviour."""

    #: Entries retained in the network-distance LRU. A city graph with a few
    #: thousand camera-adjacent nodes generates far fewer distinct origin
    #: -destination pairs than nodes squared, because queries cluster on the
    #: camera set; 1e5 entries covers the working set of a busy shift.
    distance_cache_size: int = 100_000

    #: Ceiling on Dijkstra/A* exploration, in metres. A query that would have
    #: to search beyond this returns "unreachable" rather than walking the
    #: whole graph -- which is the correct answer for anomaly Condition C and
    #: also bounds worst-case latency.
    max_search_distance_m: float = 50_000.0

    #: Use the haversine A* heuristic rather than plain Dijkstra. Admissible
    #: because a road segment's length is never shorter than the great-circle
    #: distance between its endpoints, so A* cannot return a suboptimal path.
    use_astar_heuristic: bool = True

    #: Default speed assumed for an edge with no speed_limit_kmh attribute.
    default_speed_limit_kmh: float = 50.0


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Dynamic multi-modal confidence fusion.

    The gate ``g = sigmoid(k (C_seq - tau))`` decides how much the system is
    willing to trust the plate string. Below tau the weight shifts to the
    visual embedding; above it the text dominates.
    """

    #: Sigmoid steepness. At k = 12 the gate moves from 0.03 to 0.97 across
    #: roughly a 0.3 band of OCR confidence, which is sharp enough to be
    #: decisive without being a step function that chatters on noise.
    gate_steepness_k: float = 12.0

    #: Gate midpoint. 0.65 sits just below the confidence at which Stage 1's
    #: validator stops needing confusion repair, so a repaired plate lands on
    #: the visual-leaning side of the gate.
    gate_threshold_tau: float = 0.65

    #: Weight schedule coefficients. w_text = g * a_text;
    #: w_vis = (1-g) * a_vis_low + g * a_vis_high; w_kin = 1 - w_text - w_vis.
    #: Closed form: w_kin = 0.35 - 0.20 g, so the three always form a proper
    #: simplex and w_kin never goes negative.
    a_text: float = 0.70
    a_vis_low: float = 0.65
    a_vis_high: float = 0.15

    #: Substitution cost for a known optical confusion pair (8<->B, 0<->O,
    #: 1<->I, ...). A full substitution costs 1.0, so a confusable pair costs
    #: 30% of an arbitrary one: Stage 1 already priced and possibly repaired
    #: these, and double-penalising them here would discard real matches.
    confusion_substitution_cost: float = 0.30

    #: Transposition cost for the Damerau extension.
    transposition_cost: float = 1.0

    #: Corridor free-flow speed used when a segment declares none.
    default_free_flow_kmh: float = 45.0

    #: Std-dev of the kinematic Gaussian, in km/h. Travelling one sigma away
    #: from the corridor baseline costs a factor exp(-0.5) ~= 0.61.
    kinematic_sigma_kmh: float = 18.0

    #: Widening applied to the Gaussian *below* the free-flow baseline.
    #:
    #: A symmetric Gaussian centred on free-flow is physically wrong: it scores
    #: crawling at 15 km/h exactly as improbable as doing 75 in a 45 zone. But
    #: a real vehicle is slower than free-flow most of the time (signals,
    #: congestion, a stop for chai) and almost never meaningfully faster. The
    #: likelihood of a transit slower than baseline should therefore decay far
    #: more gently than the likelihood of one faster than it. At 2.5x, a 15
    #: km/h transit on a 45 km/h corridor scores 0.83 instead of 0.25, while
    #: 75 km/h still scores 0.25.
    kinematic_slow_sigma_multiplier: float = 2.5

    #: Set True to use the literal symmetric Gaussian from the specification,
    #: ignoring the widening above. Retained so the asymmetric model can be
    #: A/B tested against the spec baseline rather than merely asserted better.
    symmetric_kinematic: bool = False

    #: Hard kinematic veto. Above this network speed S_kin is exactly 0.0.
    max_velocity_kmh: float = 140.0


@dataclass(frozen=True, slots=True)
class ViterbiConfig:
    """HMM map-matching parameters."""

    #: GPS/pole-position uncertainty in metres, the sigma of the emission
    #: Gaussian over perpendicular distance from camera to road segment.
    #: Survey-grade pole coordinates are good to a few metres; 12 m leaves
    #: room for lane offset and for the camera's oblique view of the roadway.
    sigma_gps_m: float = 12.0

    #: Candidate search radius. The specification caps this at 50 m, which is
    #: the right order: wider admits the parallel service road as a candidate
    #: for every arterial sighting and inflates the trellis quadratically.
    search_radius_m: float = 50.0

    #: Maximum candidate edges retained per sighting, best emission first.
    #: Viterbi is O(T |C|^2); capping |C| is what bounds runtime on a dense
    #: junction where a dozen segments fall inside the radius.
    max_candidates_per_sighting: int = 8

    #: Scale of the exponential transition penalty, in metres. Newson & Krumm
    #: fit beta from data; 120 m is a reasonable urban prior for the
    #: discrepancy between network distance and speed-implied distance.
    beta_m: float = 120.0

    #: Floor on the heading-alignment factor. ``max(0, cos(dtheta))`` is
    #: exactly zero at 90 degrees, and log(0) = -inf would annihilate an
    #: otherwise excellent path because of one perpendicular candidate. The
    #: floor keeps the penalty severe (about -9.2 nats) but finite and
    #: recoverable.
    heading_floor: float = 1e-4

    #: Treat an edge as bidirectionally plausible for heading purposes when
    #: the network is known to be undirected at that point. Off by default:
    #: the graph is directed and one-way compliance is a real signal.
    allow_reverse_heading: bool = False

    #: Ceiling on segments interpolated into one blind-spot gap. Prevents a
    #: pathological pair of sightings from expanding into a path with
    #: thousands of segments.
    max_path_segments_per_gap: int = 400


@dataclass(frozen=True, slots=True)
class AnomalyConfig:
    """Kinematic and visual anomaly thresholds."""

    #: Condition A. Above this network-distance speed, one physical vehicle
    #: cannot have produced both sightings.
    max_velocity_kmh: float = 140.0

    #: Condition B. Below this Re-ID cosine similarity the two sightings are
    #: visually different vehicles. Stage 1 measured int8 quantisation drift
    #: at 0.00179, two orders below this, so the threshold is not sensitive
    #: to the edge's wire encoding.
    min_reid_similarity: float = 0.35

    #: Condition B requires an exact plate-string match before a visual
    #: divergence is meaningful. Exposed so a deployment can relax it.
    swapped_plate_requires_exact_text: bool = True

    #: Minimum elapsed time between sightings for a speed computation to mean
    #: anything. Two cameras firing 0.4 s apart produce a division that is
    #: dominated by timestamp quantisation, not by vehicle motion.
    min_delta_t_s: float = 1.0

    #: Ignore Condition A when the two sightings are closer than this. At very
    #: short network distances the speed estimate is dominated by pole-position
    #: error rather than by travel.
    min_network_distance_m: float = 25.0

    #: Confidence floor below which a sighting is too unreliable to accuse a
    #: motorist on. A cloned-plate flag on a 0.3-confidence OCR read is far
    #: more likely to be a misread than a crime.
    min_confidence_for_flag: float = 0.55


@dataclass(frozen=True, slots=True)
class TrajectoryConfig:
    """Root Stage 3 configuration.

    The flat aliases (``sigma_gps``, ``beta``, ``max_velocity_kmh``,
    ``min_reid_similarity``, ``search_radius_m``) are exposed as properties so
    that the specification's vocabulary works directly against this object
    without forcing every caller to know which sub-config owns which value.
    """

    graph: GraphConfig = field(default_factory=GraphConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    viterbi: ViterbiConfig = field(default_factory=ViterbiConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)

    #: Minimum fused score for two sightings to be accepted as the same
    #: vehicle when chaining a trajectory.
    min_match_score: float = 0.55

    #: Maximum wall-clock gap between consecutive sightings before the engine
    #: splits them into separate journeys rather than interpolating a path.
    max_journey_gap_s: float = 3_600.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not 0.0 < self.fusion.gate_threshold_tau < 1.0:
            raise ValueError("gate_threshold_tau must lie in (0, 1)")
        if self.fusion.gate_steepness_k <= 0.0:
            raise ValueError("gate_steepness_k must be positive")
        if self.viterbi.sigma_gps_m <= 0.0:
            raise ValueError("sigma_gps_m must be positive")
        if self.viterbi.beta_m <= 0.0:
            raise ValueError("beta_m must be positive")
        if self.viterbi.search_radius_m <= 0.0:
            raise ValueError("search_radius_m must be positive")
        if self.viterbi.max_candidates_per_sighting < 1:
            raise ValueError("max_candidates_per_sighting must be >= 1")
        if not 0.0 <= self.anomaly.min_reid_similarity <= 1.0:
            raise ValueError("min_reid_similarity must lie in [0, 1]")
        if self.anomaly.max_velocity_kmh <= 0.0:
            raise ValueError("max_velocity_kmh must be positive")

        # The weight schedule must produce a proper simplex at both extremes.
        for gate in (0.0, 1.0):
            weights = self.weights_for_gate(gate)
            if any(weight < -1e-12 for weight in weights):
                raise ValueError(f"weight schedule yields a negative weight at g={gate}")
            if abs(sum(weights) - 1.0) > 1e-9:
                raise ValueError(f"weight schedule does not sum to 1 at g={gate}")

    # -- specification-vocabulary aliases -----------------------------------

    @property
    def sigma_gps(self) -> float:
        return self.viterbi.sigma_gps_m

    @property
    def beta(self) -> float:
        return self.viterbi.beta_m

    @property
    def max_velocity_kmh(self) -> float:
        return self.anomaly.max_velocity_kmh

    @property
    def min_reid_similarity(self) -> float:
        return self.anomaly.min_reid_similarity

    @property
    def search_radius_m(self) -> float:
        return self.viterbi.search_radius_m

    # -- weight schedule -----------------------------------------------------

    def gate(self, sequence_confidence: float) -> float:
        """g = 1 / (1 + exp(-k (C_seq - tau))).

        Computed in a numerically stable branch so that a large negative
        exponent cannot overflow ``math.exp``.
        """
        z = self.fusion.gate_steepness_k * (sequence_confidence - self.fusion.gate_threshold_tau)
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def weights_for_gate(self, gate: float) -> tuple[float, float, float]:
        """(w_text, w_vis, w_kin) for a given gate value."""
        w_text = gate * self.fusion.a_text
        w_vis = (1.0 - gate) * self.fusion.a_vis_low + gate * self.fusion.a_vis_high
        w_kin = 1.0 - (w_text + w_vis)
        return w_text, w_vis, w_kin

    def weights_for_confidence(self, sequence_confidence: float) -> tuple[float, float, float]:
        return self.weights_for_gate(self.gate(sequence_confidence))

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryConfig":
        return cls(
            graph=GraphConfig(**payload.get("graph", {})),
            fusion=FusionConfig(**payload.get("fusion", {})),
            viterbi=ViterbiConfig(**payload.get("viterbi", {})),
            anomaly=AnomalyConfig(**payload.get("anomaly", {})),
            min_match_score=float(payload.get("min_match_score", 0.55)),
            max_journey_gap_s=float(payload.get("max_journey_gap_s", 3_600.0)),
        )

    def evolve(self, **changes: Any) -> "TrajectoryConfig":
        """Return a copy with top-level fields replaced. Validation re-runs."""
        return replace(self, **changes)


def load_config(path: str | Path) -> TrajectoryConfig:
    """Load a JSON configuration file, falling back to defaults for absent keys."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"trajectory config not found: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"trajectory config is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("trajectory config must be a JSON object")
    return TrajectoryConfig.from_dict(payload)


DEFAULT_CONFIG: Final[TrajectoryConfig] = TrajectoryConfig()
