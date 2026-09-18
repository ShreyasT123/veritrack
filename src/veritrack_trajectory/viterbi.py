"""HMM Viterbi map-matching for Stage 3.

The problem
-----------
Cameras see a vehicle at a handful of discrete points. The road between those
points is unobserved -- often kilometres of it. Map matching recovers the most
probable *continuous* sequence of road segments that explains the observations,
which is what turns a scatter of pole hits into a route with street names.

The model (Newson & Krumm, with the transition term adapted to the
specification's speed-discrepancy form):

* **States** are candidate road segments near each sighting.
* **Emission** :math:`p(z_k \\mid e_i)` scores how well segment :math:`e_i`
  explains the observed position and heading.
* **Transition** :math:`p(e_j \\mid e_i)` scores how plausible it is to have
  driven from one segment to the next in the elapsed time.
* **Viterbi** finds the single highest-probability path through the trellis in
  :math:`O(T |C|^2)`, with backpointers for reconstruction.

Everything is computed in **log space**. This is not a stylistic preference: a
20-sighting trajectory multiplies roughly 40 probabilities, each often below
:math:`10^{-3}`, so the product underflows float64 long before the path is
complete. In log space the same computation is a sum of well-scaled negative
numbers.

Two numerical details that matter
---------------------------------
1. ``max(0, cos(dtheta))`` is *exactly zero* at 90 degrees, and
   :math:`\\log 0 = -\\infty`. One perpendicular candidate would then poison an
   otherwise excellent path with a NaN or an unrecoverable ``-inf``. The
   heading factor is floored at ``heading_floor`` (default 1e-4, about -9.2
   nats): still a severe penalty, but finite and recoverable if the rest of the
   evidence is strong.
2. Bearings must be compared with wraparound. A vehicle heading 359 degrees and
   an edge heading 1 degree differ by 2 degrees, not 358. Getting this wrong
   silently rejects correct candidates at the north crossing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import TrajectoryConfig, ViterbiConfig
from .graph import UNREACHABLE, EdgeCandidate, RoadNetworkGraph
from .types import (
    CameraSighting,
    GeoPoint,
    RoadSegment,
    TrajectoryWaypoint,
    angular_difference_deg,
)

__all__ = [
    "NEG_INF",
    "emission_logprob",
    "transition_logprob",
    "TrellisNode",
    "MapMatchResult",
    "ViterbiMapMatcher",
]

#: A large finite negative number rather than float("-inf"). Keeping the
#: trellis finite means a path is always recoverable: if every candidate at a
#: timestep is implausible, Viterbi still returns the least-bad one and flags
#: the trajectory as non-contiguous, rather than returning nothing at all.
NEG_INF: float = -1.0e9

_LOG_SQRT_2PI: float = 0.5 * math.log(2.0 * math.pi)


def emission_logprob(
    perpendicular_distance_m: float,
    heading_difference_deg: float,
    *,
    sigma_gps_m: float,
    heading_floor: float = 1e-4,
    allow_reverse: bool = False,
) -> float:
    """log p(z_k | e_i) for one candidate segment.

    .. math::

        p = \\frac{1}{\\sqrt{2\\pi}\\,\\sigma}
            \\exp\\!\\left(-\\frac{d_\\perp^2}{2\\sigma^2}\\right)
            \\cdot \\max(0, \\cos(\\Delta\\theta))

    The Gaussian term is the standard positional likelihood. The cosine term
    encodes that a vehicle observed heading north is not on a westbound
    carriageway -- which is what separates the two directions of a divided
    highway whose segments are metres apart and would otherwise be
    indistinguishable on distance alone.
    """
    if sigma_gps_m <= 0.0:
        raise ValueError("sigma_gps_m must be positive")

    z = perpendicular_distance_m / sigma_gps_m
    log_gaussian = -0.5 * z * z - _LOG_SQRT_2PI - math.log(sigma_gps_m)

    difference = angular_difference_deg(heading_difference_deg, 0.0)
    if allow_reverse:
        # Treat an edge as usable in either direction: fold the angle into
        # [0, 90] so a 180-degree mismatch scores as a perfect alignment.
        difference = min(difference, 180.0 - difference)
    alignment = math.cos(math.radians(difference))
    alignment = max(heading_floor, alignment)

    return log_gaussian + math.log(alignment)


def transition_logprob(
    network_distance_m: float,
    delta_t_s: float,
    corridor_speed_kmh: float,
    *,
    beta_m: float,
) -> float:
    """log p(e_j | e_i) from the speed-discrepancy exponential.

    .. math::

        p = \\frac{1}{\\beta}
            \\exp\\!\\left(-\\frac{|d_{\\text{network}} - v_{\\text{corridor}}\\,\\Delta t|}{\\beta}\\right)

    The quantity being penalised is the gap between how far the road network
    says the vehicle went and how far the corridor's free-flow speed says it
    should have gone in the elapsed time. A transition consistent with normal
    driving has a small discrepancy and a near-zero penalty; one requiring an
    implausible detour or an implausible sprint decays exponentially.

    An unreachable pair returns :data:`NEG_INF` -- there is no route, so no
    amount of timing agreement can rescue the transition. This is the
    topological half of anomaly Condition C, expressed in the transition model.
    """
    if beta_m <= 0.0:
        raise ValueError("beta_m must be positive")
    if not math.isfinite(network_distance_m):
        return NEG_INF
    if delta_t_s < 0.0:
        return NEG_INF

    expected_distance_m = (corridor_speed_kmh / 3.6) * delta_t_s
    discrepancy = abs(network_distance_m - expected_distance_m)
    return -math.log(beta_m) - (discrepancy / beta_m)


@dataclass(slots=True)
class TrellisNode:
    """One cell of the Viterbi trellis."""

    timestep: int
    candidate: EdgeCandidate
    emission: float
    #: Best log-probability of any path ending in this cell.
    score: float = NEG_INF
    #: Index of the predecessor cell at ``timestep - 1``.
    backpointer: Optional[int] = None
    #: Transition log-probability along the winning incoming edge.
    transition: float = 0.0

    @property
    def segment(self) -> RoadSegment:
        return self.candidate.segment


@dataclass(slots=True)
class MapMatchResult:
    """Output of a Viterbi map-match."""

    matched_segments: List[RoadSegment]
    matched_candidates: List[Optional[EdgeCandidate]]
    path_segments: List[RoadSegment]
    waypoints: List[TrajectoryWaypoint]
    total_logprob: float
    is_contiguous: bool
    #: Road distance actually travelled, measured projection-to-projection
    #: across consecutive matched positions. This is the authoritative journey
    #: length: summing whole ``path_segments`` over-counts the tail of the
    #: first segment and the head of the last, neither of which the vehicle
    #: was observed to traverse.
    travelled_distance_m: float = 0.0
    unreachable_gaps: List[Tuple[int, int]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def interpolated_segment_count(self) -> int:
        return sum(1 for waypoint in self.waypoints if not waypoint.is_observed)


class ViterbiMapMatcher:
    """Reconstructs the most probable road-segment sequence from sightings."""

    __slots__ = ("_graph", "_config", "_viterbi")

    def __init__(
        self, graph: RoadNetworkGraph, config: Optional[TrajectoryConfig] = None
    ) -> None:
        self._graph = graph
        self._config = config if config is not None else TrajectoryConfig()
        self._viterbi: ViterbiConfig = self._config.viterbi

    @property
    def graph(self) -> RoadNetworkGraph:
        return self._graph

    @property
    def config(self) -> TrajectoryConfig:
        return self._config

    # -----------------------------------------------------------------
    # Candidate generation
    # -----------------------------------------------------------------

    def candidates_for(self, sighting: CameraSighting) -> List[EdgeCandidate]:
        """Prune the network to the segments that could explain one sighting."""
        return self._graph.candidate_edges(
            sighting.location,
            radius_m=self._viterbi.search_radius_m,
            observed_heading_deg=sighting.travel_heading_azimuth,
            max_candidates=self._viterbi.max_candidates_per_sighting,
        )

    def _emission(self, candidate: EdgeCandidate) -> float:
        return emission_logprob(
            candidate.perpendicular_distance_m,
            candidate.heading_difference_deg,
            sigma_gps_m=self._viterbi.sigma_gps_m,
            heading_floor=self._viterbi.heading_floor,
            allow_reverse=self._viterbi.allow_reverse_heading,
        )

    def travel_distance(
        self, from_candidate: EdgeCandidate, to_candidate: EdgeCandidate
    ) -> float:
        """Road distance actually travelled between two matched positions.

        Measured **projection to projection**, not node to node. This matters
        far more than it looks. A camera at a junction sees the vehicle at a
        point that is simultaneously the end of the arriving segment
        (``along_fraction = 1.0``) and the start of the departing one
        (``along_fraction = 0.0``); the two candidates tie exactly on emission,
        since both have zero perpendicular distance and zero heading error.

        With node-to-node measurement the tie breaks on which whole-segment
        hop happens to fit the elapsed time better, and the matcher will
        happily place the vehicle on the departing segment and then route it
        *out to that segment's far node and back* -- a phantom detour that
        inflates the journey by a full block.

        Projection-aware measurement resolves this not by making the two tied
        candidates equal, but by making the transition model *discriminate*
        between them on the right grounds. From the arriving candidate the
        onward travel is the true remaining distance; from the departing one it
        includes the spurious out-and-back. The exponential penalty then scores
        the honest figure against the elapsed time and the detour loses, so
        Viterbi selects the candidate that actually explains the journey:

        .. math::

            d = (1 - f_1)\\,\\ell_1 + d_{\\text{net}}(v_1^{\\text{to}}, v_2^{\\text{from}}) + f_2\\,\\ell_2
        """
        from_segment = from_candidate.segment
        to_segment = to_candidate.segment

        if from_segment.segment_id == to_segment.segment_id:
            # Still on the same stretch of road: signed progress along it. A
            # negative value means the vehicle moved backwards along a
            # one-way, which the transition penalty should see as a large
            # discrepancy rather than as a short hop.
            return (to_candidate.along_fraction - from_candidate.along_fraction) * (
                from_segment.length_m
            )

        remaining = (1.0 - from_candidate.along_fraction) * from_segment.length_m
        entered = to_candidate.along_fraction * to_segment.length_m
        bridge = self._graph.network_distance(from_segment.to_node, to_segment.from_node)
        if not math.isfinite(bridge):
            return math.inf
        return remaining + bridge + entered

    def _transition(
        self,
        from_candidate: EdgeCandidate,
        to_candidate: EdgeCandidate,
        delta_t_s: float,
    ) -> Tuple[float, float]:
        """(log-probability, travelled distance) for one trellis transition."""
        from_segment = from_candidate.segment
        to_segment = to_candidate.segment

        distance = self.travel_distance(from_candidate, to_candidate)
        if distance < 0.0:
            # Backwards along a one-way. Keep it finite but treat the
            # magnitude as discrepancy, so the path is still recoverable.
            distance = abs(distance)

        corridor_speed = self._graph.corridor_free_flow_kmh(
            from_segment.to_node,
            to_segment.from_node,
            fallback=(from_segment.speed_limit_kmh + to_segment.speed_limit_kmh) * 0.5,
        )
        return (
            transition_logprob(
                distance, delta_t_s, corridor_speed, beta_m=self._viterbi.beta_m
            ),
            distance,
        )

    # -----------------------------------------------------------------
    # Trellis
    # -----------------------------------------------------------------

    def match(self, sightings: Sequence[CameraSighting]) -> MapMatchResult:
        """Run Viterbi over the sighting sequence and reconstruct the path."""
        if not sightings:
            return MapMatchResult([], [], [], [], 0.0, True, [], ["no sightings supplied"])

        ordered = sorted(sightings, key=lambda sighting: sighting.epoch_s)
        notes: List[str] = []

        # --- build the trellis columns --------------------------------
        columns: List[List[TrellisNode]] = []
        for timestep, sighting in enumerate(ordered):
            candidates = self.candidates_for(sighting)
            if not candidates:
                notes.append(
                    f"no road segment within {self._viterbi.search_radius_m:.0f} m of "
                    f"camera {sighting.camera_id} at t={timestep}"
                )
                columns.append([])
                continue
            columns.append(
                [
                    TrellisNode(
                        timestep=timestep,
                        candidate=candidate,
                        emission=self._emission(candidate),
                    )
                    for candidate in candidates
                ]
            )

        populated = [index for index, column in enumerate(columns) if column]
        if not populated:
            return MapMatchResult(
                [], [None] * len(ordered), [], [], NEG_INF, False, [],
                notes + ["every sighting failed candidate pruning"],
            )

        # --- forward pass ---------------------------------------------
        # Columns that produced no candidates are skipped rather than
        # terminating the match: a camera mounted on a road absent from the
        # network extract must not destroy the rest of an otherwise good
        # trajectory.
        first = populated[0]
        for node in columns[first]:
            node.score = node.emission
            node.backpointer = None

        unreachable_gaps: List[Tuple[int, int]] = []
        previous_index = first

        for current_index in populated[1:]:
            delta_t_s = ordered[current_index].epoch_s - ordered[previous_index].epoch_s
            any_reachable = False

            for current_position, current_node in enumerate(columns[current_index]):
                best_score = NEG_INF
                best_previous: Optional[int] = None
                best_transition = NEG_INF

                for previous_position, previous_node in enumerate(columns[previous_index]):
                    transition, distance = self._transition(
                        previous_node.candidate, current_node.candidate, delta_t_s
                    )
                    if math.isfinite(distance):
                        any_reachable = True
                    total = previous_node.score + transition + current_node.emission
                    if total > best_score:
                        best_score = total
                        best_previous = previous_position
                        best_transition = transition

                current_node.score = best_score
                current_node.backpointer = best_previous
                current_node.transition = best_transition

            if not any_reachable:
                unreachable_gaps.append((previous_index, current_index))
                notes.append(
                    f"no directed route between the candidate sets at t={previous_index} "
                    f"and t={current_index}"
                )
            previous_index = current_index

        # --- backtrack -------------------------------------------------
        last = populated[-1]
        best_terminal = max(range(len(columns[last])), key=lambda i: columns[last][i].score)
        total_logprob = columns[last][best_terminal].score

        reversed_path: List[TrellisNode] = []
        position: Optional[int] = best_terminal
        for column_index in reversed(populated):
            if position is None:
                break
            node = columns[column_index][position]
            reversed_path.append(node)
            position = node.backpointer
        reversed_path.reverse()

        matched_by_timestep: Dict[int, TrellisNode] = {
            node.timestep: node for node in reversed_path
        }
        matched_candidates: List[Optional[EdgeCandidate]] = [
            matched_by_timestep[index].candidate if index in matched_by_timestep else None
            for index in range(len(ordered))
        ]
        matched_segments = [node.segment for node in reversed_path]

        (
            path_segments,
            waypoints,
            expansion_notes,
            contiguous,
            travelled_m,
        ) = self._expand_path(ordered, reversed_path)
        notes.extend(expansion_notes)

        return MapMatchResult(
            matched_segments=matched_segments,
            matched_candidates=matched_candidates,
            path_segments=path_segments,
            waypoints=waypoints,
            total_logprob=total_logprob,
            is_contiguous=contiguous and not unreachable_gaps,
            travelled_distance_m=travelled_m,
            unreachable_gaps=unreachable_gaps,
            notes=notes,
        )

    # -----------------------------------------------------------------
    # Blind-spot filling
    # -----------------------------------------------------------------

    def _expand_path(
        self, sightings: Sequence[CameraSighting], matched: Sequence[TrellisNode]
    ) -> Tuple[List[RoadSegment], List[TrajectoryWaypoint], List[str], bool, float]:
        """Fill the road between consecutive matched segments.

        This is what makes the output a continuous route rather than a list of
        pole hits. Between each matched pair the shortest directed path is
        expanded into its constituent segments and emitted as *interpolated*
        waypoints, flagged ``is_observed=False`` so that a downstream consumer
        can always separate evidence from inference.
        """
        if not matched:
            return [], [], [], True, 0.0

        path_segments: List[RoadSegment] = []
        waypoints: List[TrajectoryWaypoint] = []
        notes: List[str] = []
        contiguous = True
        cumulative_m = 0.0
        sequence_index = 0

        def append_segment(segment: RoadSegment) -> None:
            nonlocal cumulative_m
            if path_segments and path_segments[-1].segment_id == segment.segment_id:
                return
            path_segments.append(segment)
            cumulative_m += segment.length_m

        first_node = matched[0]
        append_segment(first_node.segment)
        waypoints.append(
            TrajectoryWaypoint(
                sequence_index=sequence_index,
                location=first_node.candidate.projection,
                segment_id=first_node.segment.segment_id,
                street_name=first_node.segment.street_name,
                is_observed=True,
                timestamp_utc=sightings[first_node.timestep].timestamp_utc,
                sighting_id=sightings[first_node.timestep].sighting_id,
                camera_id=sightings[first_node.timestep].camera_id,
                emission_logprob=first_node.emission,
                cumulative_distance_m=0.0,
                heading_deg=first_node.segment.heading_deg,
            )
        )
        sequence_index += 1

        travelled_m = 0.0
        for previous_node, current_node in zip(matched, matched[1:]):
            previous_segment = previous_node.segment
            current_segment = current_node.segment

            leg_m = self.travel_distance(previous_node.candidate, current_node.candidate)
            if math.isfinite(leg_m):
                travelled_m += abs(leg_m)

            if previous_segment.segment_id != current_segment.segment_id:
                bridge = self._graph.shortest_path_segments(
                    previous_segment.to_node, current_segment.from_node
                )
                if bridge is None:
                    contiguous = False
                    notes.append(
                        f"no directed path from {previous_segment.segment_id} to "
                        f"{current_segment.segment_id}; path is discontinuous here"
                    )
                elif len(bridge) > self._viterbi.max_path_segments_per_gap:
                    contiguous = False
                    notes.append(
                        f"gap between {previous_segment.segment_id} and "
                        f"{current_segment.segment_id} needed {len(bridge)} segments, "
                        f"above the cap of {self._viterbi.max_path_segments_per_gap}"
                    )
                else:
                    for segment in bridge:
                        append_segment(segment)
                        waypoints.append(
                            TrajectoryWaypoint(
                                sequence_index=sequence_index,
                                location=segment.to_point,
                                segment_id=segment.segment_id,
                                street_name=segment.street_name,
                                is_observed=False,
                                cumulative_distance_m=travelled_m,
                                heading_deg=segment.heading_deg,
                            )
                        )
                        sequence_index += 1
                append_segment(current_segment)

            sighting = sightings[current_node.timestep]
            waypoints.append(
                TrajectoryWaypoint(
                    sequence_index=sequence_index,
                    location=current_node.candidate.projection,
                    segment_id=current_segment.segment_id,
                    street_name=current_segment.street_name,
                    is_observed=True,
                    timestamp_utc=sighting.timestamp_utc,
                    sighting_id=sighting.sighting_id,
                    camera_id=sighting.camera_id,
                    emission_logprob=current_node.emission,
                    cumulative_distance_m=travelled_m,
                    heading_deg=current_segment.heading_deg,
                )
            )
            sequence_index += 1

        return path_segments, waypoints, notes, contiguous, travelled_m
