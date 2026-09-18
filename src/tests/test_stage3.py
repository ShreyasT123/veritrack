"""Stage 3 verification suite.

Runs entirely offline: the road network is a synthetic grid, so every geometric
and probabilistic claim is checked against a topology whose ground truth is
known by construction.

    pytest tests/test_stage3.py -v
"""


import math
from datetime import datetime, timedelta, timezone
from typing import Optional,  Tuple

import pytest

from veritrack_trajectory.anomaly import AnomalyDetector
from veritrack_trajectory.config import (
    FusionConfig,
    GraphConfig,
    TrajectoryConfig,
    ViterbiConfig,
)
from veritrack_trajectory.engine import TrajectoryEngine
from veritrack_trajectory.fusion import (
    ConfidenceFusion,
    cosine_similarity,
    damerau_levenshtein,
    kinematic_similarity,
    text_similarity,
    visual_similarity,
)
from veritrack_trajectory.graph import (
    UNREACHABLE,
    GraphError,
    RoadNetworkGraph,
    build_synthetic_grid,
)
from veritrack_trajectory.types import (
    AnomalySeverity,
    AnomalyType,
    CameraSighting,
    GeoPoint,
    angular_difference_deg,
    haversine_m,
)
from veritrack_trajectory.viterbi import (
    NEG_INF,
    ViterbiMapMatcher,
    emission_logprob,
    transition_logprob,
)

BASE_TIME = datetime(2026, 9, 18, 9, 0, 0, tzinfo=timezone.utc)
SPACING_M = 500.0


# =====================================================================
# Fixtures and builders
# =====================================================================


def unit_embedding(seed: int, dimension: int = 128) -> Tuple[float, ...]:
    """A deterministic 128-D unit vector."""
    raw = [math.sin(seed * 1.7 + index * 0.37) for index in range(dimension)]
    norm = math.sqrt(sum(value * value for value in raw))
    return tuple(value / norm for value in raw)


def perturbed_embedding(seed: int, noise: float, dimension: int = 128) -> Tuple[float, ...]:
    """The same vehicle seen again: small per-component noise, renormalised."""
    base = unit_embedding(seed, dimension)
    raw = [
        value + noise * math.sin(index * 2.11 + seed)
        for index, value in enumerate(base)
    ]
    norm = math.sqrt(sum(value * value for value in raw))
    return tuple(value / norm for value in raw)


@pytest.fixture()
def grid() -> RoadNetworkGraph:
    """A 6x6 bidirectional Manhattan grid, 500 m spacing."""
    return build_synthetic_grid(6, 6, spacing_m=SPACING_M)


@pytest.fixture()
def config() -> TrajectoryConfig:
    return TrajectoryConfig()


@pytest.fixture()
def fusion(config: TrajectoryConfig) -> ConfidenceFusion:
    return ConfidenceFusion(config)


@pytest.fixture()
def engine(grid: RoadNetworkGraph, config: TrajectoryConfig) -> TrajectoryEngine:
    return TrajectoryEngine(grid, config)


def sighting_at(
    graph: RoadNetworkGraph,
    node_id: str,
    *,
    offset_s: float,
    sighting_id: str,
    plate: str = "MH12AB1234",
    confidence: float = 0.95,
    embedding: Optional[Tuple[float, ...]] = None,
    heading: Optional[float] = 90.0,
    camera_id: Optional[str] = None,
    lateral_offset_m: float = 0.0,
    vehicle_class: str = "car",
) -> CameraSighting:
    """Place a camera at (or just beside) a grid node."""
    point = graph.node_point(node_id)
    if lateral_offset_m:
        # Shift north by a few metres so the sighting is near the road rather
        # than exactly on the node, which is the realistic case for a pole.
        delta_lat = math.degrees(lateral_offset_m / 6_371_008.8)
        point = GeoPoint(latitude=point.latitude + delta_lat, longitude=point.longitude)
    return CameraSighting(
        sighting_id=sighting_id,
        camera_id=camera_id or f"cam-{node_id}",
        timestamp_utc=BASE_TIME + timedelta(seconds=offset_s),
        location=point,
        plate_pseudonym=plate,
        plate_text=plate,
        plate_sequence_confidence=confidence,
        reid_embedding=embedding if embedding is not None else unit_embedding(7),
        vehicle_class=vehicle_class,
        travel_heading_azimuth=heading,
    )


# =====================================================================
# 1. Geodesy and graph
# =====================================================================


def test_synthetic_grid_topology(grid: RoadNetworkGraph) -> None:
    assert grid.node_count == 36
    # 6x6 grid: 30 horizontal + 30 vertical undirected edges, doubled by
    # bidirectionality.
    assert grid.edge_count == 120


def test_grid_spacing_is_metric(grid: RoadNetworkGraph) -> None:
    """The longitude step must shrink by cos(lat) or the grid is not square."""
    east = grid.node_point("n0_0").distance_to(grid.node_point("n0_1"))
    north = grid.node_point("n0_0").distance_to(grid.node_point("n1_0"))
    assert east == pytest.approx(SPACING_M, rel=1e-3)
    assert north == pytest.approx(SPACING_M, rel=1e-3)


def test_angular_difference_wraps_at_north() -> None:
    """359 and 1 degrees differ by 2, not 358."""
    assert angular_difference_deg(359.0, 1.0) == pytest.approx(2.0)
    assert angular_difference_deg(1.0, 359.0) == pytest.approx(2.0)
    assert angular_difference_deg(0.0, 180.0) == pytest.approx(180.0)
    assert angular_difference_deg(90.0, 270.0) == pytest.approx(180.0)


def test_network_distance_is_manhattan_not_euclidean(grid: RoadNetworkGraph) -> None:
    """Two east, two north on a grid is 4 blocks of road, not 2.83 of straight line."""
    network = grid.network_distance("n0_0", "n2_2")
    straight = grid.node_point("n0_0").distance_to(grid.node_point("n2_2"))
    assert network == pytest.approx(4 * SPACING_M, rel=1e-6)
    # Straight line across a 2x2 block diagonal is sqrt(8)/2 blocks ~= 1.414 km.
    assert straight == pytest.approx(math.sqrt(8) * SPACING_M, rel=0.01)
    assert network > straight, "road distance must exceed straight-line distance on a grid"


def test_astar_matches_dijkstra(grid: RoadNetworkGraph) -> None:
    """The haversine heuristic is admissible, so it cannot change the answer."""
    astar_graph = build_synthetic_grid(
        6, 6, spacing_m=SPACING_M, config=GraphConfig(use_astar_heuristic=True)
    )
    dijkstra_graph = build_synthetic_grid(
        6, 6, spacing_m=SPACING_M, config=GraphConfig(use_astar_heuristic=False)
    )
    for target in ("n1_1", "n3_2", "n5_5", "n2_5"):
        assert astar_graph.network_distance("n0_0", target) == pytest.approx(
            dijkstra_graph.network_distance("n0_0", target), rel=1e-9
        )


def test_distance_cache_serves_repeat_queries(grid: RoadNetworkGraph) -> None:
    grid.network_distance("n0_0", "n5_5")
    for _ in range(50):
        grid.network_distance("n0_0", "n5_5")
    stats = grid.cache_stats()
    assert stats["hits"] >= 50
    assert stats["hit_rate"] > 0.9


def test_cache_is_invalidated_on_mutation(grid: RoadNetworkGraph) -> None:
    """A stale cache after a road opens would report a distance the graph no longer has."""
    before = grid.network_distance("n0_0", "n1_1")
    grid.add_node("shortcut", grid.node_point("n0_0").latitude, grid.node_point("n0_0").longitude)
    grid.add_edge("n0_0", "shortcut", length_m=10.0)
    grid.add_edge("shortcut", "n1_1", length_m=10.0)
    after = grid.network_distance("n0_0", "n1_1")
    assert after < before
    assert after == pytest.approx(20.0)


def test_one_way_grid_makes_reverse_travel_unreachable() -> None:
    """A strictly one-way lattice is the fixture for anomaly Condition C."""
    oneway = build_synthetic_grid(4, 4, spacing_m=SPACING_M, bidirectional=False)
    assert math.isfinite(oneway.network_distance("n0_0", "n3_3"))
    assert not math.isfinite(oneway.network_distance("n3_3", "n0_0"))
    assert oneway.is_reachable("n0_0", "n3_3")
    assert not oneway.is_reachable("n3_3", "n0_0")


def test_candidate_pruning_respects_radius(grid: RoadNetworkGraph) -> None:
    node = grid.node_point("n2_2")
    near = grid.candidate_edges(node, radius_m=50.0, observed_heading_deg=90.0)
    assert near, "a node sits on its own incident segments"
    assert all(c.perpendicular_distance_m <= 50.0 for c in near)

    # The centre of a city block is 250 m from each of its four bounding
    # roads. Note the midpoint of two *adjacent* nodes would not do: that point
    # lies exactly on the street joining them.
    block_centre = GeoPoint(
        latitude=(grid.node_point("n2_2").latitude + grid.node_point("n3_2").latitude) / 2,
        longitude=(grid.node_point("n2_2").longitude + grid.node_point("n2_3").longitude) / 2,
    )
    assert grid.candidate_edges(block_centre, radius_m=50.0) == []
    assert grid.candidate_edges(block_centre, radius_m=300.0)


def test_nearest_node_is_exact_against_brute_force(grid: RoadNetworkGraph) -> None:
    """The KD-tree is an optimisation, not an approximation.

    It replaced a linear scan that dominated Stage 3's wall clock. Speed is
    only worth having if the answer is identical, so this checks it against the
    definition rather than trusting the index.
    """
    import random

    random.seed(0)
    origin = grid.node_point("n0_0")
    for _ in range(200):
        probe = GeoPoint(
            latitude=origin.latitude + random.uniform(-0.005, 0.03),
            longitude=origin.longitude + random.uniform(-0.005, 0.03),
        )
        indexed, distance = grid.nearest_node(probe)
        brute = min(
            grid.graph.nodes,
            key=lambda node: haversine_m(
                probe.latitude,
                probe.longitude,
                grid.graph.nodes[node]["latitude"],
                grid.graph.nodes[node]["longitude"],
            ),
        )
        assert indexed == brute
        assert distance == pytest.approx(
            haversine_m(
                probe.latitude,
                probe.longitude,
                grid.graph.nodes[brute]["latitude"],
                grid.graph.nodes[brute]["longitude"],
            )
        )


def test_nearest_node_cache_is_invalidated_on_mutation(grid: RoadNetworkGraph) -> None:
    """A memoised nearest node must not survive a new node being added."""
    probe = grid.node_point("n0_0")
    assert grid.nearest_node(probe)[0] == "n0_0"
    grid.add_node("pole-x", probe.latitude + 1e-6, probe.longitude)
    grid.add_edge("pole-x", "n0_1", length_m=500.0)
    nearest, distance = grid.nearest_node(
        GeoPoint(latitude=probe.latitude + 1e-6, longitude=probe.longitude)
    )
    assert nearest == "pole-x"
    assert distance < 1.0


def test_candidate_max_cap_is_enforced(grid: RoadNetworkGraph) -> None:
    node = grid.node_point("n2_2")
    capped = grid.candidate_edges(node, radius_m=50.0, max_candidates=2)
    assert len(capped) <= 2


def test_self_loop_is_rejected(grid: RoadNetworkGraph) -> None:
    with pytest.raises(GraphError, match="self-loop"):
        grid.add_edge("n0_0", "n0_0", length_m=10.0)


def test_shortest_path_segments_are_contiguous(grid: RoadNetworkGraph) -> None:
    segments = grid.shortest_path_segments("n0_0", "n2_2")
    assert segments is not None
    assert len(segments) == 4
    for previous, current in zip(segments, segments[1:]):
        assert previous.to_node == current.from_node
    assert sum(s.length_m for s in segments) == pytest.approx(4 * SPACING_M)


def test_corridor_free_flow_is_length_weighted(grid: RoadNetworkGraph) -> None:
    """A 2 km arterial plus a 50 m slip road is an arterial, not their mean."""
    network = RoadNetworkGraph()
    network.add_node("a", 19.0, 73.0)
    network.add_node("b", 19.02, 73.0)
    network.add_node("c", 19.0201, 73.0)
    network.add_edge("a", "b", length_m=2000.0, speed_limit_kmh=60.0)
    network.add_edge("b", "c", length_m=50.0, speed_limit_kmh=20.0)
    weighted = network.corridor_free_flow_kmh("a", "c", fallback=45.0)
    arithmetic_mean = (60.0 + 20.0) / 2
    assert weighted == pytest.approx((60 * 2000 + 20 * 50) / 2050, rel=1e-9)
    assert weighted > 59.0
    assert weighted > arithmetic_mean


# =====================================================================
# 2. Dynamic weight shifting
# =====================================================================


def test_weights_sum_to_one_across_the_confidence_range(config: TrajectoryConfig) -> None:
    for step in range(0, 101):
        confidence = step / 100.0
        w_text, w_vis, w_kin = config.weights_for_confidence(confidence)
        assert w_text + w_vis + w_kin == pytest.approx(1.0, abs=1e-12)
        assert w_text >= 0.0 and w_vis >= 0.0 and w_kin >= 0.0


def test_weight_closed_form_matches_specification(config: TrajectoryConfig) -> None:
    """w_kin must equal 0.35 - 0.20 g, the algebraic consequence of the schedule."""
    for gate in (0.0, 0.17, 0.5, 0.83, 1.0):
        w_text, w_vis, w_kin = config.weights_for_gate(gate)
        assert w_text == pytest.approx(gate * 0.70)
        assert w_vis == pytest.approx((1.0 - gate) * 0.65 + gate * 0.15)
        assert w_kin == pytest.approx(0.35 - 0.20 * gate)


def test_high_confidence_shifts_weight_to_text(fusion: ConfidenceFusion) -> None:
    w_text, w_vis, w_kin = fusion.weights(0.99)
    assert w_text > 0.65
    assert w_text > w_vis and w_text > w_kin
    assert w_vis == pytest.approx(0.15, abs=0.01)


def test_low_confidence_shifts_weight_to_vision(fusion: ConfidenceFusion) -> None:
    w_text, w_vis, w_kin = fusion.weights(0.05)
    assert w_text < 0.01
    assert w_vis > 0.64
    assert w_vis > w_text and w_vis > w_kin


def test_gate_crosses_half_at_tau(config: TrajectoryConfig) -> None:
    assert config.gate(config.fusion.gate_threshold_tau) == pytest.approx(0.5)
    assert config.gate(0.0) < 0.01
    assert config.gate(1.0) > 0.98


def test_gate_is_monotonic_and_stable_at_the_tails(config: TrajectoryConfig) -> None:
    values = [config.gate(step / 200.0) for step in range(201)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    # A steep gate on a large negative exponent must not overflow.
    steep = TrajectoryConfig(fusion=FusionConfig(gate_steepness_k=500.0))
    assert steep.gate(0.0) == pytest.approx(0.0, abs=1e-12)
    assert steep.gate(1.0) == pytest.approx(1.0, abs=1e-12)


def test_kinematics_never_dominates(config: TrajectoryConfig) -> None:
    """Travel speed corroborates and vetoes; it must never assert identity alone."""
    for step in range(0, 101):
        w_text, w_vis, w_kin = config.weights_for_confidence(step / 100.0)
        assert w_kin <= 0.35
        assert w_kin < max(w_text, w_vis) + 1e-12


def test_pair_confidence_uses_the_weaker_read(
    grid: RoadNetworkGraph, fusion: ConfidenceFusion
) -> None:
    """Gating on the mean would trust a string one camera barely resolved."""
    strong = sighting_at(grid, "n0_0", offset_s=0, sighting_id="s1", confidence=0.98)
    weak = sighting_at(grid, "n0_1", offset_s=40, sighting_id="s2", confidence=0.30)
    score = fusion.score(strong, weak, network_distance_m=SPACING_M)
    assert score.gate == pytest.approx(fusion.gate(0.30))
    assert score.w_vis > score.w_text


def test_invalid_weight_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative weight|sum to 1"):
        TrajectoryConfig(fusion=FusionConfig(a_text=0.95, a_vis_low=0.65, a_vis_high=0.60))


# =====================================================================
# 3. Similarity channels
# =====================================================================


def test_text_similarity_is_one_for_exact_match() -> None:
    assert text_similarity("MH12AB1234", "MH12AB1234") == 1.0
    assert text_similarity("mh12ab1234", "MH12AB1234") == 1.0


def test_optical_confusion_costs_less_than_arbitrary_substitution() -> None:
    """8->B is a real ANPR error; 8->Q is not. They must not cost the same."""
    confusable = damerau_levenshtein("MH12AB1234", "MH12A81234")
    arbitrary = damerau_levenshtein("MH12AB1234", "MH12AQ1234")
    assert confusable == pytest.approx(0.30)
    assert arbitrary == pytest.approx(1.0)
    assert text_similarity("MH12AB1234", "MH12A81234") > text_similarity(
        "MH12AB1234", "MH12AQ1234"
    )


def test_transposition_is_a_single_edit() -> None:
    assert damerau_levenshtein("MH12AB1234", "MH12AB1243") == pytest.approx(1.0)


def test_text_similarity_is_symmetric_and_bounded() -> None:
    for a, b in [("MH12AB1234", "KA05MN7788"), ("AB", "ABCDEFGH"), ("", "MH12AB1234")]:
        assert text_similarity(a, b) == pytest.approx(text_similarity(b, a))
        assert 0.0 <= text_similarity(a, b) <= 1.0


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    vector = unit_embedding(3)
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero() -> None:
    a = tuple([1.0] + [0.0] * 127)
    b = tuple([0.0, 1.0] + [0.0] * 126)
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_visual_similarity_clamps_negative_cosine() -> None:
    a = unit_embedding(5)
    b = tuple(-value for value in a)
    assert cosine_similarity(a, b) == pytest.approx(-1.0)
    assert visual_similarity(a, b) == 0.0


def test_embedding_dimension_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity(unit_embedding(1), unit_embedding(1, dimension=64))


def test_kinematic_veto_above_the_ceiling() -> None:
    """The hard veto is what lets kinematics refute a perfect text+visual match."""
    score, speed, feasible = kinematic_similarity(
        10_000.0, 120.0, 45.0, max_velocity_kmh=140.0
    )
    assert speed == pytest.approx(300.0)
    assert score == 0.0
    assert feasible is False


def test_kinematic_peaks_at_the_free_flow_baseline() -> None:
    distance_m, delta_t = 450.0, 36.0  # exactly 45 km/h
    score, speed, feasible = kinematic_similarity(distance_m, delta_t, 45.0)
    assert speed == pytest.approx(45.0)
    assert score == pytest.approx(1.0)
    assert feasible is True


def test_kinematic_is_asymmetric_congestion_is_not_speeding() -> None:
    """A vehicle crawling in traffic is normal; one doing 75 in a 45 zone is not."""
    slow, _, _ = kinematic_similarity(150.0, 36.0, 45.0)   # 15 km/h
    fast, _, _ = kinematic_similarity(750.0, 36.0, 45.0)   # 75 km/h
    assert slow > fast
    assert slow > 0.75, "congestion must not be treated as improbable"
    symmetric_slow, _, _ = kinematic_similarity(150.0, 36.0, 45.0, symmetric=True)
    symmetric_fast, _, _ = kinematic_similarity(750.0, 36.0, 45.0, symmetric=True)
    assert symmetric_slow == pytest.approx(symmetric_fast, abs=1e-9)


def test_kinematic_rejects_non_positive_elapsed_time() -> None:
    score, _, feasible = kinematic_similarity(500.0, 0.0, 45.0)
    assert score == 0.0 and feasible is False


def test_kinematic_handles_unreachable_distance() -> None:
    score, speed, feasible = kinematic_similarity(UNREACHABLE, 60.0, 45.0)
    assert score == 0.0 and feasible is False and math.isinf(speed)


# =====================================================================
# 4. HMM emission and transition
# =====================================================================


def test_emission_decays_with_perpendicular_distance() -> None:
    near = emission_logprob(1.0, 0.0, sigma_gps_m=12.0)
    far = emission_logprob(40.0, 0.0, sigma_gps_m=12.0)
    assert near > far


def test_emission_penalises_heading_mismatch() -> None:
    aligned = emission_logprob(5.0, 0.0, sigma_gps_m=12.0)
    oblique = emission_logprob(5.0, 60.0, sigma_gps_m=12.0)
    assert aligned > oblique


def test_emission_is_finite_at_perpendicular_heading() -> None:
    """log(max(0, cos 90)) = -inf would annihilate an otherwise good path."""
    perpendicular = emission_logprob(5.0, 90.0, sigma_gps_m=12.0, heading_floor=1e-4)
    assert math.isfinite(perpendicular)
    assert perpendicular < emission_logprob(5.0, 0.0, sigma_gps_m=12.0)
    opposed = emission_logprob(5.0, 180.0, sigma_gps_m=12.0, heading_floor=1e-4)
    assert math.isfinite(opposed)


def test_emission_matches_the_closed_form() -> None:
    d_perp, sigma, d_theta = 8.0, 12.0, 30.0
    expected = (
        (1.0 / (math.sqrt(2.0 * math.pi) * sigma))
        * math.exp(-(d_perp**2) / (2.0 * sigma**2))
        * math.cos(math.radians(d_theta))
    )
    assert emission_logprob(d_perp, d_theta, sigma_gps_m=sigma) == pytest.approx(
        math.log(expected), rel=1e-9
    )


def test_reverse_heading_mode_folds_opposing_directions() -> None:
    opposed = emission_logprob(5.0, 180.0, sigma_gps_m=12.0, allow_reverse=True)
    aligned = emission_logprob(5.0, 0.0, sigma_gps_m=12.0, allow_reverse=True)
    assert opposed == pytest.approx(aligned)


def test_transition_peaks_when_distance_matches_expected_travel() -> None:
    """Zero discrepancy between network distance and speed-implied distance."""
    delta_t, speed_kmh = 40.0, 45.0
    expected_m = (speed_kmh / 3.6) * delta_t
    best = transition_logprob(expected_m, delta_t, speed_kmh, beta_m=120.0)
    assert best == pytest.approx(-math.log(120.0))
    worse = transition_logprob(expected_m + 400.0, delta_t, speed_kmh, beta_m=120.0)
    assert best > worse


def test_transition_decays_exponentially_in_the_discrepancy() -> None:
    delta_t, speed_kmh, beta = 40.0, 45.0, 120.0
    expected_m = (speed_kmh / 3.6) * delta_t
    one_beta = transition_logprob(expected_m + beta, delta_t, speed_kmh, beta_m=beta)
    two_beta = transition_logprob(expected_m + 2 * beta, delta_t, speed_kmh, beta_m=beta)
    assert (one_beta - two_beta) == pytest.approx(1.0, rel=1e-9)


def test_transition_rejects_unreachable_pairs() -> None:
    assert transition_logprob(UNREACHABLE, 60.0, 45.0, beta_m=120.0) == NEG_INF


def test_transition_is_finite_so_a_path_always_exists() -> None:
    """Finite NEG_INF keeps Viterbi recoverable instead of returning nothing."""
    value = transition_logprob(50_000.0, 1.0, 45.0, beta_m=120.0)
    assert math.isfinite(value)


# =====================================================================
# 5. Viterbi path recovery
# =====================================================================


def test_viterbi_recovers_a_known_straight_route(grid: RoadNetworkGraph) -> None:
    """Sightings along one avenue must match to that avenue's segments."""
    matcher = ViterbiMapMatcher(grid)
    sightings = [
        sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0", heading=90.0),
        sighting_at(grid, "n0_2", offset_s=80, sighting_id="s1", heading=90.0),
        sighting_at(grid, "n0_4", offset_s=160, sighting_id="s2", heading=90.0),
    ]
    result = matcher.match(sightings)
    assert result.is_contiguous
    assert result.matched_segments
    # 4 blocks of eastbound travel, all on 1st Avenue. Asserted on the
    # projection-to-projection travelled distance, not the sum of whole
    # matched segments -- the latter includes the tail of the final segment,
    # which the vehicle was never observed to traverse.
    assert result.travelled_distance_m == pytest.approx(4 * SPACING_M, rel=0.01)
    assert all(
        segment.street_name == "1th Avenue" for segment in result.path_segments
    )


def test_viterbi_fills_the_blind_spot_between_cameras(grid: RoadNetworkGraph) -> None:
    """The gap between two poles must be interpolated into real road segments."""
    matcher = ViterbiMapMatcher(grid)
    sightings = [
        sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0", heading=90.0),
        sighting_at(grid, "n0_5", offset_s=200, sighting_id="s1", heading=90.0),
    ]
    result = matcher.match(sightings)
    interpolated = [w for w in result.waypoints if not w.is_observed]
    observed = [w for w in result.waypoints if w.is_observed]
    assert len(observed) == 2, "both camera hits must survive as observed waypoints"
    assert interpolated, "the 2.5 km blind spot must be filled with inferred segments"
    assert result.travelled_distance_m == pytest.approx(5 * SPACING_M, rel=0.01)


def test_viterbi_recovers_an_l_shaped_turn(grid: RoadNetworkGraph) -> None:
    """Two east then two north must produce both street names in order."""
    matcher = ViterbiMapMatcher(grid)
    sightings = [
        sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0", heading=90.0),
        sighting_at(grid, "n0_2", offset_s=80, sighting_id="s1", heading=90.0),
        sighting_at(grid, "n2_2", offset_s=160, sighting_id="s2", heading=0.0),
    ]
    result = matcher.match(sightings)
    names = [segment.street_name for segment in result.path_segments]
    assert "1th Avenue" in names
    assert "3th Street" in names
    assert result.travelled_distance_m == pytest.approx(4 * SPACING_M, rel=0.01)


def test_viterbi_uses_heading_to_pick_the_right_direction(grid: RoadNetworkGraph) -> None:
    """Eastbound and westbound segments are co-located; only heading separates them."""
    matcher = ViterbiMapMatcher(grid)
    eastbound = matcher.match(
        [
            sighting_at(grid, "n0_0", offset_s=0, sighting_id="e0", heading=90.0),
            sighting_at(grid, "n0_2", offset_s=80, sighting_id="e1", heading=90.0),
        ]
    )
    for segment in eastbound.path_segments:
        assert angular_difference_deg(segment.heading_deg, 90.0) < 45.0


def test_viterbi_is_computed_in_log_space(grid: RoadNetworkGraph) -> None:
    """A long trajectory must not underflow to zero probability."""
    matcher = ViterbiMapMatcher(grid)
    sightings = [
        sighting_at(grid, f"n0_{column}", offset_s=column * 40, sighting_id=f"s{column}",
                    heading=90.0)
        for column in range(6)
    ]
    result = matcher.match(sightings)
    assert math.isfinite(result.total_logprob)
    assert result.total_logprob < 0.0
    assert result.total_logprob > NEG_INF


def test_viterbi_handles_a_camera_off_the_network(grid: RoadNetworkGraph) -> None:
    """One unmatched pole must not destroy an otherwise good trajectory."""
    matcher = ViterbiMapMatcher(grid)
    stranded = CameraSighting(
        sighting_id="s-off",
        camera_id="cam-offgrid",
        timestamp_utc=BASE_TIME + timedelta(seconds=40),
        location=GeoPoint(latitude=20.5, longitude=74.5),
        plate_pseudonym="MH12AB1234",
        plate_text="MH12AB1234",
        plate_sequence_confidence=0.9,
        reid_embedding=unit_embedding(7),
        travel_heading_azimuth=90.0,
    )
    result = matcher.match(
        [
            sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0", heading=90.0),
            stranded,
            sighting_at(grid, "n0_2", offset_s=80, sighting_id="s2", heading=90.0),
        ]
    )
    assert result.matched_segments
    assert any("no road segment within" in note for note in result.notes)
    assert result.matched_candidates[1] is None


def test_junction_tie_does_not_invent_a_detour(grid: RoadNetworkGraph) -> None:
    """A camera at a junction must not inflate the journey by a phantom block.

    At a junction the arriving and departing segments tie exactly on emission
    (both have zero perpendicular distance and zero heading error). If travel
    were measured node-to-node, the tie could break onto the departing segment
    and route the vehicle out to that segment's far node and back -- adding a
    full block that never happened. Projection-to-projection measurement makes
    both tied candidates yield the same distance, so the tie stops mattering.
    """
    matcher = ViterbiMapMatcher(grid)
    result = matcher.match(
        [
            sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0", heading=90.0),
            sighting_at(grid, "n0_2", offset_s=80, sighting_id="s1", heading=90.0),
            sighting_at(grid, "n2_2", offset_s=160, sighting_id="s2", heading=0.0),
        ]
    )
    # Two blocks east then two blocks north: exactly 2 km, never 2.5 or 3.0.
    assert result.travelled_distance_m == pytest.approx(2000.0, abs=1.0)


def test_travel_distance_is_projection_aware(grid: RoadNetworkGraph) -> None:
    """The arriving candidate must measure shorter than the departing one.

    Both tie on emission at a junction. Projection-aware travel is what breaks
    the tie on the right grounds: from the arriving segment the onward distance
    is the true 1 km, while from the departing segment it includes a spurious
    out-and-back to that segment's far node. The transition penalty then scores
    the honest figure against the elapsed time, and the detour loses.
    """
    matcher = ViterbiMapMatcher(grid)
    junction = grid.candidate_edges(grid.node_point("n0_2"), radius_m=50.0)
    arriving = next(c for c in junction if c.segment.segment_id == "n0_1->n0_2")
    departing = next(c for c in junction if c.segment.segment_id == "n0_2->n0_3")
    assert arriving.along_fraction == pytest.approx(1.0)
    assert departing.along_fraction == pytest.approx(0.0)

    destination = next(
        c for c in grid.candidate_edges(grid.node_point("n2_2"), radius_m=50.0)
        if c.segment.segment_id == "n1_2->n2_2"
    )
    from_arriving = matcher.travel_distance(arriving, destination)
    from_departing = matcher.travel_distance(departing, destination)

    # Two blocks north from the junction is exactly 1 km of real travel.
    assert from_arriving == pytest.approx(1000.0)
    # The departing candidate pays for the phantom out-and-back.
    assert from_departing == pytest.approx(2000.0)
    assert from_arriving < from_departing

    # And the transition model prefers the honest one: at 45 km/h over 80 s the
    # expected travel is 1000 m, so the arriving candidate has zero discrepancy.
    delta_t, speed = 80.0, 45.0
    honest = transition_logprob(from_arriving, delta_t, speed, beta_m=120.0)
    detour = transition_logprob(from_departing, delta_t, speed, beta_m=120.0)
    assert honest > detour


def test_travel_distance_within_one_segment(grid: RoadNetworkGraph) -> None:
    """Two cameras on the same stretch measure progress along it, not zero."""
    matcher = ViterbiMapMatcher(grid)
    start = grid.node_point("n0_0")
    quarter = GeoPoint(
        latitude=start.latitude,
        longitude=start.longitude + (grid.node_point("n0_1").longitude - start.longitude) * 0.25,
    )
    three_quarter = GeoPoint(
        latitude=start.latitude,
        longitude=start.longitude + (grid.node_point("n0_1").longitude - start.longitude) * 0.75,
    )
    left = next(
        c for c in grid.candidate_edges(quarter, radius_m=50.0, observed_heading_deg=90.0)
        if c.segment.segment_id == "n0_0->n0_1"
    )
    right = next(
        c for c in grid.candidate_edges(three_quarter, radius_m=50.0, observed_heading_deg=90.0)
        if c.segment.segment_id == "n0_0->n0_1"
    )
    assert matcher.travel_distance(left, right) == pytest.approx(0.5 * SPACING_M, rel=0.02)


def test_viterbi_on_empty_input_is_benign() -> None:
    matcher = ViterbiMapMatcher(build_synthetic_grid(3, 3))
    result = matcher.match([])
    assert result.matched_segments == []
    assert result.is_contiguous


def test_candidate_cap_bounds_the_trellis(grid: RoadNetworkGraph) -> None:
    tight = TrajectoryConfig(viterbi=ViterbiConfig(max_candidates_per_sighting=2))
    matcher = ViterbiMapMatcher(grid, tight)
    for sighting in [sighting_at(grid, "n2_2", offset_s=0, sighting_id="s0")]:
        assert len(matcher.candidates_for(sighting)) <= 2


# =====================================================================
# 6. Anomaly detection
# =====================================================================


def test_cloned_plate_detected_at_200_kmh(grid: RoadNetworkGraph) -> None:
    """2.5 km of road in 45 s is 200 km/h: one vehicle did not do that."""
    detector = AnomalyDetector(grid)
    left = sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0")
    right = sighting_at(grid, "n0_5", offset_s=45, sighting_id="s1")

    distance = grid.network_distance("n0_0", "n0_5")
    assert distance == pytest.approx(2500.0)
    implied = (distance / 45.0) * 3.6
    assert implied == pytest.approx(200.0, rel=1e-6)

    anomaly = detector.check_teleportation(left, right)
    assert anomaly is not None
    assert anomaly.anomaly_type is AnomalyType.CLONED_PLATE
    assert anomaly.implied_speed_kmh == pytest.approx(200.0, rel=1e-6)
    assert anomaly.severity >= AnomalySeverity.ELEVATED
    assert anomaly.evidence["condition"] == "A"


def test_lawful_speed_raises_no_anomaly(grid: RoadNetworkGraph) -> None:
    detector = AnomalyDetector(grid)
    left = sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0")
    right = sighting_at(grid, "n0_5", offset_s=200, sighting_id="s1")  # 45 km/h
    assert detector.check_teleportation(left, right) is None
    assert detector.analyse_pair(left, right) == []


def test_cloned_severity_scales_with_overshoot(grid: RoadNetworkGraph) -> None:
    detector = AnomalyDetector(grid)
    left = sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0")
    marginal = detector.check_teleportation(
        left, sighting_at(grid, "n0_5", offset_s=55, sighting_id="s1")  # ~164 km/h
    )
    extreme = detector.check_teleportation(
        left, sighting_at(grid, "n0_5", offset_s=15, sighting_id="s2")  # 600 km/h
    )
    assert marginal is not None and extreme is not None
    assert extreme.severity > marginal.severity
    assert extreme.severity is AnomalySeverity.CRITICAL


def test_teleportation_guard_on_tiny_elapsed_time(grid: RoadNetworkGraph) -> None:
    """Dividing by clock jitter manufactures velocities out of nothing."""
    detector = AnomalyDetector(grid)
    left = sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0")
    right = sighting_at(grid, "n0_5", offset_s=0.2, sighting_id="s1")
    assert detector.check_teleportation(left, right) is None


def test_teleportation_guard_on_low_ocr_confidence(grid: RoadNetworkGraph) -> None:
    """A clone accusation must not rest on a 0.3-confidence misread."""
    detector = AnomalyDetector(grid)
    left = sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0", confidence=0.30)
    right = sighting_at(grid, "n0_5", offset_s=45, sighting_id="s1", confidence=0.30)
    assert detector.check_teleportation(left, right) is None


def test_swapped_plate_detected_on_visual_divergence(grid: RoadNetworkGraph) -> None:
    """Same registration, different vehicle: invisible to any text-only system."""
    detector = AnomalyDetector(grid)
    left = sighting_at(
        grid, "n0_0", offset_s=0, sighting_id="s0",
        embedding=unit_embedding(11), vehicle_class="two_wheeler",
    )
    right = sighting_at(
        grid, "n0_1", offset_s=60, sighting_id="s1",
        embedding=unit_embedding(999), vehicle_class="truck",
    )
    similarity = cosine_similarity(left.reid_embedding, right.reid_embedding)
    assert similarity < 0.35

    anomaly = detector.check_visual_divergence(left, right)
    assert anomaly is not None
    assert anomaly.anomaly_type is AnomalyType.SWAPPED_PLATE
    assert anomaly.text_similarity == 1.0
    assert anomaly.reid_similarity == pytest.approx(similarity)
    assert anomaly.evidence["vehicle_class_changed"] is True


def test_same_vehicle_across_cameras_is_not_flagged(grid: RoadNetworkGraph) -> None:
    """Embedding noise from a second viewpoint must not look like a swap."""
    detector = AnomalyDetector(grid)
    left = sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0",
                       embedding=unit_embedding(11))
    right = sighting_at(grid, "n0_1", offset_s=60, sighting_id="s1",
                        embedding=perturbed_embedding(11, 0.12))
    assert cosine_similarity(left.reid_embedding, right.reid_embedding) > 0.35
    assert detector.check_visual_divergence(left, right) is None


def test_swapped_requires_exact_text_match(grid: RoadNetworkGraph) -> None:
    """Different numbers with different vehicles is two vehicles, not a swap."""
    detector = AnomalyDetector(grid)
    left = sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0",
                       plate="MH12AB1234", embedding=unit_embedding(11))
    right = sighting_at(grid, "n0_1", offset_s=60, sighting_id="s1",
                        plate="MH12AB9999", embedding=unit_embedding(999))
    assert detector.check_visual_divergence(left, right) is None


def test_swapped_needs_both_embeddings(grid: RoadNetworkGraph) -> None:
    """Absence of visual evidence is not evidence of a swap."""
    detector = AnomalyDetector(grid)
    left = sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0",
                       embedding=unit_embedding(11))
    right = CameraSighting(
        sighting_id="s1",
        camera_id="cam-b",
        timestamp_utc=BASE_TIME + timedelta(seconds=60),
        location=grid.node_point("n0_1"),
        plate_pseudonym="MH12AB1234",
        plate_text="MH12AB1234",
        plate_sequence_confidence=0.95,
        reid_embedding=(),
    )
    assert detector.check_visual_divergence(left, right) is None


def test_unreachable_transition_on_a_one_way_network() -> None:
    """Condition C: impossible at any speed, not merely too fast."""
    oneway = build_synthetic_grid(4, 4, spacing_m=SPACING_M, bidirectional=False)
    detector = AnomalyDetector(oneway)
    left = sighting_at(oneway, "n3_3", offset_s=0, sighting_id="s0")
    right = sighting_at(oneway, "n0_0", offset_s=600, sighting_id="s1")

    anomaly = detector.check_topological_impossibility(left, right)
    assert anomaly is not None
    assert anomaly.anomaly_type is AnomalyType.UNREACHABLE_TRANSITION
    assert math.isinf(anomaly.network_distance_m)
    assert anomaly.evidence["condition"] == "C"
    # The straight-line distance is reported so an operator can tell a genuine
    # disconnection from a horizon that is simply set too tight.
    assert anomaly.evidence["straight_line_m"] > 0.0


def test_conditions_a_and_c_are_mutually_exclusive() -> None:
    """One physical event must not be reported twice."""
    oneway = build_synthetic_grid(4, 4, spacing_m=SPACING_M, bidirectional=False)
    detector = AnomalyDetector(oneway)
    left = sighting_at(oneway, "n3_3", offset_s=0, sighting_id="s0")
    right = sighting_at(oneway, "n0_0", offset_s=5, sighting_id="s1")
    anomalies = detector.analyse_pair(left, right)
    types = {anomaly.anomaly_type for anomaly in anomalies}
    assert AnomalyType.UNREACHABLE_TRANSITION in types
    assert AnomalyType.CLONED_PLATE not in types


def test_sequence_analysis_reports_only_consecutive_pairs(grid: RoadNetworkGraph) -> None:
    detector = AnomalyDetector(grid)
    sightings = [
        sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0"),
        sighting_at(grid, "n0_5", offset_s=45, sighting_id="s1"),   # teleport
        sighting_at(grid, "n0_5", offset_s=400, sighting_id="s2"),
    ]
    anomalies = detector.analyse_sequence(sightings)
    assert len(anomalies) == 1
    summary = detector.summarise(anomalies)
    assert summary["total"] == 1
    assert summary["by_type"][AnomalyType.CLONED_PLATE.value] == 1


# =====================================================================
# 7. End-to-end engine
# =====================================================================


def test_engine_reconstructs_a_full_journey(engine: TrajectoryEngine,
                                            grid: RoadNetworkGraph) -> None:
    sightings = [
        sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0", heading=90.0),
        sighting_at(grid, "n0_2", offset_s=80, sighting_id="s1", heading=90.0),
        sighting_at(grid, "n2_2", offset_s=160, sighting_id="s2", heading=0.0),
    ]
    trajectory = engine.reconstruct(sightings)
    assert trajectory is not None
    assert trajectory.observed_sighting_count == 3
    assert trajectory.total_distance_km == pytest.approx(2.0, rel=0.02)
    assert trajectory.transit_time_s == pytest.approx(160.0)
    assert trajectory.average_speed_kmh == pytest.approx(45.0, rel=0.05)
    assert not trajectory.has_anomalies
    assert trajectory.is_contiguous
    assert "1th Avenue" in trajectory.street_names


def test_engine_reports_road_distance_not_straight_line(
    engine: TrajectoryEngine, grid: RoadNetworkGraph
) -> None:
    """The whole point of map matching: follow the road, not the crow."""
    sightings = [
        sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0", heading=90.0),
        sighting_at(grid, "n2_2", offset_s=240, sighting_id="s1", heading=0.0),
    ]
    trajectory = engine.reconstruct(sightings)
    assert trajectory is not None
    straight_km = grid.node_point("n0_0").distance_to(grid.node_point("n2_2")) / 1000.0
    assert trajectory.total_distance_km == pytest.approx(2.0, rel=0.02)
    assert trajectory.total_distance_km > straight_km * 1.3


def test_engine_geojson_is_well_formed(engine: TrajectoryEngine,
                                       grid: RoadNetworkGraph) -> None:
    sightings = [
        sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0", heading=90.0),
        sighting_at(grid, "n0_3", offset_s=120, sighting_id="s1", heading=90.0),
    ]
    trajectory = engine.reconstruct(sightings)
    assert trajectory is not None
    geojson = trajectory.to_geojson()

    assert geojson["type"] == "FeatureCollection"
    line = geojson["features"][0]
    assert line["geometry"]["type"] == "LineString"
    coordinates = line["geometry"]["coordinates"]
    assert len(coordinates) >= 2
    # GeoJSON is longitude-first. Navi Mumbai is roughly (73.03, 19.03), so a
    # transposed pair would put the route in the Indian Ocean.
    for longitude, latitude in coordinates:
        assert 72.0 < longitude < 74.0
        assert 18.0 < latitude < 20.0
    assert line["properties"]["street_names"]
    assert line["properties"]["total_distance_km"] > 0.0


def test_engine_flags_a_cloned_plate_end_to_end(engine: TrajectoryEngine,
                                                grid: RoadNetworkGraph) -> None:
    sightings = [
        sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0"),
        sighting_at(grid, "n0_5", offset_s=45, sighting_id="s1"),
    ]
    trajectory = engine.reconstruct(sightings)
    assert trajectory is not None
    assert trajectory.has_anomalies
    assert trajectory.anomalies[0].anomaly_type is AnomalyType.CLONED_PLATE
    assert trajectory.anomalies[0].implied_speed_kmh == pytest.approx(200.0, rel=1e-6)
    assert trajectory.max_severity is not None


def test_engine_splits_journeys_on_a_long_idle_gap(engine: TrajectoryEngine,
                                                   grid: RoadNetworkGraph) -> None:
    """A vehicle seen at 09:00 and 19:00 parked; it did not drive for ten hours."""
    sightings = [
        sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0", heading=90.0),
        sighting_at(grid, "n0_1", offset_s=40, sighting_id="s1", heading=90.0),
        sighting_at(grid, "n3_3", offset_s=36_000, sighting_id="s2", heading=90.0),
        sighting_at(grid, "n3_4", offset_s=36_040, sighting_id="s3", heading=90.0),
    ]
    journeys = engine.reconstruct_all(sightings)
    assert len(journeys) == 2
    assert all(journey.transit_time_s < 100.0 for journey in journeys)


def test_engine_groups_a_mixed_stream_by_identity(engine: TrajectoryEngine,
                                                  grid: RoadNetworkGraph) -> None:
    stream = [
        sighting_at(grid, "n0_0", offset_s=0, sighting_id="a0", plate="MH12AB1234"),
        sighting_at(grid, "n1_0", offset_s=40, sighting_id="b0", plate="KA05MN7788"),
        sighting_at(grid, "n0_1", offset_s=40, sighting_id="a1", plate="MH12AB1234"),
        sighting_at(grid, "n1_1", offset_s=80, sighting_id="b1", plate="KA05MN7788"),
    ]
    grouped = engine.reconstruct_by_identity(stream)
    assert set(grouped) == {"MH12AB1234", "KA05MN7788"}
    assert all(len(journeys) == 1 for journeys in grouped.values())


def test_trajectory_id_is_deterministic(engine: TrajectoryEngine,
                                        grid: RoadNetworkGraph) -> None:
    """Idempotent ids let the downstream store deduplicate on the primary key."""
    sightings = [
        sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0"),
        sighting_at(grid, "n0_1", offset_s=40, sighting_id="s1"),
    ]
    first = engine.reconstruct(sightings)
    second = engine.reconstruct(list(reversed(sightings)))
    assert first is not None and second is not None
    assert first.trajectory_id == second.trajectory_id


def test_engine_handles_a_single_sighting(engine: TrajectoryEngine,
                                          grid: RoadNetworkGraph) -> None:
    trajectory = engine.reconstruct(
        [sighting_at(grid, "n2_2", offset_s=0, sighting_id="solo")]
    )
    assert trajectory is not None
    assert trajectory.observed_sighting_count == 1
    assert trajectory.transit_time_s == 0.0
    assert trajectory.average_speed_kmh == 0.0
    assert trajectory.match_scores == ()


def test_engine_on_empty_input_returns_none(engine: TrajectoryEngine) -> None:
    assert engine.reconstruct([]) is None
    assert engine.reconstruct_all([]) == []


def test_engine_exposes_match_score_components(engine: TrajectoryEngine,
                                               grid: RoadNetworkGraph) -> None:
    """An analyst must be able to see *why* two sightings were linked."""
    sightings = [
        sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0"),
        sighting_at(grid, "n0_1", offset_s=40, sighting_id="s1"),
    ]
    trajectory = engine.reconstruct(sightings)
    assert trajectory is not None
    score = trajectory.match_scores[0]
    assert score.s_text == 1.0
    assert score.s_vis == pytest.approx(1.0)
    assert score.kinematically_feasible
    assert score.w_text + score.w_vis + score.w_kin == pytest.approx(1.0)
    explanation = engine.fusion.explain(score)
    assert "S=" in explanation and "km/h" in explanation


def test_engine_stats_accumulate(engine: TrajectoryEngine, grid: RoadNetworkGraph) -> None:
    engine.reconstruct(
        [
            sighting_at(grid, "n0_0", offset_s=0, sighting_id="s0"),
            sighting_at(grid, "n0_5", offset_s=45, sighting_id="s1"),
        ]
    )
    stats = engine.stats()
    assert stats["trajectories"] == 1
    assert stats["sightings"] == 2
    assert stats["anomalies"] == 1


def test_from_stage2_rows_adapts_dict_records(grid: RoadNetworkGraph) -> None:
    """Stage 3 consumes pseudonymised Stage 2 rows without needing cleartext."""
    point = grid.node_point("n0_0")
    rows = [
        {
            "pass_id": "abc123",
            "camera_id": "cam-a12",
            "timestamp_utc": BASE_TIME.isoformat(),
            "latitude": point.latitude,
            "longitude": point.longitude,
            "plate_pseudonym": "e60f189a342be370",
            "plate_sequence_confidence": 0.93,
            "reid_embedding": list(unit_embedding(7)),
            "vehicle_class": "car",
            "travel_heading_azimuth": 90.0,
            "corridor_id": "avenue-0",
        }
    ]
    sightings = TrajectoryEngine.from_stage2_rows(rows)
    assert len(sightings) == 1
    assert sightings[0].plate_text is None
    assert sightings[0].identity_key == "e60f189a342be370"
    assert len(sightings[0].reid_embedding) == 128


def test_from_stage2_rows_requires_coordinates() -> None:
    with pytest.raises(ValueError, match="no camera coordinates"):
        TrajectoryEngine.from_stage2_rows(
            [{"pass_id": "x", "camera_id": "c", "timestamp_utc": BASE_TIME.isoformat()}]
        )
