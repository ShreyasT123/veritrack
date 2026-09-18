"""Stage 4 behavioral tests: H3, O-D, CPI and streaming integration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from veritrack_analytics import (
    AnalyticsConfig,
    AnalyticsSighting,
    CongestionLevel,
    CorridorTraversal,
    CpiThresholds,
    H3SpatialIndex,
    MacroAnalyticsEngine,
    ODMatrixAccumulator,
    TripSessionizer,
    classify_cpi,
    corridor_metric,
    space_mean_speed_kph,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
MUMBAI_A = (19.0760, 72.8777)
MUMBAI_B = (19.0900, 72.8900)
MUMBAI_C = (19.1000, 72.9000)


def sighting(vehicle: str, camera: str, offset_min: int, point: tuple[float, float], *, exit: bool = False,
             corridor: str | None = None) -> AnalyticsSighting:
    return AnalyticsSighting(vehicle, camera, T0 + timedelta(minutes=offset_min), point[0], point[1], exit, corridor)


def traversal(vehicle: str, start_min: int, travel_min: int, *, corridor: str = "C1", length_km: float = 1.0) -> CorridorTraversal:
    entered = T0 + timedelta(minutes=start_min)
    return CorridorTraversal(corridor, vehicle, entered, entered + timedelta(minutes=travel_min), length_km)


def test_h3_binning_neighbor_ring_and_geojson_polygon_are_valid() -> None:
    spatial = H3SpatialIndex()
    cell = spatial.cell(*MUMBAI_A, 8)
    assert spatial.resolution(cell) == 8
    assert cell in spatial.k_ring(cell, 1)
    assert len(spatial.k_ring(cell, 1)) == 7
    feature = spatial.polygon_feature(cell, {"count": 4})
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    ring = feature["geometry"]["coordinates"][0]
    assert len(ring) == 7
    assert ring[0] == ring[-1]
    assert feature["properties"]["count"] == 4
    longitude, latitude = ring[0]
    assert -180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0


def test_sessionizer_closes_trip_when_gap_exceeds_25_minute_timeout() -> None:
    sessions = TripSessionizer.sessionize(
        [sighting("hash-a", "cam-a", 0, MUMBAI_A), sighting("hash-a", "cam-b", 5, MUMBAI_B),
         sighting("hash-a", "cam-c", 31, MUMBAI_C)], timedelta(minutes=25),
    )
    assert len(sessions) == 2
    assert sessions[0].completed is True
    assert sessions[0].termination_reason == "dwell_timeout"
    assert sessions[0].origin.camera_id == "cam-a"
    assert sessions[0].destination.camera_id == "cam-b"
    assert sessions[1].completed is False


def test_sessionizer_perimeter_exit_completes_trip_immediately() -> None:
    sessionizer = TripSessionizer(timedelta(minutes=25))
    assert sessionizer.ingest(sighting("hash-a", "cam-a", 0, MUMBAI_A)) == ()
    completed = sessionizer.ingest(sighting("hash-a", "cordon", 4, MUMBAI_B, exit=True))
    assert len(completed) == 1
    assert completed[0].termination_reason == "perimeter_exit"
    assert completed[0].duration_seconds == 240.0


def test_od_matrix_volumes_probabilities_and_duration_are_correct() -> None:
    spatial = H3SpatialIndex()
    accumulator = ODMatrixAccumulator(spatial, 7)
    sessionizer = TripSessionizer(timedelta(minutes=25))
    events = [
        sighting("one", "a", 0, MUMBAI_A), sighting("one", "b", 10, MUMBAI_B, exit=True),
        sighting("two", "a", 2, MUMBAI_A), sighting("two", "c", 22, MUMBAI_C, exit=True),
        sighting("three", "a", 3, MUMBAI_A), sighting("three", "b", 13, MUMBAI_B, exit=True),
    ]
    for event in events:
        for session in sessionizer.ingest(event):
            accumulator.add(session)
    result = accumulator.result(T0 + timedelta(hours=1), timedelta(hours=1))
    assert result.completed_trips == 3
    assert sum(entry.volume for entry in result.entries) == 3
    probabilities = [entry.probability for entry in result.entries]
    assert sum(probabilities) == pytest.approx(1.0)
    to_b = next(entry for entry in result.entries if entry.destination_h3 == spatial.cell(*MUMBAI_B, 7))
    assert to_b.volume == 2
    assert to_b.probability == pytest.approx(2.0 / 3.0)
    assert to_b.mean_duration_seconds == pytest.approx(600.0)


def test_space_mean_speed_is_harmonic_and_not_arithmetic_mean() -> None:
    samples = [traversal("fast", 0, 1), traversal("slow", 0, 4)]
    # Individual speeds: 60 and 15 km/h. Arithmetic mean is 37.5; physical
    # space mean is total 2 km / total 5 minutes = 24 km/h.
    assert space_mean_speed_kph(samples) == pytest.approx(24.0)


@pytest.mark.parametrize(("cpi", "expected"), [
    (0.80, CongestionLevel.NOMINAL), (0.50, CongestionLevel.ADAPTIVE_SIGNAL),
    (0.25, CongestionLevel.BOTTLENECK_WARNING), (0.249, CongestionLevel.CRITICAL_CONGESTION),
])
def test_cpi_classifies_all_four_congestion_levels(cpi: float, expected: CongestionLevel) -> None:
    assert classify_cpi(cpi, CpiThresholds()) is expected


def test_bottleneck_wave_needs_both_low_cpi_and_volume_above_capacity() -> None:
    # Four 1 km traversals in 15 minutes at 20 km/h; demand is 16 vph.
    samples = [traversal(f"v{i}", i, 3) for i in range(4)]
    metric = corridor_metric("C1", samples, free_flow_speed_kph=60.0, window_start=T0,
                              window_end=T0 + timedelta(minutes=15), thresholds=CpiThresholds(),
                              outflow_capacity_vph=10.0)
    assert metric.cpi == pytest.approx(1.0 / 3.0)
    assert metric.level is CongestionLevel.BOTTLENECK_WARNING
    assert metric.wave_detected is True
    assert metric.estimated_queue_vehicles == pytest.approx(1.5)


def test_engine_emits_density_od_and_corridor_reports_from_stream() -> None:
    config = AnalyticsConfig(
        corridor_lengths_km={"C1": 1.0}, corridor_free_flow_kph={"C1": 60.0},
        corridor_capacity_vph={"C1": 10.0}, cpi_window_minutes=15,
    )
    engine = MacroAnalyticsEngine(config)
    engine.ingest_sighting(sighting("hash-1", "cam-a", 0, MUMBAI_A, corridor="C1"))
    engine.ingest_sighting(sighting("hash-1", "cam-b", 3, MUMBAI_B, corridor="C1", exit=True))
    heatmap = engine.get_density_heatmap(now=T0 + timedelta(minutes=5))
    assert heatmap["type"] == "FeatureCollection"
    assert sum(feature["properties"]["count"] for feature in heatmap["features"]) == 2
    od = engine.get_od_flow_matrix(now=T0 + timedelta(minutes=5))
    assert od.completed_trips == 1
    report = engine.get_corridor_cpi_report(now=T0 + timedelta(minutes=5))
    assert len(report) == 1
    assert report[0].space_mean_speed_kph == pytest.approx(20.0)
    assert report[0].level is CongestionLevel.BOTTLENECK_WARNING


def test_stage2_row_adapter_uses_camera_registry_and_epoch_milliseconds() -> None:
    engine = MacroAnalyticsEngine()
    engine.ingest_stage2_row({"camera_id": "cam-a", "plate_pseudonym": "dayhash", "ts_first": 1767254400000},
                             {"cam-a": MUMBAI_A})
    heatmap = engine.get_density_heatmap(now=datetime(2026, 1, 1, 8, 1, tzinfo=UTC))
    assert len(heatmap["features"]) == 1
