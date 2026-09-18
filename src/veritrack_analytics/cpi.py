"""Physically correct corridor speed, CPI, and queue-wave calculations."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Optional, Tuple

from .config import CpiThresholds
from .types import CongestionLevel, CorridorMetric, CorridorTraversal

__all__ = ["space_mean_speed_kph", "classify_cpi", "corridor_metric"]


def space_mean_speed_kph(traversals: Iterable[CorridorTraversal]) -> float:
    """Compute total distance divided by total travel time in km/h.

    For a common segment length ``L``, this reduces exactly to
    ``N * L / sum(delta_t_i)``. Unlike arithmetic averaging of observed
    speeds, it gives slower vehicles their physically correct influence.
    """
    items = tuple(traversals)
    if not items:
        return 0.0
    total_km = sum(item.length_km for item in items)
    total_hours = sum(item.travel_seconds for item in items) / 3600.0
    return total_km / total_hours if total_hours > 0.0 else 0.0


def classify_cpi(cpi: float, thresholds: CpiThresholds) -> CongestionLevel:
    if cpi < 0.0:
        raise ValueError("CPI cannot be negative")
    if cpi >= thresholds.nominal:
        return CongestionLevel.NOMINAL
    if cpi >= thresholds.adaptive_signal:
        return CongestionLevel.ADAPTIVE_SIGNAL
    if cpi >= thresholds.bottleneck_warning:
        return CongestionLevel.BOTTLENECK_WARNING
    return CongestionLevel.CRITICAL_CONGESTION


def corridor_metric(
    corridor_id: str,
    traversals: Iterable[CorridorTraversal],
    *,
    free_flow_speed_kph: float,
    window_start: datetime,
    window_end: datetime,
    thresholds: CpiThresholds,
    outflow_capacity_vph: Optional[float] = None,
) -> CorridorMetric:
    """Build one corridor report and forecast a propagating queue wave.

    Incoming volume is the number of completed traversals per window hour.
    A wave requires both a sub-0.40 CPI and demand above known capacity; this
    prevents declaring a queue from low-speed but low-volume local activity.
    """
    if free_flow_speed_kph <= 0.0:
        raise ValueError("free_flow_speed_kph must be positive")
    if window_end <= window_start:
        raise ValueError("window_end must be after window_start")
    if outflow_capacity_vph is not None and outflow_capacity_vph <= 0.0:
        raise ValueError("outflow_capacity_vph must be positive when supplied")
    items = tuple(traversals)
    speed = space_mean_speed_kph(items)
    cpi = speed / free_flow_speed_kph
    window_hours = (window_end - window_start).total_seconds() / 3600.0
    incoming = len(items) / window_hours
    wave = bool(outflow_capacity_vph is not None and cpi < thresholds.wave_trigger and incoming > outflow_capacity_vph)
    queue = max(0.0, incoming - outflow_capacity_vph) * window_hours if wave and outflow_capacity_vph else 0.0
    return CorridorMetric(
        corridor_id=corridor_id,
        window_start=window_start,
        window_end=window_end,
        traversals=len(items),
        space_mean_speed_kph=speed,
        free_flow_speed_kph=free_flow_speed_kph,
        cpi=cpi,
        level=classify_cpi(cpi, thresholds),
        incoming_volume_vph=incoming,
        outflow_capacity_vph=outflow_capacity_vph,
        wave_detected=wave,
        estimated_queue_vehicles=queue,
    )
