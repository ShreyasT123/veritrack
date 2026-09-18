"""Configuration for Stage 4 macro traffic analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

__all__ = ["CpiThresholds", "AnalyticsConfig"]


@dataclass(frozen=True, slots=True)
class CpiThresholds:
    """Strictly ordered Corridor Performance Index thresholds."""

    nominal: float = 0.80
    adaptive_signal: float = 0.50
    bottleneck_warning: float = 0.25
    wave_trigger: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 < self.bottleneck_warning < self.adaptive_signal < self.nominal:
            raise ValueError("CPI thresholds must satisfy 0 < bottleneck < adaptive < nominal")
        if not 0.0 < self.wave_trigger <= self.adaptive_signal:
            raise ValueError("wave_trigger must lie in (0, adaptive_signal]")


@dataclass(frozen=True, slots=True)
class AnalyticsConfig:
    """Validated settings for density, O-D, and corridor calculations.

    Corridor maps are keyed by a stable corridor identifier. A corridor can be
    analysed only after both its length and free-flow baseline are supplied.
    Capacity is optional: without it a CPI is still reported but a queue-wave
    forecast cannot be asserted.
    """

    h3_res_density: int = 8
    h3_res_od: int = 7
    dwell_timeout_minutes: int = 25
    density_window_minutes: int = 15
    od_window_minutes: int = 60
    cpi_window_minutes: int = 15
    retention_hours: int = 24
    cpi_thresholds: CpiThresholds = field(default_factory=CpiThresholds)
    corridor_lengths_km: Mapping[str, float] = field(default_factory=dict)
    corridor_free_flow_kph: Mapping[str, float] = field(default_factory=dict)
    corridor_capacity_vph: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.h3_res_density <= 15 or not 0 <= self.h3_res_od <= 15:
            raise ValueError("H3 resolutions must lie in [0, 15]")
        if self.dwell_timeout_minutes < 1:
            raise ValueError("dwell_timeout_minutes must be >= 1")
        if min(self.density_window_minutes, self.od_window_minutes, self.cpi_window_minutes) < 1:
            raise ValueError("all analytics windows must be at least one minute")
        if self.retention_hours < 1:
            raise ValueError("retention_hours must be at least one hour")
        for corridor_id, length in self.corridor_lengths_km.items():
            if not corridor_id or length <= 0.0:
                raise ValueError("corridor lengths require non-empty ids and positive kilometres")
        for corridor_id, speed in self.corridor_free_flow_kph.items():
            if not corridor_id or speed <= 0.0:
                raise ValueError("free-flow speeds require non-empty ids and positive kph")
        for corridor_id, capacity in self.corridor_capacity_vph.items():
            if not corridor_id or capacity <= 0.0:
                raise ValueError("capacities require non-empty ids and positive vehicles/hour")
