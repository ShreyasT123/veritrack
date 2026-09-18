"""Unified read API for trajectories, macro analytics, alerts, and console ticker."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

from fastapi import APIRouter, HTTPException, Query, Request

from veritrack_analytics import MacroAnalyticsEngine

from .audit import AuditLedger
from .dispatch import CapAlert, EmergencyDispatcher

__all__ = ["UnifiedApiState", "router"]

TrajectoryProvider = Callable[[str], Awaitable[Dict[str, Any]]]


async def _empty_trajectory(_: str) -> Dict[str, Any]:
    return {"type": "FeatureCollection", "features": []}


@dataclass(slots=True)
class UnifiedApiState:
    analytics: MacroAnalyticsEngine = field(default_factory=MacroAnalyticsEngine)
    audit: AuditLedger = field(default_factory=AuditLedger)
    emergency_dispatcher: Optional[EmergencyDispatcher] = None
    trajectory_provider: TrajectoryProvider = _empty_trajectory
    trajectory_engine: Optional[Any] = None
    trajectory_rows: Dict[str, Sequence[Mapping[str, Any]]] = field(default_factory=dict)
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    recent_telemetry: List[Dict[str, Any]] = field(default_factory=list)

    def record_telemetry(self, item: Mapping[str, Any]) -> None:
        self.recent_telemetry.insert(0, dict(item))
        del self.recent_telemetry[200:]

    async def record_alert(self, alert: CapAlert) -> None:
        payload = alert.to_cap_json()
        self.active_alerts.insert(0, {"identifier": alert.identifier, "event": alert.event, "plate": alert.plate,
                                      "sent": payload["sent"], "cap": payload, "active": True})
        del self.active_alerts[200:]
        if self.emergency_dispatcher is not None:
            await self.emergency_dispatcher.submit(alert)


router = APIRouter(prefix="/api/v1", tags=["command-console"])


def _state(request: Request) -> UnifiedApiState:
    value = getattr(request.app.state, "stage5", None)
    if not isinstance(value, UnifiedApiState):
        raise HTTPException(status_code=503, detail="Stage 5 services are not initialized")
    return value


@router.get("/trajectories/{plate}")
async def trajectory(plate: str, request: Request, officer_id: str = Query("console-operator"),
                     warrant_reference: str = Query("CONSOLE-REPLAY")) -> Dict[str, Any]:
    if not plate:
        raise HTTPException(status_code=422, detail="plate is required")
    state = _state(request)
    state.audit.append(officer_id, plate, warrant_reference)
    if state.trajectory_engine is not None:
        rows = state.trajectory_rows.get(plate, ())
        sightings = state.trajectory_engine.from_stage2_rows(rows)
        trajectories = state.trajectory_engine.reconstruct_all(sightings, identity_key=plate)
        return state.trajectory_engine.to_geojson(trajectories)
    return await state.trajectory_provider(plate)


@router.get("/analytics/corridors")
async def corridors(request: Request) -> List[Dict[str, Any]]:
    state = _state(request)
    return [{"corridor_id": item.corridor_id, "space_mean_speed_kph": item.space_mean_speed_kph,
             "free_flow_speed_kph": item.free_flow_speed_kph, "cpi": item.cpi, "level": item.level.value,
             "incoming_volume_vph": item.incoming_volume_vph, "wave_detected": item.wave_detected,
             "estimated_queue_vehicles": item.estimated_queue_vehicles} for item in state.analytics.get_corridor_cpi_report()]


@router.get("/analytics/od-matrix")
async def od_matrix(request: Request, time_window_hours: float = Query(1.0, gt=0.0, le=24.0)) -> Dict[str, Any]:
    result = _state(request).analytics.get_od_flow_matrix(time_window_hours)
    return {"window_start": result.window_start.isoformat(), "window_end": result.window_end.isoformat(),
            "completed_trips": result.completed_trips,
            "entries": [{"origin_h3": entry.origin_h3, "destination_h3": entry.destination_h3, "volume": entry.volume,
                         "probability": entry.probability, "mean_duration_seconds": entry.mean_duration_seconds,
                         "origin_centroid": entry.origin_centroid, "destination_centroid": entry.destination_centroid}
                        for entry in result.entries]}


@router.get("/alerts/active")
async def active_alerts(request: Request) -> List[Dict[str, Any]]:
    return _state(request).active_alerts


@router.get("/telemetry/recent")
async def recent_telemetry(request: Request) -> List[Dict[str, Any]]:
    return _state(request).recent_telemetry[:20]
