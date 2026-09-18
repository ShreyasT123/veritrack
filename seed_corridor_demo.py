"""Seed a deterministic Gurugram Stage 5 command-console replay dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from veritrack_analytics import AnalyticsConfig, AnalyticsSighting, MacroAnalyticsEngine
from veritrack_server.api_routes import UnifiedApiState
from veritrack_server.dispatch import CapAlert, EmergencyDispatcher

CAMERAS: Dict[str, Tuple[float, float]] = {
    "GMDA-CAM-SC-01": (28.5003, 77.0870), "GMDA-CAM-IFFCO-01": (28.4735, 77.0725),
    "GMDA-CAM-MGR-02": (28.4797, 77.0802), "GMDA-CAM-SIK-03": (28.4810, 77.0921),
    "GMDA-CAM-GEN-01": (28.4932, 77.0925), "GMDA-CAM-GCR-04": (28.4430, 77.0865),
    "GMDA-CAM-CC-02": (28.4941, 77.0885), "GMDA-CAM-SEC54-01": (28.4430, 77.0865),
}


async def seed(state: UnifiedApiState, *, erss_url: str | None = None) -> UnifiedApiState:
    """Populate background flows, legal transit, then the deterministic clone event."""
    if erss_url:
        dispatcher = EmergencyDispatcher(erss_url)
        await dispatcher.start()
        state.emergency_dispatcher = dispatcher
    config = AnalyticsConfig(corridor_lengths_km={"MG Road": 1.0, "Golf Course Road": 1.2, "Cyber City Underpass": 0.8},
                             corridor_free_flow_kph={"MG Road": 44.0, "Golf Course Road": 48.0, "Cyber City Underpass": 42.0},
                             corridor_capacity_vph={"MG Road": 80.0, "Golf Course Road": 65.0, "Cyber City Underpass": 70.0})
    state.analytics = MacroAnalyticsEngine(config)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    rng = random.Random(1122026)
    corridor_cameras = [("MG Road", "GMDA-CAM-IFFCO-01", "GMDA-CAM-MGR-02"),
                        ("Golf Course Road", "GMDA-CAM-SIK-03", "GMDA-CAM-GCR-04"),
                        ("Cyber City Underpass", "GMDA-CAM-GEN-01", "GMDA-CAM-CC-02")]
    for number in range(40):
        corridor, first, second = corridor_cameras[number % len(corridor_cameras)]
        start = now - timedelta(minutes=14, seconds=number * 9)
        vehicle = f"HR26BG{number:04d}"
        for camera, timestamp in ((first, start), (second, start + timedelta(seconds=rng.randint(65, 125)))):
            lat, lon = CAMERAS[camera]
            state.analytics.ingest_sighting(AnalyticsSighting(vehicle, camera, timestamp, lat, lon, corridor_id=corridor))
            state.record_telemetry({"plate": vehicle, "camera_id": camera, "timestamp": timestamp.isoformat(), "confidence": 92})
    lawful = ["GMDA-CAM-SC-01", "GMDA-CAM-IFFCO-01", "GMDA-CAM-MGR-02", "GMDA-CAM-CC-02"]
    for index, camera in enumerate(lawful):
        timestamp = now - timedelta(minutes=8 - index * 2)
        lat, lon = CAMERAS[camera]
        state.analytics.ingest_sighting(AnalyticsSighting("ZG7497-AH", camera, timestamp, lat, lon,
                                                           is_perimeter_exit=index == len(lawful) - 1, corridor_id="MG Road"))
        state.record_telemetry({"plate": "ZG7497-AH", "camera_id": camera, "timestamp": timestamp.isoformat(), "confidence": 94})
    clone_time = now - timedelta(seconds=30)
    lat, lon = CAMERAS["GMDA-CAM-SEC54-01"]
    state.analytics.ingest_sighting(AnalyticsSighting("ZG7497-AH", "GMDA-CAM-SEC54-01", clone_time, lat, lon, is_perimeter_exit=True))
    alert = CapAlert("CLONED_PLATE_DETECTED", "ZG7497-AH", lat, lon, "Sector 54 Chowk",
                     "Same plate observed 18 km from Cyber City 90 seconds later; implied speed 195 km/h.",
                     "s3://veritrack-evidence/zg7497-ah-sector54.jpg")
    await state.record_alert(alert)
    return state


async def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Gurugram Stage 5 replay data")
    parser.add_argument("--erss-url", default=None, help="optional ERSS CAP endpoint")
    args = parser.parse_args()
    state = await seed(UnifiedApiState(), erss_url=args.erss_url)
    print(json.dumps({"cameras": len(CAMERAS), "telemetry_passes": len(state.recent_telemetry), "active_alerts": len(state.active_alerts),
                      "corridors": [item.corridor_id for item in state.analytics.get_corridor_cpi_report()]}, indent=2))
    if state.emergency_dispatcher is not None:
        await asyncio.sleep(0.2)
        await state.emergency_dispatcher.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
