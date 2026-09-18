"""Verification for CAP dispatch, audit integrity, and command-console APIs."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import FastAPI

from veritrack_analytics import AnalyticsConfig, AnalyticsSighting, MacroAnalyticsEngine
from veritrack_server.api_routes import UnifiedApiState, router
from veritrack_server.audit import AuditLedger
from veritrack_server.dispatch import CapAlert, EmergencyDispatcher


def test_cap_payload_has_required_v12_fields_and_millisecond_utc_time() -> None:
    alert = CapAlert("CLONED_PLATE_DETECTED", "ZG7497-AH", 28.443, 77.0865, "Sector 54 Chowk", "Implied speed exceeds safe limit.",
                     "s3://bucket/crop.jpg", sent_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    payload = alert.to_cap_json()
    assert payload["identifier"] == alert.identifier and payload["sender"].endswith("delhi_police")
    assert payload["sent"].endswith(".000Z") and payload["status"] == "Actual"
    info = payload["info"][0]
    assert info["category"] == ["Security"] and info["event"] == "CLONED_PLATE_DETECTED"
    assert info["urgency"] == "Immediate" and info["area"][0]["areaDesc"] == "Sector 54 Chowk"
    assert info["resource"][0]["uri"] == "s3://bucket/crop.jpg"


@pytest.mark.asyncio
async def test_dispatcher_retries_then_delivers_and_dead_letters_timeouts() -> None:
    calls = 0
    async def flaky(_: dict, __: float) -> None:
        nonlocal calls
        calls += 1
        if calls < 3: raise RuntimeError("temporary ERSS outage")
    dispatcher = EmergencyDispatcher("http://erss.invalid", max_retries=3, base_backoff_s=.001, sender=flaky)
    await dispatcher.start(); await dispatcher.submit(CapAlert("HOTLIST_WARRANT_HIT", "HR26DK4821", 28.47, 77.07, "IFFCO Chowk", "Warrant hit.")); await asyncio.sleep(.03); await dispatcher.stop()
    assert calls == 3 and dispatcher.delivered[0].attempts == 3 and not dispatcher.dead_letters
    async def slow(_: dict, __: float) -> None: await asyncio.sleep(.03)
    failed = EmergencyDispatcher("http://erss.invalid", timeout_s=.01, max_retries=1, base_backoff_s=.001, sender=slow)
    await failed.start(); await failed.submit(CapAlert("HOTLIST_WARRANT_HIT", "HR26DK4821", 28.47, 77.07, "IFFCO Chowk", "Warrant hit.")); await asyncio.sleep(.04); await failed.stop()
    assert len(failed.dead_letters) == 1 and failed.dead_letters[0].attempts == 2


def test_audit_hash_chain_verifies_and_detects_tampering() -> None:
    ledger = AuditLedger(); ledger.append("officer-7", "ZG7497-AH", "FIR-492", timestamp_utc=datetime(2026, 1, 1, tzinfo=timezone.utc)); ledger.append("officer-8", "HR26DK4821", "FIR-493", timestamp_utc=datetime(2026, 1, 2, tzinfo=timezone.utc))
    assert ledger.verify_audit_integrity() == (True, 2)
    ledger._entries[1] = replace(ledger.entries()[1], warrant_reference="FORGED")  # type: ignore[attr-defined]
    assert ledger.verify_audit_integrity() == (False, 2)


@pytest.mark.asyncio
async def test_unified_endpoints_return_live_state() -> None:
    app = FastAPI(); app.include_router(router)
    analytics = MacroAnalyticsEngine(AnalyticsConfig(corridor_lengths_km={"MG Road": 1.0}, corridor_free_flow_kph={"MG Road": 50.0}))
    t = datetime(2026, 1, 1, 8, tzinfo=timezone.utc)
    analytics.ingest_sighting(AnalyticsSighting("hash", "cam-a", t, 28.47, 77.07, corridor_id="MG Road")); analytics.ingest_sighting(AnalyticsSighting("hash", "cam-b", t.replace(minute=3), 28.48, 77.08, True, "MG Road"))
    async def trajectory(_: str) -> dict: return {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [77.07, 28.47]}, "properties": {}}]}
    state = UnifiedApiState(analytics=analytics, trajectory_provider=trajectory); state.record_telemetry({"plate": "ZG7497-AH", "camera_id": "cam-a", "timestamp": t.isoformat(), "confidence": 94})
    await state.record_alert(CapAlert("CLONED_PLATE_DETECTED", "ZG7497-AH", 28.47, 77.07, "IFFCO Chowk", "Clone.")); app.state.stage5 = state
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/api/v1/trajectories/ZG7497-AH")).json()["type"] == "FeatureCollection"
        assert (await client.get("/api/v1/analytics/corridors")).status_code == 200
        assert (await client.get("/api/v1/analytics/od-matrix")).json()["completed_trips"] == 1
        assert (await client.get("/api/v1/alerts/active")).json()[0]["event"] == "CLONED_PLATE_DETECTED"
        assert (await client.get("/api/v1/telemetry/recent")).json()[0]["plate"] == "ZG7497-AH"
    assert state.audit.verify_audit_integrity() == (True, 1)
