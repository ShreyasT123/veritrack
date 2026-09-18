"""VeriTrack Stage 2 -- high-throughput FastAPI ingestion gateway.

Request path
------------
``POST /api/v1/telemetry/ingest`` accepts either a single observation or a
batch envelope and runs each observation through four stages:

1. **Schema validation** -- Pydantic v2 rejects malformed or hostile payloads at
   the boundary, before a single byte reaches the crypto or storage layer.
2. **Hotlist check** -- one pipelined Bloom round trip for the whole batch,
   then an authoritative Redis hash confirmation for the few hits.
3. **DPDP crypto routing** -- pseudonymise, or retain cleartext under warrant
   and seal the evidence with AES-256-GCM.
4. **Persistence and dispatch** -- ordinary sightings are appended to the COPY
   buffer; hotlist hits are written synchronously and durably, then published
   to the alerts topic.

Nothing in that path blocks on disk. The buffered writer is what makes the
handler return in about the time the hotlist check takes, and the buffer is
bounded so that a database stall becomes an explicit HTTP 503 -- which an edge
node retries -- rather than silent memory growth.

Failure posture
---------------
The hotlist check **fails open**. If Redis is slow or unreachable, the sighting
is still stored, pseudonymised and marked ``ingest_degraded``, and a
reconciliation job re-tests those rows against the hotlist offline. The
alternative -- failing closed and dropping the pass -- would mean a Redis blip
erases traffic history, which is worse than a delayed alert.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple, Union

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .bloom import (
    BloomBackend,
    HotlistDecision,
    HotlistEngine,
    InMemoryBloomBackend,
    RedisBloomBackend,
    bloom_parameters,
)
from .config import Settings, get_settings
from .api_routes import UnifiedApiState, router as command_console_router
from .crypto import DpdpCryptoPipeline
from .db import (
    BackpressureError,
    Database,
    DatabaseError,
    HotlistHitRow,
    InMemoryDatabase,
    SightingRow,
    TimescaleDatabase,
    utc_from_epoch_ms,
)
from .schemas import (
    AlertSeverity,
    ComponentHealth,
    EdgeObservation,
    HealthReport,
    HotlistAlert,
    IngestBatch,
    IngestResponse,
    IngestResult,
    ProcessingDecision,
)

try:  # pragma: no cover - optional dependency
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROMETHEUS_AVAILABLE = False

__all__ = ["create_app", "AppState", "TokenBucketLimiter", "EventDispatcher", "app"]

LOGGER = logging.getLogger("veritrack.gateway")

SERVICE_VERSION = "2.0.0"


# =====================================================================
# Rate limiting
# =====================================================================


@dataclass(slots=True)
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketLimiter:
    """Per-client token bucket.

    Chosen over a fixed window because edge nodes are naturally bursty: a node
    that buffered through a 30-second network outage flushes a batch the moment
    it reconnects, and a fixed window would reject exactly the traffic we most
    want to accept. A bucket absorbs the burst up to ``burst`` and then meters
    the steady state at ``rate`` per second.
    """

    __slots__ = ("_rate", "_burst", "_buckets", "_lock", "_max_keys")

    def __init__(self, rate_per_second: float, burst: float, *, max_keys: int = 50_000) -> None:
        if rate_per_second <= 0 or burst <= 0:
            raise ValueError("rate and burst must be positive")
        self._rate = rate_per_second
        self._burst = burst
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()
        self._max_keys = max_keys

    async def acquire(self, key: str, cost: float = 1.0) -> Tuple[bool, float]:
        """Try to spend ``cost`` tokens. Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    self._evict_locked(now)
                bucket = _Bucket(tokens=self._burst, last_refill=now)
                self._buckets[key] = bucket

            elapsed = now - bucket.last_refill
            bucket.tokens = min(self._burst, bucket.tokens + elapsed * self._rate)
            bucket.last_refill = now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True, 0.0
            deficit = cost - bucket.tokens
            return False, deficit / self._rate

    def _evict_locked(self, now: float) -> None:
        """Drop buckets that have been idle long enough to be fully refilled."""
        idle_threshold = self._burst / self._rate
        stale = [key for key, bucket in self._buckets.items()
                 if now - bucket.last_refill > idle_threshold]
        for key in stale:
            del self._buckets[key]
        if not stale and self._buckets:
            self._buckets.pop(next(iter(self._buckets)))


# =====================================================================
# Event dispatch
# =====================================================================


class EventDispatcher:
    """Publishes enriched sightings and hotlist alerts to the event bus.

    Publication is fire-and-forget through a bounded queue drained by a
    background task: the ingest handler must never block on a Kafka broker. When
    the queue is full the *oldest* event is dropped, because for a live traffic
    feed a stale event is worth less than a fresh one. Alerts are exempt from
    that rule -- they are never dropped, and a full queue raises instead.
    """

    __slots__ = ("_queue", "_producer", "_task", "_settings", "_dropped", "_published", "_sink")

    def __init__(self, settings: Settings, *, sink: Optional[List[Dict[str, Any]]] = None) -> None:
        self._settings = settings
        self._queue: asyncio.Queue[Tuple[str, Dict[str, Any], Optional[str]]] = asyncio.Queue(
            maxsize=settings.kafka.max_queue_size
        )
        self._producer: Optional[Any] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._dropped = 0
        self._published = 0
        #: When Kafka is disabled, events land here. Tests assert against it;
        #: a single-box demo can tail it.
        self._sink: List[Dict[str, Any]] = [] if sink is None else sink

    @property
    def sink(self) -> List[Dict[str, Any]]:
        return self._sink

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def published(self) -> int:
        return self._published

    async def start(self) -> None:
        if self._settings.kafka.enabled:
            try:  # pragma: no cover - requires a broker
                from aiokafka import AIOKafkaProducer

                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self._settings.kafka.bootstrap_servers,
                    client_id=self._settings.kafka.client_id,
                    compression_type=(
                        None if self._settings.kafka.compression == "none"
                        else self._settings.kafka.compression
                    ),
                    acks=self._settings.kafka.acks,
                    linger_ms=self._settings.kafka.linger_ms,
                    enable_idempotence=True,
                )
                await self._producer.start()
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("kafka producer failed to start, falling back to sink: %s", exc)
                self._producer = None
        self._task = asyncio.create_task(self._drain(), name="veritrack-event-dispatcher")

    async def stop(self) -> None:
        if self._task is not None:
            await self._queue.join()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._producer is not None:  # pragma: no cover
            await self._producer.stop()
            self._producer = None

    async def publish(self, topic: str, payload: Dict[str, Any], key: Optional[str] = None,
                      *, droppable: bool = True) -> None:
        try:
            self._queue.put_nowait((topic, payload, key))
        except asyncio.QueueFull:
            if not droppable:
                raise
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._dropped += 1
                self._queue.put_nowait((topic, payload, key))
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                self._dropped += 1

    async def _drain(self) -> None:
        import json

        while True:
            topic, payload, key = await self._queue.get()
            try:
                if self._producer is not None:  # pragma: no cover
                    await self._producer.send_and_wait(
                        topic,
                        json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8"),
                        key=key.encode("utf-8") if key else None,
                    )
                else:
                    self._sink.append({"topic": topic, "key": key, "payload": payload})
                    if len(self._sink) > 10_000:
                        del self._sink[: len(self._sink) - 10_000]
                self._published += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("event publish to %s failed: %s", topic, exc)
            finally:
                self._queue.task_done()


# =====================================================================
# Metrics
# =====================================================================


class Metrics:
    """Prometheus collectors held in a private registry.

    A private ``CollectorRegistry`` rather than the global default, so that
    constructing several apps in one test session -- which the suite does --
    cannot raise a duplicate-timeseries error.
    """

    __slots__ = (
        "registry", "observations", "hotlist_hits", "rejected", "bloom_false_positives",
        "ingest_latency", "hotlist_latency", "buffered_rows", "rate_limited", "enabled",
    )

    def __init__(self) -> None:
        self.enabled = _PROMETHEUS_AVAILABLE
        if not _PROMETHEUS_AVAILABLE:  # pragma: no cover
            return
        self.registry = CollectorRegistry()
        self.observations = Counter(
            "veritrack_observations_total", "Observations accepted",
            ["decision"], registry=self.registry,
        )
        self.hotlist_hits = Counter(
            "veritrack_hotlist_hits_total", "Confirmed hotlist hits",
            ["severity"], registry=self.registry,
        )
        self.rejected = Counter(
            "veritrack_observations_rejected_total", "Observations rejected",
            ["reason"], registry=self.registry,
        )
        self.bloom_false_positives = Counter(
            "veritrack_bloom_false_positives_total",
            "Bloom hits with no confirming case record", registry=self.registry,
        )
        self.ingest_latency = Histogram(
            "veritrack_ingest_duration_seconds", "End-to-end ingest request latency",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=self.registry,
        )
        self.hotlist_latency = Histogram(
            "veritrack_hotlist_check_duration_seconds", "Hotlist decision latency",
            buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
            registry=self.registry,
        )
        self.buffered_rows = Gauge(
            "veritrack_write_buffer_rows", "Rows awaiting COPY", registry=self.registry,
        )
        self.rate_limited = Counter(
            "veritrack_rate_limited_total", "Requests rejected by the rate limiter",
            registry=self.registry,
        )

    def render(self) -> Tuple[bytes, str]:
        if not self.enabled:  # pragma: no cover
            return b"# prometheus_client is not installed\n", "text/plain; version=0.0.4"
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


# =====================================================================
# Application state
# =====================================================================


@dataclass
class AppState:
    """Everything the request handlers depend on. One object, injected once."""

    settings: Settings
    database: Database
    hotlist: HotlistEngine
    crypto: DpdpCryptoPipeline
    dispatcher: EventDispatcher
    limiter: Optional[TokenBucketLimiter]
    metrics: Metrics
    redis_client: Optional[Any] = None
    started_at: float = field(default_factory=time.monotonic)

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self.started_at


async def _build_hotlist_engine(
    settings: Settings,
) -> Tuple[HotlistEngine, Optional[Any]]:
    """Construct the hotlist engine, preferring Redis and degrading if permitted."""
    params = bloom_parameters(
        settings.bloom.capacity, settings.bloom.error_rate, max_bits=settings.bloom.max_bits
    )
    redis_client: Optional[Any] = None

    if aioredis is not None:
        try:
            redis_client = aioredis.from_url(
                settings.redis.url,
                socket_timeout=settings.redis.socket_timeout_s,
                socket_connect_timeout=settings.redis.socket_connect_timeout_s,
                max_connections=settings.redis.max_connections,
                decode_responses=True,
            )
            await asyncio.wait_for(redis_client.ping(), timeout=settings.redis.socket_connect_timeout_s)
            LOGGER.info("hotlist backed by redis at %s", settings.redis.safe_url)
        except Exception as exc:  # noqa: BLE001
            if redis_client is not None:
                try:
                    await redis_client.aclose()
                except Exception:  # noqa: BLE001
                    pass
            redis_client = None
            if settings.redis.required:
                raise RuntimeError(f"redis is required but unreachable: {exc}") from exc
            LOGGER.warning("redis unavailable (%s); using in-memory hotlist", exc)
    elif settings.redis.required:
        raise RuntimeError("redis.required is True but the redis package is not installed")

    backend: BloomBackend
    if redis_client is not None:
        backend = RedisBloomBackend(redis_client, params, key=settings.bloom.key)
    else:
        if not settings.bloom.allow_memory_fallback:
            raise RuntimeError("no Bloom backend available and memory fallback is disabled")
        # A 20M-capacity in-memory filter is ~48 MB, which is fine for a demo
        # but wasteful in CI; scale it down when Redis is absent by design.
        params = bloom_parameters(
            min(settings.bloom.capacity, 1_000_000),
            settings.bloom.error_rate,
            max_bits=settings.bloom.max_bits,
        )
        backend = InMemoryBloomBackend(params)

    engine = HotlistEngine(
        backend,
        redis_client=redis_client,
        meta_key_prefix=settings.bloom.meta_key_prefix,
        timeout_s=settings.ingest.hotlist_timeout_s,
    )
    return engine, redis_client


# =====================================================================
# Ingest pipeline
# =====================================================================


def _reject(observation: EdgeObservation, reason: str, state: AppState) -> IngestResult:
    if state.metrics.enabled:
        state.metrics.rejected.labels(reason=reason).inc()
    return IngestResult(
        pass_id=observation.stable_pass_id(),
        decision=ProcessingDecision.REJECTED,
        error=reason,
    )


def _validate_clock(observation: EdgeObservation, state: AppState, now_s: float) -> Optional[str]:
    """Reject observations whose timestamp is implausible."""
    observed_s = observation.timestamp_epoch_ms / 1000.0
    skew = observed_s - now_s
    if skew > state.settings.ingest.max_clock_skew_s:
        return f"timestamp is {skew:.1f}s in the future (unsynchronised edge clock)"
    if -skew > state.settings.ingest.max_observation_age_s:
        return f"timestamp is {-skew:.1f}s old (beyond the replay horizon)"
    return None


def _build_sighting_row(
    observation: EdgeObservation,
    state: AppState,
    *,
    hotlist_flag: bool,
    degraded: bool,
) -> SightingRow:
    """Pseudonymise and shape the row. The cleartext plate ends its life here."""
    observed_s = observation.timestamp_epoch_ms / 1000.0
    pseudonym = state.crypto.pseudonymize(
        observation.plate_number_decoded, observed_epoch_s=observed_s
    )
    location = observation.camera_location
    return SightingRow(
        timestamp_utc=utc_from_epoch_ms(observation.timestamp_epoch_ms),
        pass_id=observation.stable_pass_id(),
        edge_device_id=observation.edge_device_id,
        camera_id=observation.camera_id,
        tracklet_id=observation.tracklet_id,
        plate_pseudonym=pseudonym.digest_hex,
        plate_pseudonym_prefix=pseudonym.prefix,
        salt_epoch=pseudonym.salt_epoch,
        vehicle_class=observation.vehicle_class.value,
        plate_series=observation.plate_series.value,
        plate_sequence_confidence=observation.plate_sequence_confidence,
        min_character_confidence=observation.min_character_confidence,
        text_entropy=observation.text_entropy,
        repair_cost_nats=observation.repair_cost_nats,
        is_valid_format=bool(observation.is_valid_format),
        is_dual_line_plate=observation.is_dual_line_plate,
        vehicle_speed_kmh=observation.vehicle_speed_kmh,
        travel_heading_azimuth=observation.travel_heading_azimuth,
        reid_embedding=observation.reid_embedding_128d,
        latitude=location.latitude if location else None,
        longitude=location.longitude if location else None,
        corridor_id=observation.corridor_id,
        hotlist_flag=hotlist_flag,
        ingest_degraded=degraded,
    )


async def _handle_hotlist_hit(
    observation: EdgeObservation,
    decision: HotlistDecision,
    state: AppState,
) -> IngestResult:
    """Cleartext-under-warrant track: durable write, sealed evidence, alert."""
    match = decision.match
    assert match is not None  # guarded by decision.confirmed
    pass_id = observation.stable_pass_id()
    observed_at = utc_from_epoch_ms(observation.timestamp_epoch_ms)
    location = observation.camera_location

    evidence = {
        "pass_id": pass_id,
        "plate_number": observation.plate_number_decoded,
        "evidence_crop_s3_key": observation.evidence_crop_s3_key,
        "plate_corners": observation.plate_corners.flatten(),
        "vehicle_bounding_box": observation.vehicle_bounding_box.as_list(),
        "character_confidences": observation.character_confidences,
        "reid_embedding_128d": observation.reid_embedding_128d,
        "text_entropy": observation.text_entropy,
        "repair_cost_nats": observation.repair_cost_nats,
        "observed_at": observed_at.isoformat(),
        "warrant_reference": match.warrant_reference,
    }
    envelope = state.crypto.seal_evidence(evidence, pass_id=pass_id)

    hit_row = HotlistHitRow(
        timestamp_utc=observed_at,
        pass_id=pass_id,
        plate_number=observation.plate_number_decoded,
        edge_device_id=observation.edge_device_id,
        camera_id=observation.camera_id,
        tracklet_id=observation.tracklet_id,
        fir_number=match.fir_number,
        warrant_reference=match.warrant_reference,
        severity=int(match.severity),
        offence_category=match.offence_category,
        issuing_authority=match.issuing_authority,
        plate_sequence_confidence=observation.plate_sequence_confidence,
        vehicle_class=observation.vehicle_class.value,
        vehicle_speed_kmh=observation.vehicle_speed_kmh,
        travel_heading_azimuth=observation.travel_heading_azimuth,
        latitude=location.latitude if location else None,
        longitude=location.longitude if location else None,
        evidence_envelope=envelope.to_json(),
        evidence_key_version=envelope.key_version,
    )
    await state.database.write_hotlist_hit(hit_row)

    # The vehicle also belongs in the analytics stream, pseudonymised and
    # flagged, so corridor statistics and Stage 3 trajectory reconstruction see
    # a complete picture without reading the cleartext table.
    await state.database.enqueue_sighting(
        _build_sighting_row(observation, state, hotlist_flag=True, degraded=decision.degraded)
    )

    alert = HotlistAlert(
        alert_id=str(uuid.uuid4()),
        pass_id=pass_id,
        plate_number=observation.plate_number_decoded,
        edge_device_id=observation.edge_device_id,
        camera_id=observation.camera_id,
        tracklet_id=observation.tracklet_id,
        observed_at=observed_at,
        severity=match.severity,
        fir_number=match.fir_number,
        warrant_reference=match.warrant_reference,
        offence_category=match.offence_category,
        plate_sequence_confidence=observation.plate_sequence_confidence,
        vehicle_class=observation.vehicle_class,
        vehicle_speed_kmh=observation.vehicle_speed_kmh,
        travel_heading_azimuth=observation.travel_heading_azimuth,
        camera_location=location,
        evidence_ciphertext_b64=envelope.ciphertext_b64,
        evidence_key_version=envelope.key_version,
        dispatched_at=datetime.now(timezone.utc),
    )
    await state.dispatcher.publish(
        state.settings.kafka.alerts_topic,
        alert.model_dump(mode="json"),
        key=pass_id,
        droppable=False,
    )

    if state.metrics.enabled:
        state.metrics.hotlist_hits.labels(severity=str(int(match.severity))).inc()
        state.metrics.observations.labels(
            decision=ProcessingDecision.HOTLIST_CLEARTEXT.value
        ).inc()

    return IngestResult(
        pass_id=pass_id,
        decision=ProcessingDecision.HOTLIST_CLEARTEXT,
        hotlist_hit=True,
        severity=match.severity,
    )


async def process_batch(observations: Sequence[EdgeObservation], state: AppState) -> IngestResponse:
    """Run a validated batch through hotlist, crypto, storage and dispatch."""
    started = time.perf_counter()
    now_s = time.time()
    now_ms = int(now_s * 1000)

    results: List[IngestResult] = []
    admitted: List[EdgeObservation] = []
    for observation in observations:
        problem = _validate_clock(observation, state, now_s)
        if problem is not None:
            results.append(_reject(observation, problem, state))
        else:
            admitted.append(observation)

    hotlist_hits = 0
    if admitted:
        hotlist_started = time.perf_counter()
        decisions = await state.hotlist.check_many(
            [observation.plate_number_decoded for observation in admitted],
            now_epoch_ms=now_ms,
        )
        if state.metrics.enabled:
            state.metrics.hotlist_latency.observe(time.perf_counter() - hotlist_started)

        pseudonymous_rows: List[SightingRow] = []
        for observation, decision in zip(admitted, decisions):
            if decision.false_positive and state.metrics.enabled:
                state.metrics.bloom_false_positives.inc()
            try:
                if decision.confirmed:
                    results.append(await _handle_hotlist_hit(observation, decision, state))
                    hotlist_hits += 1
                else:
                    pseudonymous_rows.append(
                        _build_sighting_row(
                            observation, state, hotlist_flag=False, degraded=decision.degraded
                        )
                    )
                    results.append(
                        IngestResult(
                            pass_id=observation.stable_pass_id(),
                            decision=ProcessingDecision.PSEUDONYMIZED,
                        )
                    )
            except BackpressureError:
                raise
            except (DatabaseError, ValueError) as exc:
                results.append(_reject(observation, f"persistence failed: {exc}", state))

        if pseudonymous_rows:
            await state.database.enqueue_sightings(pseudonymous_rows)
            if state.metrics.enabled:
                state.metrics.observations.labels(
                    decision=ProcessingDecision.PSEUDONYMIZED.value
                ).inc(len(pseudonymous_rows))
            for row in pseudonymous_rows:
                await state.dispatcher.publish(
                    state.settings.kafka.sightings_topic,
                    {
                        "pass_id": row.pass_id,
                        "timestamp_utc": row.timestamp_utc.isoformat(),
                        "camera_id": row.camera_id,
                        "edge_device_id": row.edge_device_id,
                        "plate_pseudonym": row.plate_pseudonym,
                        "plate_pseudonym_prefix": row.plate_pseudonym_prefix,
                        "salt_epoch": row.salt_epoch,
                        "vehicle_class": row.vehicle_class,
                        "vehicle_speed_kmh": row.vehicle_speed_kmh,
                        "travel_heading_azimuth": row.travel_heading_azimuth,
                        "corridor_id": row.corridor_id,
                        "text_entropy": row.text_entropy,
                        "repair_cost_nats": row.repair_cost_nats,
                        "reid_embedding_128d": row.reid_embedding,
                    },
                    key=row.pass_id,
                )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if state.metrics.enabled:
        state.metrics.ingest_latency.observe(elapsed_ms / 1000.0)
        state.metrics.buffered_rows.set(state.database.buffered_rows)

    accepted = sum(1 for result in results if result.decision is not ProcessingDecision.REJECTED)
    return IngestResponse(
        accepted=accepted,
        rejected=len(results) - accepted,
        hotlist_hits=hotlist_hits,
        results=results,
        processing_ms=round(elapsed_ms, 3),
        buffered_rows=state.database.buffered_rows,
    )


# =====================================================================
# Application factory
# =====================================================================


def create_app(
    settings: Optional[Settings] = None,
    *,
    database: Optional[Database] = None,
    hotlist: Optional[HotlistEngine] = None,
    crypto: Optional[DpdpCryptoPipeline] = None,
) -> FastAPI:
    """Build the gateway.

    Every collaborator is injectable. That is what lets the test suite exercise
    the real request path against an in-memory database and Bloom backend,
    rather than testing a parallel mock of the pipeline that could drift from
    the code that actually runs in production.
    """
    resolved_settings = settings if settings is not None else get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(
            level=getattr(logging, resolved_settings.log_level),
            format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        )
        redis_client: Optional[Any] = None

        if hotlist is not None:
            engine = hotlist
        else:
            engine, redis_client = await _build_hotlist_engine(resolved_settings)

        db: Database = database if database is not None else TimescaleDatabase(
            resolved_settings.database, resolved_settings.ingest
        )
        await db.start()

        pipeline = crypto if crypto is not None else DpdpCryptoPipeline.from_settings(
            resolved_settings.crypto
        )
        dispatcher = EventDispatcher(resolved_settings)
        await dispatcher.start()

        limiter = (
            TokenBucketLimiter(
                resolved_settings.server.rate_limit_per_second,
                resolved_settings.server.rate_limit_burst,
            )
            if resolved_settings.server.rate_limit_enabled
            else None
        )

        salt_epoch, key_version = pipeline.describe()
        LOGGER.info(
            "gateway ready: env=%s salt_epoch=%d kek_version=%d bloom=%s",
            resolved_settings.environment, salt_epoch, key_version, engine.backend.name,
        )

        application.state.veritrack = AppState(
            settings=resolved_settings,
            database=db,
            hotlist=engine,
            crypto=pipeline,
            dispatcher=dispatcher,
            limiter=limiter,
            metrics=Metrics(),
            redis_client=redis_client,
        )
        application.state.stage5 = UnifiedApiState()
        try:
            yield
        finally:
            await dispatcher.stop()
            await db.stop()
            if redis_client is not None:
                try:
                    await redis_client.aclose()
                except Exception:  # noqa: BLE001
                    pass
            LOGGER.info("gateway shut down cleanly")

    application = FastAPI(
        title="VeriTrack Ingestion Gateway",
        description=(
            "Stage 2 central ingestion for the VeriTrack city-wide ANPR trajectory "
            "platform (SIH PS 26127). DPDP Act 2023 dual-track storage."
        ),
        version=SERVICE_VERSION,
        lifespan=lifespan,
        root_path=resolved_settings.server.root_path,
        docs_url="/docs" if resolved_settings.server.enable_docs else None,
        redoc_url="/redoc" if resolved_settings.server.enable_docs else None,
        openapi_url="/openapi.json" if resolved_settings.server.enable_docs else None,
    )

    if resolved_settings.server.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_settings.server.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    application.include_router(command_console_router)

    @application.middleware("http")
    async def _request_context(request: Request, call_next: Any) -> Response:
        """Attach a correlation id and a server-timing header to every response."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001
            LOGGER.exception("unhandled error on %s (request_id=%s)", request.url.path, request_id)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "internal error", "request_id": request_id},
                headers={"x-request-id": request_id},
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["x-request-id"] = request_id
        response.headers["server-timing"] = f"app;dur={elapsed_ms:.2f}"
        return response

    def get_state(request: Request) -> AppState:
        state: Optional[AppState] = getattr(request.app.state, "veritrack", None)
        if state is None:  # pragma: no cover - only if lifespan did not run
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="gateway is not initialised",
            )
        return state

    async def authorise(
        request: Request,
        state: AppState = Depends(get_state),
        authorization: Optional[str] = Header(default=None),
    ) -> str:
        """Bearer-token auth plus per-client rate limiting.

        Returns the rate-limit key, which is the authenticated device id when
        the edge node supplies one and the peer address otherwise.
        """
        expected = state.settings.server.ingest_api_key
        if expected is not None:
            provided = ""
            if authorization and authorization.lower().startswith("bearer "):
                provided = authorization[7:].strip()
            # Constant-time comparison: a timing oracle on the shared token
            # would be exploitable by any node on the ingest network.
            if not secrets.compare_digest(provided, expected.get_secret_value()):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="invalid or missing ingest credential",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        device = request.headers.get("x-edge-device-id")
        key = device or (request.client.host if request.client else "unknown")

        if state.limiter is not None:
            allowed, retry_after = await state.limiter.acquire(key)
            if not allowed:
                if state.metrics.enabled:
                    state.metrics.rate_limited.inc()
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="rate limit exceeded",
                    headers={"Retry-After": str(max(1, int(retry_after) + 1))},
                )
        return key

    # -----------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------

    @application.post(
        "/api/v1/telemetry/ingest",
        response_model=IngestResponse,
        status_code=status.HTTP_202_ACCEPTED,
        summary="Ingest one or many edge observations",
        tags=["telemetry"],
    )
    async def ingest(
        payload: Union[IngestBatch, EdgeObservation],
        state: AppState = Depends(get_state),
        _key: str = Depends(authorise),
    ) -> IngestResponse:
        observations = (
            payload.observations if isinstance(payload, IngestBatch) else [payload]
        )
        limit = state.settings.ingest.max_batch_observations
        if len(observations) > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"batch of {len(observations)} exceeds the limit of {limit}",
            )
        try:
            return await process_batch(observations, state)
        except BackpressureError as exc:
            # Explicit, retryable shed. The edge node buffers and comes back.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"ingest buffer saturated: {exc}",
                headers={"Retry-After": "2"},
            ) from exc
        except DatabaseError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"storage unavailable: {exc}",
            ) from exc

    @application.post(
        "/api/v1/telemetry/ingest/compact",
        response_model=IngestResponse,
        status_code=status.HTTP_202_ACCEPTED,
        summary="Ingest raw Stage 1 compact-dialect payloads",
        tags=["telemetry"],
    )
    async def ingest_compact(
        payload: Union[List[Dict[str, Any]], Dict[str, Any]],
        state: AppState = Depends(get_state),
        _key: str = Depends(authorise),
    ) -> IngestResponse:
        """Accept the byte-efficient dialect emitted by ``veritrack_edge``.

        Kept separate from the canonical endpoint so that the compact shape --
        which exists purely to protect the edge's 5 KB budget -- never has to be
        expressed in the public OpenAPI model.
        """
        raw_items = payload if isinstance(payload, list) else [payload]
        observations: List[EdgeObservation] = []
        rejections: List[IngestResult] = []
        for index, item in enumerate(raw_items):
            try:
                observations.append(EdgeObservation.from_compact_payload(item))
            except (ValidationError, ValueError, KeyError, TypeError) as exc:
                rejections.append(
                    IngestResult(
                        pass_id=str(item.get("pass_id", f"unparsed-{index}"))[:96],
                        decision=ProcessingDecision.REJECTED,
                        error=f"compact payload rejected: {exc}",
                    )
                )
        if not observations:
            return IngestResponse(
                accepted=0,
                rejected=len(rejections),
                hotlist_hits=0,
                results=rejections,
                processing_ms=0.0,
                buffered_rows=state.database.buffered_rows,
            )
        try:
            response = await process_batch(observations, state)
        except BackpressureError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"ingest buffer saturated: {exc}",
                headers={"Retry-After": "2"},
            ) from exc
        return IngestResponse(
            accepted=response.accepted,
            rejected=response.rejected + len(rejections),
            hotlist_hits=response.hotlist_hits,
            results=rejections + response.results,
            processing_ms=response.processing_ms,
            buffered_rows=response.buffered_rows,
        )

    @application.get("/healthz", response_model=HealthReport, tags=["ops"])
    async def healthz(state: AppState = Depends(get_state)) -> HealthReport:
        components: List[ComponentHealth] = []

        started = time.perf_counter()
        db_ok, db_detail = await state.database.health()
        components.append(
            ComponentHealth(
                name="database",
                healthy=db_ok,
                detail=db_detail,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
        )

        started = time.perf_counter()
        hotlist_ok, hotlist_detail = await state.hotlist.health()
        components.append(
            ComponentHealth(
                name="hotlist",
                healthy=hotlist_ok,
                detail=hotlist_detail,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
        )

        salt_epoch, key_version = state.crypto.describe()
        components.append(
            ComponentHealth(
                name="crypto",
                healthy=True,
                detail=f"salt_epoch={salt_epoch} kek_version={key_version}",
            )
        )
        components.append(
            ComponentHealth(
                name="dispatcher",
                healthy=True,
                detail=f"published={state.dispatcher.published} dropped={state.dispatcher.dropped}",
            )
        )

        overall = "ok" if all(component.healthy for component in components) else "degraded"
        return HealthReport(
            status=overall,
            version=SERVICE_VERSION,
            environment=state.settings.environment,
            uptime_s=round(state.uptime_s, 3),
            components=components,
        )

    @application.get("/readyz", tags=["ops"])
    async def readyz(state: AppState = Depends(get_state)) -> JSONResponse:
        """Kubernetes readiness: strictly stricter than liveness.

        A gateway whose write buffer is saturated is *alive* but must be pulled
        out of the load-balancer rotation until it drains.
        """
        db_ok, _ = await state.database.health()
        ready = db_ok and state.database.buffered_rows < state.settings.ingest.max_buffered_rows
        return JSONResponse(
            status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": ready, "buffered_rows": state.database.buffered_rows},
        )

    @application.get("/metrics", tags=["ops"])
    async def metrics(state: AppState = Depends(get_state)) -> Response:
        if state.metrics.enabled:
            state.metrics.buffered_rows.set(state.database.buffered_rows)
        body, content_type = state.metrics.render()
        return Response(content=body, media_type=content_type)

    @application.get("/api/v1/hotlist/stats", tags=["ops"])
    async def hotlist_stats(state: AppState = Depends(get_state)) -> Dict[str, Any]:
        params = state.hotlist.params
        inserted = await state.hotlist.backend.cardinality()
        return {
            "backend": state.hotlist.backend.name,
            "capacity": params.capacity,
            "num_bits": params.num_bits,
            "num_bytes": params.num_bytes,
            "num_hashes": params.num_hashes,
            "bits_per_item": round(params.bits_per_item, 3),
            "target_error_rate": params.target_error_rate,
            "realised_error_rate": params.realised_error_rate(inserted),
            "inserted": inserted,
            "saturation": round(params.saturation(inserted), 4),
            "counters": state.hotlist.stats(),
        }

    @application.post("/api/v1/hotlist/flush", status_code=status.HTTP_200_OK, tags=["ops"])
    async def flush_writes(state: AppState = Depends(get_state)) -> Dict[str, int]:
        """Force a COPY flush. Used by drain-before-deploy automation."""
        return {"rows_flushed": await state.database.flush()}

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    application.mount("/", StaticFiles(directory=static_dir, html=True), name="command-console")
    return application


_DEFAULT_APP: Optional[FastAPI] = None


def _default_app() -> FastAPI:
    """Uvicorn entrypoint target: ``veritrack_server.gateway:app``."""
    global _DEFAULT_APP
    if _DEFAULT_APP is None:
        _DEFAULT_APP = create_app()
    return _DEFAULT_APP


def __getattr__(name: str) -> Any:
    """Resolve ``gateway.app`` lazily.

    Building the application at import time would read the environment during
    ``import``, so a missing ``VERITRACK_CRYPTO_*`` secret would turn every
    import of this module -- including a test collecting ``create_app`` -- into
    a settings validation error. A module-level ``__getattr__`` keeps the
    ``veritrack_server.gateway:app`` entrypoint that uvicorn expects while
    deferring construction until something actually asks for it.
    """
    if name == "app":
        return _default_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    runtime_settings = get_settings()
    uvicorn.run(
        "veritrack_server.gateway:app",
        host=runtime_settings.server.host,
        port=runtime_settings.server.port,
        log_level=runtime_settings.log_level.lower(),
        workers=int(os.environ.get("VERITRACK_WORKERS", "1")),
    )
