"""Async TimescaleDB / PostGIS access layer with a COPY-based write buffer.

Throughput shape
----------------
A city deployment of a few thousand cameras produces a high-rate, append-only,
never-updated stream. That is the ideal shape for PostgreSQL's binary ``COPY``
protocol, which is roughly an order of magnitude faster than executing one
``INSERT`` per row: it amortises parse, plan and round-trip cost across the
whole batch.

So the write path is a **buffer, not a write-through**. The request handler
appends rows to an in-process deque and returns; a background task flushes via
``copy_records_to_table`` when either the row threshold or the time threshold
trips. The HTTP 202 an edge node receives means "durably queued", and the
buffer is bounded so that a database stall turns into explicit backpressure
(HTTP 503, which the edge node retries) rather than unbounded memory growth
followed by an OOM kill.

Two durability classes
----------------------
Ordinary sightings ride the buffered path with ``synchronous_commit=off``: a
sub-second window of loss on a hard node failure is an acceptable price for the
throughput, since an individual anonymous sighting is statistical input.

Hotlist hits do not. They are written immediately, on a connection with
``synchronous_commit`` forced ``ON``, inside the request. An alert that a
wanted vehicle passed a camera is evidence, and losing it is not an option we
are willing to buy performance with.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

try:  # pragma: no cover - exercised only where the driver is installed
    import asyncpg
except ImportError:  # pragma: no cover
    asyncpg = None  # type: ignore[assignment]

from .config import DatabaseSettings, IngestSettings

__all__ = [
    "DatabaseError",
    "BackpressureError",
    "SightingRow",
    "HotlistHitRow",
    "Database",
    "TimescaleDatabase",
    "InMemoryDatabase",
    "SIGHTING_COLUMNS",
    "HOTLIST_HIT_COLUMNS",
]

LOGGER = logging.getLogger("veritrack.db")

SIGHTING_COLUMNS: Tuple[str, ...] = (
    "timestamp_utc",
    "pass_id",
    "edge_device_id",
    "camera_id",
    "tracklet_id",
    "plate_pseudonym",
    "plate_pseudonym_prefix",
    "salt_epoch",
    "vehicle_class",
    "plate_series",
    "plate_sequence_confidence",
    "min_character_confidence",
    "text_entropy",
    "repair_cost_nats",
    "is_valid_format",
    "is_dual_line_plate",
    "vehicle_speed_kmh",
    "travel_heading_azimuth",
    "reid_embedding",
    "latitude",
    "longitude",
    "corridor_id",
    "hotlist_flag",
    "ingest_degraded",
)

HOTLIST_HIT_COLUMNS: Tuple[str, ...] = (
    "timestamp_utc",
    "pass_id",
    "plate_number",
    "edge_device_id",
    "camera_id",
    "tracklet_id",
    "fir_number",
    "warrant_reference",
    "severity",
    "offence_category",
    "issuing_authority",
    "plate_sequence_confidence",
    "vehicle_class",
    "vehicle_speed_kmh",
    "travel_heading_azimuth",
    "latitude",
    "longitude",
    "evidence_envelope",
    "evidence_key_version",
)


class DatabaseError(RuntimeError):
    """Persistent-store fault."""


class BackpressureError(DatabaseError):
    """The write buffer is full; the caller must shed load."""


@dataclass(slots=True)
class SightingRow:
    """One pseudonymised (or hotlist-flagged) sighting, ready for COPY.

    Deliberately holds no cleartext plate. On the non-hotlist track the plate
    string is discarded inside the crypto stage and never reaches this object,
    so there is no code path by which it could be written to the sightings
    hypertable.
    """

    timestamp_utc: datetime
    pass_id: str
    edge_device_id: str
    camera_id: str
    tracklet_id: str
    plate_pseudonym: str
    plate_pseudonym_prefix: str
    salt_epoch: int
    vehicle_class: str
    plate_series: str
    plate_sequence_confidence: float
    min_character_confidence: float
    text_entropy: Optional[float]
    repair_cost_nats: float
    is_valid_format: bool
    is_dual_line_plate: bool
    vehicle_speed_kmh: Optional[float]
    travel_heading_azimuth: Optional[float]
    reid_embedding: List[float]
    latitude: Optional[float]
    longitude: Optional[float]
    corridor_id: Optional[str]
    hotlist_flag: bool = False
    ingest_degraded: bool = False

    def as_record(self) -> Tuple[Any, ...]:
        return (
            self.timestamp_utc,
            self.pass_id,
            self.edge_device_id,
            self.camera_id,
            self.tracklet_id,
            self.plate_pseudonym,
            self.plate_pseudonym_prefix,
            self.salt_epoch,
            self.vehicle_class,
            self.plate_series,
            self.plate_sequence_confidence,
            self.min_character_confidence,
            self.text_entropy,
            self.repair_cost_nats,
            self.is_valid_format,
            self.is_dual_line_plate,
            self.vehicle_speed_kmh,
            self.travel_heading_azimuth,
            self.reid_embedding,
            self.latitude,
            self.longitude,
            self.corridor_id,
            self.hotlist_flag,
            self.ingest_degraded,
        )


@dataclass(slots=True)
class HotlistHitRow:
    """A confirmed wanted-vehicle sighting. Cleartext, warrant-scoped, encrypted evidence."""

    timestamp_utc: datetime
    pass_id: str
    plate_number: str
    edge_device_id: str
    camera_id: str
    tracklet_id: str
    fir_number: str
    warrant_reference: str
    severity: int
    offence_category: str
    issuing_authority: str
    plate_sequence_confidence: float
    vehicle_class: str
    vehicle_speed_kmh: Optional[float]
    travel_heading_azimuth: Optional[float]
    latitude: Optional[float]
    longitude: Optional[float]
    evidence_envelope: Optional[str]
    evidence_key_version: Optional[int]

    def as_record(self) -> Tuple[Any, ...]:
        return (
            self.timestamp_utc,
            self.pass_id,
            self.plate_number,
            self.edge_device_id,
            self.camera_id,
            self.tracklet_id,
            self.fir_number,
            self.warrant_reference,
            self.severity,
            self.offence_category,
            self.issuing_authority,
            self.plate_sequence_confidence,
            self.vehicle_class,
            self.vehicle_speed_kmh,
            self.travel_heading_azimuth,
            self.latitude,
            self.longitude,
            self.evidence_envelope,
            self.evidence_key_version,
        )


class Database(Protocol):
    """Persistence surface the gateway depends on."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def enqueue_sighting(self, row: SightingRow) -> None: ...
    async def enqueue_sightings(self, rows: Sequence[SightingRow]) -> None: ...
    async def write_hotlist_hit(self, row: HotlistHitRow) -> None: ...
    async def flush(self) -> int: ...
    async def health(self) -> Tuple[bool, str]: ...
    @property
    def buffered_rows(self) -> int: ...


@dataclass(slots=True)
class _WriteStats:
    rows_buffered: int = 0
    rows_flushed: int = 0
    flushes: int = 0
    flush_failures: int = 0
    hotlist_hits: int = 0
    last_flush_ms: float = 0.0
    last_error: str = ""


class TimescaleDatabase:
    """asyncpg-backed implementation against TimescaleDB + PostGIS."""

    __slots__ = (
        "_settings",
        "_ingest",
        "_pool",
        "_buffer",
        "_buffer_lock",
        "_flush_lock",
        "_flusher",
        "_stopping",
        "_stats",
        "_last_flush_at",
    )

    def __init__(self, settings: DatabaseSettings, ingest: IngestSettings) -> None:
        if asyncpg is None:  # pragma: no cover
            raise DatabaseError("asyncpg is not installed; cannot use TimescaleDatabase")
        self._settings = settings
        self._ingest = ingest
        self._pool: Optional[Any] = None
        self._buffer: List[SightingRow] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_lock = asyncio.Lock()
        self._flusher: Optional[asyncio.Task[None]] = None
        self._stopping = False
        self._stats = _WriteStats()
        self._last_flush_at = time.monotonic()

    @property
    def buffered_rows(self) -> int:
        return len(self._buffer)

    def stats(self) -> Dict[str, Any]:
        return {
            "rows_buffered": self._stats.rows_buffered,
            "rows_flushed": self._stats.rows_flushed,
            "flushes": self._stats.flushes,
            "flush_failures": self._stats.flush_failures,
            "hotlist_hits": self._stats.hotlist_hits,
            "last_flush_ms": self._stats.last_flush_ms,
            "pending": len(self._buffer),
            "last_error": self._stats.last_error,
        }

    async def start(self) -> None:
        LOGGER.info("connecting to %s", self._settings.safe_dsn)
        self._pool = await asyncpg.create_pool(
            dsn=self._settings.dsn,
            min_size=self._settings.min_pool_size,
            max_size=self._settings.max_pool_size,
            command_timeout=self._settings.command_timeout_s,
            timeout=self._settings.connect_timeout_s,
            statement_cache_size=self._settings.statement_cache_size,
            server_settings=self._settings.server_settings,
        )
        self._stopping = False
        self._flusher = asyncio.create_task(self._flush_loop(), name="veritrack-db-flusher")
        LOGGER.info("database pool ready (min=%d max=%d)",
                    self._settings.min_pool_size, self._settings.max_pool_size)

    async def stop(self) -> None:
        self._stopping = True
        if self._flusher is not None:
            self._flusher.cancel()
            try:
                await self._flusher
            except asyncio.CancelledError:
                pass
            self._flusher = None
        # Drain whatever is still buffered before tearing the pool down, so a
        # graceful shutdown never discards acknowledged rows.
        try:
            await self.flush()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("final flush failed, %d rows lost: %s", len(self._buffer), exc)
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def enqueue_sighting(self, row: SightingRow) -> None:
        await self.enqueue_sightings((row,))

    async def enqueue_sightings(self, rows: Sequence[SightingRow]) -> None:
        if not rows:
            return
        async with self._buffer_lock:
            if len(self._buffer) + len(rows) > self._ingest.max_buffered_rows:
                raise BackpressureError(
                    f"write buffer full ({len(self._buffer)}/{self._ingest.max_buffered_rows})"
                )
            self._buffer.extend(rows)
            self._stats.rows_buffered += len(rows)
            should_flush = len(self._buffer) >= self._ingest.flush_row_threshold
        if should_flush:
            # Fire and forget: the caller must not wait on disk.
            asyncio.create_task(self._safe_flush())

    async def write_hotlist_hit(self, row: HotlistHitRow) -> None:
        """Synchronous, durable write. Never buffered."""
        if self._pool is None:
            raise DatabaseError("database pool is not started")
        placeholders = ", ".join(f"${i + 1}" for i in range(len(HOTLIST_HIT_COLUMNS)))
        statement = (
            f"INSERT INTO hotlist_hits ({', '.join(HOTLIST_HIT_COLUMNS)}) "
            f"VALUES ({placeholders}) ON CONFLICT (pass_id, timestamp_utc) DO NOTHING"
        )
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("SET LOCAL synchronous_commit = ON")
                await connection.execute(statement, *row.as_record())
        self._stats.hotlist_hits += 1

    async def flush(self) -> int:
        """Drain the buffer into the hypertable via binary COPY. Returns rows written."""
        if self._pool is None:
            raise DatabaseError("database pool is not started")
        async with self._flush_lock:
            async with self._buffer_lock:
                if not self._buffer:
                    return 0
                batch = self._buffer
                self._buffer = []
            started = time.perf_counter()
            records = [row.as_record() for row in batch]
            try:
                async with self._pool.acquire() as connection:
                    await connection.copy_records_to_table(
                        "sightings", records=records, columns=list(SIGHTING_COLUMNS)
                    )
            except Exception as exc:  # noqa: BLE001
                self._stats.flush_failures += 1
                self._stats.last_error = str(exc)
                # Return the batch to the head of the buffer so a transient
                # database blip is retried rather than silently dropped. If the
                # buffer has since overflowed, the oldest rows are the ones we
                # sacrifice -- recency matters more for live traffic analytics.
                async with self._buffer_lock:
                    self._buffer = (batch + self._buffer)[-self._ingest.max_buffered_rows :]
                LOGGER.error("COPY of %d rows failed: %s", len(batch), exc)
                raise DatabaseError(f"bulk insert failed: {exc}") from exc

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._stats.rows_flushed += len(batch)
            self._stats.flushes += 1
            self._stats.last_flush_ms = elapsed_ms
            self._last_flush_at = time.monotonic()
            LOGGER.debug("flushed %d rows in %.1f ms", len(batch), elapsed_ms)
            return len(batch)

    async def _safe_flush(self) -> None:
        try:
            await self.flush()
        except DatabaseError:
            pass  # already logged and re-buffered

    async def _flush_loop(self) -> None:
        interval = self._ingest.flush_interval_s
        while not self._stopping:
            await asyncio.sleep(interval)
            if not self._buffer:
                continue
            if (time.monotonic() - self._last_flush_at) < interval and (
                len(self._buffer) < self._ingest.flush_row_threshold
            ):
                continue
            await self._safe_flush()

    async def health(self) -> Tuple[bool, str]:
        if self._pool is None:
            return False, "pool not started"
        try:
            started = time.perf_counter()
            async with self._pool.acquire() as connection:
                value = await connection.fetchval("SELECT 1")
            latency_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001
            return False, f"unreachable: {exc}"
        healthy = value == 1 and len(self._buffer) < self._ingest.max_buffered_rows
        return healthy, (
            f"ping={latency_ms:.1f}ms buffered={len(self._buffer)}"
            f"/{self._ingest.max_buffered_rows} flushed={self._stats.rows_flushed}"
        )


class InMemoryDatabase:
    """Reference implementation used by the test suite and offline replay.

    Holds the same row objects the COPY path would emit, so assertions written
    against it are assertions about what would actually hit disk.
    """

    __slots__ = ("sightings", "hotlist_hits", "_started", "_fail_next", "max_rows")

    def __init__(self, max_rows: int = 1_000_000) -> None:
        self.sightings: List[SightingRow] = []
        self.hotlist_hits: List[HotlistHitRow] = []
        self._started = False
        self._fail_next = False
        self.max_rows = max_rows

    @property
    def buffered_rows(self) -> int:
        return 0

    def fail_next_write(self) -> None:
        """Arm a single simulated write failure, for backpressure tests."""
        self._fail_next = True

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def enqueue_sighting(self, row: SightingRow) -> None:
        await self.enqueue_sightings((row,))

    async def enqueue_sightings(self, rows: Sequence[SightingRow]) -> None:
        if self._fail_next:
            self._fail_next = False
            raise BackpressureError("simulated buffer overflow")
        if len(self.sightings) + len(rows) > self.max_rows:
            raise BackpressureError("in-memory row cap exceeded")
        self.sightings.extend(rows)

    async def write_hotlist_hit(self, row: HotlistHitRow) -> None:
        self.hotlist_hits.append(row)

    async def flush(self) -> int:
        return 0

    async def health(self) -> Tuple[bool, str]:
        return self._started, f"in-memory rows={len(self.sightings)} hits={len(self.hotlist_hits)}"


def utc_from_epoch_ms(epoch_ms: int) -> datetime:
    """Convert an edge millisecond timestamp to a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
