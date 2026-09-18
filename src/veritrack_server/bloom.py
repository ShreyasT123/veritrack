"""Redis-backed Bloom filter and hotlist matching engine.

Why a Bloom filter at all
-------------------------
The national wanted-vehicle list is on the order of :math:`2 \\times 10^7`
registrations, and every single vehicle pass in the city must be tested against
it inside the sub-50 ms ingest budget. A ``SISMEMBER`` against a Redis set of 20
million plates costs roughly 20 M x (10 byte plate + ~50 byte hashtable
overhead) ~= 1.2 GB of RAM per replica. A Bloom filter answers the same question
in a fixed bit array with a tunable, *one-sided* error: it may say "maybe" for a
plate that is not on the list, but it never says "no" for a plate that is.

That asymmetry is the whole design. A false positive costs one extra Redis hash
lookup, which then authoritatively rejects it. A false negative would mean a
wanted vehicle driving past a camera unflagged -- and the filter's mathematics
guarantees that cannot happen.

Sizing
------
For capacity :math:`n` and target false-positive rate :math:`p`:

.. math::

    m = \\left\\lceil \\frac{-n \\ln p}{(\\ln 2)^2} \\right\\rceil, \\qquad
    k = \\operatorname{round}\\!\\left(\\frac{m}{n} \\ln 2\\right)

At :math:`n = 2\\times10^7` and :math:`p = 10^{-4}` this gives
:math:`m \\approx 3.83\\times10^8` bits (~48 MB, a single Redis string) and
:math:`k = 13`. Realised error after rounding :math:`k` is reported by
:meth:`BloomParameters.realised_error_rate`.

Hashing
-------
Kirsch--Mitzenmacher double hashing with a quadratic correction term:

.. math::  g_i(x) = \\bigl(h_1(x) + i\\,h_2(x) + i^2\\bigr) \\bmod m

One 128-bit BLAKE2b digest yields both :math:`h_1` and :math:`h_2`, so :math:`k`
independent-looking bit positions cost a single hash computation rather than
thirteen. The :math:`i^2` term breaks the degenerate cycles that plain double
hashing exhibits when :math:`h_2` shares a factor with :math:`m`.

Round trips
-----------
All :math:`k` ``GETBIT`` commands are issued in one pipeline, so a membership
test is **one** network round trip regardless of :math:`k`. This is what keeps
the check inside the 50 ms budget on a realistic LAN.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Final, Iterable, List, Optional, Protocol, Sequence, Tuple

from .schemas import AlertSeverity, HotlistMatch, normalise_plate

__all__ = [
    "BloomParameters",
    "bloom_parameters",
    "bit_offsets",
    "BloomBackend",
    "InMemoryBloomBackend",
    "RedisBloomBackend",
    "HotlistDecision",
    "HotlistEngine",
]

_DIGEST_BYTES: Final[int] = 16
_MAX_K: Final[int] = 64


@dataclass(frozen=True, slots=True)
class BloomParameters:
    """Derived Bloom filter geometry."""

    capacity: int
    target_error_rate: float
    num_bits: int
    num_hashes: int

    @property
    def num_bytes(self) -> int:
        return (self.num_bits + 7) // 8

    @property
    def bits_per_item(self) -> float:
        return self.num_bits / self.capacity

    def realised_error_rate(self, inserted: Optional[int] = None) -> float:
        """Actual FP rate at ``inserted`` elements, after integer rounding of k.

        :math:`p = \\left(1 - e^{-kn/m}\\right)^k`
        """
        n = self.capacity if inserted is None else max(inserted, 0)
        if n == 0:
            return 0.0
        exponent = -self.num_hashes * n / self.num_bits
        return float((1.0 - math.exp(exponent)) ** self.num_hashes)

    def saturation(self, inserted: int) -> float:
        """Expected fraction of set bits -- the operational health signal.

        An optimally-loaded filter sits near 0.5. Past ~0.75 the error rate has
        degraded badly enough that the filter should be rebuilt at a larger
        capacity rather than merely monitored.
        """
        if inserted <= 0:
            return 0.0
        return float(1.0 - math.exp(-self.num_hashes * inserted / self.num_bits))


def bloom_parameters(
    capacity: int, error_rate: float, *, max_bits: int = 1 << 32
) -> BloomParameters:
    """Solve for the bit-array size and hash count of an optimal Bloom filter."""
    if capacity < 1:
        raise ValueError("capacity must be >= 1")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must lie strictly in (0, 1)")

    ln2 = math.log(2.0)
    num_bits = int(math.ceil(-capacity * math.log(error_rate) / (ln2 * ln2)))
    num_bits = max(num_bits, 8)
    if num_bits > max_bits:
        raise ValueError(
            f"derived bit array of {num_bits} bits exceeds max_bits={max_bits}; "
            "lower the capacity or relax the error_rate"
        )
    num_hashes = max(1, min(_MAX_K, int(round((num_bits / capacity) * ln2))))
    return BloomParameters(
        capacity=capacity,
        target_error_rate=error_rate,
        num_bits=num_bits,
        num_hashes=num_hashes,
    )


def bit_offsets(item: str, num_bits: int, num_hashes: int) -> List[int]:
    """Bit positions for ``item`` via enhanced double hashing.

    Deterministic across processes and Python versions: BLAKE2b is specified,
    not ``hash()``, whose per-process randomisation would make a shared Redis
    filter incoherent across replicas.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")
    digest = hashlib.blake2b(item.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
    h1 = int.from_bytes(digest[:8], "big")
    h2 = int.from_bytes(digest[8:], "big") | 1  # odd, so it is coprime with any power of two
    return [((h1 + index * h2 + index * index) % num_bits) for index in range(num_hashes)]


class BloomBackend(Protocol):
    """Storage strategy for the bit array."""

    params: BloomParameters

    async def add(self, item: str) -> None: ...
    async def add_many(self, items: Sequence[str]) -> int: ...
    async def contains(self, item: str) -> bool: ...
    async def contains_many(self, items: Sequence[str]) -> List[bool]: ...
    async def cardinality(self) -> int: ...
    async def clear(self) -> None: ...
    async def ping(self) -> bool: ...
    @property
    def name(self) -> str: ...


class InMemoryBloomBackend:
    """Process-local bitset. Used for tests, CI and single-box demonstrations.

    Explicitly *not* production: two gateway replicas would each hold a private
    filter and disagree about the hotlist. ``config.RedisSettings.required``
    exists to make that misconfiguration a startup failure.
    """

    __slots__ = ("params", "_bits", "_count", "_lock")

    def __init__(self, params: BloomParameters) -> None:
        self.params = params
        self._bits = bytearray(params.num_bytes)
        self._count = 0
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "memory"

    def _set_bit(self, offset: int) -> None:
        self._bits[offset >> 3] |= 1 << (offset & 7)

    def _get_bit(self, offset: int) -> bool:
        return bool(self._bits[offset >> 3] & (1 << (offset & 7)))

    async def add(self, item: str) -> None:
        async with self._lock:
            for offset in bit_offsets(item, self.params.num_bits, self.params.num_hashes):
                self._set_bit(offset)
            self._count += 1

    async def add_many(self, items: Sequence[str]) -> int:
        async with self._lock:
            for item in items:
                for offset in bit_offsets(item, self.params.num_bits, self.params.num_hashes):
                    self._set_bit(offset)
            self._count += len(items)
        return len(items)

    async def contains(self, item: str) -> bool:
        return all(
            self._get_bit(offset)
            for offset in bit_offsets(item, self.params.num_bits, self.params.num_hashes)
        )

    async def contains_many(self, items: Sequence[str]) -> List[bool]:
        return [await self.contains(item) for item in items]

    async def cardinality(self) -> int:
        return self._count

    async def clear(self) -> None:
        async with self._lock:
            self._bits = bytearray(self.params.num_bytes)
            self._count = 0

    async def ping(self) -> bool:
        return True


class RedisBloomBackend:
    """Bit array held in a single Redis string, addressed with SETBIT/GETBIT.

    Deliberately built on vanilla Redis primitives rather than the RedisBloom
    module: BEL's deployment target may be a managed Redis without custom
    modules, and ``SETBIT``/``GETBIT`` are available everywhere. The cost is
    that we own the sizing arithmetic -- which we wanted to own anyway, since
    the realised error rate is an operational metric the dashboard reports.
    """

    __slots__ = ("params", "_redis", "_key", "_count_key")

    def __init__(self, redis_client: Any, params: BloomParameters, *, key: str) -> None:
        self.params = params
        self._redis = redis_client
        self._key = key
        self._count_key = f"{key}:count"

    @property
    def name(self) -> str:
        return "redis"

    async def add(self, item: str) -> None:
        pipe = self._redis.pipeline(transaction=False)
        for offset in bit_offsets(item, self.params.num_bits, self.params.num_hashes):
            pipe.setbit(self._key, offset, 1)
        pipe.incr(self._count_key)
        await pipe.execute()

    async def add_many(self, items: Sequence[str]) -> int:
        if not items:
            return 0
        pipe = self._redis.pipeline(transaction=False)
        for item in items:
            for offset in bit_offsets(item, self.params.num_bits, self.params.num_hashes):
                pipe.setbit(self._key, offset, 1)
        pipe.incrby(self._count_key, len(items))
        await pipe.execute()
        return len(items)

    async def contains(self, item: str) -> bool:
        pipe = self._redis.pipeline(transaction=False)
        for offset in bit_offsets(item, self.params.num_bits, self.params.num_hashes):
            pipe.getbit(self._key, offset)
        results = await pipe.execute()
        return all(bool(bit) for bit in results)

    async def contains_many(self, items: Sequence[str]) -> List[bool]:
        """Test several plates in a single round trip.

        Offsets are laid out per item so the flat result array can be sliced
        back apart by ``num_hashes``.
        """
        if not items:
            return []
        k = self.params.num_hashes
        pipe = self._redis.pipeline(transaction=False)
        for item in items:
            for offset in bit_offsets(item, self.params.num_bits, k):
                pipe.getbit(self._key, offset)
        flat = await pipe.execute()
        return [all(bool(bit) for bit in flat[i * k : (i + 1) * k]) for i in range(len(items))]

    async def cardinality(self) -> int:
        raw = await self._redis.get(self._count_key)
        if raw is None:
            return 0
        return int(raw)

    async def clear(self) -> None:
        await self._redis.delete(self._key, self._count_key)

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:  # noqa: BLE001 - any transport fault means unhealthy
            return False


@dataclass(frozen=True, slots=True)
class HotlistDecision:
    """Outcome of a hotlist test, with enough detail for metrics and audit."""

    plate: str
    bloom_hit: bool
    confirmed: bool
    match: Optional[HotlistMatch]
    latency_ms: float
    false_positive: bool
    degraded: bool = False

    @property
    def severity(self) -> Optional[AlertSeverity]:
        return self.match.severity if self.match else None


class HotlistEngine:
    """Two-stage matcher: probabilistic filter, then authoritative confirmation.

    Stage A -- Bloom membership. Constant time, one round trip, no false
    negatives. The overwhelming majority of city traffic is not wanted, so this
    stage answers ``False`` for ~99.99% of passes and the pipeline moves on.

    Stage B -- reached only on a Bloom hit. Fetches the case record from a Redis
    hash. This is the authority: it resolves Bloom false positives, enforces
    warrant expiry, and supplies the FIR number and severity the alert needs.

    A confirmed match therefore *never* rests on probabilistic evidence, which
    matters because the downstream consequence is a police dispatch.
    """

    __slots__ = (
        "_backend",
        "_redis",
        "_meta_prefix",
        "_timeout_s",
        "_local_meta",
        "_stats",
    )

    def __init__(
        self,
        backend: BloomBackend,
        *,
        redis_client: Optional[Any] = None,
        meta_key_prefix: str = "veritrack:hotlist:meta:",
        timeout_s: float = 0.05,
    ) -> None:
        self._backend = backend
        self._redis = redis_client
        self._meta_prefix = meta_key_prefix
        self._timeout_s = timeout_s
        self._local_meta: Dict[str, HotlistMatch] = {}
        self._stats: Dict[str, int] = {
            "checks": 0,
            "bloom_hits": 0,
            "confirmed": 0,
            "false_positives": 0,
            "timeouts": 0,
            "expired": 0,
        }

    @property
    def backend(self) -> BloomBackend:
        return self._backend

    @property
    def params(self) -> BloomParameters:
        return self._backend.params

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    async def register(self, match: HotlistMatch) -> None:
        """Add a wanted vehicle to both the filter and the metadata store."""
        plate = normalise_plate(match.plate_number)
        await self._backend.add(plate)
        payload = {
            "plate_number": plate,
            "fir_number": match.fir_number,
            "warrant_reference": match.warrant_reference,
            "severity": str(int(match.severity)),
            "issuing_authority": match.issuing_authority,
            "offence_category": match.offence_category,
        }
        if match.registered_at_epoch_ms is not None:
            payload["registered_at_epoch_ms"] = str(match.registered_at_epoch_ms)
        if match.expires_at_epoch_ms is not None:
            payload["expires_at_epoch_ms"] = str(match.expires_at_epoch_ms)
        if match.notes:
            payload["notes"] = match.notes

        if self._redis is not None:
            await self._redis.hset(f"{self._meta_prefix}{plate}", mapping=payload)
        else:
            self._local_meta[plate] = match

    async def register_many(self, matches: Iterable[HotlistMatch]) -> int:
        count = 0
        for match in matches:
            await self.register(match)
            count += 1
        return count

    async def _load_metadata(self, plate: str) -> Optional[HotlistMatch]:
        if self._redis is None:
            return self._local_meta.get(plate)
        raw = await self._redis.hgetall(f"{self._meta_prefix}{plate}")
        if not raw:
            return None
        decoded: Dict[str, Any] = {}
        for key, value in raw.items():
            key_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)
            val_str = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            decoded[key_str] = val_str
        for numeric in ("registered_at_epoch_ms", "expires_at_epoch_ms"):
            if numeric in decoded:
                try:
                    decoded[numeric] = int(decoded[numeric])
                except ValueError:
                    decoded.pop(numeric)
        decoded.setdefault("plate_number", plate)
        try:
            return HotlistMatch.model_validate(decoded)
        except Exception:  # noqa: BLE001 - a corrupt case record must not stall ingest
            return None

    async def check(self, plate: str, *, now_epoch_ms: Optional[int] = None) -> HotlistDecision:
        """Test one plate against the hotlist inside the configured budget."""
        normalised = normalise_plate(plate)
        reference_ms = int(time.time() * 1000) if now_epoch_ms is None else now_epoch_ms
        started = time.perf_counter()
        self._stats["checks"] += 1

        try:
            bloom_hit = await asyncio.wait_for(
                self._backend.contains(normalised), timeout=self._timeout_s
            )
        except asyncio.TimeoutError:
            self._stats["timeouts"] += 1
            # Fail *open* on the ingest path: a slow Redis must not drop a
            # sighting. The pass is stored pseudonymised and flagged degraded so
            # the reconciliation job can re-test it against the hotlist offline.
            return HotlistDecision(
                plate=normalised,
                bloom_hit=False,
                confirmed=False,
                match=None,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                false_positive=False,
                degraded=True,
            )
        except Exception:  # noqa: BLE001
            self._stats["timeouts"] += 1
            return HotlistDecision(
                plate=normalised,
                bloom_hit=False,
                confirmed=False,
                match=None,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                false_positive=False,
                degraded=True,
            )

        if not bloom_hit:
            return HotlistDecision(
                plate=normalised,
                bloom_hit=False,
                confirmed=False,
                match=None,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                false_positive=False,
            )

        self._stats["bloom_hits"] += 1
        remaining = max(self._timeout_s - (time.perf_counter() - started), 0.005)
        try:
            match = await asyncio.wait_for(self._load_metadata(normalised), timeout=remaining)
        except (asyncio.TimeoutError, Exception):  # noqa: B014 - deliberate catch-all
            self._stats["timeouts"] += 1
            match = None
            degraded = True
        else:
            degraded = False

        latency_ms = (time.perf_counter() - started) * 1000.0

        if match is None:
            self._stats["false_positives"] += 1
            return HotlistDecision(
                plate=normalised,
                bloom_hit=True,
                confirmed=False,
                match=None,
                latency_ms=latency_ms,
                false_positive=not degraded,
                degraded=degraded,
            )

        if not match.is_active(reference_ms):
            self._stats["expired"] += 1
            return HotlistDecision(
                plate=normalised,
                bloom_hit=True,
                confirmed=False,
                match=None,
                latency_ms=latency_ms,
                false_positive=False,
            )

        self._stats["confirmed"] += 1
        return HotlistDecision(
            plate=normalised,
            bloom_hit=True,
            confirmed=True,
            match=match,
            latency_ms=latency_ms,
            false_positive=False,
        )

    async def check_many(
        self, plates: Sequence[str], *, now_epoch_ms: Optional[int] = None
    ) -> List[HotlistDecision]:
        """Batch test: one pipelined Bloom round trip, then confirm only the hits."""
        if not plates:
            return []
        normalised = [normalise_plate(plate) for plate in plates]
        reference_ms = int(time.time() * 1000) if now_epoch_ms is None else now_epoch_ms
        started = time.perf_counter()
        self._stats["checks"] += len(normalised)

        try:
            flags = await asyncio.wait_for(
                self._backend.contains_many(normalised),
                timeout=max(self._timeout_s, 0.01 * len(normalised)),
            )
        except Exception:  # noqa: BLE001
            self._stats["timeouts"] += len(normalised)
            elapsed = (time.perf_counter() - started) * 1000.0
            return [
                HotlistDecision(
                    plate=plate,
                    bloom_hit=False,
                    confirmed=False,
                    match=None,
                    latency_ms=elapsed,
                    false_positive=False,
                    degraded=True,
                )
                for plate in normalised
            ]

        decisions: List[HotlistDecision] = []
        for plate, hit in zip(normalised, flags):
            if not hit:
                decisions.append(
                    HotlistDecision(
                        plate=plate,
                        bloom_hit=False,
                        confirmed=False,
                        match=None,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        false_positive=False,
                    )
                )
                continue
            self._stats["bloom_hits"] += 1
            match = await self._load_metadata(plate)
            latency_ms = (time.perf_counter() - started) * 1000.0
            if match is None:
                self._stats["false_positives"] += 1
                decisions.append(
                    HotlistDecision(plate, True, False, None, latency_ms, True)
                )
            elif not match.is_active(reference_ms):
                self._stats["expired"] += 1
                decisions.append(
                    HotlistDecision(plate, True, False, None, latency_ms, False)
                )
            else:
                self._stats["confirmed"] += 1
                decisions.append(
                    HotlistDecision(plate, True, True, match, latency_ms, False)
                )
        return decisions

    async def health(self) -> Tuple[bool, str]:
        alive = await self._backend.ping()
        try:
            inserted = await self._backend.cardinality()
        except Exception:  # noqa: BLE001
            inserted = 0
        saturation = self.params.saturation(inserted)
        detail = (
            f"backend={self._backend.name} bits={self.params.num_bits} "
            f"k={self.params.num_hashes} inserted={inserted} "
            f"saturation={saturation:.3f} "
            f"fp_rate={self.params.realised_error_rate(inserted):.2e}"
        )
        return alive and saturation < 0.75, detail
