"""Stage 2 verification suite.

Runs entirely without PostgreSQL, Redis or Kafka: the gateway's collaborators
are injected, so these tests exercise the *real* request path against in-memory
implementations rather than a parallel mock of the pipeline.

    pytest tests/test_stage2.py -v
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from veritrack_server.bloom import (
    HotlistEngine,
    InMemoryBloomBackend,
    bit_offsets,
    bloom_parameters,
)
from veritrack_server.config import CryptoSettings, Settings, build_dev_settings
from veritrack_server.crypto import (
    AesGcmEnvelopeEncryptor,
    DecryptionError,
    DerivedSaltSource,
    DpdpCryptoPipeline,
    EphemeralSaltSource,
    PlatePseudonymizer,
    RotatingSaltManager,
    SealedEnvelope,
)
from veritrack_server.db import BackpressureError, InMemoryDatabase
from veritrack_server.gateway import TokenBucketLimiter, create_app
from veritrack_server.schemas import (
    AlertSeverity,
    EdgeObservation,
    HotlistMatch,
    IngestBatch,
    PlateCorners,
    ProcessingDecision,
    VehicleClass,
    is_valid_indian_plate,
    normalise_plate,
)

pytestmark = pytest.mark.asyncio

NOW_MS = int(time.time() * 1000)


# =====================================================================
# Fixtures and builders
# =====================================================================


def unit_embedding(seed: int = 0) -> List[float]:
    """A deterministic 128-D unit vector."""
    raw = [math.sin(seed + index * 0.37) for index in range(128)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


def quantised_embedding(seed: int = 0) -> str:
    """The same vector in Stage 1's int8 + base64 wire form."""
    vector = unit_embedding(seed)
    scale = max(abs(value) for value in vector)
    quantised = bytes(
        (int(round(value / scale * 127)) & 0xFF) for value in vector
    )
    return base64.b64encode(quantised).decode("ascii")


def make_observation(
    plate: str = "MH12AB1234",
    *,
    device: str = "edge-a12",
    tracklet: str = "t-0001",
    timestamp_ms: int | None = None,
    **overrides: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "edge_device_id": device,
        "camera_id": "cam-a12-north",
        "tracklet_id": tracklet,
        # Stage 1 derives this as blake2b(node|camera|track|first_ts): opaque,
        # deterministic, and containing no plate text. Mirror that here.
        "pass_id": hashlib.blake2b(
            f"{device}|{tracklet}|{timestamp_ms or NOW_MS}".encode(), digest_size=10
        ).hexdigest(),
        "timestamp_epoch_ms": timestamp_ms if timestamp_ms is not None else NOW_MS,
        "plate_number_decoded": plate,
        "plate_sequence_confidence": 0.94,
        "character_confidences": [0.95] * len(plate),
        "is_dual_line_plate": False,
        "plate_series": "private_white",
        "text_entropy": 0.12,
        "repair_cost_nats": 0.0,
        "plate_corners": {
            "top_left": [100.0, 200.0],
            "top_right": [260.0, 204.0],
            "bottom_right": [258.0, 252.0],
            "bottom_left": [98.0, 248.0],
        },
        "vehicle_bounding_box": {"x1": 40.0, "y1": 90.0, "x2": 380.0, "y2": 420.0},
        "reid_embedding_128d": unit_embedding(7),
        "vehicle_class": "car",
        "vehicle_speed_kmh": 42.5,
        "travel_heading_azimuth": 187.4,
        "evidence_crop_s3_key": "evidence/2026/09/18/edge-a12/t-0001.jpg",
        "camera_location": {"latitude": 19.0330, "longitude": 73.0297},
        "corridor_id": "corridor-palm-beach",
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def settings() -> Settings:
    return build_dev_settings()


@pytest.fixture()
def crypto_pipeline(settings: Settings) -> DpdpCryptoPipeline:
    return DpdpCryptoPipeline.from_settings(settings.crypto)


@pytest.fixture()
def database() -> InMemoryDatabase:
    return InMemoryDatabase()


@pytest.fixture()
def hotlist_engine() -> HotlistEngine:
    params = bloom_parameters(50_000, 1e-4)
    return HotlistEngine(InMemoryBloomBackend(params), timeout_s=0.5)


@pytest_asyncio.fixture()
async def client(
    settings: Settings,
    database: InMemoryDatabase,
    hotlist_engine: HotlistEngine,
    crypto_pipeline: DpdpCryptoPipeline,
) -> AsyncClient:
    await hotlist_engine.register(
        HotlistMatch(
            plate_number="MH02WANTED1"[:10],
            fir_number="FIR/2026/00412",
            warrant_reference="WNT-MH-2026-00412",
            severity=AlertSeverity.CRITICAL,
            offence_category="VEHICLE_THEFT",
            issuing_authority="Navi Mumbai Police",
        )
    )
    await hotlist_engine.register(
        HotlistMatch(
            plate_number="DL8CAF5511",
            fir_number="FIR/2026/00981",
            warrant_reference="WNT-DL-2026-00981",
            severity=AlertSeverity.HIGH,
            offence_category="TRAFFIC_VIOLATION",
            issuing_authority="Delhi Traffic Police",
        )
    )
    app = create_app(
        settings, database=database, hotlist=hotlist_engine, crypto=crypto_pipeline
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://gateway") as http_client:
        # Trigger the lifespan so app.state.veritrack exists.
        async with app.router.lifespan_context(app):
            yield http_client


# =====================================================================
# 1. Schema validation
# =====================================================================


async def test_valid_observation_parses_and_derives_fields() -> None:
    observation = EdgeObservation.model_validate(make_observation())
    assert observation.plate_number_decoded == "MH12AB1234"
    assert observation.is_valid_format is True
    assert observation.timestamp_utc.tzinfo is timezone.utc
    assert abs(sum(c * c for c in observation.reid_embedding_128d) - 1.0) < 1e-9


async def test_plate_is_normalised_before_validation() -> None:
    observation = EdgeObservation.model_validate(
        make_observation(plate="mh 12-ab 1234", character_confidences=[0.9] * 10)
    )
    assert observation.plate_number_decoded == "MH12AB1234"


@pytest.mark.parametrize(
    "plate,expected",
    [
        ("MH12AB1234", True),
        ("DL8CAF5511", True),
        ("22BH1234AB", True),      # Bharat series
        ("KA01A1234", True),       # single series letter
        ("ZZ12AB1234", False),     # ZZ is not an RTO code
        ("MH12AB123", False),      # three trailing digits
        ("1234ABCD", False),
    ],
)
async def test_plate_grammar(plate: str, expected: bool) -> None:
    assert is_valid_indian_plate(plate) is expected


async def test_character_confidence_length_must_match_plate() -> None:
    with pytest.raises(ValidationError, match="character_confidences length"):
        EdgeObservation.model_validate(
            make_observation(character_confidences=[0.9, 0.9, 0.9])
        )


async def test_embedding_wrong_dimension_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EdgeObservation.model_validate(
            make_observation(reid_embedding_128d=unit_embedding(1)[:64])
        )


async def test_base64_int8_embedding_is_dequantised_and_renormalised() -> None:
    observation = EdgeObservation.model_validate(
        make_observation(reid_embedding_128d=quantised_embedding(7))
    )
    assert len(observation.reid_embedding_128d) == 128
    norm = math.sqrt(sum(v * v for v in observation.reid_embedding_128d))
    assert abs(norm - 1.0) < 1e-9
    # int8 quantisation must not move the vector meaningfully: cosine against
    # the float original stays far above the 0.35 Stage 3 divergence threshold.
    reference = unit_embedding(7)
    cosine = sum(a * b for a, b in zip(observation.reid_embedding_128d, reference))
    assert cosine > 0.999


async def test_degenerate_plate_quad_is_rejected() -> None:
    with pytest.raises(ValidationError, match="degenerate"):
        EdgeObservation.model_validate(
            make_observation(
                plate_corners={
                    "top_left": [100.0, 200.0],
                    "top_right": [100.0, 200.0],
                    "bottom_right": [100.0, 200.0],
                    "bottom_left": [100.0, 200.0],
                }
            )
        )


async def test_inverted_bounding_box_is_rejected() -> None:
    with pytest.raises(ValidationError, match="x2 > x1"):
        EdgeObservation.model_validate(
            make_observation(vehicle_bounding_box={"x1": 400.0, "y1": 90.0,
                                                   "x2": 40.0, "y2": 420.0})
        )


async def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EdgeObservation.model_validate(make_observation(injected_field="malicious"))


async def test_evidence_key_path_traversal_is_rejected() -> None:
    with pytest.raises(ValidationError, match="traversal"):
        EdgeObservation.model_validate(
            make_observation(evidence_crop_s3_key="../../etc/passwd")
        )


@pytest.mark.parametrize("field_name", ["pass_id", "tracklet_id", "corridor_id"])
async def test_identifier_embedding_the_plate_is_rejected(field_name: str) -> None:
    """No accepted payload may smuggle cleartext into the pseudonymised store.

    `pass_id`, `tracklet_id` and `corridor_id` are copied verbatim into the
    `sightings` hypertable, which holds no cleartext plate column by design. If
    an identifier contained the plate, pseudonymisation would be silently
    defeated for that row.
    """
    with pytest.raises(ValidationError, match="embeds the decoded plate"):
        EdgeObservation.model_validate(
            make_observation(**{field_name: "prefix-MH12AB1234-suffix"})
        )


async def test_opaque_identifiers_are_accepted() -> None:
    """The guard must not reject a legitimate hashed identifier."""
    observation = EdgeObservation.model_validate(
        make_observation(pass_id="a1b2c3d4e5f60718", tracklet_id="t-0001")
    )
    assert observation.stable_pass_id() == "a1b2c3d4e5f60718"


async def test_batch_must_come_from_a_single_device() -> None:
    with pytest.raises(ValidationError, match="single edge_device_id"):
        IngestBatch.model_validate(
            {
                "observations": [
                    make_observation(device="edge-a12"),
                    make_observation(device="edge-b07", tracklet="t-0002"),
                ]
            }
        )


async def test_compact_stage1_dialect_round_trips() -> None:
    compact = {
        "node_id": "edge-a12",
        "camera_id": "cam-a12-north",
        "track_id": "t-0042",
        "pass_id": "a1b2c3d4e5",
        "ts_first": NOW_MS,
        "ts_last": NOW_MS + 900,
        "plate": {
            "text": "MH12AB1234",
            "conf": 0.91,
            "char_confidence": [0.9] * 10,
            "layout": "single_line",
            "series": "private_white",
            "entropy": 0.21,
            "repair_cost": 0.318,
            "valid": True,
        },
        "geometry": {"q": [100.0, 200.0, 260.0, 204.0, 258.0, 252.0, 98.0, 248.0]},
        "box": [40.0, 90.0, 380.0, 420.0],
        "reid": {"int8": quantised_embedding(3)},
        "vehicle_class": "car",
        "speed_kmh": 51.0,
        "heading_deg": 12.0,
    }
    observation = EdgeObservation.from_compact_payload(compact)
    assert observation.edge_device_id == "edge-a12"
    assert observation.tracklet_id == "t-0042"
    assert observation.plate_number_decoded == "MH12AB1234"
    assert observation.repair_cost_nats == pytest.approx(0.318)
    assert observation.stable_pass_id() == "a1b2c3d4e5"
    assert len(observation.reid_embedding_128d) == 128


# =====================================================================
# 2. Bloom filter
# =====================================================================


async def test_bloom_sizing_matches_the_closed_form() -> None:
    params = bloom_parameters(20_000_000, 1e-4)
    expected_bits = math.ceil(-20_000_000 * math.log(1e-4) / (math.log(2) ** 2))
    assert params.num_bits == expected_bits
    assert params.num_hashes == round((params.num_bits / params.capacity) * math.log(2))
    # ~48 MB and k = 13 for the national hotlist.
    assert 45 <= params.num_bytes / (1024 * 1024) <= 50
    assert params.num_hashes == 13
    assert params.realised_error_rate() < 2e-4


async def test_bloom_offsets_are_deterministic_and_in_range() -> None:
    params = bloom_parameters(10_000, 1e-3)
    first = bit_offsets("MH12AB1234", params.num_bits, params.num_hashes)
    second = bit_offsets("MH12AB1234", params.num_bits, params.num_hashes)
    assert first == second
    assert len(set(first)) == len(first), "offsets must not collide within one item"
    assert all(0 <= offset < params.num_bits for offset in first)
    assert first != bit_offsets("MH12AB1235", params.num_bits, params.num_hashes)


async def test_bloom_has_no_false_negatives() -> None:
    """The load-bearing guarantee: a wanted vehicle is never missed."""
    params = bloom_parameters(20_000, 1e-4)
    backend = InMemoryBloomBackend(params)
    plates = [f"MH{index % 90:02d}AB{index:04d}" for index in range(5_000)]
    await backend.add_many(plates)
    for plate in plates:
        assert await backend.contains(plate), f"false negative on {plate}"


async def test_bloom_false_positive_rate_is_within_the_predicted_bound() -> None:
    params = bloom_parameters(20_000, 1e-3)
    backend = InMemoryBloomBackend(params)
    inserted = [f"MH01AB{index:04d}" for index in range(10_000)]
    await backend.add_many(inserted)

    absent = [f"KA05XY{index:04d}" for index in range(20_000)]
    false_positives = 0
    for plate in absent:
        if await backend.contains(plate):
            false_positives += 1
    observed = false_positives / len(absent)
    predicted = params.realised_error_rate(len(inserted))
    # Generous headroom: this is a probabilistic assertion, so it is checked
    # against a multiple of the analytic bound rather than the bound itself.
    assert observed <= max(predicted * 5.0, 1e-3), (
        f"observed FP {observed:.2e} exceeds predicted {predicted:.2e}"
    )


async def test_hotlist_confirms_hits_and_rejects_bloom_false_positives(
    hotlist_engine: HotlistEngine,
) -> None:
    await hotlist_engine.register(
        HotlistMatch(
            plate_number="MH12AB1234",
            fir_number="FIR/2026/00001",
            warrant_reference="WNT-1",
            severity=AlertSeverity.HIGH,
        )
    )
    hit = await hotlist_engine.check("MH12AB1234")
    assert hit.bloom_hit and hit.confirmed
    assert hit.match is not None and hit.match.fir_number == "FIR/2026/00001"
    assert hit.severity is AlertSeverity.HIGH

    miss = await hotlist_engine.check("KA05ZZ9999")
    assert not miss.bloom_hit and not miss.confirmed and miss.match is None

    # A plate the filter believes in but the case store does not know is a
    # false positive: the authoritative lookup must reject it.
    forced = await hotlist_engine.check("MH12AB1234")
    assert forced.confirmed
    hotlist_engine._local_meta.pop("MH12AB1234")  # simulate a purged case record
    resolved = await hotlist_engine.check("MH12AB1234")
    assert resolved.bloom_hit and not resolved.confirmed and resolved.false_positive


async def test_hotlist_honours_warrant_expiry(hotlist_engine: HotlistEngine) -> None:
    await hotlist_engine.register(
        HotlistMatch(
            plate_number="GJ01KL4567",
            fir_number="FIR/2025/77",
            warrant_reference="WNT-EXPIRED",
            severity=AlertSeverity.ELEVATED,
            expires_at_epoch_ms=NOW_MS - 86_400_000,
        )
    )
    decision = await hotlist_engine.check("GJ01KL4567", now_epoch_ms=NOW_MS)
    assert decision.bloom_hit
    assert not decision.confirmed, "an expired warrant must not raise an alert"


async def test_hotlist_check_meets_the_sub_50ms_budget(hotlist_engine: HotlistEngine) -> None:
    await hotlist_engine.register(
        HotlistMatch(
            plate_number="MH43Q1234",
            fir_number="FIR/2026/5",
            warrant_reference="WNT-5",
            severity=AlertSeverity.ADVISORY,
        )
    )
    for plate in ("MH43Q1234", "KA51CD9876"):
        decision = await hotlist_engine.check(plate)
        assert decision.latency_ms < 50.0, f"{plate} took {decision.latency_ms:.2f} ms"


async def test_hotlist_batch_check_matches_per_item_results(
    hotlist_engine: HotlistEngine,
) -> None:
    await hotlist_engine.register(
        HotlistMatch(
            plate_number="TN10BB2222",
            fir_number="FIR/2026/9",
            warrant_reference="WNT-9",
            severity=AlertSeverity.HIGH,
        )
    )
    plates = ["TN10BB2222", "KL07CD3333", "TN10BB2222"]
    batch = await hotlist_engine.check_many(plates)
    assert [d.confirmed for d in batch] == [True, False, True]


# =====================================================================
# 3. DPDP cryptography
# =====================================================================


async def test_pseudonymization_is_deterministic_within_an_epoch(
    crypto_pipeline: DpdpCryptoPipeline,
) -> None:
    epoch_s = 1_780_000_000.0
    first = crypto_pipeline.pseudonymize("MH12AB1234", observed_epoch_s=epoch_s)
    second = crypto_pipeline.pseudonymize("MH12AB1234", observed_epoch_s=epoch_s + 3600)
    assert first.digest_hex == second.digest_hex
    assert first.salt_epoch == second.salt_epoch
    assert len(first.digest_hex) == 64
    assert first.prefix == first.digest_hex[: len(first.prefix)]


async def test_pseudonymization_differs_across_plates(
    crypto_pipeline: DpdpCryptoPipeline,
) -> None:
    epoch_s = 1_780_000_000.0
    a = crypto_pipeline.pseudonymize("MH12AB1234", observed_epoch_s=epoch_s)
    b = crypto_pipeline.pseudonymize("MH12AB1235", observed_epoch_s=epoch_s)
    assert a.digest_hex != b.digest_hex


async def test_salt_rotation_breaks_cross_day_linkage(
    crypto_pipeline: DpdpCryptoPipeline,
) -> None:
    """The privacy property: same vehicle, different day, unlinkable pseudonym."""
    day_one = 1_780_000_000.0
    day_two = day_one + 86_400.0
    first = crypto_pipeline.pseudonymize("MH12AB1234", observed_epoch_s=day_one)
    second = crypto_pipeline.pseudonymize("MH12AB1234", observed_epoch_s=day_two)
    assert first.salt_epoch != second.salt_epoch
    assert first.digest_hex != second.digest_hex


async def test_pseudonym_repr_never_leaks_the_digest(
    crypto_pipeline: DpdpCryptoPipeline,
) -> None:
    pseudonym = crypto_pipeline.pseudonymize("MH12AB1234", observed_epoch_s=time.time())
    assert pseudonym.digest_hex not in repr(pseudonym)


async def test_warrant_backed_rematch_is_constant_time(
    crypto_pipeline: DpdpCryptoPipeline,
) -> None:
    epoch_s = time.time()
    pseudonym = crypto_pipeline.pseudonymize("MH12AB1234", observed_epoch_s=epoch_s)
    assert crypto_pipeline.pseudonymizer.matches(
        "MH12AB1234", pseudonym, observed_epoch_s=epoch_s
    )
    assert not crypto_pipeline.pseudonymizer.matches(
        "MH12AB9999", pseudonym, observed_epoch_s=epoch_s
    )


async def test_derived_salts_agree_across_independent_managers() -> None:
    """Two gateway replicas must produce the same pseudonym for the same plate."""
    pepper = bytes(range(64))
    left = PlatePseudonymizer(RotatingSaltManager(DerivedSaltSource(pepper)))
    right = PlatePseudonymizer(RotatingSaltManager(DerivedSaltSource(pepper)))
    epoch_s = 1_780_000_000.0
    assert (
        left.pseudonymize("MH12AB1234", observed_epoch_s=epoch_s).digest_hex
        == right.pseudonymize("MH12AB1234", observed_epoch_s=epoch_s).digest_hex
    )


async def test_ephemeral_salts_do_not_agree_across_instances() -> None:
    """The maximally-private mode is, by construction, not horizontally scalable."""
    epoch_s = 1_780_000_000.0
    left = PlatePseudonymizer(RotatingSaltManager(EphemeralSaltSource()))
    right = PlatePseudonymizer(RotatingSaltManager(EphemeralSaltSource()))
    assert (
        left.pseudonymize("MH12AB1234", observed_epoch_s=epoch_s).digest_hex
        != right.pseudonymize("MH12AB1234", observed_epoch_s=epoch_s).digest_hex
    )


async def test_aes_gcm_envelope_round_trip(crypto_pipeline: DpdpCryptoPipeline) -> None:
    payload = {"plate_number": "MH12AB1234", "crop": "s3://evidence/a.jpg", "speed": 42.5}
    envelope = crypto_pipeline.seal_evidence(payload, pass_id="pass-123")
    assert "MH12AB1234" not in envelope.ciphertext_b64
    recovered = crypto_pipeline.open_evidence(envelope, pass_id="pass-123")
    assert recovered == payload


async def test_aes_gcm_rejects_a_transplanted_ciphertext(
    crypto_pipeline: DpdpCryptoPipeline,
) -> None:
    """AAD binding: a ciphertext lifted onto another row must not decrypt."""
    envelope = crypto_pipeline.seal_evidence({"plate_number": "MH12AB1234"}, pass_id="pass-A")
    with pytest.raises(DecryptionError, match="AAD mismatch"):
        crypto_pipeline.open_evidence(envelope, pass_id="pass-B")


async def test_aes_gcm_detects_ciphertext_tampering(
    crypto_pipeline: DpdpCryptoPipeline,
) -> None:
    envelope = crypto_pipeline.seal_evidence({"plate_number": "MH12AB1234"}, pass_id="pass-A")
    raw = bytearray(base64.b64decode(envelope.ciphertext_b64))
    raw[0] ^= 0x01
    tampered = SealedEnvelope(
        key_version=envelope.key_version,
        wrapped_dek_b64=envelope.wrapped_dek_b64,
        dek_nonce_b64=envelope.dek_nonce_b64,
        ciphertext_b64=base64.b64encode(bytes(raw)).decode("ascii"),
        nonce_b64=envelope.nonce_b64,
        aad=envelope.aad,
    )
    with pytest.raises(DecryptionError, match="authentication failed"):
        crypto_pipeline.open_evidence(tampered, pass_id="pass-A")


async def test_every_envelope_uses_a_fresh_dek_and_nonce(
    crypto_pipeline: DpdpCryptoPipeline,
) -> None:
    """Identical plaintexts must not produce identical ciphertexts."""
    payload = {"plate_number": "MH12AB1234"}
    first = crypto_pipeline.seal_evidence(payload, pass_id="pass-A")
    second = crypto_pipeline.seal_evidence(payload, pass_id="pass-A")
    assert first.ciphertext_b64 != second.ciphertext_b64
    assert first.nonce_b64 != second.nonce_b64
    assert first.wrapped_dek_b64 != second.wrapped_dek_b64


async def test_envelope_json_serialisation_round_trips(
    crypto_pipeline: DpdpCryptoPipeline,
) -> None:
    envelope = crypto_pipeline.seal_evidence({"a": 1}, pass_id="pass-A")
    restored = SealedEnvelope.from_json(envelope.to_json())
    assert crypto_pipeline.open_evidence(restored, pass_id="pass-A") == {"a": 1}


async def test_wrong_kek_cannot_unwrap(crypto_pipeline: DpdpCryptoPipeline) -> None:
    envelope = crypto_pipeline.seal_evidence({"a": 1}, pass_id="pass-A")
    foreign = AesGcmEnvelopeEncryptor(bytes(range(100, 132)), key_version=1)
    with pytest.raises(DecryptionError, match="DEK unwrap failed"):
        foreign.open_json(envelope, aad="pass-A")


async def test_production_settings_reject_placeholder_key_material() -> None:
    with pytest.raises(ValidationError, match="all-zero development placeholder"):
        CryptoSettings(allow_insecure_defaults=False)
    # Explicitly opting in is allowed, for a lab bring-up.
    assert CryptoSettings(allow_insecure_defaults=True) is not None


# =====================================================================
# 4. Rate limiter
# =====================================================================


async def test_token_bucket_allows_a_burst_then_meters() -> None:
    limiter = TokenBucketLimiter(rate_per_second=10.0, burst=5.0)
    for _ in range(5):
        allowed, _ = await limiter.acquire("edge-a12")
        assert allowed
    allowed, retry_after = await limiter.acquire("edge-a12")
    assert not allowed and retry_after > 0.0
    # A different node has its own bucket.
    allowed, _ = await limiter.acquire("edge-b07")
    assert allowed


async def test_token_bucket_refills_over_time() -> None:
    limiter = TokenBucketLimiter(rate_per_second=100.0, burst=2.0)
    await limiter.acquire("edge-a12")
    await limiter.acquire("edge-a12")
    assert not (await limiter.acquire("edge-a12"))[0]
    await asyncio.sleep(0.05)
    assert (await limiter.acquire("edge-a12"))[0]


# =====================================================================
# 5. API ingestion
# =====================================================================


async def test_ingest_single_observation_is_pseudonymized(
    client: AsyncClient, database: InMemoryDatabase
) -> None:
    response = await client.post("/api/v1/telemetry/ingest", json=make_observation())
    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] == 1
    assert body["rejected"] == 0
    assert body["hotlist_hits"] == 0
    assert body["results"][0]["decision"] == ProcessingDecision.PSEUDONYMIZED.value

    assert len(database.sightings) == 1
    row = database.sightings[0]
    assert len(row.plate_pseudonym) == 64
    assert row.hotlist_flag is False
    # The load-bearing DPDP assertion: no cleartext plate anywhere in the row.
    assert "MH12AB1234" not in json.dumps(row.as_record(), default=str)


async def test_ingest_batch_accepts_all_observations(
    client: AsyncClient, database: InMemoryDatabase
) -> None:
    batch = {
        "observations": [
            make_observation(plate="MH12AB1234", tracklet="t-1"),
            make_observation(plate="KA05MN7788", tracklet="t-2",
                             character_confidences=[0.9] * 10),
            make_observation(plate="GJ01KL4567", tracklet="t-3",
                             character_confidences=[0.9] * 10),
        ],
        "edge_batch_id": "batch-001",
    }
    response = await client.post("/api/v1/telemetry/ingest", json=batch)
    assert response.status_code == 202
    assert response.json()["accepted"] == 3
    assert len(database.sightings) == 3
    assert len({row.plate_pseudonym for row in database.sightings}) == 3


async def test_hotlist_hit_stores_cleartext_and_sealed_evidence(
    client: AsyncClient,
    database: InMemoryDatabase,
    crypto_pipeline: DpdpCryptoPipeline,
) -> None:
    response = await client.post(
        "/api/v1/telemetry/ingest",
        json=make_observation(plate="DL8CAF5511", character_confidences=[0.93] * 10),
    )
    assert response.status_code == 202
    body = response.json()
    assert body["hotlist_hits"] == 1
    result = body["results"][0]
    assert result["decision"] == ProcessingDecision.HOTLIST_CLEARTEXT.value
    assert result["hotlist_hit"] is True
    assert result["severity"] == int(AlertSeverity.HIGH)

    assert len(database.hotlist_hits) == 1
    hit = database.hotlist_hits[0]
    assert hit.plate_number == "DL8CAF5511"
    assert hit.fir_number == "FIR/2026/00981"
    assert hit.warrant_reference == "WNT-DL-2026-00981"
    assert hit.evidence_envelope is not None

    # The sealed evidence must decrypt only under its own pass_id.
    envelope = SealedEnvelope.from_json(hit.evidence_envelope)
    evidence = crypto_pipeline.open_evidence(envelope, pass_id=hit.pass_id)
    assert evidence["plate_number"] == "DL8CAF5511"
    assert "DL8CAF5511" not in envelope.ciphertext_b64

    # The vehicle also appears in the analytics stream, pseudonymised and flagged.
    assert len(database.sightings) == 1
    assert database.sightings[0].hotlist_flag is True
    assert database.sightings[0].plate_pseudonym != "DL8CAF5511"


async def test_same_plate_yields_the_same_pseudonym_across_cameras(
    client: AsyncClient, database: InMemoryDatabase
) -> None:
    """Same-day linkage is what corridor and O-D analytics depend on."""
    await client.post(
        "/api/v1/telemetry/ingest",
        json=make_observation(tracklet="t-a", camera_id="cam-1",
                              pass_id="p-a", device="edge-a12"),
    )
    await client.post(
        "/api/v1/telemetry/ingest",
        json=make_observation(tracklet="t-b", camera_id="cam-2",
                              pass_id="p-b", device="edge-a12"),
    )
    assert len(database.sightings) == 2
    assert database.sightings[0].plate_pseudonym == database.sightings[1].plate_pseudonym
    assert database.sightings[0].camera_id != database.sightings[1].camera_id


async def test_future_timestamp_is_rejected(
    client: AsyncClient, database: InMemoryDatabase
) -> None:
    response = await client.post(
        "/api/v1/telemetry/ingest",
        json=make_observation(timestamp_ms=NOW_MS + 600_000),
    )
    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] == 0 and body["rejected"] == 1
    assert "future" in body["results"][0]["error"]
    assert database.sightings == []


async def test_stale_timestamp_beyond_replay_horizon_is_rejected(
    client: AsyncClient, database: InMemoryDatabase
) -> None:
    response = await client.post(
        "/api/v1/telemetry/ingest",
        json=make_observation(timestamp_ms=NOW_MS - 5 * 86_400_000),
    )
    body = response.json()
    assert body["rejected"] == 1
    assert "replay horizon" in body["results"][0]["error"]
    assert database.sightings == []


async def test_malformed_payload_is_rejected_with_422(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/telemetry/ingest",
        json=make_observation(plate_sequence_confidence=1.7),
    )
    assert response.status_code == 422


async def test_backpressure_surfaces_as_503(
    client: AsyncClient, database: InMemoryDatabase
) -> None:
    database.fail_next_write()
    response = await client.post("/api/v1/telemetry/ingest", json=make_observation())
    assert response.status_code == 503
    assert response.headers.get("Retry-After") == "2"


async def test_compact_endpoint_ingests_stage1_payloads(
    client: AsyncClient, database: InMemoryDatabase
) -> None:
    compact = {
        "node_id": "edge-a12",
        "camera_id": "cam-a12-north",
        "track_id": "t-9001",
        "pass_id": "compact-pass-1",
        "ts_first": NOW_MS,
        "plate": {"text": "MH12AB1234", "conf": 0.88, "char_confidence": [0.88] * 10,
                  "series": "private_white", "valid": True},
        "geometry": {"q": [100.0, 200.0, 260.0, 204.0, 258.0, 252.0, 98.0, 248.0]},
        "box": [40.0, 90.0, 380.0, 420.0],
        "reid": {"int8": quantised_embedding(11)},
        "vehicle_class": "car",
    }
    response = await client.post("/api/v1/telemetry/ingest/compact", json=[compact])
    assert response.status_code == 202
    assert response.json()["accepted"] == 1
    assert database.sightings[0].pass_id == "compact-pass-1"


async def test_compact_endpoint_reports_unparseable_items(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/telemetry/ingest/compact", json=[{"node_id": "edge-a12"}]
    )
    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] == 0 and body["rejected"] == 1
    assert "compact payload rejected" in body["results"][0]["error"]


async def test_request_id_is_echoed(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/telemetry/ingest",
        json=make_observation(),
        headers={"x-request-id": "trace-abc-123"},
    )
    assert response.headers["x-request-id"] == "trace-abc-123"
    assert "server-timing" in response.headers


# =====================================================================
# 6. Operational endpoints
# =====================================================================


async def test_healthz_reports_every_component(client: AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    names = {component["name"] for component in body["components"]}
    assert names == {"database", "hotlist", "crypto", "dispatcher"}
    crypto_component = next(c for c in body["components"] if c["name"] == "crypto")
    assert "salt_epoch=" in crypto_component["detail"]
    # /healthz must never expose key material.
    assert "pepper" not in response.text.lower()


async def test_metrics_endpoint_exposes_prometheus_text(client: AsyncClient) -> None:
    await client.post("/api/v1/telemetry/ingest", json=make_observation())
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "veritrack_observations_total" in response.text
    assert "veritrack_ingest_duration_seconds" in response.text


async def test_hotlist_stats_reports_filter_geometry(client: AsyncClient) -> None:
    response = await client.get("/api/v1/hotlist/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["num_hashes"] >= 1
    assert body["inserted"] >= 2
    assert 0.0 <= body["saturation"] < 1.0
    assert body["realised_error_rate"] < 1e-2


async def test_readyz_reflects_buffer_state(client: AsyncClient) -> None:
    response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["ready"] is True


# =====================================================================
# 7. Authentication
# =====================================================================


async def test_bearer_token_is_enforced_when_configured(
    database: InMemoryDatabase,
    hotlist_engine: HotlistEngine,
    crypto_pipeline: DpdpCryptoPipeline,
) -> None:
    from pydantic import SecretStr

    secured = build_dev_settings()
    secured.server.ingest_api_key = SecretStr("s3cret-edge-token")
    app = create_app(secured, database=database, hotlist=hotlist_engine, crypto=crypto_pipeline)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://gateway") as http_client:
        async with app.router.lifespan_context(app):
            unauthenticated = await http_client.post(
                "/api/v1/telemetry/ingest", json=make_observation()
            )
            assert unauthenticated.status_code == 401

            wrong = await http_client.post(
                "/api/v1/telemetry/ingest",
                json=make_observation(),
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert wrong.status_code == 401

            authorised = await http_client.post(
                "/api/v1/telemetry/ingest",
                json=make_observation(),
                headers={"Authorization": "Bearer s3cret-edge-token"},
            )
            assert authorised.status_code == 202
