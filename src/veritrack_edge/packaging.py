"""Edge -> gateway metadata packaging.

The wire contract is a compact JSON object under 5 KB. At 3,000 cameras and a
peak of 8 passes per camera per minute that is a sustained ~2 MB/s across the
city backhaul, which a single Kafka partition set absorbs comfortably.

Budget, worst case
------------------
=========================  ========
field group                bytes
=========================  ========
identity + timestamps           ~230
plate + grammar metadata        ~220
per-character confidences       ~180
Re-ID embedding (int8 b64)       172
geometry + quality               ~200
stage timings                    ~230
=========================  ========

Comfortably inside 5,120 bytes with headroom for a warrant/hotlist correlation
id added at the gateway. If a payload nevertheless exceeds the ceiling - an
unusually long alternatives list, say - optional fields are shed in a fixed
priority order rather than the record being dropped: a pass with no debug
timings is still a usable trajectory point, whereas a dropped pass is a hole in
the HMM.

Privacy note
------------
The plate leaves the edge node in **plaintext**. Pseudonymisation is applied at
the gateway (Stage 2), not here, and deliberately so: the rolling HMAC salt is
rotated centrally and must never be resident on a pole-mounted device that can
be physically removed. The edge-to-gateway link is mTLS; the optional device
HMAC below authenticates the *origin* of a record, which is what makes the
audit trail non-repudiable, and is not a confidentiality control.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .config import PackagingConfig
from .errors import PayloadTooLargeError
from .reid import quantize_embedding
from .types import VehiclePass

logger = logging.getLogger(__name__)

# Shed order when the payload exceeds the ceiling. Earlier entries go first.
_SHED_ORDER: tuple[str, ...] = (
    "timings",
    "alternatives",
    "char_confidence",
    "geometry",
)


def _round_floats(value: Any, digits: int = 4) -> Any:
    """Recursively round floats; JSON of a float32 otherwise costs 17 bytes."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(v, digits) for v in value]
    return value


class EdgePayloadBuilder:
    """Serialises a :class:`VehiclePass` into the gateway wire format."""

    __slots__ = ("_config", "_key")

    def __init__(self, config: PackagingConfig) -> None:
        self._config = config
        self._key: Optional[bytes] = None
        if config.device_hmac_key_hex:
            try:
                self._key = bytes.fromhex(config.device_hmac_key_hex)
            except ValueError as exc:
                raise ValueError("device_hmac_key_hex is not valid hexadecimal") from exc

    def build(self, vehicle_pass: VehiclePass) -> Dict[str, Any]:
        """Assemble the payload dictionary (pre-serialisation)."""
        cfg = self._config
        reading = vehicle_pass.reading

        payload: Dict[str, Any] = {
            "v": cfg.schema_version,
            "pass_id": vehicle_pass.pass_id,
            "camera_id": vehicle_pass.camera_id,
            "node_id": vehicle_pass.node_id,
            "track_id": vehicle_pass.track_id,
            "vehicle_class": vehicle_pass.vehicle_class,
            "ts_first": round(vehicle_pass.first_seen, 3),
            "ts_last": round(vehicle_pass.last_seen, 3),
            "dwell_s": round(max(0.0, vehicle_pass.last_seen - vehicle_pass.first_seen), 3),
        }

        if reading is not None:
            payload["plate"] = {
                "text": reading.text,
                "raw": reading.raw_text if reading.raw_text != reading.text else None,
                "conf": round(reading.confidence, 4),
                "entropy": round(reading.sequence_entropy, 4),
                "template": reading.template_id,
                "valid": reading.is_valid_format,
                "repaired": reading.was_repaired,
                "repair_cost": round(reading.repair_cost, 3) if reading.was_repaired else None,
                "state": reading.state_code,
                "layout": reading.layout.value,
                "series": reading.series.value,
                "obs": reading.observation_count,
            }
            payload["plate"] = {k: v for k, v in payload["plate"].items() if v is not None}
            if cfg.include_char_confidence and reading.char_confidences:
                payload["char_confidence"] = [round(c, 3) for c in reading.char_confidences]
            if reading.alternatives:
                payload["alternatives"] = [
                    {"text": alt.text, "lp": round(alt.log_prob, 3)}
                    for alt in reading.alternatives[:3]
                ]
        else:
            payload["plate"] = None

        if vehicle_pass.embedding is not None:
            embedding = np.asarray(vehicle_pass.embedding, dtype=np.float32)
            if cfg.quantize_embedding:
                payload["reid"] = {"q": "int8", "d": int(embedding.size), "e": quantize_embedding(embedding)}
            else:
                payload["reid"] = {
                    "q": "f32",
                    "d": int(embedding.size),
                    "e": [round(float(x), 5) for x in embedding],
                }

        payload["geometry"] = {
            "entry": [round(v, 1) for v in vehicle_pass.entry_box.as_tuple()],
            "exit": [round(v, 1) for v in vehicle_pass.exit_box.as_tuple()],
        }

        if cfg.include_debug_timings:
            payload["timings"] = vehicle_pass.timings.as_dict()

        return _round_floats(payload)

    def _sign(self, body: str) -> str:
        assert self._key is not None
        return hmac.new(self._key, body.encode("utf-8"), hashlib.sha256).hexdigest()

    def serialize(self, vehicle_pass: VehiclePass) -> bytes:
        """Serialise to compact UTF-8 JSON within the byte ceiling.

        Raises:
            PayloadTooLargeError: the payload exceeds the ceiling even after all
                optional fields have been shed.
        """
        payload = self.build(vehicle_pass)
        shed: List[str] = []

        for attempt in range(len(_SHED_ORDER) + 1):
            body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True, sort_keys=False)
            if self._key is not None:
                envelope = {"body": payload, "sig": self._sign(body)}
                encoded = json.dumps(
                    envelope, separators=(",", ":"), ensure_ascii=True
                ).encode("utf-8")
            else:
                encoded = body.encode("utf-8")

            if len(encoded) <= self._config.max_payload_bytes:
                if shed:
                    logger.warning(
                        "pass %s exceeded budget; shed %s", vehicle_pass.pass_id, ", ".join(shed)
                    )
                return encoded

            if attempt >= len(_SHED_ORDER):
                break
            field = _SHED_ORDER[attempt]
            if payload.pop(field, None) is not None:
                shed.append(field)

        raise PayloadTooLargeError(
            f"Pass {vehicle_pass.pass_id} exceeds {self._config.max_payload_bytes} bytes "
            "after shedding every optional field"
        )
