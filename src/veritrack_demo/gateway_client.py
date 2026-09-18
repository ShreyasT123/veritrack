"""Optional forwarding of demo detections to the Stage 2 ingestion gateway.

This closes the loop for a full end-to-end demo — webcam on the laptop,
gateway in Docker on the same machine — but it is honest about what it can and
cannot claim to have measured.

Stage 2's :class:`~veritrack_server.schemas.EdgeObservation` requires a vehicle
bounding box and a 128-D Re-ID embedding, because a real Stage 1 pole node
runs a vehicle detector and an OSNet extractor to produce them. This demo runs
neither: it is a plate localizer plus a text recognizer. Two fields are
therefore filled with clearly-marked stand-ins rather than silently forwarded
as if they were real detections:

* ``vehicle_bounding_box`` — the *plate's own* box, expanded by a fixed
  margin. It is not a vehicle detection.
* ``reid_embedding_128d`` — a deterministic pseudo-embedding derived from the
  decoded text and geometry, purely so that repeated sightings of the same
  demo plate are self-consistent within one session. It is not a visual
  appearance vector and must never be treated as one.

Every payload this module builds carries ``"demo_stand_in": true`` alongside
those two fields so nothing downstream can mistake them for production data
by accident.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import GatewayConfig
from .pipeline import DemoDetection

__all__ = ["ForwardResult", "build_compact_payload", "GatewayForwarder"]

#: Fraction by which the plate box is expanded to stand in for a vehicle box.
#: Purely cosmetic for the demo — see the module docstring.
_VEHICLE_BOX_MARGIN = 2.5


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """Outcome of one forward attempt."""

    success: bool
    status_code: Optional[int]
    error: Optional[str]
    latency_ms: float


def _pseudo_embedding(text: str, seed_extra: float) -> list:
    """128 deterministic, non-degenerate floats. Not a Re-ID vector — see module docstring."""
    digest = hashlib.blake2b(f"{text}|{seed_extra:.4f}".encode("utf-8"), digest_size=64).digest()
    # Two bytes per component keeps the values spread rather than clustering
    # near zero, which matters because Stage 2 rejects a near-zero-norm vector.
    values = []
    for index in range(128):
        byte_pair = digest[(index * 2) % 64], digest[(index * 2 + 1) % 64]
        values.append((byte_pair[0] - 128) / 128.0 + (byte_pair[1] - 128) / 32768.0)
    return values


def build_compact_payload(
    detection: DemoDetection,
    *,
    node_id: str,
    camera_id: str,
    session_id: str,
    sequence_number: int,
) -> Dict[str, Any]:
    """Build a Stage 1 compact-dialect payload for one detection.

    Matches ``EdgeObservation.from_compact_payload``'s expected keys exactly:
    ``node_id``, ``camera_id``, ``track_id``, ``pass_id``, ``ts_first``,
    ``plate.{text,conf,char_confidence,layout,series,valid}``,
    ``geometry.q`` (flat 8-value quad), ``box`` (vehicle box), and
    ``reid.f32``.

    ``track_id`` and ``pass_id`` are derived from ``session_id`` and a
    sequence counter, never from the plate text — Stage 2 rejects any
    identifier that embeds the decoded plate, since those fields are stored
    verbatim in the pseudonymised table.
    """
    candidate = detection.candidate
    text = detection.validated_text or detection.raw_text
    now_ms = int(time.time() * 1000)
    track_id = hashlib.blake2b(f"{session_id}|{candidate.source}".encode(), digest_size=8).hexdigest()
    pass_id = hashlib.blake2b(f"{session_id}|{sequence_number}|{now_ms}".encode(), digest_size=10).hexdigest()

    char_confidence = [detection.recognition.confidence] * max(len(text), 1)

    margin_x = candidate.width * (_VEHICLE_BOX_MARGIN - 1.0) / 2.0
    margin_y = candidate.height * (_VEHICLE_BOX_MARGIN - 1.0) / 2.0
    vehicle_box = [
        max(0.0, candidate.x1 - margin_x),
        max(0.0, candidate.y1 - margin_y),
        candidate.x2 + margin_x,
        candidate.y2 + margin_y,
    ]

    return {
        "node_id": node_id,
        "camera_id": camera_id,
        "track_id": track_id,
        "pass_id": pass_id,
        "ts_first": now_ms,
        "plate": {
            "text": text,
            "conf": detection.recognition.confidence,
            "char_confidence": char_confidence,
            "layout": "single_line",
            "series": "unknown",
            "repair_cost": detection.repair_cost,
            "valid": detection.is_valid_format,
        },
        "geometry": {"q": [float(v) for v in candidate.quad.flatten()]},
        "box": vehicle_box,
        "reid": {"f32": _pseudo_embedding(text, candidate.x1 + candidate.y1)},
        "vehicle_class": "unknown",
        # Explicit marker: neither the vehicle box nor the embedding above is
        # a real detection. See the module docstring for why they exist at all.
        "demo_stand_in": True,
    }


class GatewayForwarder:
    """POSTs compact-dialect payloads to the Stage 2 gateway over plain HTTP.

    Uses only the standard library (``urllib``) so the demo's dependency list
    does not grow just to support an optional feature most people running the
    webcam-only path will never enable.
    """

    __slots__ = ("_config", "_session_id", "_sequence", "_sent_count")

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._session_id = uuid.uuid4().hex
        self._sequence = 0
        self._sent_count = 0

    @property
    def sent_count(self) -> int:
        return self._sent_count

    def build_payload(self, detection: DemoDetection) -> Dict[str, Any]:
        self._sequence += 1
        return build_compact_payload(
            detection,
            node_id=self._config.node_id,
            camera_id=self._config.camera_id,
            session_id=self._session_id,
            sequence_number=self._sequence,
        )

    def forward(self, detection: DemoDetection) -> ForwardResult:
        """Best-effort POST. Never raises — a live demo must not crash because
        the gateway happened to be unreachable for one frame."""
        payload = self.build_payload(detection)
        body = json.dumps([payload]).encode("utf-8")
        url = self._config.base_url.rstrip("/") + self._config.ingest_path
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self._config.timeout_s) as response:
                status = response.getcode()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self._sent_count += 1
                return ForwardResult(success=200 <= status < 300, status_code=status,
                                     error=None, latency_ms=elapsed_ms)
        except urllib.error.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return ForwardResult(success=False, status_code=exc.code,
                                 error=exc.reason, latency_ms=elapsed_ms)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return ForwardResult(success=False, status_code=None,
                                 error=str(exc), latency_ms=elapsed_ms)
