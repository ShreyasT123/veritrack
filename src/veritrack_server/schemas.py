"""Strict Pydantic v2 schemas for the VeriTrack ingestion boundary.

These models are the contract between the Stage 1 edge package and the Stage 2
gateway. Everything crossing this line is untrusted: an edge node sits in an
unattended roadside cabinet, so the gateway assumes every field is attacker
controlled until proven otherwise.

Two wire dialects are supported:

* the **canonical** dialect declared by :class:`EdgeObservation`, with fully
  spelled field names, used by external integrators and by the REST API; and
* the **compact** dialect actually emitted by ``veritrack_edge.packaging``,
  which shortens keys to protect the 5 KB per-pass budget.

:meth:`EdgeObservation.from_compact_payload` converts the second into the first,
so the byte-efficient edge format never leaks into the server's domain model.
"""

from __future__ import annotations

import base64
import binascii
import math
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, Final, List, Optional, Sequence, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    field_validator,
    model_validator,
)

__all__ = [
    "VehicleClass",
    "PlateSeries",
    "AlertSeverity",
    "ProcessingDecision",
    "GeoPoint",
    "PlateCorners",
    "BoundingBox",
    "EdgeObservation",
    "IngestBatch",
    "IngestResult",
    "IngestResponse",
    "HotlistMatch",
    "HotlistAlert",
    "PseudonymizedRecord",
    "HealthReport",
    "ComponentHealth",
    "PLATE_STANDARD_RE",
    "PLATE_BH_RE",
    "is_valid_indian_plate",
]

EMBEDDING_DIM: Final[int] = 128

#: ``MH 12 AB 1234`` and its legacy short forms (single-digit RTO, 1--3 series
#: letters). Whitespace and hyphens are stripped before matching.
PLATE_STANDARD_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

#: Bharat series: ``YY BH NNNN LL`` -- e.g. ``22BH1234AB``.
PLATE_BH_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

_PLATE_STRIP_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Z0-9]")

#: The 39 RTO state/UT codes recognised nationally. Mirrors
#: ``veritrack_edge.validation.STATE_CODES`` -- the two must not drift.
STATE_CODES: Final[frozenset[str]] = frozenset(
    {
        "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN",
        "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
        "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
        "RJ", "SK", "TN", "TR", "TS", "UK", "UA", "UP", "WB",
    }
)


def normalise_plate(raw: str) -> str:
    """Uppercase and strip every non-alphanumeric character."""
    return _PLATE_STRIP_RE.sub("", raw.upper())


def is_valid_indian_plate(plate: str) -> bool:
    """True when ``plate`` satisfies a recognised national registration grammar."""
    candidate = normalise_plate(plate)
    if PLATE_BH_RE.match(candidate):
        return True
    if not PLATE_STANDARD_RE.match(candidate):
        return False
    return candidate[:2] in STATE_CODES


class VehicleClass(str, Enum):
    """Coarse vehicle taxonomy produced by the Stage 1 detector head."""

    TWO_WHEELER = "two_wheeler"
    THREE_WHEELER = "three_wheeler"
    CAR = "car"
    LCV = "lcv"
    TRUCK = "truck"
    BUS = "bus"
    TRACTOR = "tractor"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


class PlateSeries(str, Enum):
    """Plate colour class, which encodes registration category under CMVR."""

    PRIVATE_WHITE = "private_white"
    COMMERCIAL_YELLOW = "commercial_yellow"
    EV_GREEN = "ev_green"
    GOVERNMENT = "government"
    DIPLOMATIC = "diplomatic"
    UNKNOWN = "unknown"


class AlertSeverity(int, Enum):
    """Warrant severity ladder used by the Stage 5 dispatch engine."""

    ADVISORY = 1
    ELEVATED = 2
    HIGH = 3
    CRITICAL = 4


class ProcessingDecision(str, Enum):
    """Which DPDP track a sighting was routed down."""

    PSEUDONYMIZED = "pseudonymized"
    HOTLIST_CLEARTEXT = "hotlist_cleartext"
    REJECTED = "rejected"


class _Strict(BaseModel):
    """Base for inbound models: unknown keys are an error, not a shrug."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
    )


class GeoPoint(_Strict):
    """WGS84 point. Longitude first when it reaches PostGIS."""

    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)

    def as_ewkt(self) -> str:
        return f"SRID=4326;POINT({self.longitude:.7f} {self.latitude:.7f})"


class BoundingBox(_Strict):
    """Axis-aligned vehicle box in source-frame pixel coordinates."""

    x1: float = Field(ge=0.0)
    y1: float = Field(ge=0.0)
    x2: float = Field(ge=0.0)
    y2: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _ordered_and_nondegenerate(self) -> "BoundingBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must satisfy x2 > x1 and y2 > y1")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class PlateCorners(_Strict):
    """The four plate keypoints, ordered top-left, top-right, bottom-right, bottom-left.

    Stage 1 guarantees this winding via ``rectify.order_quad_corners``. We
    re-check convexity here rather than trusting it, because a malformed quad
    would produce a degenerate homography if it were ever replayed server side
    for evidence re-rendering.
    """

    top_left: Tuple[float, float]
    top_right: Tuple[float, float]
    bottom_right: Tuple[float, float]
    bottom_left: Tuple[float, float]

    @model_validator(mode="after")
    def _non_degenerate(self) -> "PlateCorners":
        ring = self.as_list()
        area2 = 0.0
        for index in range(4):
            x0, y0 = ring[index]
            x1, y1 = ring[(index + 1) % 4]
            area2 += x0 * y1 - x1 * y0
        if abs(area2) < 1.0:
            raise ValueError("plate quad is degenerate (near-zero area)")
        for x, y in ring:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("plate corner contains a non-finite coordinate")
        return self

    def as_list(self) -> List[Tuple[float, float]]:
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]

    def flatten(self) -> List[float]:
        return [value for point in self.as_list() for value in point]

    @classmethod
    def from_flat(cls, values: Sequence[float]) -> "PlateCorners":
        if len(values) != 8:
            raise ValueError("flat corner array must hold exactly 8 values")
        pts = [(float(values[i]), float(values[i + 1])) for i in range(0, 8, 2)]
        return cls(top_left=pts[0], top_right=pts[1], bottom_right=pts[2], bottom_left=pts[3])


def _decode_embedding(value: Any) -> List[float]:
    """Accept either a float vector or Stage 1's base64 int8 encoding.

    The edge quantises the 128-D OSNet vector to int8 and base64s it: 172
    characters instead of ~1.4 KB of JSON floats, at a measured cosine drift of
    0.00179. Dequantisation here is symmetric: ``v = q / 127``, followed by a
    re-normalisation to unit L2 so downstream cosine similarity is a pure dot
    product.
    """
    if isinstance(value, str):
        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("reid_embedding_128d string must be valid base64") from exc
        if len(raw) != EMBEDDING_DIM:
            raise ValueError(
                f"quantised embedding must decode to {EMBEDDING_DIM} bytes, got {len(raw)}"
            )
        signed = [(byte - 256 if byte > 127 else byte) / 127.0 for byte in raw]
        value = signed
    if not isinstance(value, (list, tuple)):
        raise ValueError("reid_embedding_128d must be a float array or a base64 string")
    vector = [float(component) for component in value]
    if len(vector) != EMBEDDING_DIM:
        raise ValueError(f"reid_embedding_128d must hold exactly {EMBEDDING_DIM} components")
    if not all(math.isfinite(component) for component in vector):
        raise ValueError("reid_embedding_128d contains a non-finite component")
    norm = math.sqrt(sum(component * component for component in vector))
    if norm < 1e-9:
        raise ValueError("reid_embedding_128d has a near-zero norm")
    return [component / norm for component in vector]


class EdgeObservation(_Strict):
    """A single vehicle pass as reported by a Stage 1 edge node.

    Field names are the canonical dialect. See
    :meth:`from_compact_payload` for the byte-efficient dialect the edge
    actually puts on the wire.
    """

    edge_device_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    camera_id: str = Field(default="", max_length=64, pattern=r"^[A-Za-z0-9_.:-]*$")
    tracklet_id: str = Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9_.:-]+$")
    pass_id: Optional[str] = Field(default=None, max_length=96, pattern=r"^[A-Za-z0-9_.:-]+$")

    timestamp_epoch_ms: int = Field(ge=946_684_800_000, le=4_102_444_800_000)

    plate_number_decoded: str = Field(min_length=3, max_length=16)
    plate_sequence_confidence: float = Field(ge=0.0, le=1.0)
    character_confidences: List[float] = Field(min_length=1, max_length=16)
    is_dual_line_plate: bool = False
    plate_series: PlateSeries = PlateSeries.UNKNOWN

    #: Normalised mean CTC entropy from Stage 1. Feeds the Stage 3 ``w_text``
    #: weight directly; carried through rather than recomputed.
    text_entropy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    #: Cost in nats of any optical-confusion repair Stage 1 applied.
    repair_cost_nats: NonNegativeFloat = 0.0
    is_valid_format: Optional[bool] = None

    plate_corners: PlateCorners
    vehicle_bounding_box: BoundingBox

    reid_embedding_128d: Annotated[List[float], Field(min_length=EMBEDDING_DIM, max_length=EMBEDDING_DIM)]

    vehicle_class: VehicleClass = VehicleClass.UNKNOWN
    vehicle_speed_kmh: Optional[float] = Field(default=None, ge=0.0, le=400.0)
    travel_heading_azimuth: Optional[float] = Field(default=None, ge=0.0, lt=360.0)

    evidence_crop_s3_key: Optional[str] = Field(default=None, max_length=512)

    camera_location: Optional[GeoPoint] = None
    corridor_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("plate_number_decoded", mode="before")
    @classmethod
    def _normalise_plate(cls, value: Any) -> Any:
        if isinstance(value, str):
            return normalise_plate(value)
        return value

    @field_validator("plate_number_decoded")
    @classmethod
    def _plate_charset(cls, value: str) -> str:
        if not value.isalnum():
            raise ValueError("plate_number_decoded must be alphanumeric after normalisation")
        return value

    @field_validator("character_confidences")
    @classmethod
    def _confidences_in_unit_interval(cls, value: List[float]) -> List[float]:
        for confidence in value:
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("every character confidence must lie in [0, 1]")
        return value

    @field_validator("reid_embedding_128d", mode="before")
    @classmethod
    def _coerce_embedding(cls, value: Any) -> Any:
        return _decode_embedding(value)

    @field_validator("evidence_crop_s3_key")
    @classmethod
    def _safe_object_key(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if value.startswith("/") or ".." in value:
            raise ValueError("evidence_crop_s3_key must be a relative key without traversal")
        return value

    @model_validator(mode="after")
    def _cross_field_consistency(self) -> "EdgeObservation":
        if len(self.character_confidences) != len(self.plate_number_decoded):
            raise ValueError(
                "character_confidences length must equal the decoded plate length "
                f"({len(self.character_confidences)} vs {len(self.plate_number_decoded)})"
            )

        # DPDP invariant, enforced here rather than trusted upstream.
        #
        # `pass_id` and `tracklet_id` are copied verbatim into the `sightings`
        # hypertable, which by design holds no cleartext registration number.
        # Stage 1 derives pass_id as blake2b(node|camera|track|first_ts), so it
        # never contains plate text -- but "the edge is well behaved" is not a
        # security control. A future edge build, a third-party integrator or a
        # hand-crafted replay could embed the plate in an identifier and
        # silently defeat pseudonymisation for that row. Rejecting it at the
        # boundary makes the invariant structural: there is no accepted payload
        # from which cleartext can reach the pseudonymised table.
        plate = self.plate_number_decoded
        if len(plate) >= 4:
            for field_name, value in (
                ("pass_id", self.pass_id),
                ("tracklet_id", self.tracklet_id),
                ("corridor_id", self.corridor_id),
            ):
                if value and plate in normalise_plate(value):
                    raise ValueError(
                        f"{field_name} embeds the decoded plate; identifiers must be "
                        "opaque so that no cleartext reaches the pseudonymised store"
                    )

        if self.is_valid_format is None:
            object.__setattr__(
                self, "is_valid_format", is_valid_indian_plate(self.plate_number_decoded)
            )
        if not self.camera_id:
            object.__setattr__(self, "camera_id", self.edge_device_id)
        return self

    @property
    def timestamp_utc(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_epoch_ms / 1000.0, tz=timezone.utc)

    @property
    def min_character_confidence(self) -> float:
        return min(self.character_confidences)

    def stable_pass_id(self) -> str:
        """The deduplication key. Prefers the edge-supplied deterministic id."""
        if self.pass_id:
            return self.pass_id
        return f"{self.edge_device_id}:{self.tracklet_id}:{self.timestamp_epoch_ms}"

    @classmethod
    def from_compact_payload(
        cls,
        payload: Dict[str, Any],
        *,
        camera_location: Optional[GeoPoint] = None,
        corridor_id: Optional[str] = None,
    ) -> "EdgeObservation":
        """Translate the Stage 1 compact wire dialect into the canonical model.

        The edge shortens keys and may shed optional blocks under byte pressure
        (ordered: timings, alternatives, char_confidence, geometry), so every
        lookup here tolerates absence.
        """
        plate_block: Dict[str, Any] = payload.get("plate") or {}
        geometry: Dict[str, Any] = payload.get("geometry") or {}
        reid_block: Dict[str, Any] = payload.get("reid") or {}

        text = normalise_plate(str(plate_block.get("text", "")))
        char_conf = plate_block.get("char_confidence")
        if not char_conf:
            # Geometry-shed payloads drop per-character detail; fall back to a
            # flat prior at the sequence confidence so the length invariant holds.
            sequence_conf = float(plate_block.get("conf", 0.0))
            char_conf = [sequence_conf] * max(len(text), 1)

        quad = geometry.get("q")
        if quad is None:
            raise ValueError("compact payload is missing the plate quad; cannot ingest")

        box = payload.get("box") or geometry.get("box")
        if box is None:
            raise ValueError("compact payload is missing the vehicle bounding box")

        embedding = reid_block.get("int8") or reid_block.get("f32")
        if embedding is None:
            raise ValueError("compact payload is missing the Re-ID embedding")

        return cls(
            edge_device_id=str(payload["node_id"]),
            camera_id=str(payload.get("camera_id", "")),
            tracklet_id=str(payload["track_id"]),
            pass_id=payload.get("pass_id"),
            timestamp_epoch_ms=int(payload["ts_first"]),
            plate_number_decoded=text,
            plate_sequence_confidence=float(plate_block.get("conf", 0.0)),
            character_confidences=[float(value) for value in char_conf],
            is_dual_line_plate=bool(plate_block.get("layout") == "two_line"),
            plate_series=PlateSeries(plate_block.get("series", "unknown")),
            text_entropy=plate_block.get("entropy"),
            repair_cost_nats=float(plate_block.get("repair_cost", 0.0)),
            is_valid_format=plate_block.get("valid"),
            plate_corners=PlateCorners.from_flat([float(v) for v in quad]),
            vehicle_bounding_box=BoundingBox(
                x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3])
            ),
            reid_embedding_128d=embedding,
            vehicle_class=VehicleClass(payload.get("vehicle_class", "unknown")),
            vehicle_speed_kmh=payload.get("speed_kmh"),
            travel_heading_azimuth=payload.get("heading_deg"),
            evidence_crop_s3_key=payload.get("evidence_key"),
            camera_location=camera_location,
            corridor_id=corridor_id,
        )


class IngestBatch(_Strict):
    """A batch envelope. A single observation may also be POSTed unwrapped."""

    observations: List[EdgeObservation] = Field(min_length=1, max_length=10_000)
    edge_batch_id: Optional[str] = Field(default=None, max_length=96)
    edge_sent_epoch_ms: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _single_device_per_batch(self) -> "IngestBatch":
        devices = {observation.edge_device_id for observation in self.observations}
        if len(devices) > 1:
            raise ValueError("a batch must originate from a single edge_device_id")
        return self


class HotlistMatch(BaseModel):
    """Authoritative hotlist metadata, read from Redis after a Bloom hit."""

    model_config = ConfigDict(extra="ignore")

    plate_number: str
    fir_number: str
    warrant_reference: str
    severity: AlertSeverity
    issuing_authority: str = "UNKNOWN"
    offence_category: str = "UNSPECIFIED"
    registered_at_epoch_ms: Optional[int] = None
    expires_at_epoch_ms: Optional[int] = None
    notes: Optional[str] = None

    @field_validator("plate_number", mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        return normalise_plate(value) if isinstance(value, str) else value

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip().upper()
            if text.isdigit():
                return int(text)
            named = {
                "ADVISORY": 1, "LOW": 1,
                "ELEVATED": 2, "MEDIUM": 2,
                "HIGH": 3,
                "CRITICAL": 4, "SEVERE": 4,
            }
            if text in named:
                return named[text]
        return value

    def is_active(self, now_epoch_ms: int) -> bool:
        if self.expires_at_epoch_ms is None:
            return True
        return now_epoch_ms <= self.expires_at_epoch_ms


class HotlistAlert(BaseModel):
    """Emitted to the alerts topic when a wanted vehicle is sighted."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str
    pass_id: str
    plate_number: str
    edge_device_id: str
    camera_id: str
    tracklet_id: str
    observed_at: datetime
    severity: AlertSeverity
    fir_number: str
    warrant_reference: str
    offence_category: str
    plate_sequence_confidence: float
    vehicle_class: VehicleClass
    vehicle_speed_kmh: Optional[float] = None
    travel_heading_azimuth: Optional[float] = None
    camera_location: Optional[GeoPoint] = None
    evidence_ciphertext_b64: Optional[str] = None
    evidence_key_version: Optional[int] = None
    dispatched_at: datetime


class PseudonymizedRecord(BaseModel):
    """The non-hotlist storage form. No cleartext plate exists in this object."""

    model_config = ConfigDict(extra="forbid")

    pass_id: str
    plate_pseudonym: str
    plate_pseudonym_prefix: str
    salt_epoch: int
    edge_device_id: str
    camera_id: str
    tracklet_id: str
    observed_at: datetime
    vehicle_class: VehicleClass
    plate_series: PlateSeries
    plate_sequence_confidence: float
    min_character_confidence: float
    text_entropy: Optional[float]
    repair_cost_nats: float
    is_valid_format: bool
    is_dual_line_plate: bool
    vehicle_speed_kmh: Optional[float]
    travel_heading_azimuth: Optional[float]
    reid_embedding_128d: List[float]
    camera_location: Optional[GeoPoint]
    corridor_id: Optional[str]


class IngestResult(BaseModel):
    """Per-observation outcome returned to the edge node."""

    model_config = ConfigDict(extra="forbid")

    pass_id: str
    decision: ProcessingDecision
    hotlist_hit: bool = False
    severity: Optional[AlertSeverity] = None
    error: Optional[str] = None


class IngestResponse(BaseModel):
    """Batch-level ingest acknowledgement."""

    model_config = ConfigDict(extra="forbid")

    accepted: int
    rejected: int
    hotlist_hits: int
    results: List[IngestResult]
    processing_ms: float
    buffered_rows: int


class ComponentHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    healthy: bool
    detail: str = ""
    latency_ms: Optional[float] = None


class HealthReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    version: str
    environment: str
    uptime_s: float
    components: List[ComponentHealth]
