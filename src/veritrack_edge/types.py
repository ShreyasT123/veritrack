"""Core value types.

Everything crossing a module boundary is one of these. They are intentionally
plain (`slots=True` dataclasses over numpy arrays) so that per-frame allocation
stays cheap on the edge SoC.

Coordinate conventions
----------------------
* Boxes are absolute **full-frame** pixels, ``(x1, y1, x2, y2)``, x1 < x2.
* Quads are ``(4, 2)`` float32 arrays in full-frame pixels, ordered
  top-left, top-right, bottom-right, bottom-left (clockwise, image axes).
* Timestamps are UNIX epoch seconds (float, UTC) captured at frame grab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

BoxArray = np.ndarray  # shape (N, 4), float32
Quad = np.ndarray      # shape (4, 2), float32


@dataclass(frozen=True, slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"Degenerate box: {self.as_tuple()}")

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_array(self) -> np.ndarray:
        return np.array(self.as_tuple(), dtype=np.float32)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    def pad(self, ratio: float) -> "BBox":
        """Isotropically expand by ``ratio`` of each side length."""
        dx = self.width * ratio
        dy = self.height * ratio
        return BBox(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def clip(self, width: int, height: int) -> "BBox":
        return BBox(
            max(0.0, min(self.x1, width - 2.0)),
            max(0.0, min(self.y1, height - 2.0)),
            min(float(width), max(self.x2, 1.0)),
            min(float(height), max(self.y2, 1.0)),
        )

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return float(inter / union) if union > 0.0 else 0.0

    @staticmethod
    def from_array(arr: Sequence[float]) -> "BBox":
        return BBox(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detector output in full-frame coordinates."""

    bbox: BBox
    score: float
    class_id: int
    class_name: str = ""


@dataclass(frozen=True, slots=True)
class PlateQuad:
    """A localised plate: axis-aligned box plus regressed 4-point corners."""

    bbox: BBox
    quad: Quad
    score: float

    def __post_init__(self) -> None:
        if self.quad.shape != (4, 2):
            raise ValueError(f"Quad must be (4, 2), got {self.quad.shape}")


class PlateLayout(str, Enum):
    SINGLE_LINE = "single_line"
    TWO_LINE = "two_line"


class PlateSeries(str, Enum):
    """Background colour class, which encodes the vehicle's usage category."""

    PRIVATE_WHITE = "private_white"
    COMMERCIAL_YELLOW = "commercial_yellow"
    ELECTRIC_GREEN = "electric_green"
    RENTAL_BLACK = "rental_black"
    DIPLOMATIC_LIGHT_BLUE = "diplomatic_light_blue"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PlatePose:
    """Recovered orientation of the plate plane relative to the camera."""

    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    edge_symmetry: float
    analytic: bool  # True when derived from intrinsics, False when heuristic

    @property
    def max_skew_deg(self) -> float:
        return max(abs(self.yaw_deg), abs(self.pitch_deg))


@dataclass(frozen=True, slots=True)
class RectifiedPlate:
    """A plate warped to the canonical recognition strip."""

    image: np.ndarray        # (H, W, 3) uint8, canonical strip
    layout: PlateLayout
    series: PlateSeries
    pose: PlatePose
    homography: np.ndarray   # (3, 3) float64, full-frame -> canonical canvas
    focus_score: float       # Var(Laplacian) of the rectified strip
    source_quad: Quad


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlateObservation:
    """One scored recognition attempt for one tracklet at one frame."""

    frame_index: int
    timestamp: float
    log_probs: np.ndarray    # (T, C) float32, log-softmax over the charset
    layout: PlateLayout
    series: PlateSeries
    detection_score: float
    focus_score: float
    pose: PlatePose

    @property
    def num_steps(self) -> int:
        return int(self.log_probs.shape[0])


@dataclass(frozen=True, slots=True)
class CharPosterior:
    """Per-character posterior at its Viterbi peak frame."""

    char: str
    log_prob: float
    alt_chars: Tuple[str, ...]
    alt_log_probs: Tuple[float, ...]

    @property
    def probability(self) -> float:
        return float(np.exp(self.log_prob))


@dataclass(frozen=True, slots=True)
class PlateHypothesis:
    """A decoded candidate string with its CTC score."""

    text: str
    log_prob: float

    @property
    def probability(self) -> float:
        return float(np.exp(self.log_prob))


@dataclass(frozen=True, slots=True)
class PlateReading:
    """The final, grammar-checked plate for one tracklet."""

    text: str
    raw_text: str
    confidence: float
    char_confidences: Tuple[float, ...]
    sequence_entropy: float
    template_id: str
    is_valid_format: bool
    was_repaired: bool
    repair_cost: float
    state_code: Optional[str]
    layout: PlateLayout
    series: PlateSeries
    observation_count: int
    alternatives: Tuple[PlateHypothesis, ...] = ()


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


class TrackState(str, Enum):
    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


@dataclass(slots=True)
class TrackletSummary:
    """Everything the pipeline accumulated about one vehicle pass."""

    track_id: int
    class_name: str
    first_frame: int
    last_frame: int
    first_timestamp: float
    last_timestamp: float
    hit_count: int
    boxes_first: BBox
    boxes_last: BBox
    embedding: Optional[np.ndarray] = None      # (128,) float32, L2-normalised
    observations: List[PlateObservation] = field(default_factory=list)
    mean_detection_score: float = 0.0

    @property
    def dwell_seconds(self) -> float:
        return max(0.0, self.last_timestamp - self.first_timestamp)


@dataclass(frozen=True, slots=True)
class StageTimings:
    """Per-pass latency accounting, in milliseconds."""

    gating: float = 0.0
    vehicle_detect: float = 0.0
    track: float = 0.0
    plate_detect: float = 0.0
    rectify: float = 0.0
    ocr: float = 0.0
    reid: float = 0.0
    decode: float = 0.0
    package: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.gating
            + self.vehicle_detect
            + self.track
            + self.plate_detect
            + self.rectify
            + self.ocr
            + self.reid
            + self.decode
            + self.package
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "gating": round(self.gating, 3),
            "vehicle_detect": round(self.vehicle_detect, 3),
            "track": round(self.track, 3),
            "plate_detect": round(self.plate_detect, 3),
            "rectify": round(self.rectify, 3),
            "ocr": round(self.ocr, 3),
            "reid": round(self.reid, 3),
            "decode": round(self.decode, 3),
            "package": round(self.package, 3),
            "total": round(self.total, 3),
        }


@dataclass(frozen=True, slots=True)
class VehiclePass:
    """The complete, emit-ready record of one vehicle crossing this camera."""

    pass_id: str
    camera_id: str
    node_id: str
    track_id: int
    vehicle_class: str
    first_seen: float
    last_seen: float
    embedding: Optional[np.ndarray]
    reading: Optional[PlateReading]
    entry_box: BBox
    exit_box: BBox
    timings: StageTimings
