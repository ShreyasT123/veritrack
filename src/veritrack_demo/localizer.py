"""Where to look for a plate in a webcam frame, with no downloaded weights.

Three independent strategies behind one interface. None of them is a real
object detector — Stage 1's ``VehicleDetector`` and ``PlateKeypointDetector``
are trained anchor-free heads that need exported model weights this demo does
not have — but a hackathon demo has a different set of constraints than a
production pole camera: the operator controls where the plate is held, the
distance is short, and *reliability in front of an audience* outranks
generality. All three strategies below run on stock OpenCV with nothing to
download.

``ManualRoiLocalizer``
    A fixed or interactively-adjusted rectangle. This is the generalisation of
    the hardcoded ``crop_box`` percentages in the laptop script you already
    validated — same idea, drawn live on screen so you know exactly where to
    hold the plate. It cannot fail to find a candidate, which is exactly the
    property you want for the one take that matters.

``HaarCascadePlateLocalizer``
    OpenCV ships a Haar cascade trained for (Russian) plate-like rectangular
    patterns as a data file, no separate download required. It is coarse and
    was not trained on Indian plates, but the feature it looks for —
    a high-contrast rectangular region with a particular horizontal/vertical
    gradient structure — transfers reasonably across plate designs at close
    range and good lighting.

``ContourPlateLocalizer``
    The classic "poor man's ANPR" approach: edge detection, contour
    extraction, and a filter for near-rectangular contours in a plate-like
    aspect-ratio band. Different failure modes than Haar (sensitive to
    background clutter, robust to lighting Haar is not tuned for), which is
    why ``FallbackLocalizer`` tries both.

Every candidate carries **both** an axis-aligned box and a 4-point quad. For
the manual ROI and the contour search (`cv2.minAreaRect`) the quad can reflect
real rotation; for Haar it is just the box's own corners. Either is enough for
``veritrack_edge.rectify.PlateRectifier`` to attempt a homography warp, and
the demo pipeline falls back to a plain crop when that raises
``GeometryError`` — see ``pipeline.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from .config import LocalizerConfig

__all__ = [
    "PlateCandidate",
    "PlateLocalizer",
    "ManualRoiLocalizer",
    "HaarCascadePlateLocalizer",
    "ContourPlateLocalizer",
    "FallbackLocalizer",
    "build_localizer",
    "box_to_quad",
]


def box_to_quad(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Axis-aligned box corners in (TL, TR, BR, BL) order, as ``rectify`` expects.

    ``rectify.order_quad_corners`` re-derives the correct winding from the
    geometry regardless of input order, so this ordering is a convenience for
    readability rather than a strict requirement.
    """
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class PlateCandidate:
    """One proposed plate region, in source-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    quad: np.ndarray
    score: float
    source: str

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"degenerate candidate box: {(self.x1, self.y1, self.x2, self.y2)}")
        if self.quad.shape != (4, 2):
            raise ValueError(f"quad must be (4, 2), got {self.quad.shape}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def as_int_box(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Axis-aligned crop, clamped to the frame bounds."""
        height, width = frame.shape[:2]
        x1 = max(0, int(round(self.x1)))
        y1 = max(0, int(round(self.y1)))
        x2 = min(width, int(round(self.x2)))
        y2 = min(height, int(round(self.y2)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("candidate box lies entirely outside the frame")
        return frame[y1:y2, x1:x2]


class PlateLocalizer(Protocol):
    """Anything that proposes plate candidates in a frame."""

    def locate(self, frame: np.ndarray) -> List[PlateCandidate]:
        ...

    @property
    def name(self) -> str:
        ...


class ManualRoiLocalizer:
    """A fixed or live-adjustable region of interest.

    This is the demo-safe default. It always returns exactly one candidate —
    the configured box — so the rest of the pipeline always has something to
    work with, and the operator has full control over where that is by holding
    the plate inside the drawn rectangle. ``nudge`` lets the run loop respond
    to keyboard input to reposition the box live, for the seconds before the
    demo starts when you are lining the shot up.
    """

    __slots__ = ("_fractional",)

    def __init__(self, config: LocalizerConfig) -> None:
        self._fractional = config.manual_roi_fractional

    @property
    def name(self) -> str:
        return "manual"

    @property
    def fractional_roi(self) -> Tuple[float, float, float, float]:
        return self._fractional

    def with_roi(self, left: float, top: float, right: float, bottom: float) -> "ManualRoiLocalizer":
        """Return a copy with a new ROI. Immutability keeps the run loop's
        current state explicit rather than mutating a shared object mid-frame."""
        clone = ManualRoiLocalizer.__new__(ManualRoiLocalizer)
        clone._fractional = (
            max(0.0, min(left, right - 0.01)),
            max(0.0, min(top, bottom - 0.01)),
            min(1.0, right),
            min(1.0, bottom),
        )
        return clone

    def nudge(self, dx_frac: float, dy_frac: float) -> "ManualRoiLocalizer":
        """Shift the ROI by a fraction of the frame, clamped to stay in bounds."""
        left, top, right, bottom = self._fractional
        width, height = right - left, bottom - top
        new_left = max(0.0, min(1.0 - width, left + dx_frac))
        new_top = max(0.0, min(1.0 - height, top + dy_frac))
        return self.with_roi(new_left, new_top, new_left + width, new_top + height)

    def resize(self, scale: float) -> "ManualRoiLocalizer":
        """Grow or shrink the ROI about its own centre."""
        left, top, right, bottom = self._fractional
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        half_w = (right - left) / 2.0 * scale
        half_h = (bottom - top) / 2.0 * scale
        return self.with_roi(
            max(0.0, cx - half_w), max(0.0, cy - half_h),
            min(1.0, cx + half_w), min(1.0, cy + half_h),
        )

    def locate(self, frame: np.ndarray) -> List[PlateCandidate]:
        height, width = frame.shape[:2]
        left, top, right, bottom = self._fractional
        x1, y1, x2, y2 = left * width, top * height, right * width, bottom * height
        return [
            PlateCandidate(
                x1=x1, y1=y1, x2=x2, y2=y2,
                quad=box_to_quad(x1, y1, x2, y2),
                score=1.0,
                source=self.name,
            )
        ]


class HaarCascadePlateLocalizer:
    """OpenCV's bundled plate-shaped Haar cascade. No download required."""

    __slots__ = ("_config", "_cascade")

    def __init__(self, config: LocalizerConfig, cascade_file: Optional[str] = None) -> None:
        self._config = config
        path = cascade_file or (cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            raise RuntimeError(
                f"failed to load Haar cascade from {path!r}; the installed opencv-python "
                "build may not ship its data files, or the path is wrong"
            )
        self._cascade = cascade

    @property
    def name(self) -> str:
        return "haar"

    def locate(self, frame: np.ndarray) -> List[PlateCandidate]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv2.equalizeHist(gray)

        # Detect on a downscaled copy; see LocalizerConfig.detection_max_width
        # for why this is a ~9x speedup, not a rounding-error optimisation.
        height, width = gray.shape[:2]
        scale = min(1.0, self._config.detection_max_width / float(width))
        detection_image = (
            cv2.resize(gray, (int(round(width * scale)), int(round(height * scale))))
            if scale < 1.0
            else gray
        )

        detections = self._cascade.detectMultiScale(
            detection_image,
            scaleFactor=self._config.haar_scale_factor,
            minNeighbors=self._config.haar_min_neighbors,
            minSize=(40, 12),
        )
        frame_area = float(frame.shape[0] * frame.shape[1])
        inverse_scale = 1.0 / scale
        candidates: List[PlateCandidate] = []
        for x, y, w, h in detections:
            aspect = w / float(h)
            area_fraction = (w * h) * (inverse_scale ** 2) / frame_area
            if not (self._config.min_aspect_ratio <= aspect <= self._config.max_aspect_ratio):
                continue
            if not (self._config.min_area_fraction <= area_fraction <= self._config.max_area_fraction):
                continue
            x1, y1 = x * inverse_scale, y * inverse_scale
            x2, y2 = (x + w) * inverse_scale, (y + h) * inverse_scale
            candidates.append(
                PlateCandidate(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    quad=box_to_quad(x1, y1, x2, y2),
                    # Haar's detectMultiScale does not expose a calibrated
                    # score in this API; the aspect ratio's closeness to the
                    # nominal single-line plate (4.17) is used as a proxy so
                    # candidates can still be ranked.
                    score=1.0 / (1.0 + abs(aspect - 4.17)),
                    source=self.name,
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates


class ContourPlateLocalizer:
    """Classic edge + contour rectangle search. No download required.

    Finds high-contrast quadrilateral regions in a plate-like aspect band.
    Deliberately returns a rotated quad from ``cv2.minAreaRect`` rather than
    only its bounding box, so a plate held at a modest angle to the camera can
    still get a real (if approximate) perspective correction out of
    ``PlateRectifier`` instead of a naive axis-aligned crop.
    """

    __slots__ = ("_config",)

    def __init__(self, config: LocalizerConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "contour"

    def locate(self, frame: np.ndarray) -> List[PlateCandidate]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        height, width = gray.shape[:2]
        scale = min(1.0, self._config.detection_max_width / float(width))
        detection_image = (
            cv2.resize(gray, (int(round(width * scale)), int(round(height * scale))))
            if scale < 1.0
            else gray
        )
        inverse_scale = 1.0 / scale

        blurred = cv2.bilateralFilter(detection_image, 11, 17, 17)
        edges = cv2.Canny(blurred, self._config.contour_canny_low, self._config.contour_canny_high)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(frame.shape[0] * frame.shape[1])
        candidates: List[PlateCandidate] = []

        for contour in contours:
            # Contour area is measured in the downscaled image, so it is
            # compared against a fraction computed in that same downscaled
            # frame area, not the full-resolution one.
            area = cv2.contourArea(contour)
            area_fraction = area / (frame_area * (scale ** 2))
            if not (self._config.min_area_fraction <= area_fraction <= self._config.max_area_fraction):
                continue

            rotated = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rotated
            if rw <= 1.0 or rh <= 1.0:
                continue
            long_side, short_side = max(rw, rh), min(rw, rh)
            aspect = long_side / short_side
            if not (self._config.min_aspect_ratio <= aspect <= self._config.max_aspect_ratio):
                continue

            # How well the contour fills its own minimum-area rectangle.
            # A real plate's silhouette is close to that rectangle; a cluttered
            # patch of background edges is not, so this is the discriminator
            # that keeps this strategy from firing constantly on foliage.
            rectangularity = area / (rw * rh)
            if rectangularity < 0.55:
                continue

            # Rescale the rotated rectangle back to full-frame coordinates by
            # scaling its centre and size, then re-deriving the box points —
            # scaling the four corner points independently would also work,
            # but scaling (centre, size) first keeps the rectangle exact
            # rather than accumulating independent per-corner rounding.
            full_res_rect = ((cx * inverse_scale, cy * inverse_scale),
                             (rw * inverse_scale, rh * inverse_scale), angle)
            box_points = cv2.boxPoints(full_res_rect).astype(np.float32)
            x1, y1 = float(box_points[:, 0].min()), float(box_points[:, 1].min())
            x2, y2 = float(box_points[:, 0].max()), float(box_points[:, 1].max())
            candidates.append(
                PlateCandidate(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    quad=box_points,
                    score=rectangularity,
                    source=self.name,
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates


class FallbackLocalizer:
    """Tries each localizer in order, returning the first non-empty result.

    Order matters: put the fastest and most trustworthy strategy first. Falling
    all the way through to a manual ROI guarantees the pipeline never has
    nothing to work with, at the cost of that being a much larger region than
    an actual plate.
    """

    __slots__ = ("_stages",)

    def __init__(self, stages: Sequence[PlateLocalizer]) -> None:
        if not stages:
            raise ValueError("FallbackLocalizer needs at least one stage")
        self._stages = list(stages)

    @property
    def name(self) -> str:
        return "fallback(" + "->".join(stage.name for stage in self._stages) + ")"

    @property
    def stages(self) -> List[PlateLocalizer]:
        return list(self._stages)

    def locate(self, frame: np.ndarray) -> List[PlateCandidate]:
        for stage in self._stages:
            candidates = stage.locate(frame)
            if candidates:
                return candidates
        return []


def build_localizer(config: LocalizerConfig) -> PlateLocalizer:
    """Construct the configured strategy (or chain, for ``"auto"``)."""
    if config.strategy == "manual":
        return ManualRoiLocalizer(config)
    if config.strategy == "haar":
        return HaarCascadePlateLocalizer(config)
    if config.strategy == "contour":
        return ContourPlateLocalizer(config)
    if config.strategy == "auto":
        return FallbackLocalizer(
            [
                HaarCascadePlateLocalizer(config),
                ContourPlateLocalizer(config),
                ManualRoiLocalizer(config),
            ]
        )
    raise ValueError(f"unknown localizer strategy {config.strategy!r}")
