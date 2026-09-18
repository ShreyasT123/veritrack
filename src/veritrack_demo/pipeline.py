"""Orchestrates one frame through the live demo pipeline.

    frame -> FrameGate (reused from veritrack_edge) -> localize -> rectify-or-crop
          -> recognize (batched) -> PlateValidator (reused from veritrack_edge)
          -> DemoDetection

Two pieces of Stage 1 are reused verbatim rather than reimplemented:

``veritrack_edge.gating.FrameGate``
    The same Var(Laplacian) + motion pre-filter that keeps a pole node from
    burning NPU cycles on empty frames keeps this demo from burning CPU cycles
    on an empty desk between cars. Motion detection is disabled by default
    here (see ``build_pipeline``) because a live demo usually wants to
    recognize a plate someone is *holding still* in front of the camera, which
    a motion detector would actively fight; it is one flag to re-enable for a
    "drive-by" style demo instead.

``veritrack_edge.validation.PlateValidator``
    The Indian registration grammar and optical-confusion repair. The HF
    recognizer gives a raw string and one confidence score, no per-character
    posteriors, so repair falls back to a uniform substitution cost — the
    documented behaviour when ``posteriors`` is empty — which still prefers a
    minimal-edit legal plate over a larger one.

``veritrack_edge.rectify.PlateRectifier`` is attempted, not assumed. A
localizer that only found an axis-aligned box (Haar, the manual ROI) produces
a "quad" that is really just that box's four corners; ``PlateRectifier`` will
happily rectify it as a zero-skew plate, which is a reasonable approximation
for a plate held roughly square to the webcam. If the geometry check ever
rejects it (degenerate, too skewed, below the corner-symmetry sanity floor),
the pipeline falls back to a plain resized crop rather than dropping the
candidate — the HF image processor does its own resizing internally and does
not require the exact 48x160 canonical strip Stage 1's custom CTC head needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from veritrack_edge.errors import GeometryError
from veritrack_edge.gating import FrameGate
from veritrack_edge.rectify import PlateRectifier
from veritrack_edge.types import PlateQuad
from veritrack_edge.validation import PlateValidator

from .config import DemoConfig
from .localizer import PlateCandidate, PlateLocalizer
from .recognizer import Recognizer, RecognitionResult, to_pil_image

__all__ = ["DemoDetection", "DemoPipeline", "build_pipeline"]


@dataclass(frozen=True, slots=True)
class DemoDetection:
    """One fully-processed candidate: geometry, raw OCR, and validated text."""

    candidate: PlateCandidate
    recognition: RecognitionResult
    validated_text: str
    raw_text: str
    is_valid_format: bool
    was_repaired: bool
    repair_cost: float
    state_code: Optional[str]
    rectified: bool
    total_latency_ms: float

    @property
    def display_text(self) -> str:
        return self.validated_text or self.raw_text or "?"


class DemoPipeline:
    """Runs the full detect -> recognize -> validate chain for one frame."""

    __slots__ = (
        "_config", "_localizer", "_recognizer", "_rectifier", "_validator",
        "_gate", "_frame_index", "_last_detections",
    )

    def __init__(
        self,
        config: DemoConfig,
        localizer: PlateLocalizer,
        recognizer: Recognizer,
        rectifier: PlateRectifier,
        validator: PlateValidator,
        gate: Optional[FrameGate] = None,
    ) -> None:
        self._config = config
        self._localizer = localizer
        self._recognizer = recognizer
        self._rectifier = rectifier
        self._validator = validator
        self._gate = gate
        self._frame_index = 0
        self._last_detections: List[DemoDetection] = []

    @property
    def localizer(self) -> PlateLocalizer:
        return self._localizer

    @property
    def recognizer(self) -> Recognizer:
        return self._recognizer

    @property
    def last_detections(self) -> List[DemoDetection]:
        """The most recent processed result, held between skipped frames.

        ``process_frame`` only actually runs the (comparatively expensive)
        localize+recognize chain on every Nth frame; on the frames in between,
        the caller re-displays this instead so the on-screen box does not
        blink out between recognizer calls.
        """
        return self._last_detections

    def _prepare_crop(self, frame: np.ndarray, candidate: PlateCandidate) -> Tuple[np.ndarray, bool]:
        """Return (image_for_recognizer, was_rectified)."""
        try:
            plate_quad = PlateQuad(
                bbox=_bbox_from_candidate(candidate),
                quad=candidate.quad.astype(np.float32),
                score=candidate.score,
            )
            rectified = self._rectifier.rectify(frame, plate_quad)
            return rectified.image, True
        except GeometryError:
            return candidate.crop(frame), False

    def process_frame(self, frame: np.ndarray) -> List[DemoDetection]:
        """Process one frame. Returns the (possibly cached) detection list."""
        started = time.perf_counter()
        self._frame_index += 1

        if self._frame_index % self._config.process_every_n_frames != 0:
            return self._last_detections

        if self._gate is not None:
            decision = self._gate.evaluate(frame)
            if not decision.accepted:
                self._last_detections = []
                return self._last_detections

        candidates = self._localizer.locate(frame)
        if not candidates:
            self._last_detections = []
            return self._last_detections

        crops: List[np.ndarray] = []
        rectified_flags: List[bool] = []
        for candidate in candidates[: self._config.recognizer.max_batch_size]:
            image, was_rectified = self._prepare_crop(frame, candidate)
            crops.append(to_pil_image(image))
            rectified_flags.append(was_rectified)

        used_candidates = candidates[: len(crops)]
        results = self._recognizer.recognize_batch(crops)

        detections: List[DemoDetection] = []
        total_ms = (time.perf_counter() - started) * 1000.0
        for candidate, recognition, was_rectified in zip(used_candidates, results, rectified_flags):
            validation = self._validator.validate(recognition.text)
            detections.append(
                DemoDetection(
                    candidate=candidate,
                    recognition=recognition,
                    validated_text=validation.text,
                    raw_text=validation.raw_text,
                    is_valid_format=validation.is_valid,
                    was_repaired=validation.was_repaired,
                    repair_cost=validation.repair_cost,
                    state_code=validation.state_code,
                    rectified=was_rectified,
                    total_latency_ms=total_ms,
                )
            )

        detections.sort(key=lambda d: d.recognition.confidence, reverse=True)
        self._last_detections = detections
        return detections


def _bbox_from_candidate(candidate: PlateCandidate):
    from veritrack_edge.types import BBox

    return BBox(x1=candidate.x1, y1=candidate.y1, x2=candidate.x2, y2=candidate.y2)


def build_pipeline(
    config: DemoConfig,
    *,
    localizer: Optional[PlateLocalizer] = None,
    recognizer: Optional[Recognizer] = None,
) -> DemoPipeline:
    """Wire up a pipeline from a :class:`DemoConfig`.

    ``localizer`` and ``recognizer`` are injectable so tests (and a future
    swap to a real ONNX plate detector) never have to touch this function's
    body — only the object passed in changes.
    """
    from veritrack_edge.config import GateConfig, RectifyConfig, ValidationConfig

    from .localizer import build_localizer
    from .recognizer import PPOcrRecognizer

    resolved_localizer = localizer if localizer is not None else build_localizer(config.localizer)
    resolved_recognizer = recognizer if recognizer is not None else PPOcrRecognizer(
        config.recognizer.model_id,
        device=config.recognizer.device,
        warmup_on_load=config.recognizer.warmup_on_load,
    )

    # Motion detection off by default: a demo plate is usually held still in
    # front of the camera, which the motion gate would otherwise reject.
    gate = FrameGate(GateConfig(motion_enabled=False))
    rectifier = PlateRectifier(RectifyConfig())
    validator = PlateValidator(ValidationConfig())

    return DemoPipeline(
        config=config,
        localizer=resolved_localizer,
        recognizer=resolved_recognizer,
        rectifier=rectifier,
        validator=validator,
        gate=gate,
    )
