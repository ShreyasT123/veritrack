"""Edge pipeline orchestration.

Per-frame flow::

    frame -> motion + focus gate
          -> vehicle detector (dual threshold)
          -> ByteTrack
          -> per track: plate quad -> homography -> line split -> CTC logits
          -> per track (every N frames): OSNet embedding -> EMA
          -> on track exit or observation quota: fuse -> beam search
             -> grammar validate/repair -> package -> emit

Latency accounting
------------------
The <15 ms budget is *per vehicle pass*, not per frame. A pass amortises its
cost over the frames the vehicle is visible: the detector and tracker run once
per frame regardless of occupancy, while the recogniser runs once per track per
frame and the fusion/decode runs once per pass. :class:`LatencyBudget`
accumulates per-stage milliseconds against the track that caused them, so the
figure reported on the wire is the true marginal cost of that vehicle. Passes
that breach the budget are logged with a per-stage breakdown - that log line is
the primary field-tuning instrument.

Emission policy
---------------
A pass is emitted once, at whichever comes first: the observation quota
(``early_emit_observations``) or track termination. Emitting early matters for
hotlist latency - a stolen vehicle should be dispatched while it is still in
frame, not after it has left - and the quota is chosen so that fusion has
already converged. After emission the tracklet stops accumulating, which bounds
memory on a node watching a jam.
"""

from __future__ import annotations

import hashlib
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .detection import PlateKeypointDetector, VehicleDetector
from .errors import RecoverableEdgeError
from .gating import FrameGate
from .ingest import Frame
from .ocr import CtcRecognizer
from .packaging import EdgePayloadBuilder
from .rectify import PlateRectifier
from .reid import EmbeddingAccumulator, OsNetExtractor
from .splitter import PlateLineSplitter
from .tracking import ByteTracker, STrack
from .types import (
    BBox,
    PlateObservation,
    PlateReading,
    StageTimings,
    TrackletSummary,
    VehiclePass,
)
from .validation import PlateValidator

logger = logging.getLogger(__name__)


class LatencyBudget:
    """Accumulates per-stage milliseconds, attributable per track."""

    __slots__ = ("_frame", "_per_track")

    def __init__(self) -> None:
        self._frame: Dict[str, float] = {}
        self._per_track: Dict[int, Dict[str, float]] = {}

    @contextmanager
    def measure(self, stage: str, track_id: Optional[int] = None) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if track_id is None:
                self._frame[stage] = self._frame.get(stage, 0.0) + elapsed_ms
            else:
                bucket = self._per_track.setdefault(track_id, {})
                bucket[stage] = bucket.get(stage, 0.0) + elapsed_ms

    def add(self, stage: str, milliseconds: float, track_id: Optional[int] = None) -> None:
        if track_id is None:
            self._frame[stage] = self._frame.get(stage, 0.0) + milliseconds
        else:
            bucket = self._per_track.setdefault(track_id, {})
            bucket[stage] = bucket.get(stage, 0.0) + milliseconds

    def share_frame_cost(self, track_count: int) -> Dict[str, float]:
        """Split frame-level stage costs evenly across the tracks in frame."""
        if track_count <= 0:
            return {}
        return {stage: value / track_count for stage, value in self._frame.items()}

    def snapshot(self, track_id: int, shared: Dict[str, float]) -> StageTimings:
        own = self._per_track.get(track_id, {})
        merged = {stage: shared.get(stage, 0.0) + own.get(stage, 0.0) for stage in set(shared) | set(own)}
        return StageTimings(
            gating=merged.get("gating", 0.0),
            vehicle_detect=merged.get("vehicle_detect", 0.0),
            track=merged.get("track", 0.0),
            plate_detect=merged.get("plate_detect", 0.0),
            rectify=merged.get("rectify", 0.0),
            ocr=merged.get("ocr", 0.0),
            reid=merged.get("reid", 0.0),
            decode=merged.get("decode", 0.0),
            package=merged.get("package", 0.0),
        )

    def reset_track(self, track_id: int) -> None:
        self._per_track.pop(track_id, None)

    def reset_frame(self) -> None:
        self._frame.clear()


@dataclass(slots=True)
class _TrackContext:
    """Mutable per-track accumulator held by the pipeline."""

    summary: TrackletSummary
    embeddings: EmbeddingAccumulator
    emitted: bool = False
    last_reid_frame: int = -10_000
    detection_scores: List[float] = field(default_factory=list)


def make_pass_id(node_id: str, camera_id: str, track_id: int, first_seen: float) -> str:
    """Deterministic 16-hex-char pass identifier.

    Deterministic rather than random so that an at-least-once Kafka producer
    retrying after a backhaul drop produces a byte-identical key, letting the
    gateway deduplicate on primary key instead of on heuristics.
    """
    material = f"{node_id}|{camera_id}|{track_id}|{first_seen:.3f}".encode("utf-8")
    return hashlib.blake2b(material, digest_size=8).hexdigest()


class EdgePipeline:
    """The full Stage 1 edge vision and feature extraction pipeline."""

    __slots__ = (
        "_config",
        "_gate",
        "_vehicle_detector",
        "_plate_detector",
        "_rectifier",
        "_splitter",
        "_recognizer",
        "_reid",
        "_validator",
        "_builder",
        "_tracker",
        "_contexts",
        "_budget",
        "_frames_processed",
        "_frames_gated",
        "_passes_emitted",
        "_budget_breaches",
    )

    def __init__(
        self,
        config: PipelineConfig,
        vehicle_detector: Optional[VehicleDetector] = None,
        plate_detector: Optional[PlateKeypointDetector] = None,
        recognizer: Optional[CtcRecognizer] = None,
        reid: Optional[OsNetExtractor] = None,
    ) -> None:
        self._config = config
        self._gate = FrameGate(config.gate)
        self._vehicle_detector = vehicle_detector or VehicleDetector(
            config.vehicle_detector, config.backend
        )
        self._plate_detector = plate_detector or PlateKeypointDetector(
            config.plate_detector, config.backend
        )
        self._rectifier = PlateRectifier(config.rectify)
        self._splitter = PlateLineSplitter(config.rectify, config.ocr.input_size)
        self._recognizer = recognizer or CtcRecognizer(
            config.ocr, config.backend, config.gate.laplacian_reference_variance
        )
        self._reid = reid or OsNetExtractor(config.reid, config.backend)
        self._validator = PlateValidator(config.validation)
        self._builder = EdgePayloadBuilder(config.packaging)
        self._tracker = ByteTracker(config.tracker)
        self._contexts: Dict[int, _TrackContext] = {}
        self._budget = LatencyBudget()
        self._frames_processed = 0
        self._frames_gated = 0
        self._passes_emitted = 0
        self._budget_breaches = 0

    # -- statistics --------------------------------------------------------

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "frames_processed": self._frames_processed,
            "frames_gated": self._frames_gated,
            "passes_emitted": self._passes_emitted,
            "budget_breaches": self._budget_breaches,
            "live_tracks": len(self._contexts),
        }

    @property
    def payload_builder(self) -> EdgePayloadBuilder:
        return self._builder

    # -- per-frame ---------------------------------------------------------

    def process(self, frame: Frame) -> List[VehiclePass]:
        """Process one frame and return any passes that completed on it."""
        self._frames_processed += 1
        self._budget.reset_frame()
        image = frame.image

        with self._budget.measure("gating"):
            decision = self._gate.evaluate(image)
        if not decision.accepted:
            self._frames_gated += 1
            # Tracks still age out while the carriageway is empty.
            _, terminated = self._tracker.update([], [], frame.timestamp)
            return self._finalize_many(terminated)

        with self._budget.measure("vehicle_detect"):
            high, low = self._vehicle_detector.detect(image)

        with self._budget.measure("track"):
            active, terminated = self._tracker.update(high, low, frame.timestamp)

        for track in active:
            self._observe(frame, track)

        completed = self._finalize_many(terminated)
        completed.extend(self._emit_quota_reached(frame))
        return completed

    def _context_for(self, track: STrack, frame: Frame) -> _TrackContext:
        context = self._contexts.get(track.track_id)
        if context is not None:
            return context
        summary = TrackletSummary(
            track_id=track.track_id,
            class_name=track.class_name or "vehicle",
            first_frame=frame.index,
            last_frame=frame.index,
            first_timestamp=track.start_timestamp or frame.timestamp,
            last_timestamp=frame.timestamp,
            hit_count=track.hits,
            boxes_first=BBox.from_array(track.first_tlbr),
            boxes_last=track.bbox,
        )
        context = _TrackContext(summary=summary, embeddings=EmbeddingAccumulator(self._config.reid.ema_alpha))
        self._contexts[track.track_id] = context
        return context

    def _observe(self, frame: Frame, track: STrack) -> None:
        """Accumulate plate and appearance evidence for one track on one frame."""
        context = self._context_for(track, frame)
        if context.emitted:
            return

        summary = context.summary
        summary.last_frame = frame.index
        summary.last_timestamp = frame.timestamp
        summary.hit_count = track.hits
        summary.boxes_last = track.bbox
        context.detection_scores.append(track.score)

        self._observe_appearance(frame, track, context)

        if frame.index % max(1, self._config.plate_detect_every_n_frames) != 0:
            return
        self._observe_plate(frame, track, context)

    def _observe_appearance(self, frame: Frame, track: STrack, context: _TrackContext) -> None:
        interval = max(1, self._config.reid.refresh_every_n_frames)
        if frame.index - context.last_reid_frame < interval:
            return
        try:
            with self._budget.measure("reid", track.track_id):
                embedding = self._reid.embed_box(frame.image, track.bbox)
            context.embeddings.update(embedding)
            context.summary.embedding = context.embeddings.value
            context.last_reid_frame = frame.index
        except (RecoverableEdgeError, ValueError) as exc:
            logger.debug("track %d: re-id skipped (%s)", track.track_id, exc)

    def _observe_plate(self, frame: Frame, track: STrack, context: _TrackContext) -> None:
        track_id = track.track_id
        try:
            with self._budget.measure("plate_detect", track_id):
                plate = self._plate_detector.detect(frame.image, track.bbox)
            if plate is None:
                return

            with self._budget.measure("rectify", track_id):
                rectified = self._rectifier.rectify(frame.image, plate)
                # Scale-invariant focus gate, applied on the canonical strip.
                if rectified.focus_score < self._config.gate.laplacian_min_variance:
                    logger.debug(
                        "track %d: strip rejected, Var(Laplacian)=%.1f < %.1f",
                        track_id,
                        rectified.focus_score,
                        self._config.gate.laplacian_min_variance,
                    )
                    return
                strip = self._splitter.to_strip(rectified)

            with self._budget.measure("ocr", track_id):
                log_probs = self._recognizer.infer_log_probs(strip)

            context.summary.observations.append(
                PlateObservation(
                    frame_index=frame.index,
                    timestamp=frame.timestamp,
                    log_probs=log_probs,
                    layout=rectified.layout,
                    series=rectified.series,
                    detection_score=plate.score,
                    focus_score=rectified.focus_score,
                    pose=rectified.pose,
                )
            )
            # Bound memory: keep only the most recent window; fusion re-ranks by
            # quality inside CtcRecognizer.fuse_and_decode.
            limit = self._config.ocr.max_observations * 2
            if len(context.summary.observations) > limit:
                del context.summary.observations[:-limit]

        except RecoverableEdgeError as exc:
            logger.debug("track %d: plate observation skipped (%s)", track_id, exc)

    # -- emission ----------------------------------------------------------

    def _emit_quota_reached(self, frame: Frame) -> List[VehiclePass]:
        """Emit any track that has gathered enough evidence to decide early."""
        quota = self._config.early_emit_observations
        if quota <= 0:
            return []
        out: List[VehiclePass] = []
        for track_id, context in self._contexts.items():
            if context.emitted or len(context.summary.observations) < quota:
                continue
            vehicle_pass = self._finalize(track_id, context)
            if vehicle_pass is not None:
                out.append(vehicle_pass)
        return out

    def _finalize_many(self, terminated: List[STrack]) -> List[VehiclePass]:
        out: List[VehiclePass] = []
        for track in terminated:
            context = self._contexts.pop(track.track_id, None)
            if context is None or context.emitted:
                self._budget.reset_track(track.track_id)
                continue
            vehicle_pass = self._finalize(track.track_id, context)
            if vehicle_pass is not None:
                out.append(vehicle_pass)
        return out

    def _decode(self, context: _TrackContext) -> Optional[PlateReading]:
        summary = context.summary
        if len(summary.observations) < max(1, self._config.min_emit_observations):
            return None
        try:
            fused = self._recognizer.fuse_and_decode(summary.observations)
        except RecoverableEdgeError as exc:
            logger.info("track %d: decode failed (%s)", summary.track_id, exc)
            return None

        best = fused.best
        result = self._validator.validate(best.text, fused.posteriors)

        # Sequence confidence: geometric mean of per-character posteriors, which
        # is length-normalised and therefore comparable across plate formats.
        char_probs = tuple(float(np.exp(p.log_prob)) for p in fused.posteriors)
        if char_probs:
            confidence = float(np.exp(np.mean(np.log(np.clip(char_probs, 1e-12, 1.0)))))
        else:
            confidence = float(np.exp(best.log_prob))
        if result.was_repaired:
            # A repair is evidence the raw read was wrong; discount accordingly.
            confidence *= float(np.exp(-0.25 * result.repair_cost))
        if not result.is_valid:
            confidence *= 0.6

        if confidence < self._config.ocr.min_fused_confidence:
            logger.debug(
                "track %d: plate '%s' below confidence floor (%.3f)",
                summary.track_id,
                result.text,
                confidence,
            )

        latest = summary.observations[-1]
        return PlateReading(
            text=result.text,
            raw_text=result.raw_text,
            confidence=round(confidence, 5),
            char_confidences=tuple(round(p, 4) for p in char_probs),
            sequence_entropy=fused.entropy,
            template_id=result.template_id,
            is_valid_format=result.is_valid,
            was_repaired=result.was_repaired,
            repair_cost=result.repair_cost,
            state_code=result.state_code,
            layout=latest.layout,
            series=latest.series,
            observation_count=fused.observation_count,
            alternatives=fused.hypotheses[1:4],
        )

    def _finalize(self, track_id: int, context: _TrackContext) -> Optional[VehiclePass]:
        summary = context.summary
        with self._budget.measure("decode", track_id):
            reading = self._decode(context)

        embedding = context.embeddings.value
        if reading is None and embedding is None:
            context.emitted = True
            self._budget.reset_track(track_id)
            return None

        shared = self._budget.share_frame_cost(max(1, len(self._contexts)))
        timings = self._budget.snapshot(track_id, shared)

        if context.detection_scores:
            summary.mean_detection_score = float(np.mean(context.detection_scores))

        vehicle_pass = VehiclePass(
            pass_id=make_pass_id(
                self._config.node_id, self._config.camera_id, track_id, summary.first_timestamp
            ),
            camera_id=self._config.camera_id,
            node_id=self._config.node_id,
            track_id=track_id,
            vehicle_class=summary.class_name,
            first_seen=summary.first_timestamp,
            last_seen=summary.last_timestamp,
            embedding=embedding,
            reading=reading,
            entry_box=summary.boxes_first,
            exit_box=summary.boxes_last,
            timings=timings,
        )

        context.emitted = True
        context.summary.observations.clear()
        self._budget.reset_track(track_id)
        self._passes_emitted += 1

        if timings.total > self._config.latency_budget_ms:
            self._budget_breaches += 1
            logger.warning(
                "pass %s breached %.1fms budget: %s",
                vehicle_pass.pass_id,
                self._config.latency_budget_ms,
                timings.as_dict(),
            )
        return vehicle_pass

    def flush(self) -> List[VehiclePass]:
        """Terminate and emit every live track, e.g. at shutdown."""
        out: List[VehiclePass] = []
        for track in self._tracker.flush():
            context = self._contexts.pop(track.track_id, None)
            if context is None or context.emitted:
                continue
            vehicle_pass = self._finalize(track.track_id, context)
            if vehicle_pass is not None:
                out.append(vehicle_pass)
        self._contexts.clear()
        return out
