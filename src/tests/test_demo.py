"""Verification suite for ``veritrack_demo``.

Runs entirely without a camera, a GPU, ``torch``, ``transformers``, or network
access: the recognizer is a :class:`~veritrack_demo.recognizer.FakeRecognizer`
injected via dependency injection, and every image the localizers see is
synthesised with plain ``cv2``/``numpy`` drawing calls. What this suite proves
is the pipeline *wiring* — localizer to rectifier-or-crop to recognizer to
validator to (optionally) a Stage 2 payload — not the recognizer's own
accuracy, which was already validated on the real target laptop (see
``recognizer.py``'s module docstring for exactly what was and was not
re-verified here).

    pytest tests/test_demo.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from veritrack_demo.config import (
    CaptureConfig,
    DemoConfig,
    GatewayConfig,
    LocalizerConfig,
    RecognizerConfig,
    load_demo_config,
)
from veritrack_demo.gateway_client import GatewayForwarder, build_compact_payload
from veritrack_demo.localizer import (
    ContourPlateLocalizer,
    FallbackLocalizer,
    HaarCascadePlateLocalizer,
    ManualRoiLocalizer,
    PlateCandidate,
    box_to_quad,
    build_localizer,
)
from veritrack_demo.overlay import draw_detections, draw_hud, draw_manual_roi
from veritrack_demo.pipeline import DemoDetection, build_pipeline
from veritrack_demo.recognizer import (
    FakeRecognizer,
    ModelNotLoadedError,
    RecognitionResult,
    to_pil_image,
)

from veritrack_server.schemas import EdgeObservation


# =====================================================================
# Synthetic image builders
# =====================================================================


def blank_frame(width: int = 640, height: int = 480, color=(30, 30, 30)) -> np.ndarray:
    """A frame with a solid base colour plus mild synthetic sensor noise.

    A perfectly flat, single-colour array has zero Laplacian variance, which
    ``FrameGate`` correctly rejects as "defocused" — real camera sensors never
    produce a frame that flat, so a bare solid-colour array is not a realistic
    stand-in for one. A small amount of noise restores that realism without
    approaching the noise level the focus gate is actually meant to catch.
    """
    rng = np.random.default_rng(0)
    frame = np.full((height, width, 3), color, dtype=np.int16)
    frame += rng.integers(-4, 5, size=frame.shape, dtype=np.int16)
    return np.clip(frame, 0, 255).astype(np.uint8)


def perfectly_flat_frame(width: int = 640, height: int = 480, color=(30, 30, 30)) -> np.ndarray:
    """A truly uniform frame: zero Laplacian variance, no sensor noise at all.

    Used only to test the focus gate's own rejection behaviour — this is not a
    realistic camera frame, which is exactly the point.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def frame_with_plate_rectangle(
    width: int = 640,
    height: int = 480,
    *,
    box=(220, 260, 420, 320),
    angle_deg: float = 0.0,
) -> np.ndarray:
    """A bright, high-contrast rectangle on a dark background.

    Stands in for a physical plate under a webcam: light background, dark
    border, plate-like aspect ratio — exactly the feature both the contour
    search and (approximately) the Haar cascade key on.
    """
    frame = blank_frame(width, height)
    x1, y1, x2, y2 = box
    if angle_deg == 0.0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 235, 235), thickness=-1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 10, 10), thickness=3)
    else:
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        size = (x2 - x1, y2 - y1)
        rect = (center, size, angle_deg)
        points = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(frame, [points], (235, 235, 235))
        cv2.polylines(frame, [points], isClosed=True, color=(10, 10, 10), thickness=3)
    return frame


def make_detection(
    text: str = "MH12AB1234",
    *,
    confidence: float = 0.9,
    is_valid: bool = True,
    was_repaired: bool = False,
    box=(100.0, 100.0, 300.0, 160.0),
) -> DemoDetection:
    x1, y1, x2, y2 = box
    candidate = PlateCandidate(
        x1=x1, y1=y1, x2=x2, y2=y2,
        quad=box_to_quad(x1, y1, x2, y2),
        score=1.0,
        source="manual",
    )
    return DemoDetection(
        candidate=candidate,
        recognition=RecognitionResult(text=text, confidence=confidence, inference_ms=12.0),
        validated_text=text,
        raw_text=text,
        is_valid_format=is_valid,
        was_repaired=was_repaired,
        repair_cost=0.0 if not was_repaired else 0.5,
        state_code="MH" if is_valid else None,
        rectified=True,
        total_latency_ms=15.0,
    )


# =====================================================================
# 1. Config
# =====================================================================


def test_default_config_is_valid() -> None:
    config = DemoConfig()
    assert config.capture.backend == "dshow"
    assert config.localizer.strategy == "manual"
    assert config.recognizer.device == "cpu"
    assert not config.gateway.enabled


@pytest.mark.parametrize("backend", ["dshow", "msmf", "any"])
def test_capture_backend_accepts_known_values(backend: str) -> None:
    assert CaptureConfig(backend=backend).backend == backend


def test_capture_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        CaptureConfig(backend="v4l2-please")


def test_manual_roi_bounds_are_validated() -> None:
    with pytest.raises(ValueError, match="manual_roi_fractional"):
        LocalizerConfig(manual_roi_fractional=(0.5, 0.5, 0.4, 0.9))  # left > right
    with pytest.raises(ValueError, match="manual_roi_fractional"):
        LocalizerConfig(manual_roi_fractional=(0.0, 0.0, 1.1, 0.5))  # out of [0, 1]


def test_recognizer_confidence_bounds_are_validated() -> None:
    with pytest.raises(ValueError, match="min_confidence_for_display"):
        RecognizerConfig(min_confidence_for_display=1.5)


def test_gateway_config_requires_ids() -> None:
    with pytest.raises(ValueError, match="node_id"):
        GatewayConfig(node_id="")


def test_config_json_round_trips_through_a_file() -> None:
    config = DemoConfig(
        capture=CaptureConfig(device_index=1, backend="any"),
        localizer=LocalizerConfig(strategy="auto"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.json"
        path.write_text(config.to_json(), encoding="utf-8")
        restored = load_demo_config(path)
    assert restored.capture.device_index == 1
    assert restored.capture.backend == "any"
    assert restored.localizer.strategy == "auto"


def test_load_demo_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_demo_config("/nonexistent/path/demo.json")


def test_config_evolve_only_changes_named_fields() -> None:
    config = DemoConfig()
    changed = config.evolve(process_every_n_frames=5)
    assert changed.process_every_n_frames == 5
    assert changed.capture == config.capture  # untouched


# =====================================================================
# 2. Localizers
# =====================================================================


def test_box_to_quad_shape_and_order() -> None:
    quad = box_to_quad(10.0, 20.0, 110.0, 70.0)
    assert quad.shape == (4, 2)
    assert tuple(quad[0]) == (10.0, 20.0)  # TL
    assert tuple(quad[2]) == (110.0, 70.0)  # BR


def test_plate_candidate_rejects_degenerate_box() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        PlateCandidate(x1=50, y1=50, x2=50, y2=80, quad=box_to_quad(50, 50, 50, 80), score=1.0, source="x")


def test_plate_candidate_crop_matches_requested_region() -> None:
    frame = blank_frame(200, 100, color=(0, 0, 0))
    frame[10:40, 20:60] = (255, 255, 255)
    candidate = PlateCandidate(x1=20, y1=10, x2=60, y2=40, quad=box_to_quad(20, 10, 60, 40),
                               score=1.0, source="manual")
    crop = candidate.crop(frame)
    assert crop.shape == (30, 40, 3)
    assert crop.mean() > 200  # entirely inside the white region


def test_plate_candidate_crop_clamps_to_frame_bounds() -> None:
    frame = blank_frame(100, 100)
    candidate = PlateCandidate(x1=-20, y1=-20, x2=50, y2=50, quad=box_to_quad(-20, -20, 50, 50),
                               score=1.0, source="manual")
    crop = candidate.crop(frame)
    assert crop.shape[0] <= 100 and crop.shape[1] <= 100


def test_manual_roi_locates_exactly_the_configured_fraction() -> None:
    config = LocalizerConfig(manual_roi_fractional=(0.25, 0.25, 0.75, 0.75))
    localizer = ManualRoiLocalizer(config)
    candidates = localizer.locate(blank_frame(400, 200))
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.x1 == pytest.approx(100.0)
    assert candidate.y1 == pytest.approx(50.0)
    assert candidate.x2 == pytest.approx(300.0)
    assert candidate.y2 == pytest.approx(150.0)


def test_manual_roi_nudge_moves_and_clamps() -> None:
    localizer = ManualRoiLocalizer(LocalizerConfig(manual_roi_fractional=(0.0, 0.0, 0.2, 0.2)))
    moved = localizer.nudge(-0.5, -0.5)  # should clamp at the frame edge, not go negative
    left, top, right, bottom = moved.fractional_roi
    assert left == pytest.approx(0.0)
    assert top == pytest.approx(0.0)
    assert (right - left) == pytest.approx(0.2)


def test_manual_roi_resize_keeps_center() -> None:
    localizer = ManualRoiLocalizer(LocalizerConfig(manual_roi_fractional=(0.4, 0.4, 0.6, 0.6)))
    grown = localizer.resize(2.0)
    left, top, right, bottom = grown.fractional_roi
    assert (left + right) / 2.0 == pytest.approx(0.5, abs=1e-6)
    assert (right - left) == pytest.approx(0.4, abs=1e-6)


def test_contour_localizer_finds_a_synthetic_rectangle() -> None:
    config = LocalizerConfig(min_aspect_ratio=1.5, max_aspect_ratio=5.0,
                             min_area_fraction=0.001, max_area_fraction=0.5)
    localizer = ContourPlateLocalizer(config)
    frame = frame_with_plate_rectangle(box=(220, 260, 420, 320))  # 200x60, aspect 3.33
    candidates = localizer.locate(frame)
    assert candidates, "the contour localizer must find the synthetic plate rectangle"
    best = candidates[0]
    assert config.min_aspect_ratio <= best.aspect_ratio <= config.max_aspect_ratio
    # Centre of the found box should be near the centre of the drawn one.
    assert abs((best.x1 + best.x2) / 2.0 - 320.0) < 20.0
    assert abs((best.y1 + best.y2) / 2.0 - 290.0) < 20.0


def test_contour_localizer_recovers_rotation() -> None:
    """A plate held at an angle should still yield a quad reflecting that angle."""
    config = LocalizerConfig(min_aspect_ratio=1.5, max_aspect_ratio=5.0,
                             min_area_fraction=0.001, max_area_fraction=0.5)
    localizer = ContourPlateLocalizer(config)
    frame = frame_with_plate_rectangle(box=(220, 260, 420, 320), angle_deg=15.0)
    candidates = localizer.locate(frame)
    assert candidates
    # The bounding box of a rotated rectangle is larger than the unrotated
    # one; the quad itself should not be axis-aligned.
    quad = candidates[0].quad
    xs = sorted(quad[:, 0])
    assert xs[0] != pytest.approx(xs[1], abs=0.5)  # not two pairs of identical x's


def test_contour_localizer_rejects_out_of_band_aspect_ratio() -> None:
    """A near-square blob (e.g. a sign, not a plate) must not be accepted."""
    config = LocalizerConfig(min_aspect_ratio=2.0, max_aspect_ratio=6.0,
                             min_area_fraction=0.001, max_area_fraction=0.5)
    localizer = ContourPlateLocalizer(config)
    frame = frame_with_plate_rectangle(box=(250, 250, 350, 350))  # square, aspect 1.0
    candidates = localizer.locate(frame)
    assert candidates == []


def test_contour_localizer_finds_nothing_on_a_blank_frame() -> None:
    localizer = ContourPlateLocalizer(LocalizerConfig())
    assert localizer.locate(blank_frame()) == []


def test_haar_cascade_loads_and_runs_without_crashing() -> None:
    """The bundled cascade must load; detection on a blank frame must not error."""
    localizer = HaarCascadePlateLocalizer(LocalizerConfig())
    candidates = localizer.locate(blank_frame())
    assert isinstance(candidates, list)
    for candidate in candidates:
        assert LocalizerConfig().min_aspect_ratio <= candidate.aspect_ratio <= LocalizerConfig().max_aspect_ratio


def test_haar_cascade_missing_file_raises_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="failed to load"):
        HaarCascadePlateLocalizer(LocalizerConfig(), cascade_file="/nonexistent/cascade.xml")


def test_fallback_localizer_uses_first_non_empty_stage() -> None:
    class EmptyLocalizer:
        name = "empty"
        def locate(self, frame):
            return []

    class AlwaysHitLocalizer:
        name = "always"
        def locate(self, frame):
            return [PlateCandidate(x1=0, y1=0, x2=10, y2=10, quad=box_to_quad(0, 0, 10, 10),
                                   score=1.0, source="always")]

    fallback = FallbackLocalizer([EmptyLocalizer(), AlwaysHitLocalizer()])
    candidates = fallback.locate(blank_frame())
    assert len(candidates) == 1
    assert candidates[0].source == "always"
    assert "empty->always" in fallback.name


def test_fallback_localizer_requires_at_least_one_stage() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        FallbackLocalizer([])


def test_build_localizer_dispatches_on_strategy() -> None:
    assert isinstance(build_localizer(LocalizerConfig(strategy="manual")), ManualRoiLocalizer)
    assert isinstance(build_localizer(LocalizerConfig(strategy="haar")), HaarCascadePlateLocalizer)
    assert isinstance(build_localizer(LocalizerConfig(strategy="contour")), ContourPlateLocalizer)
    assert isinstance(build_localizer(LocalizerConfig(strategy="auto")), FallbackLocalizer)


# =====================================================================
# 3. Recognizer wrapper (FakeRecognizer + image conversion)
# =====================================================================


def test_recognition_result_validates_confidence_range() -> None:
    with pytest.raises(ValueError, match="confidence"):
        RecognitionResult(text="X", confidence=1.5, inference_ms=1.0)


def test_to_pil_image_converts_bgr_array_to_rgb() -> None:
    """A pure-blue OpenCV (BGR) pixel must become a pure-blue RGB pixel, not red."""
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)  # BGR blue
    pil_image = to_pil_image(frame)
    r, g, b = pil_image.getpixel((5, 5))
    assert (r, g, b) == (0, 0, 255)


def test_to_pil_image_passes_through_an_existing_pil_image() -> None:
    from PIL import Image

    original = Image.new("RGB", (4, 4), color=(1, 2, 3))
    assert to_pil_image(original) is original


def test_to_pil_image_rejects_unsupported_shapes() -> None:
    with pytest.raises(ValueError, match="unsupported array shape"):
        to_pil_image(np.zeros((4, 4, 5), dtype=np.uint8))
    with pytest.raises(TypeError):
        to_pil_image("not an image")


def test_fake_recognizer_returns_fixed_result_by_default() -> None:
    recognizer = FakeRecognizer()
    result = recognizer.recognize(blank_frame(20, 20))
    assert result.text == "MH12AB1234"
    assert recognizer.call_count == 1


def test_fake_recognizer_batch_calls_responder_per_image() -> None:
    calls: List[int] = []

    def responder(image):
        calls.append(1)
        return RecognitionResult(text="KA05MN7788", confidence=0.8, inference_ms=5.0)

    recognizer = FakeRecognizer(responder=responder)
    results = recognizer.recognize_batch([blank_frame(10, 10), blank_frame(10, 10)])
    assert len(results) == 2 and len(calls) == 2
    assert all(r.text == "KA05MN7788" for r in results)


def test_fake_recognizer_batch_of_zero_is_empty() -> None:
    assert FakeRecognizer().recognize_batch([]) == []


# =====================================================================
# 4. Pipeline end-to-end (with FakeRecognizer, real localizer/rectifier/validator)
# =====================================================================


def test_pipeline_end_to_end_with_manual_roi_and_valid_plate() -> None:
    config = DemoConfig(process_every_n_frames=1)
    recognizer = FakeRecognizer(RecognitionResult(text="MH12AB1234", confidence=0.93, inference_ms=90.0))
    pipeline = build_pipeline(config, recognizer=recognizer)

    frame = blank_frame(640, 480)
    detections = pipeline.process_frame(frame)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.validated_text == "MH12AB1234"
    assert detection.is_valid_format is True
    assert detection.was_repaired is False
    assert detection.rectified in (True, False)  # a plain box always rectifies as zero-skew
    assert detection.recognition.inference_ms == pytest.approx(90.0)


def test_pipeline_repairs_a_confusable_character() -> None:
    """8<->B is a real optical confusion; the validator should repair it."""
    config = DemoConfig(process_every_n_frames=1)
    recognizer = FakeRecognizer(RecognitionResult(text="MH12A81234", confidence=0.88, inference_ms=80.0))
    pipeline = build_pipeline(config, recognizer=recognizer)

    detections = pipeline.process_frame(blank_frame())
    assert len(detections) == 1
    assert detections[0].validated_text == "MH12AB1234"
    assert detections[0].was_repaired is True
    assert detections[0].is_valid_format is True


def test_pipeline_reports_invalid_format_without_crashing() -> None:
    config = DemoConfig(process_every_n_frames=1)
    recognizer = FakeRecognizer(RecognitionResult(text="???###", confidence=0.4, inference_ms=80.0))
    pipeline = build_pipeline(config, recognizer=recognizer)
    detections = pipeline.process_frame(blank_frame())
    assert len(detections) == 1
    assert detections[0].is_valid_format is False


def test_pipeline_respects_process_every_n_frames() -> None:
    """Frames between the configured stride must reuse the cached detections."""
    config = DemoConfig(process_every_n_frames=3)
    recognizer = FakeRecognizer(RecognitionResult(text="MH12AB1234", confidence=0.9, inference_ms=10.0))
    pipeline = build_pipeline(config, recognizer=recognizer)

    frame = blank_frame()
    first = pipeline.process_frame(frame)   # frame_index=1, not a multiple of 3 -> cached (empty)
    second = pipeline.process_frame(frame)  # frame_index=2 -> still cached (empty)
    third = pipeline.process_frame(frame)   # frame_index=3 -> runs
    assert first == [] and second == []
    assert len(third) == 1
    assert recognizer.call_count == 1  # only the third frame actually invoked recognition


def test_pipeline_gate_rejects_a_truly_flat_frame() -> None:
    """A frame with zero Laplacian variance is correctly treated as defocused.

    This is the reused Stage 1 ``FrameGate`` doing its job: no real camera
    sensor ever produces a perfectly flat frame, so this is not a false
    rejection of realistic input — it is what should happen to an
    all-one-colour array, and it is why every other pipeline test in this
    suite uses ``blank_frame`` (with mild synthetic noise) rather than this.
    """
    config = DemoConfig(process_every_n_frames=1)
    recognizer = FakeRecognizer(RecognitionResult(text="MH12AB1234", confidence=0.9, inference_ms=10.0))
    pipeline = build_pipeline(config, recognizer=recognizer)
    detections = pipeline.process_frame(perfectly_flat_frame())
    assert detections == []
    assert recognizer.call_count == 0


def test_pipeline_returns_nothing_when_localizer_finds_nothing() -> None:
    class EmptyLocalizer:
        name = "empty"
        def locate(self, frame):
            return []

    config = DemoConfig(process_every_n_frames=1)
    pipeline = build_pipeline(config, localizer=EmptyLocalizer(), recognizer=FakeRecognizer())
    assert pipeline.process_frame(blank_frame()) == []


def test_pipeline_sorts_detections_by_confidence_descending() -> None:
    class TwoBoxLocalizer:
        name = "two-box"
        def locate(self, frame):
            return [
                PlateCandidate(x1=10, y1=10, x2=110, y2=40, quad=box_to_quad(10, 10, 110, 40),
                               score=1.0, source="a"),
                PlateCandidate(x1=200, y1=10, x2=300, y2=40, quad=box_to_quad(200, 10, 300, 40),
                               score=1.0, source="b"),
            ]

    responses = iter([
        RecognitionResult(text="MH12AB1234", confidence=0.5, inference_ms=5.0),
        RecognitionResult(text="KA05MN7788", confidence=0.95, inference_ms=5.0),
    ])
    recognizer = FakeRecognizer(responder=lambda image: next(responses))
    config = DemoConfig(process_every_n_frames=1)
    pipeline = build_pipeline(config, localizer=TwoBoxLocalizer(), recognizer=recognizer)

    detections = pipeline.process_frame(blank_frame(640, 480))
    assert len(detections) == 2
    assert detections[0].recognition.confidence > detections[1].recognition.confidence


def test_pipeline_caps_batch_at_max_batch_size() -> None:
    class ManyBoxLocalizer:
        name = "many"
        def locate(self, frame):
            return [
                PlateCandidate(x1=i * 10, y1=0, x2=i * 10 + 5, y2=5, quad=box_to_quad(i * 10, 0, i * 10 + 5, 5),
                               score=1.0, source="many")
                for i in range(10)
            ]

    config = DemoConfig(process_every_n_frames=1,
                        recognizer=RecognizerConfig(max_batch_size=2))
    recognizer = FakeRecognizer(RecognitionResult(text="MH12AB1234", confidence=0.9, inference_ms=1.0))
    pipeline = build_pipeline(config, localizer=ManyBoxLocalizer(), recognizer=recognizer)

    detections = pipeline.process_frame(blank_frame(200, 50))
    assert len(detections) == 2


# =====================================================================
# 5. Overlay (must not crash; pure drawing)
# =====================================================================


def test_draw_detections_does_not_crash_and_returns_a_frame() -> None:
    frame = blank_frame()
    detections = [make_detection(is_valid=True), make_detection(text="???", is_valid=False)]
    result = draw_detections(frame, detections)
    assert result is frame
    assert result.shape == frame.shape


def test_draw_detections_on_empty_list() -> None:
    frame = blank_frame()
    assert draw_detections(frame, []) is frame


def test_draw_manual_roi_only_draws_for_manual_localizer() -> None:
    frame_a = blank_frame()
    manual = ManualRoiLocalizer(LocalizerConfig())
    draw_manual_roi(frame_a, manual, active=True)
    assert not np.array_equal(frame_a, blank_frame())  # something was drawn

    frame_b = blank_frame()
    haar = HaarCascadePlateLocalizer(LocalizerConfig())
    draw_manual_roi(frame_b, haar, active=True)
    assert np.array_equal(frame_b, blank_frame())  # nothing drawn for a non-manual localizer


def test_draw_hud_does_not_crash() -> None:
    frame = blank_frame()
    draw_hud(frame, fps=29.7, localizer_name="manual", forwarding=True, forwarded_count=3)


# =====================================================================
# 6. Gateway forwarding
# =====================================================================


def test_build_compact_payload_round_trips_through_stage2_schema() -> None:
    """The strongest cross-package check: Stage 2's own parser must accept this."""
    detection = make_detection(text="MH12AB1234", confidence=0.91)
    payload = build_compact_payload(
        detection, node_id="DEMO-01", camera_id="webcam", session_id="session-abc", sequence_number=1
    )
    observation = EdgeObservation.from_compact_payload(payload)
    assert observation.plate_number_decoded == "MH12AB1234"
    assert observation.edge_device_id == "DEMO-01"
    assert observation.plate_sequence_confidence == pytest.approx(0.91)
    assert len(observation.reid_embedding_128d) == 128
    assert observation.vehicle_bounding_box.x2 > observation.vehicle_bounding_box.x1


def test_build_compact_payload_marks_stand_ins_explicitly() -> None:
    detection = make_detection()
    payload = build_compact_payload(detection, node_id="n", camera_id="c", session_id="s", sequence_number=1)
    assert payload["demo_stand_in"] is True


def test_build_compact_payload_identifiers_never_embed_the_plate() -> None:
    """Stage 2 rejects any pass_id/track_id that embeds the decoded plate."""
    detection = make_detection(text="MH12AB1234")
    payload = build_compact_payload(detection, node_id="n", camera_id="c", session_id="s", sequence_number=1)
    assert "MH12AB1234" not in payload["pass_id"]
    assert "MH12AB1234" not in payload["track_id"]
    # And Stage 2's own boundary validator must agree.
    EdgeObservation.from_compact_payload(payload)  # must not raise


def test_build_compact_payload_is_deterministic_within_a_session() -> None:
    detection = make_detection()
    first = build_compact_payload(detection, node_id="n", camera_id="c", session_id="fixed-session", sequence_number=1)
    second = build_compact_payload(detection, node_id="n", camera_id="c", session_id="fixed-session", sequence_number=1)
    assert first["track_id"] == second["track_id"]
    assert first["pass_id"] == second["pass_id"]
    assert first["reid"]["f32"] == second["reid"]["f32"]


def test_gateway_forwarder_never_raises_on_unreachable_host() -> None:
    """A live demo must not crash because the gateway happened to be down."""
    config = GatewayConfig(enabled=True, base_url="http://127.0.0.1:1", timeout_s=0.3)
    forwarder = GatewayForwarder(config)
    result = forwarder.forward(make_detection())
    assert result.success is False
    assert result.status_code is None
    assert result.error is not None
    assert forwarder.sent_count == 0


def test_gateway_forwarder_sequence_increments() -> None:
    config = GatewayConfig(enabled=True, base_url="http://127.0.0.1:1", timeout_s=0.2)
    forwarder = GatewayForwarder(config)
    forwarder.build_payload(make_detection())
    forwarder.build_payload(make_detection())
    assert forwarder._sequence == 2  # noqa: SLF001
