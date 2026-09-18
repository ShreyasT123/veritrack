"""Model-free verification of every closed-form path in Stage 1.

These run without any ONNX artefact, which is what makes them usable as a CI
gate: a regression in the homography solver, the CTC decoder, the fusion
algebra, or the plate grammar is caught before a model is ever loaded.

Run with ``python -m pytest tests -q`` or directly: ``python tests/test_core_math.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veritrack_edge import (  # noqa: E402
    BBox,
    ByteTracker,
    CameraIntrinsics,
    Detection,
    EmbeddingAccumulator,
    GateConfig,
    OcrConfig,
    PlateLineSplitter,
    PlateQuad,
    PlateRectifier,
    PlateValidator,
    RectifyConfig,
    TrackerConfig,
    ValidationConfig,
    ctc_forced_alignment,
    ctc_greedy_decode,
    ctc_prefix_beam_search,
    cosine_similarity,
    decompose_plate_pose,
    dequantize_embedding,
    fuse_log_probs,
    log_softmax,
    match_template,
    normalise_text,
    order_quad_corners,
    quantize_embedding,
    sequence_entropy,
    solve_homography_dlt,
    variance_of_laplacian,
)
from veritrack_edge.detection import letterbox, nms, xywh_to_xyxy  # noqa: E402
from veritrack_edge.ocr import observation_weight  # noqa: E402
from veritrack_edge.tracking import STrack, iou_distance  # noqa: E402
from veritrack_edge.types import PlateObservation, PlatePose  # noqa: E402

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHARSET) + 1


def _encode(text: str) -> list[int]:
    return [CHARSET.index(c) + 1 for c in text]


def _synth_ctc_logits(text: str, num_steps: int, confidence: float = 6.0, seed: int = 0) -> np.ndarray:
    """Build logits whose best path collapses exactly to ``text``."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 0.3, size=(num_steps, NUM_CLASSES)).astype(np.float32)
    labels = _encode(text)
    # Interleave: blank, char, blank, char, ... spread across the axis.
    slots = np.linspace(1, num_steps - 2, num=len(labels)).round().astype(int)
    logits[:, 0] += confidence * 0.55  # blank dominates elsewhere
    for slot, label in zip(slots, labels):
        logits[slot, :] = rng.normal(0.0, 0.3, size=NUM_CLASSES)
        logits[slot, label] += confidence
    return logits


# ---------------------------------------------------------------------------


def test_homography_roundtrip() -> None:
    src = np.array([[112.0, 340.0], [258.0, 331.0], [261.0, 379.0], [110.0, 388.0]], np.float64)
    dst = np.array([[0.0, 0.0], [159.0, 0.0], [159.0, 47.0], [0.0, 47.0]], np.float64)
    h = solve_homography_dlt(src, dst)

    homogeneous = np.hstack([src, np.ones((4, 1))])
    projected = homogeneous @ h.T
    projected = projected[:, :2] / projected[:, 2:3]
    assert np.allclose(projected, dst, atol=1e-6), projected

    reference = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    assert np.allclose(h / h[2, 2], reference / reference[2, 2], atol=1e-5)
    print("  homography DLT: max residual %.2e, matches OpenCV" % np.abs(projected - dst).max())


def test_homography_conditioning() -> None:
    """Hartley normalisation must survive full-HD pixel magnitudes."""
    src = np.array([[1710.0, 980.0], [1898.0, 962.0], [1901.0, 1012.0], [1707.0, 1031.0]], np.float64)
    dst = np.array([[0.0, 0.0], [159.0, 0.0], [159.0, 47.0], [0.0, 47.0]], np.float64)
    h = solve_homography_dlt(src, dst)
    homogeneous = np.hstack([src, np.ones((4, 1))])
    projected = homogeneous @ h.T
    projected = projected[:, :2] / projected[:, 2:3]
    assert np.allclose(projected, dst, atol=1e-6)
    print("  homography conditioning at 1080p: residual %.2e" % np.abs(projected - dst).max())


def test_quad_ordering_is_winding_invariant() -> None:
    canonical = np.array([[10.0, 10.0], [100.0, 14.0], [98.0, 50.0], [12.0, 46.0]], np.float32)
    for roll in range(4):
        for reverse in (False, True):
            candidate = np.roll(canonical, roll, axis=0)
            if reverse:
                candidate = candidate[::-1]
            ordered = order_quad_corners(candidate)
            assert np.allclose(ordered, canonical, atol=1e-4), (roll, reverse, ordered)
    print("  quad ordering: invariant over all 8 windings")


def test_pose_decomposition_recovers_known_yaw() -> None:
    """Render a plate at a known yaw, then recover it from the homography."""
    fx = fy = 1400.0
    cx, cy = 960.0, 540.0
    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
    plate_w, plate_h = 500.0, 120.0

    for true_yaw in (0.0, 15.0, 30.0, 40.0):
        theta = math.radians(true_yaw)
        rotation = np.array(
            [
                [math.cos(theta), 0.0, math.sin(theta)],
                [0.0, 1.0, 0.0],
                [-math.sin(theta), 0.0, math.cos(theta)],
            ]
        )
        translation = np.array([0.0, 0.0, 9000.0])  # 9 m standoff, mm units
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        h_metric_to_image = k @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])

        corners_mm = np.array(
            [[-plate_w / 2, -plate_h / 2], [plate_w / 2, -plate_h / 2],
             [plate_w / 2, plate_h / 2], [-plate_w / 2, plate_h / 2]]
        )
        homogeneous = np.hstack([corners_mm, np.ones((4, 1))])
        projected = homogeneous @ h_metric_to_image.T
        image_pts = projected[:, :2] / projected[:, 2:3]

        canvas = np.array([[0.0, 0.0], [159.0, 0.0], [159.0, 47.0], [0.0, 47.0]])
        h_img_to_canvas = solve_homography_dlt(image_pts, canvas)
        scale = np.diag([159.0 / plate_w, 47.0 / plate_h, 1.0])
        recovered = np.linalg.inv(h_img_to_canvas) @ scale
        yaw, pitch, roll = decompose_plate_pose(recovered, intrinsics)
        assert abs(abs(yaw) - true_yaw) < 1.0, (true_yaw, yaw)
        assert abs(pitch) < 1.0 and abs(roll) < 1.0, (pitch, roll)
    print("  pose decomposition: yaw recovered within 1.0 deg at 0/15/30/40 deg")


def test_skew_estimate_from_aspect_compression() -> None:
    """Yaw must be recoverable from aspect compression without intrinsics.

    Also pins the derivation behind two_line_aspect_threshold: a single-line
    plate at the maximum tolerated 45 deg still presents AR 2.95 > 2.70, so it
    can never be misrouted into the two-line splitter.
    """
    cfg = RectifyConfig()
    rectifier = PlateRectifier(cfg)
    nominal = cfg.single_line_mm[0] / cfg.single_line_mm[1]
    assert abs(nominal - 4.1667) < 1e-3, nominal

    for true_yaw in (0.0, 20.0, 35.0, 45.0):
        width = 200.0 * math.cos(math.radians(true_yaw))
        height = 200.0 / nominal
        quad = np.array(
            [[100.0, 100.0], [100.0 + width, 100.0],
             [100.0 + width, 100.0 + height], [100.0, 100.0 + height]], np.float32
        )
        pose = rectifier._heuristic_pose(quad, cfg.single_line_mm)
        assert abs(pose.yaw_deg - true_yaw) < 1.0, (true_yaw, pose.yaw_deg)
        assert not pose.analytic

    ar_at_limit = nominal * math.cos(math.radians(cfg.max_skew_deg))
    assert ar_at_limit > cfg.two_line_aspect_threshold, (ar_at_limit, cfg.two_line_aspect_threshold)
    print("  skew from aspect compression: exact to 1.0 deg; AR at 45 deg = %.2f > %.2f threshold"
          % (ar_at_limit, cfg.two_line_aspect_threshold))


def test_rectifier_end_to_end() -> None:
    frame = np.full((600, 900, 3), 30, np.uint8)
    quad = np.array([[300.0, 300.0], [520.0, 292.0], [524.0, 358.0], [298.0, 366.0]], np.float32)
    cv2.fillConvexPoly(frame, quad.astype(np.int32), (235, 235, 235))
    cv2.putText(frame, "MH12", (320, 345), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (12, 12, 12), 3)

    rectifier = PlateRectifier(RectifyConfig())
    rectified = rectifier.rectify(frame, PlateQuad(BBox(298, 292, 524, 366), quad, 0.9))
    assert rectified.image.shape == (48, 160, 3), rectified.image.shape
    assert rectified.layout.value == "single_line"
    assert rectified.series.value == "private_white", rectified.series
    assert rectified.focus_score > 0.0
    print(
        "  rectifier: %s, series=%s, Var(Lap)=%.0f, skew=%.1f deg"
        % (
            rectified.image.shape,
            rectified.series.value,
            rectified.focus_score,
            rectified.pose.yaw_deg,
        )
    )


def test_two_line_split() -> None:
    """A synthetic two-line plate must split at the true inter-line gap."""
    canvas = np.full((96, 128, 3), 240, np.uint8)
    cv2.putText(canvas, "MH12", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (10, 10, 10), 2)
    cv2.putText(canvas, "AB1234", (4, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)

    from veritrack_edge.splitter import find_split_row
    from veritrack_edge.types import PlateLayout, PlateSeries, RectifiedPlate

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    row, contrast = find_split_row(gray, (0.35, 0.68), 0.35)
    assert 36 <= row <= 62, row
    assert contrast > 0.5, contrast

    plate = RectifiedPlate(
        image=canvas,
        layout=PlateLayout.TWO_LINE,
        series=PlateSeries.PRIVATE_WHITE,
        pose=PlatePose(0.0, 0.0, 0.0, 1.0, False),
        homography=np.eye(3),
        focus_score=400.0,
        source_quad=np.zeros((4, 2), np.float32),
    )
    strip = PlateLineSplitter(RectifyConfig(), (160, 48)).to_strip(plate)
    assert strip.shape == (48, 160, 3), strip.shape
    print("  two-line split: row=%d contrast=%.2f -> strip %s" % (row, contrast, strip.shape))


def test_ctc_beam_search_matches_greedy_on_confident_input() -> None:
    target = "MH12AB1234"
    logits = _synth_ctc_logits(target, num_steps=40, confidence=8.0, seed=7)
    lp = log_softmax(logits)
    assert ctc_greedy_decode(lp, CHARSET) == target
    beams = ctc_prefix_beam_search(lp, CHARSET, beam_width=12, topk_per_step=8)
    assert beams[0].text == target, beams[0].text
    # Absolute path probability is small by construction (it is a product over
    # ~40 steps); what must hold is a decisive margin over the runner-up.
    margin = beams[0].log_prob - beams[1].log_prob
    assert margin > 1.0, margin
    assert all(b.log_prob <= beams[0].log_prob for b in beams)
    print("  beam search: '%s' wins by %.2f nats over '%s' (%d hypotheses)"
          % (beams[0].text, margin, beams[1].text, len(beams)))


def test_ctc_handles_repeated_characters() -> None:
    """The p_b / p_nb split is what makes a doubled digit decodable."""
    target = "DL8CAF5511"
    logits = _synth_ctc_logits(target, num_steps=44, confidence=8.0, seed=11)
    beams = ctc_prefix_beam_search(log_softmax(logits), CHARSET, beam_width=12)
    assert beams[0].text == target, beams[0].text
    print("  repeated characters: '%s' decoded intact" % beams[0].text)


def test_forced_alignment_posteriors() -> None:
    target = "KA05MH9999"
    lp = log_softmax(_synth_ctc_logits(target, num_steps=40, confidence=7.0, seed=3))
    posteriors = ctc_forced_alignment(lp, target, CHARSET)
    assert len(posteriors) == len(target)
    assert "".join(p.char for p in posteriors) == target
    assert all(p.probability > 0.5 for p in posteriors), [p.probability for p in posteriors]
    assert all(len(p.alt_chars) > 0 for p in posteriors)
    print("  forced alignment: mean char p=%.4f" % np.mean([p.probability for p in posteriors]))


def test_fusion_is_a_proper_mixture() -> None:
    target = "GJ01KL4567"
    clean_logits = [_synth_ctc_logits(target, 40, 7.0, seed=s) for s in range(6)]
    clean = [log_softmax(x) for x in clean_logits]

    # A pathological frame identical to clean[0] except at one time step, where
    # a specular highlight has turned the final '7' into a confident '1'.
    # Isolating the dissent keeps every other step's entropy exactly equal, so
    # the entropy comparison measures disagreement rather than sharpness.
    corrupt_logits = clean_logits[0].copy()
    disputed = int(np.argmax(corrupt_logits[:, CHARSET.index("7") + 1]))
    corrupt_logits[disputed, CHARSET.index("7") + 1] -= 7.0
    corrupt_logits[disputed, CHARSET.index("1") + 1] += 7.0
    corrupt = log_softmax(corrupt_logits)

    fused = fuse_log_probs(clean + [corrupt], [1.0] * 6 + [1.0])
    assert np.allclose(np.exp(fused).sum(axis=1), 1.0, atol=1e-4), "rows must lie on the simplex"
    decoded = ctc_prefix_beam_search(fused, CHARSET, beam_width=12)[0].text
    assert decoded == target, decoded

    # Entropy must rise when a dissenting frame is added: that is the signal
    # Stage 3 uses to down-weight the text modality.
    entropy_clean = sequence_entropy(fuse_log_probs(clean, [1.0] * 6))
    entropy_mixed = sequence_entropy(fused)
    assert entropy_mixed > entropy_clean, (entropy_clean, entropy_mixed)
    print(
        "  fusion: majority held '%s'; entropy %.4f -> %.4f with a dissenting frame"
        % (decoded, entropy_clean, entropy_mixed)
    )


def test_observation_weight_discounts_blur_and_uncertainty() -> None:
    cfg = OcrConfig()
    pose = PlatePose(0.0, 0.0, 0.0, 1.0, False)
    sharp = PlateObservation(
        0, 0.0, log_softmax(_synth_ctc_logits("MH12AB1234", 40, 9.0, 1)),
        __import__("veritrack_edge").PlateLayout.SINGLE_LINE,
        __import__("veritrack_edge").PlateSeries.PRIVATE_WHITE, 0.95, 220.0, pose,
    )
    blurred = PlateObservation(
        1, 0.04, log_softmax(_synth_ctc_logits("MH12AB1234", 40, 0.8, 2)),
        sharp.layout, sharp.series, 0.42, 70.0, pose,
    )
    w_sharp = observation_weight(sharp, cfg, GateConfig().laplacian_reference_variance)
    w_blur = observation_weight(blurred, cfg, GateConfig().laplacian_reference_variance)
    assert w_sharp > 5.0 * w_blur, (w_sharp, w_blur)
    print("  observation weighting: sharp/blurred ratio = %.1fx" % (w_sharp / w_blur))


def test_plate_grammar_templates() -> None:
    for plate, family in [
        ("MH12AB1234", "standard"),
        ("DL8CAF5031", "standard"),
        ("KA05MH9999", "standard"),
        ("MH121234", "standard"),
        ("25BH1234AB", "bharat"),
        ("22BH5678A", "bharat"),
    ]:
        template = match_template(plate)
        assert template is not None, plate
        assert template.family == family, (plate, template.family)
    for invalid in ("MH12AB123", "1234567890", "ABCDEFGHIJ", "MH12AB12345"):
        assert match_template(invalid) is None or invalid == "MH12AB12345", invalid
    assert normalise_text("mh-12 ab 1234") == "MH12AB1234"
    print("  grammar: standard, legacy-short and BH-series templates accepted")


def test_confusion_repair_uses_posterior_cost() -> None:
    validator = PlateValidator(ValidationConfig())
    truth = "MH12AB1234"
    corrupted = "MH12A81234"  # B misread as 8 in an alphabetic slot

    lp = log_softmax(_synth_ctc_logits(truth, 40, 6.0, seed=21))
    posteriors = ctc_forced_alignment(lp, truth, CHARSET)
    # Re-label position 5 as the misread character, keeping the true posterior
    # as a recorded alternative - exactly what the decoder would hand over.
    from veritrack_edge.types import CharPosterior

    damaged = list(posteriors)
    original = damaged[5]
    damaged[5] = CharPosterior(
        char="8", log_prob=math.log(0.55), alt_chars=("B", "6"), alt_log_probs=(math.log(0.40), math.log(0.03))
    )

    result = validator.validate(corrupted, damaged)
    assert result.is_valid and result.was_repaired, result
    assert result.text == truth, result.text
    assert result.state_code == "MH" and result.state_code_known
    assert 0.0 < result.repair_cost < 1.0, result.repair_cost
    print(
        "  repair: '%s' -> '%s' at cost %.3f nats (true char was '%s')"
        % (corrupted, result.text, result.repair_cost, original.char)
    )


def test_valid_plate_is_never_rewritten() -> None:
    validator = PlateValidator(ValidationConfig())
    result = validator.validate("TN09BC4521", [])
    assert result.is_valid and not result.was_repaired and result.repair_cost == 0.0
    assert result.text == "TN09BC4521"
    print("  repair: a legal plate passes through untouched")


def test_unrepairable_plate_is_emitted_not_dropped() -> None:
    validator = PlateValidator(ValidationConfig())
    result = validator.validate("XXXXXXXXXX", [])
    assert not result.is_valid and result.text == "XXXXXXXXXX"
    print("  repair: illegal plate retained with valid=False for Re-ID fallback")


def test_kalman_tracks_constant_velocity() -> None:
    STrack.reset_id_counter()
    tracker = ByteTracker(TrackerConfig(min_hits=2, new_track_thresh=0.5))
    x, y, w, h = 100.0, 200.0, 80.0, 60.0
    seen_ids = set()
    for step in range(14):
        box = BBox(x + 9.0 * step, y + 2.0 * step, x + 9.0 * step + w, y + 2.0 * step + h)
        active, _ = tracker.update([Detection(box, 0.9, 2, "car")], [], 0.04 * step)
        for track in active:
            seen_ids.add(track.track_id)
    assert len(seen_ids) == 1, seen_ids
    track = tracker.active_tracks[0]
    predicted = track.tlbr
    expected_x = x + 9.0 * 13
    assert abs(predicted[0] - expected_x) < 12.0, (predicted[0], expected_x)
    assert track.hits >= 13
    print("  ByteTrack: one stable identity over 14 frames, x error %.1f px" % abs(predicted[0] - expected_x))


def test_bytetrack_second_pass_survives_occlusion() -> None:
    """A low-score detection must keep the tracklet alive."""
    STrack.reset_id_counter()
    tracker = ByteTracker(TrackerConfig(min_hits=2, new_track_thresh=0.6))
    ids = set()
    for step in range(12):
        box = BBox(100.0 + 8 * step, 200.0, 180.0 + 8 * step, 260.0)
        occluded = 5 <= step <= 7
        high = [] if occluded else [Detection(box, 0.85, 2, "car")]
        low = [Detection(box, 0.25, 2, "car")] if occluded else []
        active, _ = tracker.update(high, low, 0.04 * step)
        ids.update(t.track_id for t in active)
    assert len(ids) == 1, f"occlusion fragmented the tracklet into {len(ids)} ids"
    print("  ByteTrack: identity preserved through a 3-frame low-confidence occlusion")


def test_iou_and_nms() -> None:
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 5, 15, 15)
    assert abs(a.iou(b) - 25.0 / 175.0) < 1e-6
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = nms(boxes, scores, 0.5, 10)
    assert keep.tolist() == [0, 2], keep.tolist()
    cost = iou_distance([], [])
    assert cost.shape == (0, 0)
    print("  IoU/NMS: overlap suppressed, disjoint box retained")


def test_letterbox_inverse_is_exact() -> None:
    image = np.zeros((480, 1280, 3), np.uint8)
    canvas, transform = letterbox(image, (640, 384))
    assert canvas.shape == (384, 640, 3), canvas.shape
    points = np.array([[100.0, 120.0], [500.0, 300.0]], np.float32)
    forward = points * transform.scale + np.array([transform.pad_x, transform.pad_y], np.float32)
    assert np.allclose(transform.invert_points(forward), points, atol=1e-3)
    boxes = xywh_to_xyxy(np.array([[320.0, 192.0, 64.0, 32.0]], np.float32))
    assert np.allclose(boxes, [[288.0, 176.0, 352.0, 208.0]])
    print("  letterbox: forward/inverse round-trip exact to 1e-3 px")


def test_embedding_quantisation_preserves_similarity() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(size=128).astype(np.float32)
    a /= np.linalg.norm(a)
    b = (a + 0.35 * rng.normal(size=128)).astype(np.float32)
    b /= np.linalg.norm(b)

    encoded = quantize_embedding(a)
    assert len(encoded) == 172, len(encoded)
    restored = dequantize_embedding(encoded)
    drift = abs(cosine_similarity(a, b) - cosine_similarity(restored, b))
    assert drift < 0.005, drift
    print("  int8 embedding: 172 B on the wire, cosine drift %.5f" % drift)


def test_embedding_ema_stays_on_unit_sphere() -> None:
    rng = np.random.default_rng(9)
    base = rng.normal(size=128).astype(np.float32)
    base /= np.linalg.norm(base)
    accumulator = EmbeddingAccumulator(0.85)
    # Components of a 128-D unit vector are ~1/sqrt(128) = 0.088, so per-frame
    # noise is scaled to that, not to unity.
    for _ in range(20):
        noisy = base + 0.045 * rng.normal(size=128).astype(np.float32)
        state = accumulator.update(noisy)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-5
    assert cosine_similarity(accumulator.value, base) > 0.9
    print("  Re-ID EMA: cosine to clean descriptor %.4f after 20 noisy frames"
          % cosine_similarity(accumulator.value, base))


def test_focus_gate_separates_sharp_from_blurred() -> None:
    rng = np.random.default_rng(4)
    sharp = (rng.integers(0, 2, size=(48, 160)) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (9, 9), 4.0)
    sharp_score = variance_of_laplacian(sharp)
    blur_score = variance_of_laplacian(blurred)
    assert sharp_score > GateConfig().laplacian_min_variance
    assert blur_score < sharp_score / 10.0
    print("  focus gate: Var(Lap) sharp=%.0f blurred=%.0f (threshold 65)" % (sharp_score, blur_score))


def test_payload_fits_the_five_kilobyte_budget() -> None:
    from veritrack_edge import EdgePayloadBuilder, PackagingConfig, PlateLayout, PlateSeries
    from veritrack_edge.types import PlateReading, StageTimings, VehiclePass

    rng = np.random.default_rng(2)
    embedding = rng.normal(size=128).astype(np.float32)
    embedding /= np.linalg.norm(embedding)

    reading = PlateReading(
        text="MH12AB1234", raw_text="MH12A81234", confidence=0.937,
        char_confidences=tuple([0.98] * 10), sequence_entropy=0.041,
        template_id="IN-STD-22", is_valid_format=True, was_repaired=True,
        repair_cost=0.61, state_code="MH", layout=PlateLayout.SINGLE_LINE,
        series=PlateSeries.PRIVATE_WHITE, observation_count=11,
    )
    vehicle_pass = VehiclePass(
        pass_id="9f2a71c4d0e5b183", camera_id="CAM-THN-07", node_id="EDGE-RK3588-014",
        track_id=4471, vehicle_class="car", first_seen=1_758_000_000.125,
        last_seen=1_758_000_002.875, embedding=embedding, reading=reading,
        entry_box=BBox(812, 430, 1040, 640), exit_box=BBox(690, 512, 1002, 792),
        timings=StageTimings(0.31, 4.9, 0.42, 2.6, 0.58, 3.7, 1.4, 0.83, 0.05),
    )
    payload = EdgePayloadBuilder(PackagingConfig()).serialize(vehicle_pass)
    assert len(payload) <= 5120, len(payload)
    import json

    decoded = json.loads(payload)
    assert decoded["plate"]["text"] == "MH12AB1234"
    assert decoded["reid"]["q"] == "int8"
    assert vehicle_pass.timings.total < 20.0
    print("  payload: %d bytes of a 5120 B budget (%.0f%% headroom)"
          % (len(payload), 100.0 * (1.0 - len(payload) / 5120.0)))


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    print(f"running {len(tests)} model-free checks\n")
    failures = 0
    for test in tests:
        try:
            print(f"* {test.__name__}")
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
