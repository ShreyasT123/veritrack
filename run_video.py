"""Real-time, single-stream ANPR demonstration for a local video file.

Example:
    .venv\Scripts\python.exe run_video.py --video sample_30fps_1440.mp4 --show
    .venv\Scripts\python.exe run_video.py --output annotated_demo.mp4
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from veritrack_demo.config import DEFAULT_MODEL_ID, GatewayConfig, LocalizerConfig
from veritrack_demo.gateway_client import GatewayForwarder
from veritrack_demo.localizer import ManualRoiLocalizer, PlateCandidate, PlateLocalizer, build_localizer
from veritrack_demo.overlay import draw_detections, draw_hud
from veritrack_demo.pipeline import DemoDetection
from veritrack_demo.recognizer import PPOcrRecognizer
from veritrack_edge.config import RectifyConfig, ValidationConfig
from veritrack_edge.errors import GeometryError
from veritrack_edge.rectify import PlateRectifier
from veritrack_edge.types import BBox, PlateQuad
from veritrack_edge.validation import PlateValidator


@dataclass(slots=True)
class PlateRecord:
    first_seen_s: float
    confidences: List[float]
    latencies_ms: List[float]


def parse_args() -> argparse.Namespace:
    default = ROOT / "sample_30fps_1440.mp4"
    parser = argparse.ArgumentParser(description="Run annotated PP-OCR ANPR over a video stream.")
    parser.add_argument("--video", type=Path, default=default if default.is_file() else ROOT / "sample_traffic.mp4")
    parser.add_argument("--skip-frames", type=int, default=2, help="process every N+1th frame (default: 2)")
    parser.add_argument("--localizer", choices=("auto", "contour", "haar", "manual"), default="auto")
    parser.add_argument("--show", action="store_true", help="show the annotated OpenCV display window")
    parser.add_argument("--output", type=Path, default=None, help="optional annotated MP4 destination")
    parser.add_argument("--forward", action="store_true", help="POST confident demo reads to the gateway")
    parser.add_argument("--gateway-url", default="http://localhost:8080", help="gateway base URL used with --forward")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames; 0 processes the full source")
    return parser.parse_args()


def cached_model_source() -> str:
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / "models--PaddlePaddle--PP-OCRv6_tiny_rec_safetensors" / "snapshots"
    if snapshots.is_dir():
        for folder in snapshots.iterdir():
            if all((folder / name).exists() for name in ("config.json", "preprocessor_config.json", "model.safetensors")):
                return str(folder)
    return DEFAULT_MODEL_ID


def make_localizer(strategy: str) -> Tuple[PlateLocalizer, str]:
    """Construct a requested localizer, degrading auto/Haar gracefully on OpenCV builds without Haar support."""
    config = LocalizerConfig(strategy=strategy)
    try:
        return (ManualRoiLocalizer(config) if strategy == "manual" else build_localizer(config)), strategy
    except (AttributeError, RuntimeError) as exc:
        if strategy not in ("auto", "haar"):
            raise
        fallback_config = LocalizerConfig(strategy="contour")
        print(f"[localizer] {strategy} unavailable ({exc}); using contour fallback.", flush=True)
        return build_localizer(fallback_config), "contour"


def crop_candidate(frame: np.ndarray, candidate: PlateCandidate, rectifier: PlateRectifier) -> np.ndarray:
    try:
        quad = PlateQuad(BBox(candidate.x1, candidate.y1, candidate.x2, candidate.y2), candidate.quad.astype(np.float32), candidate.score)
        return rectifier.rectify(frame, quad).image
    except GeometryError:
        return candidate.crop(frame)


def make_detection(frame: np.ndarray, candidate: PlateCandidate, recognizer: PPOcrRecognizer,
                   rectifier: PlateRectifier, validator: PlateValidator) -> Tuple[DemoDetection, np.ndarray]:
    started = time.perf_counter()
    crop = crop_candidate(frame, candidate, rectifier)
    result = recognizer.recognize(crop)
    validation = validator.validate(result.text)
    return DemoDetection(candidate=candidate, recognition=result, validated_text=validation.text, raw_text=validation.raw_text,
                         is_valid_format=validation.is_valid, was_repaired=validation.was_repaired,
                         repair_cost=validation.repair_cost, state_code=validation.state_code,
                         rectified=True, total_latency_ms=(time.perf_counter() - started) * 1000.0), crop


def inset_crop(frame: np.ndarray, crop: Optional[np.ndarray]) -> None:
    if crop is None or crop.size == 0:
        return
    height, width = frame.shape[:2]
    inset_width = min(310, max(120, width // 4))
    inset_height = max(48, int(inset_width * crop.shape[0] / max(crop.shape[1], 1)))
    inset_height = min(inset_height, height // 4)
    view = cv2.resize(crop, (inset_width, inset_height), interpolation=cv2.INTER_CUBIC)
    x, y = width - inset_width - 16, 16
    frame[y:y + inset_height, x:x + inset_width] = view
    cv2.rectangle(frame, (x - 2, y - 22), (x + inset_width + 2, y + inset_height + 2), (0, 200, 0), 2)
    cv2.putText(frame, "PLATE CROP / RECTIFIED", (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)


def draw_video_banner(frame: np.ndarray, source_fps: float, elapsed_s: float, unique_plates: int, mean_latency_ms: float) -> None:
    text = f"VIDEO FPS {source_fps:.1f} | ELAPSED {elapsed_s:06.1f}s | UNIQUE PLATES {unique_plates} | MEAN E2E {mean_latency_ms:.1f}ms"
    cv2.rectangle(frame, (8, 92), (min(frame.shape[1] - 8, 980), 122), (18, 58, 138), -1)
    cv2.putText(frame, text, (16, 113), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 2, cv2.LINE_AA)


def print_summary(frames_read: int, frames_processed: int, records: Dict[str, PlateRecord]) -> None:
    latencies = [item for record in records.values() for item in record.latencies_ms]
    print("\n" + "=" * 78)
    print("                        VERITRACK VIDEO ANPR SUMMARY")
    print("=" * 78)
    print(f"Frames read / processed : {frames_read} / {frames_processed}")
    print(f"Unique plates recognized: {len(records)}")
    print(f"Mean end-to-end latency  : {statistics.fmean(latencies) if latencies else 0.0:.2f} ms")
    print("-" * 78)
    if not records:
        print("No plate text was recognized. Try --localizer manual or contour for this footage.")
    for plate, record in sorted(records.items(), key=lambda item: item[1].first_seen_s):
        print(f"{record.first_seen_s:8.2f}s  {plate:<20} mean confidence {statistics.fmean(record.confidences) * 100:5.2f}%  reads {len(record.confidences)}")
    print("=" * 78)


def main() -> int:
    args = parse_args()
    if args.skip_frames < 0 or args.max_frames < 0:
        raise ValueError("--skip-frames and --max-frames must be non-negative")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {args.video}")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {args.video.name}: {width}x{height}, {fps:.3f} FPS, {total} frames", flush=True)

    localizer, strategy = make_localizer(args.localizer)
    recognizer = PPOcrRecognizer(cached_model_source(), device="cpu", warmup_on_load=False)
    rectifier, validator = PlateRectifier(RectifyConfig()), PlateValidator(ValidationConfig())
    forwarder = GatewayForwarder(GatewayConfig(enabled=True, base_url=args.gateway_url, camera_id="DEMO-VIDEO")) if args.forward else None
    writer: Optional[cv2.VideoWriter] = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"unable to create output video: {args.output}")

    records: Dict[str, PlateRecord] = {}
    recently_seen: Dict[str, Tuple[float, Tuple[int, int, int, int]]] = {}
    latencies: List[float] = []
    last_detections: Sequence[DemoDetection] = ()
    last_crop: Optional[np.ndarray] = None
    paused, frame_index, frames_read, frames_processed = False, 0, 0, 0
    playback_started = time.perf_counter()
    try:
        while True:
            if not paused:
                ok, frame = capture.read()
                if not ok:
                    break
                frames_read += 1
                frame_index += 1
                if args.max_frames and frames_read >= args.max_frames:
                    break
                if frame_index % (args.skip_frames + 1) == 1:
                    processed_started = time.perf_counter()
                    candidates = localizer.locate(frame)
                    detections: List[DemoDetection] = []
                    crop: Optional[np.ndarray] = None
                    for candidate in candidates[:1]:
                        detection, crop = make_detection(frame, candidate, recognizer, rectifier, validator)
                        detections.append(detection)
                    last_detections, last_crop = detections, crop
                    frames_processed += 1
                    now_s = frame_index / fps
                    for detection in detections:
                        plate = detection.display_text.strip()
                        box = detection.candidate.as_int_box()
                        previous = recently_seen.get(plate)
                        same_nearby = previous is not None and now_s - previous[0] <= 1.5 and sum(abs(a - b) for a, b in zip(box, previous[1])) < 80
                        recently_seen[plate] = (now_s, box)
                        if plate and plate != "?" and not same_nearby:
                            entry = records.setdefault(plate, PlateRecord(now_s, [], []))
                            entry.confidences.append(detection.recognition.confidence)
                            entry.latencies_ms.append(detection.total_latency_ms)
                            if forwarder is not None and detection.recognition.confidence >= 0.60:
                                forwarder.forward(detection)
                    latencies.append((time.perf_counter() - processed_started) * 1000.0)

                annotated = frame.copy()
                draw_detections(annotated, last_detections)
                draw_hud(annotated, fps=fps, localizer_name=strategy, forwarding=forwarder is not None,
                         forwarded_count=forwarder.sent_count if forwarder else 0, help_text="[q/ESC] quit  [SPACE] pause  [s] switch localizer")
                inset_crop(annotated, last_crop)
                draw_video_banner(annotated, fps, frame_index / fps, len(records), statistics.fmean(latencies) if latencies else 0.0)
                if writer is not None:
                    writer.write(annotated)
                if args.show:
                    cv2.imshow("VeriTrack - Video ANPR", annotated)

                target = playback_started + frame_index / fps
                delay = target - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
            if args.show:
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                if key == ord("s"):
                    choices = ("auto", "contour", "haar", "manual")
                    strategy_request = choices[(choices.index(strategy) + 1) % len(choices)] if strategy in choices else "auto"
                    localizer, strategy = make_localizer(strategy_request)
                    print(f"[localizer] switched to {strategy}", flush=True)
            elif paused:
                paused = False
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    print_summary(frames_read, frames_processed, records)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VIDEO DEMO FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
