"""One-frame, headless VeriTrack ANPR demonstration.

Examples:
    .venv\Scripts\python dry_run.py --image samplecarimg.png
    .venv\Scripts\python dry_run.py --webcam --localizer auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# The project keeps importable packages under src while this convenience
# driver intentionally lives at the repository root.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from veritrack_demo.config import LocalizerConfig, DEFAULT_MODEL_ID
from veritrack_demo.localizer import ManualRoiLocalizer, build_localizer
from veritrack_demo.recognizer import PPOcrRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one headless ANPR inference and print telemetry.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="static image to process")
    source.add_argument("--webcam", action="store_true", help="capture one frame from camera index 0")
    parser.add_argument(
        "--localizer", choices=("manual", "haar", "contour", "auto"), default="manual",
        help="plate localization strategy (default: manual central ROI)",
    )
    return parser.parse_args()


def get_frame(args: argparse.Namespace) -> Tuple[np.ndarray, str]:
    image_path = args.image
    if image_path is None and not args.webcam:
        default_image = ROOT / "image.png"
        image_path = default_image if default_image.is_file() else ROOT / "samplecarimg.png"

    if image_path is not None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"could not read static image: {image_path}")
        return frame, f"Static Image ({image_path.name})"

    camera = cv2.VideoCapture(0)
    try:
        if not camera.isOpened():
            raise RuntimeError("could not open laptop webcam at index 0")
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError("webcam opened but did not return a frame")
        return frame, "Laptop Webcam (index 0)"
    finally:
        camera.release()


def localize(frame: np.ndarray, strategy: str):
    config = LocalizerConfig(strategy=strategy)
    localizer = ManualRoiLocalizer(config) if strategy == "manual" else build_localizer(config)
    started = time.perf_counter()
    candidates = localizer.locate(frame)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not candidates:
        raise RuntimeError(f"{strategy} localizer found no plate candidate")
    candidate = candidates[0]
    crop = candidate.crop(frame)
    if crop.size == 0:
        raise RuntimeError("localized plate crop is empty")
    return crop, candidate, elapsed_ms, localizer.name


def resolve_model_source() -> str:
    """Use a complete local HF snapshot when available, otherwise the model ID.

    Passing the snapshot directory directly prevents Transformers from making a
    network metadata check before it loads already-cached weights.
    """
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    snapshots = cache_root / "models--PaddlePaddle--PP-OCRv6_tiny_rec_safetensors" / "snapshots"
    if snapshots.is_dir():
        for snapshot in snapshots.iterdir():
            if all((snapshot / name).exists() for name in ("config.json", "preprocessor_config.json", "model.safetensors")):
                return str(snapshot)
    return DEFAULT_MODEL_ID


def main() -> int:
    args = parse_args()
    print("[1/3] Acquiring input frame...", flush=True)
    frame, source_name = get_frame(args)
    print(f"[2/3] Localizing plate with {args.localizer} strategy...", flush=True)
    crop, candidate, localization_ms, localizer_name = localize(frame, args.localizer)

    print(f"[3/3] Loading PP-OCRv6-Tiny on CPU and recognizing {crop.shape[1]}x{crop.shape[0]} crop...", flush=True)
    recognizer = PPOcrRecognizer(resolve_model_source(), device="cpu", warmup_on_load=False)
    result = recognizer.recognize(crop)
    telemetry = {
        "schema": "veritrack.edge.telemetry.v1",
        "event_type": "plate_read",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_source": "static_image" if source_name.startswith("Static") else "laptop_webcam",
        "model": DEFAULT_MODEL_ID,
        "device": recognizer.resolved_device or "cpu",
        "localizer": localizer_name,
        "plate": {"text": result.text, "confidence": round(result.confidence, 6)},
        "crop": {
            "width": int(crop.shape[1]), "height": int(crop.shape[0]),
            "bbox_xyxy": list(candidate.as_int_box()), "candidate_score": round(candidate.score, 6),
        },
        "timing_ms": {"preprocess_localize": round(localization_ms, 2), "ocr_inference": round(result.inference_ms, 2)},
    }

    print("\n" + "=" * 72)
    print("                 VERITRACK EDGE - HEADLESS DRY RUN")
    print("=" * 72)
    print(f"Input source                 : {source_name}")
    print(f"Localizer                    : {localizer_name}")
    print(f"Crop dimensions              : {crop.shape[1]} x {crop.shape[0]} px")
    print(f"Pre-processing / localization: {localization_ms:.2f} ms")
    print(f"OCR inference                : {result.inference_ms:.2f} ms")
    print(f"Decoded plate text           : {result.text or '<empty>'}")
    print(f"Confidence score             : {result.confidence * 100:.2f}%")
    print("-" * 72)
    print("Edge Telemetry JSON packet:")
    print(json.dumps(telemetry, indent=2))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"DRY RUN FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
