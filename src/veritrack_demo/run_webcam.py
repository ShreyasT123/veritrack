"""CLI entrypoint: ``python -m veritrack_demo.run_webcam``.

Windows note: the default capture backend is DirectShow (``cv2.CAP_DSHOW``),
which is the fast, reliable choice for a native Windows Python process talking
to a built-in or USB webcam. If you are running this *inside WSL2* instead
(after attaching the camera with ``usbipd``), pass ``--backend any`` — DirectShow
does not exist there. See README.md ("Stage 1 laptop demo") for the full
Windows / WSL2 / Docker decision and why native Windows is the recommended
path for the camera specifically.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import cv2

from .config import CaptureConfig, DemoConfig
from .gateway_client import GatewayForwarder
from .localizer import ManualRoiLocalizer
from .overlay import draw_detections, draw_hud, draw_manual_roi
from .pipeline import DemoPipeline, build_pipeline

LOGGER = logging.getLogger("veritrack.demo")

_BACKEND_FLAGS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
_ROI_STEP = 0.02
_ROI_GROW = 1.08
_ROI_SHRINK = 0.93

_HELP_TEXT = (
    "[q] quit  [s] snapshot  [f] toggle forward  [1/2/3/4] manual/haar/contour/auto  "
    "[wasd] move ROI  [+/-] resize ROI"
)


def list_cameras(max_index: int = 6) -> None:
    """Probe device indices 0..max_index-1 and report which ones open.

    Run this first on an unfamiliar machine: the built-in camera is not
    guaranteed to be index 0 once an external USB webcam is also plugged in,
    and guessing wrong just gets you "could not open camera" mid-setup.
    """
    print("Probing camera indices (each is opened and immediately released)...")
    for index in range(max_index):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        opened = capture.isOpened()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        capture.release()
        status = f"OK  {width}x{height}" if opened else "not available"
        print(f"  index {index}: {status}")


def open_capture(config: CaptureConfig) -> cv2.VideoCapture:
    """Open the configured camera, or raise with an actionable message."""
    backend_flag = _BACKEND_FLAGS[config.backend]
    capture = cv2.VideoCapture(config.device_index, backend_flag)
    if not capture.isOpened():
        raise RuntimeError(
            f"could not open camera index {config.device_index} with backend "
            f"{config.backend!r}. Run with --list-cameras to see what is available, "
            f"or try --backend any (needed inside WSL2, where DirectShow does not exist)."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    capture.set(cv2.CAP_PROP_FPS, config.requested_fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, config.buffer_size)
    return capture


def _handle_key(key: int, pipeline: DemoPipeline, config: DemoConfig) -> DemoPipeline:
    """Apply one keypress. Returns the (possibly rebuilt) pipeline.

    Switching localizer strategy mid-run rebuilds the pipeline with a new
    localizer instance rather than mutating strategy state in place — the
    three strategies are different classes entirely, so "switch strategy" is
    naturally "construct a different object", not a flag flip.
    """
    from .localizer import ContourPlateLocalizer, FallbackLocalizer, HaarCascadePlateLocalizer

    if key == ord("1"):
        return build_pipeline(config, localizer=ManualRoiLocalizer(config.localizer),
                              recognizer=pipeline.recognizer)
    if key == ord("2"):
        return build_pipeline(config, localizer=HaarCascadePlateLocalizer(config.localizer),
                              recognizer=pipeline.recognizer)
    if key == ord("3"):
        return build_pipeline(config, localizer=ContourPlateLocalizer(config.localizer),
                              recognizer=pipeline.recognizer)
    if key == ord("4"):
        return build_pipeline(
            config,
            localizer=FallbackLocalizer([
                HaarCascadePlateLocalizer(config.localizer),
                ContourPlateLocalizer(config.localizer),
                ManualRoiLocalizer(config.localizer),
            ]),
            recognizer=pipeline.recognizer,
        )

    if isinstance(pipeline.localizer, ManualRoiLocalizer):
        moves = {ord("w"): (0.0, -_ROI_STEP), ord("a"): (-_ROI_STEP, 0.0),
                 ord("s"): (0.0, _ROI_STEP), ord("d"): (_ROI_STEP, 0.0)}
        if key in moves:
            new_localizer = pipeline.localizer.nudge(*moves[key])
            return build_pipeline(config, localizer=new_localizer, recognizer=pipeline.recognizer)
        if key in (ord("+"), ord("=")):
            return build_pipeline(config, localizer=pipeline.localizer.resize(_ROI_GROW),
                                  recognizer=pipeline.recognizer)
        if key == ord("-"):
            return build_pipeline(config, localizer=pipeline.localizer.resize(_ROI_SHRINK),
                                  recognizer=pipeline.recognizer)
    return pipeline


def run(config: DemoConfig, pipeline: DemoPipeline) -> None:
    capture = open_capture(config.capture)
    forwarder = GatewayForwarder(config.gateway) if config.gateway.enabled else None
    forwarding_enabled = config.gateway.enabled

    if config.save_snapshots_dir is not None:
        config.save_snapshots_dir.mkdir(parents=True, exist_ok=True)

    fps_smoothed = 0.0
    last_tick = time.perf_counter()
    frame_counter = 0

    print(_HELP_TEXT)
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                LOGGER.warning("frame grab failed; retrying")
                time.sleep(0.05)
                continue

            detections = pipeline.process_frame(frame)

            if forwarding_enabled and forwarder is not None:
                for detection in detections:
                    if detection.recognition.confidence >= config.recognizer.min_confidence_for_forward:
                        forwarder.forward(detection)

            visible = [
                d for d in detections
                if d.recognition.confidence >= config.recognizer.min_confidence_for_display
            ]
            draw_manual_roi(frame, pipeline.localizer, active=isinstance(pipeline.localizer, ManualRoiLocalizer))
            draw_detections(frame, visible)

            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - last_tick, 1e-6)
            fps_smoothed = fps_smoothed * 0.9 + instantaneous_fps * 0.1
            last_tick = now

            draw_hud(
                frame,
                fps=fps_smoothed,
                localizer_name=pipeline.localizer.name,
                forwarding=forwarding_enabled,
                forwarded_count=forwarder.sent_count if forwarder else 0,
                help_text=_HELP_TEXT,
            )

            cv2.imshow(config.window_title, frame)
            frame_counter += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f") and forwarder is not None:
                forwarding_enabled = not forwarding_enabled
                print(f"forward-to-gateway: {'ON' if forwarding_enabled else 'off'}")
            elif key == ord("s") and config.save_snapshots_dir is not None:
                out_path = config.save_snapshots_dir / f"snapshot_{frame_counter:06d}.png"
                cv2.imwrite(str(out_path), frame)
                print(f"saved {out_path}")
            else:
                pipeline = _handle_key(key, pipeline, config)
    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VeriTrack live webcam ANPR demo")
    parser.add_argument("--list-cameras", action="store_true", help="probe camera indices and exit")
    parser.add_argument("--config", type=Path, default=None, help="path to a DemoConfig JSON file")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--backend", choices=sorted(_BACKEND_FLAGS), default=None)
    parser.add_argument("--localizer", choices=["manual", "haar", "contour", "auto"], default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None,
                        help="recognizer device; this laptop has no GPU, so leave at cpu")
    parser.add_argument("--forward", action="store_true", help="enable Stage 2 gateway forwarding")
    parser.add_argument("--gateway-url", type=str, default=None)
    parser.add_argument("--snapshots-dir", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _apply_overrides(config: DemoConfig, args: argparse.Namespace) -> DemoConfig:
    """CLI flags override file/default config, field by field, never silently."""
    capture = config.capture
    if args.camera_index is not None:
        capture = replace(capture, device_index=args.camera_index)
    if args.backend is not None:
        capture = replace(capture, backend=args.backend)

    localizer_cfg = config.localizer
    if args.localizer is not None:
        localizer_cfg = replace(localizer_cfg, strategy=args.localizer)

    recognizer_cfg = config.recognizer
    if args.device is not None:
        recognizer_cfg = replace(recognizer_cfg, device=args.device)

    gateway_cfg = config.gateway
    if args.forward:
        gateway_cfg = replace(gateway_cfg, enabled=True)
    if args.gateway_url is not None:
        gateway_cfg = replace(gateway_cfg, base_url=args.gateway_url, enabled=True)

    return config.evolve(
        capture=capture,
        localizer=localizer_cfg,
        recognizer=recognizer_cfg,
        gateway=gateway_cfg,
        save_snapshots_dir=args.snapshots_dir or config.save_snapshots_dir,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    if args.list_cameras:
        list_cameras()
        return 0

    config = (
        DemoConfig.from_dict({})
        if args.config is None
        else DemoConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    )
    config = _apply_overrides(config, args)

    LOGGER.info("starting with config:\n%s", config.to_json())
    pipeline = build_pipeline(config)
    try:
        run(config, pipeline)
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
