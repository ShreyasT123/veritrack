"""Configuration for the VeriTrack laptop demo runner.

This is a separate deployment target from ``veritrack_edge``: it targets a
CPU-only Windows/WSL2 laptop with a USB or built-in webcam, standing in for
the RK3588/Jetson pole hardware Stage 1 was written for. It reuses Stage 1's
geometry and grammar code (``rectify``, ``splitter``, ``validation``) because
those are hardware-independent math, but it swaps the model backend for a
HuggingFace-hosted PP-OCRv6-Tiny recognizer running on CPU, and swaps RTSP
ingestion for a local ``cv2.VideoCapture`` device.

Frozen dataclass, matching the style of ``veritrack_edge.config`` — this is
read once at startup from a trusted local file or CLI args, not from the
network, so there is no case for Pydantic's validation cost here either.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Final, Mapping, Optional, Tuple

__all__ = ["CaptureConfig", "LocalizerConfig", "RecognizerConfig", "GatewayConfig", "DemoConfig", "load_demo_config"]

#: The exact model the recognizer was validated against on the user's laptop:
#: 0.098 s CPU inference, 0.947 confidence, correct read on a real photograph.
DEFAULT_MODEL_ID: Final[str] = "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors"


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Webcam capture parameters."""

    #: OpenCV device index. 0 is almost always the built-in laptop camera;
    #: an external USB webcam usually enumerates as 1, but this varies with
    #: what else is plugged in, so the CLI exposes a ``--list-cameras`` probe
    #: rather than asking the user to guess.
    device_index: int = 0

    #: "dshow" (Windows DirectShow — fast, reliable, the right default on
    #: native Windows Python), "msmf" (Windows Media Foundation — the fallback
    #: when DirectShow can't open a given device), or "any" (let OpenCV pick,
    #: which is the right choice inside WSL2/Linux where DirectShow does not
    #: exist). Never "auto-detect the OS": the same Python interpreter can run
    #: under native Windows or under WSL2 on the same machine, and only the
    #: caller knows which one this process is in.
    backend: str = "dshow"

    frame_width: int = 1280
    frame_height: int = 720
    #: Requested capture FPS. Actual FPS is capped by the recognizer's
    #: inference time once the gate opens (~100 ms/candidate on the reference
    #: i5-13th-gen laptop), not by this value; it mainly controls exposure/
    #: buffering behaviour on cameras that support it.
    requested_fps: int = 30
    #: OpenCV internally buffers a few frames; on a slow consumer loop that
    #: buffer fills and every frame you read is stale. Setting the capture
    #: buffer size to 1 trades a little smoothness for "what you see is what
    #: is being processed right now", which matters far more in a live demo.
    buffer_size: int = 1

    def __post_init__(self) -> None:
        if self.device_index < 0:
            raise ValueError("device_index must be >= 0")
        if self.backend not in ("dshow", "msmf", "any"):
            raise ValueError(f"unknown capture backend {self.backend!r}")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame_width and frame_height must be positive")
        if self.requested_fps <= 0:
            raise ValueError("requested_fps must be positive")
        if self.buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")


@dataclass(frozen=True, slots=True)
class LocalizerConfig:
    """Plate-region localization: where in the frame to look for a plate."""

    #: "manual" (fixed/adjustable on-screen ROI — the reliable default for a
    #: live demo: you hold the plate inside a drawn box), "haar" (OpenCV's
    #: bundled cascade classifier — automatic, zero extra download, coarse),
    #: "contour" (classic edge+contour rectangle search — automatic, zero
    #: extra download, different failure modes than Haar), or "auto" (try haar,
    #: then contour, then fall back to the manual ROI if both find nothing —
    #: the most robust option once you trust the demo, but not the one to
    #: bet a live audience on the first time).
    strategy: str = "manual"

    #: Manual ROI as fractions of frame (left, top, right, bottom), matching
    #: the hardcoded ``crop_box`` percentages from the validated laptop script
    #: rather than absolute pixels, so it is resolution-independent.
    manual_roi_fractional: Tuple[float, float, float, float] = (0.28, 0.42, 0.75, 0.64)

    #: Minimum plate-like aspect ratio (width/height) accepted by the
    #: automatic localizers. An Indian single-line plate is nominally
    #: 500:120 = 4.17; 2.0 gives headroom for a two-line plate (nominal 1.43)
    #: without accepting near-square noise.
    min_aspect_ratio: float = 1.3
    max_aspect_ratio: float = 6.5
    #: Reject candidates smaller than this fraction of the frame area — noise
    #: and false positives cluster at the small end.
    min_area_fraction: float = 0.0015
    max_area_fraction: float = 0.35

    haar_scale_factor: float = 1.05
    haar_min_neighbors: int = 4

    #: Canny thresholds for the contour strategy.
    contour_canny_low: int = 50
    contour_canny_high: int = 150

    #: Both automatic strategies run their detection pass on a copy of the
    #: frame downscaled so its width does not exceed this, then rescale any
    #: found boxes back to full-frame coordinates. This is not a minor
    #: optimisation: Haar's ``detectMultiScale`` measured at **420 ms** on a
    #: 1280x720 frame on the reference i5-13th-gen laptop — roughly 2 fps —
    #: and at **44 ms** (~9.4x faster) downscaled to width 480, which is the
    #: difference between a live demo and a slideshow. A demo plate is held
    #: close to the camera, so the resolution lost is resolution the detector
    #: never needed.
    detection_max_width: int = 480

    def __post_init__(self) -> None:
        if self.strategy not in ("manual", "haar", "contour", "auto"):
            raise ValueError(f"unknown localizer strategy {self.strategy!r}")
        left, top, right, bottom = self.manual_roi_fractional
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise ValueError(
                f"manual_roi_fractional must satisfy 0<=left<right<=1 and "
                f"0<=top<bottom<=1, got {self.manual_roi_fractional}"
            )
        if self.min_aspect_ratio <= 0.0 or self.max_aspect_ratio <= self.min_aspect_ratio:
            raise ValueError("aspect ratio bounds must be positive and increasing")
        if not 0.0 < self.min_area_fraction < self.max_area_fraction <= 1.0:
            raise ValueError("area fraction bounds must satisfy 0 < min < max <= 1")
        if self.detection_max_width < 64:
            raise ValueError("detection_max_width must be >= 64")


@dataclass(frozen=True, slots=True)
class RecognizerConfig:
    """PP-OCRv6-Tiny recognizer, run locally via HuggingFace ``transformers``."""

    model_id: str = DEFAULT_MODEL_ID
    #: "cpu" is the honest default for the target laptop (Intel i5, no GPU).
    #: "auto" checks ``torch.cuda.is_available()`` at load time, matching the
    #: pattern in the validated script, so the same code also runs well on a
    #: GPU machine without edits.
    device: str = "cpu"
    #: Recognition below this score is kept but visually flagged, rather than
    #: discarded — a low-confidence read is still worth showing an operator,
    #: it is just not worth forwarding to Stage 2 as a confirmed sighting.
    min_confidence_for_display: float = 0.30
    min_confidence_for_forward: float = 0.60
    #: Batch every candidate crop from one frame through a single forward
    #: pass. On CPU, model launch overhead dominates a 1-image batch, so
    #: batching 2-3 ROI candidates costs little more than one.
    max_batch_size: int = 4
    #: Run one dummy inference at startup so the first live detection is not
    #: the one that pays for lazy weight materialisation and any JIT/kernel
    #: warm-up inside torch.
    warmup_on_load: bool = True

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError(f"unknown device {self.device!r}")
        if not 0.0 <= self.min_confidence_for_display <= 1.0:
            raise ValueError("min_confidence_for_display must lie in [0, 1]")
        if not 0.0 <= self.min_confidence_for_forward <= 1.0:
            raise ValueError("min_confidence_for_forward must lie in [0, 1]")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    """Optional forwarding to the Stage 2 ingestion gateway.

    Everything this demo sends that Stage 1 would normally have measured for
    real — the vehicle bounding box and the Re-ID embedding — the demo cannot
    produce, because it has no vehicle detector or Re-ID model. Forwarded
    payloads carry an explicit placeholder marker (see
    ``gateway_client.build_compact_payload``) rather than silently passing off
    stand-in values as real detections.
    """

    enabled: bool = False
    base_url: str = "http://localhost:8080"
    ingest_path: str = "/api/v1/telemetry/ingest/compact"
    api_key: Optional[str] = None
    timeout_s: float = 2.0
    node_id: str = "DEMO-LAPTOP-01"
    camera_id: str = "DEMO-WEBCAM"

    def __post_init__(self) -> None:
        if self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if not self.node_id or not self.camera_id:
            raise ValueError("node_id and camera_id must be non-empty")


@dataclass(frozen=True, slots=True)
class DemoConfig:
    """Root configuration for the webcam demo."""

    capture: CaptureConfig = field(default_factory=CaptureConfig)
    localizer: LocalizerConfig = field(default_factory=LocalizerConfig)
    recognizer: RecognizerConfig = field(default_factory=RecognizerConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)

    #: Run the (cheap, CPU-only) recognizer at most once every N frames, and
    #: reuse the last result the rest of the time. At 30 fps capture and a
    #: ~100 ms recognizer, running it every frame would fall further and
    #: further behind; skipping frames keeps display smooth and keeps the
    #: recognizer's input queue from ever growing unbounded.
    process_every_n_frames: int = 3
    window_title: str = "VeriTrack - Live Demo"
    save_snapshots_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.process_every_n_frames < 1:
            raise ValueError("process_every_n_frames must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        payload = self.to_dict()
        if payload.get("save_snapshots_dir") is not None:
            payload["save_snapshots_dir"] = str(payload["save_snapshots_dir"])
        return json.dumps(payload, indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DemoConfig":
        snapshots = payload.get("save_snapshots_dir")
        return cls(
            capture=CaptureConfig(**payload.get("capture", {})),
            localizer=LocalizerConfig(**payload.get("localizer", {})),
            recognizer=RecognizerConfig(**payload.get("recognizer", {})),
            gateway=GatewayConfig(**payload.get("gateway", {})),
            process_every_n_frames=int(payload.get("process_every_n_frames", 3)),
            window_title=str(payload.get("window_title", "VeriTrack - Live Demo")),
            save_snapshots_dir=(Path(snapshots) if snapshots else None),
        )

    def evolve(self, **changes: Any) -> "DemoConfig":
        return replace(self, **changes)


def load_demo_config(path: str | Path) -> DemoConfig:
    """Load a JSON demo config, or raise a clear error if the file is missing."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"demo config not found: {file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("demo config must be a JSON object")
    return DemoConfig.from_dict(payload)
