"""Typed, immutable configuration for the VeriTrack edge node.

Deliberately dataclass-based rather than Pydantic: the edge node targets an
RK3588 / Orin Nano where the import cost and per-object validation overhead of
Pydantic are not justified. Pydantic is introduced at the central gateway
(Stage 2) where untrusted payloads actually arrive.

All thresholds that appear in the problem statement are surfaced here as named
fields so they can be re-tuned per camera without touching algorithm code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Tuple

BackendKind = Literal["onnxruntime", "rknnlite"]
DetectorLayout = Literal["yolov8", "yolox"]

# Canonical recognition charset. Index 0 is reserved for the CTC blank.
DEFAULT_CHARSET: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Inference runtime selection and thread/NPU affinity."""

    kind: BackendKind = "onnxruntime"
    providers: Tuple[str, ...] = ("CPUExecutionProvider",)
    intra_op_threads: int = 2
    inter_op_threads: int = 1
    # RKNNLite core mask: 0 = auto, 1 = NPU core 0, 2 = core 1, 4 = core 2, 7 = all.
    npu_core_mask: int = 0
    warmup_iterations: int = 3


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """RTSP / file ingestion parameters."""

    uri: str = "rtsp://127.0.0.1:8554/cam"
    # OpenCV backend hint; CAP_FFMPEG == 1900.
    capture_api: int = 1900
    reconnect_backoff_s: float = 1.0
    reconnect_backoff_max_s: float = 15.0
    read_timeout_s: float = 5.0
    # Keep-latest queue: the grabber thread drops stale frames so the pipeline
    # always operates on the freshest available image.
    queue_size: int = 2
    target_fps: float = 25.0


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Cheap pre-filters that keep the NPU idle on empty or unusable frames."""

    # Var(Laplacian) below this is discarded as out-of-focus / motion-blurred.
    laplacian_min_variance: float = 65.0
    # Reference variance at which focus quality saturates to 1.0.
    laplacian_reference_variance: float = 180.0
    motion_enabled: bool = True
    motion_downscale: float = 0.25
    motion_history: int = 250
    motion_var_threshold: float = 16.0
    motion_detect_shadows: bool = False
    motion_min_fg_ratio: float = 0.004
    motion_warmup_frames: int = 60
    motion_open_kernel: int = 3


@dataclass(frozen=True, slots=True)
class VehicleDetectorConfig:
    """Anchor-free vehicle detector (YOLO-family export)."""

    model_path: Path = Path("models/vehicle_det.onnx")
    input_size: Tuple[int, int] = (640, 384)  # (width, height)
    layout: DetectorLayout = "yolov8"
    strides: Tuple[int, ...] = (8, 16, 32)
    # ByteTrack's two association tiers.
    conf_high: float = 0.55
    conf_low: float = 0.15
    nms_iou: float = 0.60
    max_detections: int = 64
    # COCO-style vehicle classes: car, motorcycle, bus, truck.
    keep_class_ids: Tuple[int, ...] = (2, 3, 5, 7)
    class_names: Tuple[str, ...] = ("car", "motorcycle", "bus", "truck")
    min_box_area_px: float = 480.0


@dataclass(frozen=True, slots=True)
class PlateDetectorConfig:
    """Plate detector with 4-point corner keypoint regression.

    Expected ONNX output contract: float32 ``(1, N, 13)`` where each row is
    ``[cx, cy, w, h, score, k0x, k0y, k1x, k1y, k2x, k2y, k3x, k3y]`` expressed
    in letterboxed input-image pixels. Keypoints are emitted in raster order
    (top-left, top-right, bottom-right, bottom-left) by the training head; the
    rectifier re-canonicalises them defensively regardless.
    """

    model_path: Path = Path("models/plate_kpt.onnx")
    input_size: Tuple[int, int] = (192, 192)
    conf_threshold: float = 0.40
    nms_iou: float = 0.45
    max_detections: int = 4
    min_quad_area_px: float = 240.0
    # Vehicle-box expansion before cropping, as a fraction of box size.
    roi_pad_ratio: float = 0.06


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics in *full-frame* pixels. Optional.

    When supplied, plate pose (yaw/pitch/roll) is recovered analytically from
    the plate-plane homography instead of being approximated from edge
    foreshortening. See :func:`veritrack_edge.rectify.decompose_plate_pose`.
    """

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True, slots=True)
class RectifyConfig:
    """Homography rectification and line-splitting geometry."""

    # Canonical recognition strip, (width, height).
    single_line_size: Tuple[int, int] = (160, 48)
    # Intermediate canvas for two-line plates before the row split.
    two_line_size: Tuple[int, int] = (128, 96)
    # Quad aspect ratio (mean horizontal edge / mean vertical edge) at or above
    # which the plate is treated as single-line. Derived, not guessed: a
    # 500x120 mm plate has a nominal AR of 4.17, which compresses to
    # 4.17 * cos(45 deg) = 2.95 at the maximum tolerated skew. 2.70 sits just
    # below that (equivalent to 49.7 deg), so a single-line plate anywhere in
    # the supported envelope is never mistaken for a two-line one, while a
    # two-line plate (nominal AR 1.43) stays far clear of the boundary.
    two_line_aspect_threshold: float = 2.70
    # Physical plate dimensions in mm, used for metric pose decomposition and
    # for the aspect-compression skew estimate.
    single_line_mm: Tuple[float, float] = (500.0, 120.0)
    two_line_mm: Tuple[float, float] = (285.0, 200.0)
    max_skew_deg: float = 45.0
    # Sanity gate on the corner regression, not a skew gate: the ratio of the
    # shorter to the longer vertical edge. At ANPR standoff a genuine plate is
    # near 1.0 even at 45 deg yaw (perspective is weak at 10-25 m), so anything
    # this lopsided indicates a bad keypoint prediction rather than a hard view.
    min_edge_symmetry: float = 0.35
    border_mode_replicate: bool = True
    # Row-split search band, as fractions of the two-line canvas height.
    split_band: Tuple[float, float] = (0.35, 0.68)
    min_valley_contrast: float = 0.35
    intrinsics: Optional[CameraIntrinsics] = None


@dataclass(frozen=True, slots=True)
class OcrConfig:
    """SVTR / PP-OCRv6-Tiny CTC recogniser and multi-frame fusion."""

    model_path: Path = Path("models/svtr_tiny_rec.onnx")
    input_size: Tuple[int, int] = (160, 48)  # (width, height)
    charset: str = DEFAULT_CHARSET
    blank_index: int = 0
    beam_width: int = 10
    topk_per_step: int = 6
    # Tracklet fusion.
    max_observations: int = 12
    entropy_lambda: float = 3.0
    detection_conf_exponent: float = 0.5
    focus_exponent: float = 0.5
    min_observations: int = 1
    min_fused_confidence: float = 0.45
    # Mean/std applied after scaling to [0, 1]; SVTR convention.
    normalize_mean: float = 0.5
    normalize_std: float = 0.5


@dataclass(frozen=True, slots=True)
class ReidConfig:
    """OSNet appearance embedding."""

    model_path: Path = Path("models/osnet_x0_25.onnx")
    input_size: Tuple[int, int] = (128, 256)  # (width, height)
    embedding_dim: int = 128
    # Exponential moving average over the tracklet; 1.0 disables smoothing.
    ema_alpha: float = 0.85
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    refresh_every_n_frames: int = 5


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """ByteTrack association parameters."""

    track_high_thresh: float = 0.55
    track_low_thresh: float = 0.15
    new_track_thresh: float = 0.70
    # These are maximum IoU *distances* (1 - IoU), matching upstream ByteTrack.
    # 0.80 therefore admits any pair overlapping by IoU >= 0.20 -- deliberately
    # permissive, because the Kalman prediction carries the burden of proof and
    # a strict IoU floor shreds tracklets in the first frames, before the
    # velocity estimate has converged.
    first_match_max_distance: float = 0.80    # IoU >= 0.20
    second_match_max_distance: float = 0.50   # IoU >= 0.50, low-score tier
    unconfirmed_match_max_distance: float = 0.70  # IoU >= 0.30
    track_buffer_frames: int = 30
    min_hits: int = 3
    frame_rate: float = 25.0
    # Mahalanobis gating on the first association. Off by default, matching
    # reference ByteTrack: for the first few frames of a tracklet the velocity
    # estimate is still zero, so the predicted box lags the true one by a full
    # inter-frame displacement. At 25 fps and 60 km/h that lag alone exceeds
    # chi2(0.95, 4) and the gate shreds new tracklets into fresh identities.
    # Enable only on fixed-geometry cameras, where min_hits_for_gate below
    # ensures it engages once the velocity estimate has converged.
    use_mahalanobis_gate: bool = False
    gating_chi2_thresh: float = 9.4877
    min_hits_for_gate: int = 5
    # Optional appearance veto using the OSNet embedding.
    appearance_veto_enabled: bool = False
    appearance_min_cosine: float = 0.25
    allow_greedy_assignment_fallback: bool = True


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Deterministic Indian registration grammar and optical repair."""

    max_substitutions: int = 3
    max_search_expansions: int = 2048
    # Log-probability budget the repairer may spend to reach a legal plate.
    max_repair_cost: float = 9.0
    enforce_state_code: bool = True
    # Penalty added when a repaired plate resolves to an unknown state code.
    unknown_state_penalty: float = 2.5


@dataclass(frozen=True, slots=True)
class PackagingConfig:
    """Edge -> gateway payload contract."""

    schema_version: str = "1.0"
    max_payload_bytes: int = 5120
    quantize_embedding: bool = True
    include_char_confidence: bool = True
    include_debug_timings: bool = True
    # Optional HMAC device key for payload integrity (hex). Pseudonymisation of
    # the plate itself happens centrally in Stage 2, per the DPDP design.
    device_hmac_key_hex: Optional[str] = None


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Root configuration."""

    camera_id: str = "CAM-UNSET"
    node_id: str = "EDGE-UNSET"
    latency_budget_ms: float = 15.0
    # Emit a pass once the tracklet has this many scored plate observations,
    # even if the vehicle has not yet left the frame.
    early_emit_observations: int = 8
    # Minimum observations required to emit at track termination.
    min_emit_observations: int = 1
    plate_detect_every_n_frames: int = 1
    log_level: str = "INFO"

    backend: BackendConfig = field(default_factory=BackendConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    vehicle_detector: VehicleDetectorConfig = field(default_factory=VehicleDetectorConfig)
    plate_detector: PlateDetectorConfig = field(default_factory=PlateDetectorConfig)
    rectify: RectifyConfig = field(default_factory=RectifyConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    reid: ReidConfig = field(default_factory=ReidConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    packaging: PackagingConfig = field(default_factory=PackagingConfig)


def _coerce(target_type: Any, value: Any) -> Any:
    """Recursively coerce JSON primitives into the declared dataclass types."""
    if is_dataclass(target_type) and isinstance(value, Mapping):
        return _build(target_type, value)
    origin = getattr(target_type, "__origin__", None)
    if origin is tuple and isinstance(value, list):
        args = getattr(target_type, "__args__", ())
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v) for v in value)
        return tuple(_coerce(a, v) for a, v in zip(args, value))
    if target_type is Path or target_type == Optional[Path]:
        return Path(value) if value is not None else None
    return value


def _build(cls: Any, payload: Mapping[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ValueError(f"Unknown configuration keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, raw in payload.items():
        declared = known[name].type
        if isinstance(declared, str):  # PEP 563 stringified annotation
            declared = _RESOLVED_TYPES.get((cls.__name__, name), declared)
        kwargs[name] = _coerce(declared, raw)
    return cls(**kwargs)


# Explicit resolution table for the nested dataclass fields, so that
# ``from __future__ import annotations`` does not defeat the loader.
_RESOLVED_TYPES: dict[tuple[str, str], Any] = {
    ("PipelineConfig", "backend"): BackendConfig,
    ("PipelineConfig", "stream"): StreamConfig,
    ("PipelineConfig", "gate"): GateConfig,
    ("PipelineConfig", "vehicle_detector"): VehicleDetectorConfig,
    ("PipelineConfig", "plate_detector"): PlateDetectorConfig,
    ("PipelineConfig", "rectify"): RectifyConfig,
    ("PipelineConfig", "ocr"): OcrConfig,
    ("PipelineConfig", "reid"): ReidConfig,
    ("PipelineConfig", "tracker"): TrackerConfig,
    ("PipelineConfig", "validation"): ValidationConfig,
    ("PipelineConfig", "packaging"): PackagingConfig,
    ("RectifyConfig", "intrinsics"): CameraIntrinsics,
    ("VehicleDetectorConfig", "model_path"): Path,
    ("PlateDetectorConfig", "model_path"): Path,
    ("OcrConfig", "model_path"): Path,
    ("ReidConfig", "model_path"): Path,
}


def load_config(path: str | Path) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a JSON document.

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the document contains keys not present on the schema.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a JSON object, got {type(payload).__name__}")
    return _build(PipelineConfig, payload)
