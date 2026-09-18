"""VeriTrack Stage 1 - edge vision and feature extraction.

Public surface::

    from veritrack_edge import EdgePipeline, PipelineConfig, VideoStream, load_config

    config = load_config("config/cam_thane_07.json")
    pipeline = EdgePipeline(config)
    with VideoStream(config.stream) as stream:
        for frame in stream.frames():
            for vehicle_pass in pipeline.process(frame):
                producer.send(pipeline.payload_builder.serialize(vehicle_pass))
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import (
    BackendConfig,
    CameraIntrinsics,
    GateConfig,
    OcrConfig,
    PackagingConfig,
    PipelineConfig,
    PlateDetectorConfig,
    RectifyConfig,
    ReidConfig,
    StreamConfig,
    TrackerConfig,
    ValidationConfig,
    VehicleDetectorConfig,
    load_config,
)
from .detection import PlateKeypointDetector, VehicleDetector, letterbox, nms
from .errors import (
    BackendError,
    FatalEdgeError,
    GeometryError,
    ModelContractError,
    PayloadTooLargeError,
    RecognitionError,
    RecoverableEdgeError,
    StreamError,
    VeriTrackError,
)
from .gating import FrameGate, normalized_focus, variance_of_laplacian
from .ingest import Frame, VideoStream
from .ocr import (
    CtcRecognizer,
    ctc_forced_alignment,
    ctc_greedy_decode,
    ctc_prefix_beam_search,
    fuse_log_probs,
    log_softmax,
    sequence_entropy,
)
from .packaging import EdgePayloadBuilder
from .pipeline import EdgePipeline, LatencyBudget, make_pass_id
from .rectify import (
    PlateRectifier,
    classify_plate_series,
    decompose_plate_pose,
    order_quad_corners,
    solve_homography_dlt,
)
from .reid import (
    EmbeddingAccumulator,
    OsNetExtractor,
    cosine_similarity,
    dequantize_embedding,
    l2_normalize,
    quantize_embedding,
)
from .splitter import PlateLineSplitter, find_split_row, row_ink_profile
from .tracking import ByteTracker, KalmanFilterXYAH, STrack, iou_distance
from .types import (
    BBox,
    CharPosterior,
    Detection,
    PlateHypothesis,
    PlateLayout,
    PlateObservation,
    PlatePose,
    PlateQuad,
    PlateReading,
    PlateSeries,
    RectifiedPlate,
    StageTimings,
    TrackletSummary,
    TrackState,
    VehiclePass,
)
from .validation import (
    CONFUSION_MAP,
    STATE_CODES,
    TEMPLATES,
    PlateTemplate,
    PlateValidator,
    ValidationResult,
    match_template,
    normalise_text,
)

__all__ = [
    "__version__",
    # config
    "BackendConfig",
    "CameraIntrinsics",
    "GateConfig",
    "OcrConfig",
    "PackagingConfig",
    "PipelineConfig",
    "PlateDetectorConfig",
    "RectifyConfig",
    "ReidConfig",
    "StreamConfig",
    "TrackerConfig",
    "ValidationConfig",
    "VehicleDetectorConfig",
    "load_config",
    # errors
    "BackendError",
    "FatalEdgeError",
    "GeometryError",
    "ModelContractError",
    "PayloadTooLargeError",
    "RecognitionError",
    "RecoverableEdgeError",
    "StreamError",
    "VeriTrackError",
    # ingestion and gating
    "Frame",
    "FrameGate",
    "VideoStream",
    "normalized_focus",
    "variance_of_laplacian",
    # detection and tracking
    "ByteTracker",
    "KalmanFilterXYAH",
    "PlateKeypointDetector",
    "STrack",
    "VehicleDetector",
    "iou_distance",
    "letterbox",
    "nms",
    # geometry
    "PlateLineSplitter",
    "PlateRectifier",
    "classify_plate_series",
    "decompose_plate_pose",
    "find_split_row",
    "order_quad_corners",
    "row_ink_profile",
    "solve_homography_dlt",
    # recognition
    "CtcRecognizer",
    "ctc_forced_alignment",
    "ctc_greedy_decode",
    "ctc_prefix_beam_search",
    "fuse_log_probs",
    "log_softmax",
    "sequence_entropy",
    # re-id
    "EmbeddingAccumulator",
    "OsNetExtractor",
    "cosine_similarity",
    "dequantize_embedding",
    "l2_normalize",
    "quantize_embedding",
    # validation
    "CONFUSION_MAP",
    "PlateTemplate",
    "PlateValidator",
    "STATE_CODES",
    "TEMPLATES",
    "ValidationResult",
    "match_template",
    "normalise_text",
    # pipeline
    "EdgePayloadBuilder",
    "EdgePipeline",
    "LatencyBudget",
    "make_pass_id",
    # types
    "BBox",
    "CharPosterior",
    "Detection",
    "PlateHypothesis",
    "PlateLayout",
    "PlateObservation",
    "PlatePose",
    "PlateQuad",
    "PlateReading",
    "PlateSeries",
    "RectifiedPlate",
    "StageTimings",
    "TrackState",
    "TrackletSummary",
    "VehiclePass",
]
