"""VeriTrack laptop demo — live webcam ANPR on CPU-only hardware.

A separate deployment target from ``veritrack_edge`` (the RK3588/Jetson pole
node) and ``veritrack_trajectory``/``veritrack_server`` (the backend). This
package runs entirely on a Windows or WSL2 laptop with no GPU, using a webcam
in place of an RTSP camera feed and a HuggingFace-hosted PP-OCRv6-Tiny model in
place of the exported ONNX recognizer, while reusing Stage 1's geometry
(``rectify``) and grammar (``validation``) code, which is hardware-independent.

Quick start::

    pip install -r requirements-demo.txt
    huggingface-cli download PaddlePaddle/PP-OCRv6_tiny_rec_safetensors
    python -m veritrack_demo.run_webcam

See README.md ("Stage 1 laptop demo") for Windows/WSL2/Docker guidance and
keyboard controls.
"""

from __future__ import annotations

from .config import (
    CaptureConfig,
    DemoConfig,
    GatewayConfig,
    LocalizerConfig,
    RecognizerConfig,
    load_demo_config,
)
from .gateway_client import ForwardResult, GatewayForwarder, build_compact_payload
from .localizer import (
    ContourPlateLocalizer,
    FallbackLocalizer,
    HaarCascadePlateLocalizer,
    ManualRoiLocalizer,
    PlateCandidate,
    PlateLocalizer,
    build_localizer,
)
from .pipeline import DemoDetection, DemoPipeline, build_pipeline
from .recognizer import (
    FakeRecognizer,
    ModelNotLoadedError,
    PPOcrRecognizer,
    RecognitionResult,
    Recognizer,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "DemoConfig", "CaptureConfig", "LocalizerConfig", "RecognizerConfig",
    "GatewayConfig", "load_demo_config",
    "PlateCandidate", "PlateLocalizer", "ManualRoiLocalizer",
    "HaarCascadePlateLocalizer", "ContourPlateLocalizer", "FallbackLocalizer",
    "build_localizer",
    "RecognitionResult", "Recognizer", "PPOcrRecognizer", "FakeRecognizer",
    "ModelNotLoadedError",
    "DemoDetection", "DemoPipeline", "build_pipeline",
    "ForwardResult", "GatewayForwarder", "build_compact_payload",
]
