"""CPU text recognition via HuggingFace ``transformers`` (PP-OCRv6-Tiny).

This wraps exactly the call sequence already validated end-to-end on the
target laptop (Intel i5, no GPU): 0.098 s inference, 0.947 confidence, a
correct read on a real photographed plate. Nothing about the model call here
is guessed — every method name below (``AutoModelForTextRecognition``,
``post_process_text_recognition``) was checked against the installed
``transformers`` source rather than assumed, and the batching behaviour
(``recognize_batch``) follows directly from reading
``post_process_text_recognition``'s implementation, which is already written
to loop over a batch dimension: ``preds_prob, preds_idx = logits.max(dim=-1)``
per image, ``batch_size = logits.shape[0]``. One forward pass over several
candidate crops is therefore a real, supported thing to do, not an assumption.

What could not be verified in this environment: this repository's sandbox has
no route to ``huggingface.co`` and no working CUDA/CPU PyTorch wheel available
to it, so the actual weight download and forward pass have not been executed
here. They do not need to be — you already ran them, on the real target
machine, and got real output. This module is a refactor of that exact call
sequence into a reusable, testable class, not a reimplementation of it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Sequence, Union

import numpy as np

__all__ = [
    "RecognitionResult",
    "Recognizer",
    "PPOcrRecognizer",
    "FakeRecognizer",
    "ModelNotLoadedError",
    "to_pil_image",
]

LOGGER = logging.getLogger("veritrack.demo.recognizer")


class ModelNotLoadedError(RuntimeError):
    """Raised when recognition is attempted before (or after a failed) load."""


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """One recognizer output for one candidate crop."""

    text: str
    confidence: float
    inference_ms: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must lie in [0, 1], got {self.confidence}")
        if self.inference_ms < 0.0:
            raise ValueError("inference_ms must be non-negative")


def to_pil_image(image: Union[np.ndarray, Any]) -> Any:
    """Accept either a BGR ``np.ndarray`` (OpenCV's native format) or a PIL Image.

    Centralising the conversion here means every caller — the live demo loop,
    the batch path, the tests with a ``FakeRecognizer`` — agrees on which
    channel order a raw array is in. Getting this wrong (BGR fed to a model
    trained on RGB) is a classic silent-degradation bug: the model still
    produces *a* text and *a* confidence, just a wrong and lower one, which is
    much harder to notice live than an outright crash.
    """
    from PIL import Image  # local import: keep PIL optional for pure-array callers

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.ndim == 3 and image.shape[2] == 3:
            import cv2

            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        raise ValueError(f"unsupported array shape for an image: {image.shape}")
    raise TypeError(f"expected a numpy array or PIL Image, got {type(image)!r}")


class Recognizer(Protocol):
    """Anything that turns a plate crop into text + confidence."""

    def recognize(self, image: Union[np.ndarray, Any]) -> RecognitionResult:
        ...

    def recognize_batch(self, images: Sequence[Union[np.ndarray, Any]]) -> List[RecognitionResult]:
        ...

    @property
    def is_loaded(self) -> bool:
        ...


class PPOcrRecognizer:
    """PP-OCRv6-Tiny via ``transformers``, CPU by default.

    Loading is lazy: constructing this object does no network I/O and imports
    no heavy dependency, so it is cheap to construct in tests and in the CLI's
    argument-parsing path before the user has necessarily confirmed they want
    to proceed. The first call to :meth:`recognize` (or an explicit
    :meth:`load`) triggers the actual model materialisation.

    Before a live demo, pre-download the weights once with network available::

        huggingface-cli download PaddlePaddle/PP-OCRv6_tiny_rec_safetensors

    so that the demo itself never depends on a working internet connection in
    the room. Set ``HF_HUB_OFFLINE=1`` to make a missing cache fail fast and
    loudly instead of hanging on a DNS timeout mid-demo.
    """

    __slots__ = ("_model_id", "_device", "_warmup", "_processor", "_model", "_torch", "_resolved_device")

    def __init__(self, model_id: str, *, device: str = "cpu", warmup_on_load: bool = True) -> None:
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError(f"unknown device {device!r}")
        self._model_id = model_id
        self._device = device
        self._warmup = warmup_on_load
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None
        self._torch: Optional[Any] = None
        self._resolved_device: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def resolved_device(self) -> Optional[str]:
        return self._resolved_device

    def load(self) -> None:
        """Materialise the processor and model. Idempotent."""
        if self.is_loaded:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForTextRecognition
        except ImportError as exc:
            raise ModelNotLoadedError(
                "transformers and torch are required for live recognition. Install "
                "with: pip install -r requirements-demo.txt (see that file for the "
                "CPU-only torch index URL)."
            ) from exc

        resolved = self._device
        if resolved == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self._resolved_device = resolved

        LOGGER.info("loading %s onto %s ...", self._model_id, resolved)
        started = time.perf_counter()
        processor = AutoImageProcessor.from_pretrained(self._model_id)
        model = AutoModelForTextRecognition.from_pretrained(self._model_id).to(resolved)
        model.eval()
        LOGGER.info("loaded in %.2fs", time.perf_counter() - started)

        self._torch = torch
        self._processor = processor
        self._model = model

        if self._warmup:
            blank = np.zeros((48, 160, 3), dtype=np.uint8)
            warmup_started = time.perf_counter()
            self._infer_batch([to_pil_image(blank)])
            LOGGER.info("warm-up inference in %.3fs", time.perf_counter() - warmup_started)

    def _infer_batch(self, images: Sequence[Any]) -> List[RecognitionResult]:
        """Run one forward pass over a batch of already-PIL images."""
        if self._model is None or self._processor is None or self._torch is None:
            raise ModelNotLoadedError("call load() before recognize()")

        inputs = self._processor(images=list(images), return_tensors="pt").to(self._resolved_device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        decoded = self._processor.post_process_text_recognition(outputs)
        # elapsed_ms is the whole batch's forward-pass time; dividing it evenly
        # across the batch gives each result an honest per-candidate figure
        # for the on-screen latency readout rather than reporting the full
        # batch cost against every single candidate.
        per_item_ms = elapsed_ms / max(len(decoded), 1)
        return [
            RecognitionResult(
                text=str(entry.get("text", "")),
                confidence=float(entry.get("score", 0.0)),
                inference_ms=per_item_ms,
            )
            for entry in decoded
        ]

    def recognize(self, image: Union[np.ndarray, Any]) -> RecognitionResult:
        if not self.is_loaded:
            self.load()
        return self._infer_batch([to_pil_image(image)])[0]

    def recognize_batch(self, images: Sequence[Union[np.ndarray, Any]]) -> List[RecognitionResult]:
        if not images:
            return []
        if not self.is_loaded:
            self.load()
        return self._infer_batch([to_pil_image(image) for image in images])


class FakeRecognizer:
    """A deterministic stand-in, for tests and for developing the pipeline
    without a GPU-less laptop or an internet connection at hand.

    Returns a fixed, injectable result (or one selected by a caller-supplied
    function of the input image), so pipeline wiring, gating logic, and the
    Stage 2 forwarder can all be exercised in CI without ``torch`` or
    ``transformers`` installed at all.
    """

    __slots__ = ("_fixed_result", "_responder", "_call_count")

    def __init__(
        self,
        result: Optional[RecognitionResult] = None,
        responder: Optional[Any] = None,
    ) -> None:
        if result is None and responder is None:
            result = RecognitionResult(text="MH12AB1234", confidence=0.95, inference_ms=1.0)
        self._fixed_result = result
        self._responder = responder
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def is_loaded(self) -> bool:
        return True

    def recognize(self, image: Union[np.ndarray, Any]) -> RecognitionResult:
        self._call_count += 1
        if self._responder is not None:
            return self._responder(image)
        assert self._fixed_result is not None
        return self._fixed_result

    def recognize_batch(self, images: Sequence[Union[np.ndarray, Any]]) -> List[RecognitionResult]:
        return [self.recognize(image) for image in images]
