"""OSNet vehicle re-identification embeddings.

The embedding is the system's answer to a plate that is missing, obscured by a
tow bar, deliberately smeared with mud, or swapped. Stage 3 compares a 128-D
L2-normalised descriptor across cameras; Stage 1's only job is to produce a
*stable* descriptor per tracklet.

Stability comes from temporal smoothing. A single frame's embedding is noisy
under headlight glare or partial occlusion, so the tracklet descriptor is an
exponential moving average, re-normalised after each update:

.. math::
   \\mathbf{e}_t = \\frac{\\alpha\\,\\mathbf{e}_{t-1}
                        + (1 - \\alpha)\\,\\mathbf{f}_t}
                       {\\lVert \\alpha\\,\\mathbf{e}_{t-1}
                        + (1 - \\alpha)\\,\\mathbf{f}_t \\rVert_2}

Renormalising every step keeps the descriptor on the unit hypersphere, so
cosine similarity reduces to a dot product and the >140 km/h kinematic check
and the 0.35 cosine divergence check in Stage 3 stay directly comparable.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import BackendConfig, ReidConfig
from .errors import ModelContractError
from .runtime import InferenceBackend, load_backend
from .types import BBox

logger = logging.getLogger(__name__)


def l2_normalize(vector: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Project a vector onto the unit hypersphere."""
    norm = float(np.linalg.norm(vector))
    if norm < epsilon:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; exact dot product for pre-normalised inputs."""
    if a.shape != b.shape:
        raise ValueError(f"Embedding shape mismatch: {a.shape} vs {b.shape}")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def quantize_embedding(embedding: np.ndarray) -> str:
    """Symmetric int8 quantisation of a unit vector, base64-encoded.

    A 128-D float32 descriptor is 1.4 KB as JSON, which alone would consume a
    quarter of the 5 KB per-pass budget. Since the vector is L2-normalised its
    components lie in ``[-1, 1]``, so ``q = round(127 x)`` is exact to 1/254.
    Empirically this perturbs pairwise cosine similarity by under 0.002 - two
    orders of magnitude below the 0.35 divergence threshold - for 172 bytes.
    """
    clipped = np.clip(embedding.astype(np.float32), -1.0, 1.0)
    quantized = np.round(clipped * 127.0).astype(np.int8)
    return base64.b64encode(quantized.tobytes()).decode("ascii")


def dequantize_embedding(encoded: str, dim: int = 128) -> np.ndarray:
    """Inverse of :func:`quantize_embedding`; re-normalised after decode."""
    raw = np.frombuffer(base64.b64decode(encoded), dtype=np.int8)
    if raw.size != dim:
        raise ValueError(f"Expected {dim} bytes, decoded {raw.size}")
    return l2_normalize(raw.astype(np.float32) / 127.0)


class EmbeddingAccumulator:
    """Per-tracklet EMA over frame embeddings."""

    __slots__ = ("_alpha", "_state", "_count")

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {alpha}")
        self._alpha = float(alpha)
        self._state: Optional[np.ndarray] = None
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def value(self) -> Optional[np.ndarray]:
        return None if self._state is None else self._state.copy()

    def update(self, embedding: np.ndarray) -> np.ndarray:
        """Fold a new frame embedding into the running descriptor."""
        normalised = l2_normalize(embedding)
        if self._state is None:
            self._state = normalised
        else:
            blended = self._alpha * self._state + (1.0 - self._alpha) * normalised
            self._state = l2_normalize(blended)
        self._count += 1
        return self._state.copy()


class OsNetExtractor:
    """OSNet appearance feature extractor."""

    __slots__ = ("_config", "_backend", "_mean", "_std")

    def __init__(
        self,
        config: ReidConfig,
        backend_config: BackendConfig,
        backend: Optional[InferenceBackend] = None,
    ) -> None:
        self._config = config
        self._backend = backend if backend is not None else load_backend(config.model_path, backend_config)
        self._mean = np.asarray(config.imagenet_mean, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.asarray(config.imagenet_std, dtype=np.float32).reshape(3, 1, 1)

    @property
    def backend(self) -> InferenceBackend:
        return self._backend

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Vehicle crop -> ``(1, 3, H, W)`` ImageNet-normalised tensor."""
        width, height = self._config.input_size
        if crop is None or crop.size == 0:
            raise ValueError("Cannot embed an empty crop")
        interp = cv2.INTER_AREA if crop.shape[1] > width else cv2.INTER_LINEAR
        resized = cv2.resize(crop, (width, height), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = rgb.transpose(2, 0, 1)
        return np.ascontiguousarray(((chw - self._mean) / self._std)[None, ...])

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Return the ``(128,)`` L2-normalised descriptor for one crop."""
        raw = self._backend.run(self.preprocess(crop))[0]
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        if arr.size != self._config.embedding_dim:
            raise ModelContractError(
                f"Re-ID model emits {arr.size} dimensions, expected {self._config.embedding_dim}"
            )
        return l2_normalize(arr)

    def embed_box(self, frame: np.ndarray, box: BBox) -> np.ndarray:
        """Crop ``box`` from ``frame`` and embed it."""
        height, width = frame.shape[:2]
        clipped = box.clip(width, height)
        x1, y1 = int(clipped.x1), int(clipped.y1)
        x2, y2 = int(np.ceil(clipped.x2)), int(np.ceil(clipped.y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            raise ValueError(f"Vehicle crop too small to embed: {x2 - x1}x{y2 - y1}")
        return self.embed(frame[y1:y2, x1:x2])

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Embed several crops in one graph execution where the backend allows it."""
        if not crops:
            return np.empty((0, self._config.embedding_dim), dtype=np.float32)
        batch = np.concatenate([self.preprocess(c) for c in crops], axis=0)
        raw = self._backend.run(batch)[0]
        arr = np.asarray(raw, dtype=np.float32).reshape(len(crops), -1)
        if arr.shape[1] != self._config.embedding_dim:
            raise ModelContractError(
                f"Re-ID model emits {arr.shape[1]} dimensions, "
                f"expected {self._config.embedding_dim}"
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return (arr / np.maximum(norms, 1e-12)).astype(np.float32)
