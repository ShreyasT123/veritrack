"""Inference backend abstraction.

The same model graph is deployed three ways during the programme:

* development / CI  -> ONNXRuntime CPU
* Jetson Orin Nano  -> ONNXRuntime with the TensorRT or CUDA EP
* RK3588 field node -> ``rknn-toolkit-lite2`` INT8 on the 3-core NPU

Every consumer (detector, recogniser, Re-ID) talks only to
:class:`InferenceBackend`, so swapping silicon is a config change. The backend
validates the declared tensor contract at load time and raises
:class:`ModelContractError` immediately rather than producing silently wrong
detections in the field.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import BackendConfig
from .errors import BackendError, ModelContractError

logger = logging.getLogger(__name__)


class InferenceBackend(ABC):
    """Uniform synchronous inference interface."""

    def __init__(self, model_path: Path, config: BackendConfig) -> None:
        self.model_path = Path(model_path)
        self.config = config
        if not self.model_path.exists():
            raise BackendError(f"Model not found: {self.model_path}")
        self._infer_count = 0
        self._infer_seconds = 0.0

    @abstractmethod
    def _run(self, inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Execute the graph. Implemented per runtime."""

    @property
    @abstractmethod
    def input_shapes(self) -> Tuple[Tuple[Optional[int], ...], ...]:
        """Declared input shapes, ``None`` for dynamic axes."""

    def run(self, *inputs: np.ndarray) -> List[np.ndarray]:
        """Run the graph, timing the call.

        Raises:
            BackendError: the runtime raised, or returned no outputs.
        """
        if not inputs:
            raise BackendError("At least one input tensor is required")
        started = time.perf_counter()
        try:
            outputs = self._run(inputs)
        except BackendError:
            raise
        except Exception as exc:  # pragma: no cover - runtime specific
            raise BackendError(f"Inference failed for {self.model_path.name}: {exc}") from exc
        if not outputs:
            raise BackendError(f"{self.model_path.name} produced no outputs")
        self._infer_count += 1
        self._infer_seconds += time.perf_counter() - started
        return outputs

    def warmup(self, sample: np.ndarray) -> None:
        """Run a few throwaway passes so the first real frame is not penalised."""
        for _ in range(max(0, self.config.warmup_iterations)):
            try:
                self.run(sample)
            except BackendError:  # pragma: no cover - warmup is best-effort
                logger.warning("warmup pass failed for %s", self.model_path.name)
                return

    @property
    def mean_latency_ms(self) -> float:
        if self._infer_count == 0:
            return 0.0
        return 1000.0 * self._infer_seconds / self._infer_count

    def close(self) -> None:  # pragma: no cover - overridden where needed
        """Release runtime resources."""


class OnnxRuntimeBackend(InferenceBackend):
    """ONNXRuntime execution provider backend."""

    def __init__(self, model_path: Path, config: BackendConfig) -> None:
        super().__init__(model_path, config)
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - deployment dependent
            raise BackendError("onnxruntime is not installed") from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, config.intra_op_threads)
        options.inter_op_num_threads = max(1, config.inter_op_threads)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        available = set(ort.get_available_providers())
        providers = [p for p in config.providers if p in available]
        if not providers:
            fallback = "CPUExecutionProvider"
            if fallback not in available:  # pragma: no cover
                raise BackendError(f"No usable execution provider among {sorted(available)}")
            logger.warning(
                "requested providers %s unavailable; falling back to CPU", list(config.providers)
            )
            providers = [fallback]

        try:
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=providers
            )
        except Exception as exc:
            raise BackendError(f"Failed to load {self.model_path}: {exc}") from exc

        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._input_shapes = tuple(
            tuple(d if isinstance(d, int) else None for d in i.shape)
            for i in self._session.get_inputs()
        )
        logger.info(
            "loaded %s via %s (inputs=%s)",
            self.model_path.name,
            self._session.get_providers()[0],
            self._input_names,
        )

    @property
    def input_shapes(self) -> Tuple[Tuple[Optional[int], ...], ...]:
        return self._input_shapes

    def _run(self, inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
        if len(inputs) != len(self._input_names):
            raise ModelContractError(
                f"{self.model_path.name} expects {len(self._input_names)} input(s), "
                f"received {len(inputs)}"
            )
        feed: Dict[str, np.ndarray] = {
            name: np.ascontiguousarray(tensor) for name, tensor in zip(self._input_names, inputs)
        }
        return list(self._session.run(self._output_names, feed))


class RknnLiteBackend(InferenceBackend):
    """Rockchip RK3588 NPU backend via ``rknn-toolkit-lite2``.

    Expects an INT8-quantised ``.rknn`` artefact produced offline by
    ``rknn-toolkit2`` with the same preprocessing constants declared in
    :mod:`veritrack_edge.config`. NHWC uint8 input is the native layout for the
    RKNN runtime, so tensors are transposed here rather than in each consumer.
    """

    def __init__(self, model_path: Path, config: BackendConfig) -> None:
        super().__init__(model_path, config)
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:  # pragma: no cover - device only
            raise BackendError(
                "rknn-toolkit-lite2 is not installed; this backend requires an RK3588 host"
            ) from exc

        self._rknn = RKNNLite()
        if self._rknn.load_rknn(str(self.model_path)) != 0:
            raise BackendError(f"RKNNLite.load_rknn failed for {self.model_path}")
        if self._rknn.init_runtime(core_mask=config.npu_core_mask) != 0:
            raise BackendError(f"RKNNLite.init_runtime failed for {self.model_path}")
        logger.info("loaded %s on RKNN NPU (core_mask=%d)", self.model_path.name, config.npu_core_mask)

    @property
    def input_shapes(self) -> Tuple[Tuple[Optional[int], ...], ...]:
        # RKNN does not expose shapes before the first inference; the caller's
        # declared contract in config.py is authoritative on this backend.
        return ((None,),)

    @staticmethod
    def _to_nhwc(tensor: np.ndarray) -> np.ndarray:
        if tensor.ndim == 4 and tensor.shape[1] in (1, 3):
            return np.ascontiguousarray(np.transpose(tensor, (0, 2, 3, 1)))
        return np.ascontiguousarray(tensor)

    def _run(self, inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
        payload = [self._to_nhwc(t) for t in inputs]
        outputs = self._rknn.inference(inputs=payload)
        if outputs is None:
            raise BackendError(f"RKNN inference returned None for {self.model_path.name}")
        return [np.asarray(o) for o in outputs]

    def close(self) -> None:  # pragma: no cover - device only
        try:
            self._rknn.release()
        except Exception:
            logger.exception("RKNNLite.release() failed")


_BACKENDS = {
    "onnxruntime": OnnxRuntimeBackend,
    "rknnlite": RknnLiteBackend,
}


def load_backend(model_path: Path, config: BackendConfig) -> InferenceBackend:
    """Instantiate the configured backend for ``model_path``.

    Raises:
        BackendError: unknown backend kind, or the runtime failed to load.
    """
    factory = _BACKENDS.get(config.kind)
    if factory is None:
        raise BackendError(f"Unknown backend kind '{config.kind}'; expected one of {sorted(_BACKENDS)}")
    return factory(model_path, config)
