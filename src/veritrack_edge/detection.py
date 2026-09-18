"""Anchor-free detection heads.

Two models share the preprocessing and NMS machinery here:

``VehicleDetector``
    Full-frame vehicle boxes. Supports the two dominant export layouts:

    * ``yolov8``  - ``(1, 4 + nc, A)``, boxes already decoded to input-image
      pixels by the exported graph (Ultralytics default).
    * ``yolox``   - ``(1, A, 5 + nc)``, raw grid-relative predictions that must
      be decoded against the stride grid:

      .. math::
         x = (\\hat{x} + g_x)\\,s, \\quad
         y = (\\hat{y} + g_y)\\,s, \\quad
         w = e^{\\hat{w}} s, \\quad
         h = e^{\\hat{h}} s

``PlateKeypointDetector``
    Plate box plus regressed quadrilateral corners, run on the vehicle ROI so
    that a 192x192 crop carries far more plate pixels than a full 1080p frame
    would at the same cost. Output contract is documented on
    :class:`~veritrack_edge.config.PlateDetectorConfig`.

Letterboxing preserves aspect ratio; the inverse affine is applied to both
boxes and keypoints so all downstream geometry lives in full-frame pixels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import BackendConfig, PlateDetectorConfig, VehicleDetectorConfig
from .errors import ModelContractError
from .runtime import InferenceBackend, load_backend
from .types import BBox, Detection, PlateQuad

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """Affine mapping from source image space to letterboxed network input."""

    scale: float
    pad_x: float
    pad_y: float

    def invert_points(self, points: np.ndarray) -> np.ndarray:
        """Map ``(..., 2)`` network-space points back to source pixels."""
        out = points.astype(np.float32, copy=True)
        out[..., 0] = (out[..., 0] - self.pad_x) / self.scale
        out[..., 1] = (out[..., 1] - self.pad_y) / self.scale
        return out

    def invert_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Map ``(N, 4)`` xyxy network-space boxes back to source pixels."""
        out = boxes.astype(np.float32, copy=True)
        out[:, [0, 2]] = (out[:, [0, 2]] - self.pad_x) / self.scale
        out[:, [1, 3]] = (out[:, [1, 3]] - self.pad_y) / self.scale
        return out


def letterbox(
    image: np.ndarray, target: Tuple[int, int], pad_value: int = 114
) -> Tuple[np.ndarray, LetterboxTransform]:
    """Resize preserving aspect ratio and pad to ``target`` = ``(width, height)``."""
    if image is None or image.size == 0:
        raise ValueError("Cannot letterbox an empty image")
    target_w, target_h = int(target[0]), int(target[1])
    src_h, src_w = image.shape[:2]
    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    pad_x = (target_w - new_w) * 0.5
    pad_y = (target_h - new_h) * 0.5
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3
    )
    if canvas.shape[0] != target_h or canvas.shape[1] != target_w:
        canvas = cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return canvas, LetterboxTransform(scale=scale, pad_x=float(left), pad_y=float(top))


def to_nchw_float(image: np.ndarray, scale: float = 1.0 / 255.0) -> np.ndarray:
    """BGR HWC uint8 -> RGB NCHW float32 in ``[0, 1]``."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) * scale
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, limit: int) -> np.ndarray:
    """Greedy class-agnostic NMS. Returns indices ordered by descending score."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0 and len(keep) < limit:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """``(N, 4)`` centre-form -> corner-form."""
    out = np.empty_like(boxes)
    half_w = boxes[:, 2] * 0.5
    half_h = boxes[:, 3] * 0.5
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def _build_grid(input_size: Tuple[int, int], strides: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(centres[A, 2], strides[A, 1])`` for an anchor-free FPN."""
    width, height = input_size
    centres: List[np.ndarray] = []
    stride_col: List[np.ndarray] = []
    for stride in strides:
        gw, gh = width // stride, height // stride
        xs, ys = np.meshgrid(np.arange(gw, dtype=np.float32), np.arange(gh, dtype=np.float32))
        centres.append(np.stack([xs.ravel(), ys.ravel()], axis=-1))
        stride_col.append(np.full((gw * gh, 1), float(stride), dtype=np.float32))
    return np.concatenate(centres, axis=0), np.concatenate(stride_col, axis=0)


class VehicleDetector:
    """Anchor-free vehicle detector with ByteTrack-compatible dual thresholds."""

    __slots__ = ("_config", "_backend", "_grid", "_grid_strides")

    def __init__(
        self,
        config: VehicleDetectorConfig,
        backend_config: BackendConfig,
        backend: Optional[InferenceBackend] = None,
    ) -> None:
        self._config = config
        self._backend = backend if backend is not None else load_backend(config.model_path, backend_config)
        if config.layout == "yolox":
            self._grid, self._grid_strides = _build_grid(config.input_size, config.strides)
        else:
            self._grid, self._grid_strides = np.empty((0, 2), np.float32), np.empty((0, 1), np.float32)

    @property
    def backend(self) -> InferenceBackend:
        return self._backend

    def _decode(self, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(xyxy[N, 4], scores[N], class_ids[N])`` in network pixels."""
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 3:
            raise ModelContractError(f"Detector output must be rank 3, got shape {arr.shape}")
        arr = arr[0]

        if self._config.layout == "yolov8":
            # (4 + nc, A) -> (A, 4 + nc)
            if arr.shape[0] < arr.shape[1]:
                arr = arr.transpose(1, 0)
            if arr.shape[1] < 5:
                raise ModelContractError(f"yolov8 head needs >=5 channels, got {arr.shape[1]}")
            boxes = xywh_to_xyxy(arr[:, :4])
            class_scores = arr[:, 4:]
            class_ids = class_scores.argmax(axis=1)
            scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
            return boxes, scores, class_ids

        # YOLOX: (A, 5 + nc) grid-relative.
        if arr.shape[1] < 6:
            raise ModelContractError(f"yolox head needs >=6 channels, got {arr.shape[1]}")
        if arr.shape[0] != self._grid.shape[0]:
            raise ModelContractError(
                f"yolox anchor count mismatch: model={arr.shape[0]} grid={self._grid.shape[0]}"
            )
        strides = self._grid_strides
        cxcy = (arr[:, 0:2] + self._grid) * strides
        wh = np.exp(np.clip(arr[:, 2:4], -12.0, 12.0)) * strides
        boxes = xywh_to_xyxy(np.concatenate([cxcy, wh], axis=1))
        objectness = arr[:, 4:5]
        class_scores = arr[:, 5:] * objectness
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        return boxes, scores, class_ids

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        """Detect vehicles.

        Returns:
            ``(high_confidence, low_confidence)`` detections. ByteTrack consumes
            both tiers: the low tier recovers occluded vehicles that a single
            threshold would discard.
        """
        canvas, transform = letterbox(frame, self._config.input_size)
        tensor = to_nchw_float(canvas)
        raw = self._backend.run(tensor)[0]
        boxes, scores, class_ids = self._decode(raw)

        keep_classes = np.asarray(self._config.keep_class_ids, dtype=np.int64)
        mask = (scores >= self._config.conf_low) & np.isin(class_ids, keep_classes)
        if not np.any(mask):
            return [], []
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

        keep = nms(boxes, scores, self._config.nms_iou, self._config.max_detections)
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        boxes = transform.invert_boxes(boxes)

        frame_h, frame_w = frame.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, frame_w - 1.0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, frame_h - 1.0)

        high: List[Detection] = []
        low: List[Detection] = []
        names = self._config.class_names
        for box, score, class_id in zip(boxes, scores, class_ids):
            width, height = box[2] - box[0], box[3] - box[1]
            if width <= 1.0 or height <= 1.0 or width * height < self._config.min_box_area_px:
                continue
            index = int(class_id)
            name = names[index] if 0 <= index < len(names) else str(index)
            detection = Detection(BBox.from_array(box), float(score), index, name)
            (high if score >= self._config.conf_high else low).append(detection)
        return high, low


class PlateKeypointDetector:
    """Plate localiser producing an axis-aligned box and 4 corner keypoints."""

    __slots__ = ("_config", "_backend")

    def __init__(
        self,
        config: PlateDetectorConfig,
        backend_config: BackendConfig,
        backend: Optional[InferenceBackend] = None,
    ) -> None:
        self._config = config
        self._backend = backend if backend is not None else load_backend(config.model_path, backend_config)

    @property
    def backend(self) -> InferenceBackend:
        return self._backend

    def detect(self, frame: np.ndarray, roi: BBox) -> Optional[PlateQuad]:
        """Localise the highest-scoring plate inside ``roi``.

        ``roi`` is a vehicle box in full-frame coordinates; the returned quad is
        mapped back to full-frame pixels so the rectifier can sample from the
        original, un-resampled image (double resampling is the single largest
        avoidable contributor to character error on small plates).
        """
        frame_h, frame_w = frame.shape[:2]
        padded = roi.pad(self._config.roi_pad_ratio).clip(frame_w, frame_h)
        x1, y1 = int(padded.x1), int(padded.y1)
        x2, y2 = int(np.ceil(padded.x2)), int(np.ceil(padded.y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2]

        canvas, transform = letterbox(crop, self._config.input_size)
        raw = self._backend.run(to_nchw_float(canvas))[0]
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] < 13:
            raise ModelContractError(
                f"Plate head must emit (N, 13) [box, score, 4x(x, y)], got {arr.shape}"
            )

        scores = arr[:, 4]
        mask = scores >= self._config.conf_threshold
        if not np.any(mask):
            return None
        arr, scores = arr[mask], scores[mask]

        boxes = xywh_to_xyxy(arr[:, :4])
        keep = nms(boxes, scores, self._config.nms_iou, self._config.max_detections)
        if keep.size == 0:
            return None
        best = int(keep[0])

        box = transform.invert_boxes(boxes[best : best + 1])[0]
        quad = transform.invert_points(arr[best, 5:13].reshape(4, 2))
        offset = np.array([x1, y1], dtype=np.float32)
        quad += offset
        box[[0, 2]] += x1
        box[[1, 3]] += y1

        quad[:, 0] = np.clip(quad[:, 0], 0.0, frame_w - 1.0)
        quad[:, 1] = np.clip(quad[:, 1], 0.0, frame_h - 1.0)
        box[[0, 2]] = np.clip(box[[0, 2]], 0.0, frame_w - 1.0)
        box[[1, 3]] = np.clip(box[[1, 3]], 0.0, frame_h - 1.0)

        if box[2] - box[0] < 2.0 or box[3] - box[1] < 2.0:
            return None
        return PlateQuad(BBox.from_array(box), quad.astype(np.float32), float(scores[best]))
