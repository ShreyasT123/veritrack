"""Pre-inference gates.

Two cheap CPU filters stand in front of the NPU:

1. **Motion gate** - an MOG2 background model on a quarter-resolution frame.
   An empty carriageway costs ~0.3 ms instead of a full detector pass, which
   is what makes a <15 ms *per vehicle pass* budget achievable on an RK3588
   that is simultaneously decoding 1080p25.

2. **Focus gate** - variance of the Laplacian.

   .. math::
      \\nabla^2 I = \\frac{\\partial^2 I}{\\partial x^2}
                  + \\frac{\\partial^2 I}{\\partial y^2},
      \\qquad
      F = \\operatorname{Var}\\bigl(\\nabla^2 I\\bigr)

   The discrete kernel is the 4-neighbour form
   ``[[0, 1, 0], [1, -4, 1], [0, 1, 0]]``. Because the Laplacian is a
   zero-mean high-pass operator, its variance is dominated by edge energy;
   a blurred or defocused plate collapses toward zero. Frames scoring
   ``F < 65`` are discarded per the acceptance criteria.

   Note that ``F`` scales with contrast, so it is computed on the *rectified*
   plate strip rather than the raw ROI wherever possible - that removes the
   dependence on plate size in the source image, which would otherwise make a
   single global threshold meaningless across near and far lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import GateConfig

_LAPLACIAN_KERNEL = np.array(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
)


def variance_of_laplacian(image: np.ndarray) -> float:
    """Return ``Var(∇²I)`` for a BGR or grayscale image.

    Raises:
        ValueError: the image is empty or has an unsupported rank.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot compute focus score on an empty image")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"Expected 2D or 3D image, got rank {image.ndim}")
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = cv2.filter2D(gray.astype(np.float32), ddepth=cv2.CV_32F, kernel=_LAPLACIAN_KERNEL)
    return float(lap.var())


def normalized_focus(score: float, reference: float) -> float:
    """Map a raw Laplacian variance onto ``[0, 1]`` for fusion weighting.

    Uses a saturating square-root response: focus improves sharply up to the
    reference variance then plateaus, which matches the empirical relationship
    between ``Var(∇²I)`` and character error rate far better than a linear map.
    """
    if reference <= 0.0:
        return 0.0
    return float(np.clip(np.sqrt(max(score, 0.0) / reference), 0.0, 1.0))


@dataclass(slots=True)
class GateDecision:
    """Outcome of the per-frame gates."""

    accepted: bool
    reason: str
    focus_score: float
    foreground_ratio: float
    motion_mask: Optional[np.ndarray] = None


class FocusGate:
    """Stateless Var(Laplacian) threshold."""

    __slots__ = ("_min_variance", "_reference")

    def __init__(self, config: GateConfig) -> None:
        self._min_variance = float(config.laplacian_min_variance)
        self._reference = float(config.laplacian_reference_variance)

    @property
    def threshold(self) -> float:
        return self._min_variance

    def score(self, image: np.ndarray) -> float:
        return variance_of_laplacian(image)

    def normalized(self, score: float) -> float:
        return normalized_focus(score, self._reference)

    def accepts(self, image: np.ndarray) -> Tuple[bool, float]:
        """Return ``(passed, raw_variance)``."""
        value = self.score(image)
        return value >= self._min_variance, value


class MotionGate:
    """MOG2 foreground gate operating on a decimated frame."""

    __slots__ = ("_config", "_subtractor", "_kernel", "_frames_seen")

    def __init__(self, config: GateConfig) -> None:
        self._config = config
        self._frames_seen = 0
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=int(config.motion_history),
            varThreshold=float(config.motion_var_threshold),
            detectShadows=bool(config.motion_detect_shadows),
        )
        k = max(1, int(config.motion_open_kernel))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    @property
    def warmed_up(self) -> bool:
        return self._frames_seen >= self._config.motion_warmup_frames

    def update(self, frame: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """Advance the background model and report activity.

        Returns ``(has_motion, foreground_ratio, mask)``. During warm-up the
        gate always reports motion so that the model is not starved of the
        very vehicles it needs to learn to ignore.
        """
        scale = float(self._config.motion_downscale)
        if scale <= 0.0 or scale > 1.0:
            raise ValueError(f"motion_downscale must be in (0, 1], got {scale}")
        small = (
            cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else frame
        )
        mask = self._subtractor.apply(small)
        # MOG2 marks shadows as 127 when detectShadows is on; treat only hard
        # foreground (255) as motion.
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        self._frames_seen += 1
        if not self.warmed_up:
            return True, ratio, mask
        return ratio >= self._config.motion_min_fg_ratio, ratio, mask


class FrameGate:
    """Composite gate applied to every decoded frame."""

    __slots__ = ("_config", "_motion", "_focus")

    def __init__(self, config: GateConfig) -> None:
        self._config = config
        self._focus = FocusGate(config)
        self._motion = MotionGate(config) if config.motion_enabled else None

    @property
    def focus(self) -> FocusGate:
        return self._focus

    def evaluate(self, frame: np.ndarray) -> GateDecision:
        """Decide whether ``frame`` is worth spending NPU cycles on.

        The focus gate is applied at frame level only as a coarse reject for
        globally destroyed frames (a 4x-relaxed threshold); the strict
        ``Var(∇²I) < 65`` rule is enforced per rectified plate strip inside the
        pipeline, where it is scale-invariant and therefore meaningful.
        """
        if frame is None or frame.size == 0:
            return GateDecision(False, "empty_frame", 0.0, 0.0)

        ratio = 1.0
        mask: Optional[np.ndarray] = None
        if self._motion is not None:
            has_motion, ratio, mask = self._motion.update(frame)
            if not has_motion:
                return GateDecision(False, "no_motion", 0.0, ratio, mask)

        focus_score = variance_of_laplacian(frame)
        if focus_score < self._config.laplacian_min_variance * 0.25:
            return GateDecision(False, "frame_defocused", focus_score, ratio, mask)

        return GateDecision(True, "accepted", focus_score, ratio, mask)
