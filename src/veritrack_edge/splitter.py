"""Single-line vs two-line plate handling.

Two-wheelers, auto-rickshaws and most commercial goods carriers in India use
the 285x200 mm two-line format, where the registration is split as
``[state][district]`` over ``[series][number]``. A recogniser trained on a
single horizontal strip reads such a plate as garbage.

Rather than maintaining two recognisers, the two rows are *re-flowed into one
strip*: line 1 is placed left, line 2 right, each resized to half the canonical
width. Because CTC is translation-equivariant along the width axis and the
class alphabet is shared, one 48xW recogniser handles both layouts, and the
resulting character order matches the legal registration order exactly.

The split row is found by a projection profile rather than a fixed midpoint.
Let :math:`B(r, c)` be the binarised ink mask. The row ink profile is

.. math::
   p(r) = \\frac{1}{W}\\sum_{c=0}^{W-1} B(r, c)

smoothed with a box filter. Within the central band the split row is

.. math::
   r^{*} = \\arg\\min_{r \\in [\\alpha H,\\, \\beta H]} \\tilde{p}(r)

and the split is accepted only when the valley is deep relative to the two
line peaks - the *valley contrast* - otherwise a fixed midpoint is used. That
guard matters for damaged or mud-caked plates where the profile is flat.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

from .config import RectifyConfig
from .errors import GeometryError
from .types import PlateLayout, RectifiedPlate

logger = logging.getLogger(__name__)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binarise to an ink mask where 1 = character stroke.

    Otsu is applied to the contrast-equalised image; the polarity is then
    resolved by assuming characters occupy the minority of the plate area,
    which holds for every Indian format including white-on-black rental plates.
    """
    equalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    _, binary = cv2.threshold(equalised, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if binary.mean() > 0.5:
        binary = 1 - binary
    return binary.astype(np.float32)


def row_ink_profile(gray: np.ndarray, smooth: int = 3) -> np.ndarray:
    """Normalised, smoothed per-row ink density."""
    mask = _ink_mask(gray)
    profile = mask.mean(axis=1)
    if smooth > 1 and profile.size >= smooth:
        kernel = np.ones(smooth, dtype=np.float32) / float(smooth)
        profile = np.convolve(profile, kernel, mode="same")
    return profile.astype(np.float32)


def find_split_row(gray: np.ndarray, band: Tuple[float, float], min_contrast: float) -> Tuple[int, float]:
    """Locate the inter-line gap.

    Returns:
        ``(row_index, valley_contrast)`` where contrast is
        ``1 - valley / mean(peak_above, peak_below)`` in ``[0, 1]``.
    """
    height = gray.shape[0]
    if height < 8:
        raise GeometryError(f"Canvas too short to split: {height}px")

    profile = row_ink_profile(gray)
    lo = max(1, int(round(band[0] * height)))
    hi = min(height - 2, int(round(band[1] * height)))
    if hi <= lo:
        lo, hi = height // 2 - 1, height // 2 + 1

    window = profile[lo : hi + 1]
    valley_index = lo + int(np.argmin(window))
    valley = float(profile[valley_index])

    peak_above = float(profile[:valley_index].max()) if valley_index > 0 else 0.0
    peak_below = float(profile[valley_index + 1 :].max()) if valley_index + 1 < height else 0.0
    peak = 0.5 * (peak_above + peak_below)
    contrast = 0.0 if peak <= 1e-6 else float(np.clip(1.0 - valley / peak, 0.0, 1.0))

    if contrast < min_contrast:
        logger.debug("weak valley contrast %.2f; using geometric midpoint", contrast)
        return height // 2, contrast
    return valley_index, contrast


class PlateLineSplitter:
    """Converts any rectified plate into the canonical single recognition strip."""

    __slots__ = ("_config", "_target")

    def __init__(self, config: RectifyConfig, target_size: Tuple[int, int]) -> None:
        """
        Args:
            config: geometry configuration (split band, contrast floor).
            target_size: the recogniser input ``(width, height)``.
        """
        self._config = config
        self._target = (int(target_size[0]), int(target_size[1]))

    @property
    def target_size(self) -> Tuple[int, int]:
        return self._target

    def _resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        interp = cv2.INTER_AREA if image.shape[1] > size[0] else cv2.INTER_CUBIC
        return cv2.resize(image, size, interpolation=interp)

    def to_strip(self, plate: RectifiedPlate) -> np.ndarray:
        """Return the canonical ``(H, W, 3)`` uint8 recognition strip.

        Raises:
            GeometryError: the rectified canvas is unusable.
        """
        image = plate.image
        if image is None or image.size == 0:
            raise GeometryError("Rectified plate is empty")
        width, height = self._target

        if plate.layout is PlateLayout.SINGLE_LINE:
            if (image.shape[1], image.shape[0]) == (width, height):
                return image
            return self._resize(image, (width, height))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        split_row, contrast = find_split_row(
            gray, self._config.split_band, self._config.min_valley_contrast
        )
        logger.debug("two-line split at row %d (contrast %.2f)", split_row, contrast)

        top = image[:split_row]
        bottom = image[split_row:]
        if top.shape[0] < 4 or bottom.shape[0] < 4:
            raise GeometryError(
                f"Two-line split produced a degenerate half ({top.shape[0]}/{bottom.shape[0]}px)"
            )

        half_width = width // 2
        left = self._resize(top, (half_width, height))
        right = self._resize(bottom, (width - half_width, height))
        return np.ascontiguousarray(np.hstack([left, right]))
