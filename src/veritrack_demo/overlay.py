"""Drawing helpers for the live demo window. Pure ``cv2``, no model logic here.

Kept separate from ``pipeline.py`` on purpose: the pipeline should be testable
(and was tested) with zero GUI dependency, and nothing about "where the text
goes on screen" belongs in the same module as "is this plate reading valid".
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .localizer import ManualRoiLocalizer, PlateLocalizer
from .pipeline import DemoDetection

__all__ = ["draw_detections", "draw_manual_roi", "draw_hud", "COLOR_VALID", "COLOR_INVALID", "COLOR_ROI"]

COLOR_VALID: Tuple[int, int, int] = (0, 200, 0)      # BGR green
COLOR_INVALID: Tuple[int, int, int] = (0, 165, 255)  # BGR orange
COLOR_ROI: Tuple[int, int, int] = (255, 200, 0)      # BGR cyan-ish
COLOR_TEXT_BG: Tuple[int, int, int] = (0, 0, 0)


def _put_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    """Text on a filled background rectangle, so it stays legible over any frame content."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 4),
        (x + text_width + 8, y + baseline - 4),
        COLOR_TEXT_BG,
        thickness=-1,
    )
    cv2.putText(frame, text, (x + 4, y - 6), font, scale, color, thickness, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, detections: Sequence[DemoDetection]) -> np.ndarray:
    """Draw each candidate's quad, decoded text, confidence, and validity."""
    for detection in detections:
        color = COLOR_VALID if detection.is_valid_format else COLOR_INVALID
        quad = detection.candidate.quad.astype(int)
        cv2.polylines(frame, [quad], isClosed=True, color=color, thickness=2)

        repaired_tag = " (repaired)" if detection.was_repaired else ""
        label = (
            f"{detection.display_text}{repaired_tag}  "
            f"{detection.recognition.confidence * 100:.0f}%  "
            f"{detection.recognition.inference_ms:.0f}ms"
        )
        x1, y1 = quad[:, 0].min(), quad[:, 1].min()
        _put_label(frame, label, (int(x1), max(20, int(y1))), color)
    return frame


def draw_manual_roi(frame: np.ndarray, localizer: PlateLocalizer, *, active: bool) -> np.ndarray:
    """Draw the manual ROI rectangle, when the active localizer has one.

    Shown even when a candidate was found elsewhere, as a framing guide while
    the operator lines up the shot before the strategy is switched to auto.
    """
    if not isinstance(localizer, ManualRoiLocalizer) or not active:
        return frame
    height, width = frame.shape[:2]
    left, top, right, bottom = localizer.fractional_roi
    pt1 = (int(left * width), int(top * height))
    pt2 = (int(right * width), int(bottom * height))
    cv2.rectangle(frame, pt1, pt2, COLOR_ROI, thickness=2)
    _put_label(frame, "hold plate here", (pt1[0], pt1[1]), COLOR_ROI)
    return frame


def draw_hud(
    frame: np.ndarray,
    *,
    fps: float,
    localizer_name: str,
    forwarding: bool,
    forwarded_count: int,
    help_text: Optional[str] = None,
) -> np.ndarray:
    """Top-left status readout: FPS, active strategy, gateway forwarding state."""
    lines = [
        f"FPS: {fps:.1f}   localizer: {localizer_name}",
        f"forward-to-gateway: {'ON' if forwarding else 'off'}  (sent: {forwarded_count})",
    ]
    if help_text:
        lines.append(help_text)
    y = 24
    for line in lines:
        _put_label(frame, line, (8, y), (255, 255, 255))
        y += 30
    return frame
