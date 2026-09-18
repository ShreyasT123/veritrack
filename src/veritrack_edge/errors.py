"""Exception hierarchy.

The edge node runs unattended for weeks. The distinction that matters
operationally is *recoverable per-frame faults* (skip the frame, keep running)
versus *fatal configuration or hardware faults* (exit non-zero so the
supervisor restarts the process).
"""

from __future__ import annotations


class VeriTrackError(Exception):
    """Base class for every error raised by the edge package."""


class FatalEdgeError(VeriTrackError):
    """Unrecoverable: bad model, missing NPU driver, malformed config."""


class BackendError(FatalEdgeError):
    """Inference runtime could not be initialised or produced invalid output."""


class ModelContractError(FatalEdgeError):
    """A model's tensor shapes/dtypes do not match the declared contract."""


class RecoverableEdgeError(VeriTrackError):
    """Per-frame or per-tracklet fault; the caller should skip and continue."""


class StreamError(RecoverableEdgeError):
    """Capture device dropped, timed out, or returned a corrupt frame."""


class GeometryError(RecoverableEdgeError):
    """A quadrilateral was degenerate, non-convex, or too skewed to rectify."""


class RecognitionError(RecoverableEdgeError):
    """OCR produced no usable hypothesis for a tracklet."""


class PayloadTooLargeError(RecoverableEdgeError):
    """Metadata could not be reduced below the configured byte ceiling."""
