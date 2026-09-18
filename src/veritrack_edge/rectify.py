"""Perspective rectification of the plate quadrilateral.

Normalised DLT
--------------
For each correspondence :math:`(x, y) \\mapsto (u, v)` the homography
:math:`H` with :math:`h_{33} = 1` satisfies

.. math::
   u = \\frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + 1}, \\qquad
   v = \\frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + 1}

which linearises to two rows per point:

.. math::
   \\begin{bmatrix}
   x & y & 1 & 0 & 0 & 0 & -ux & -uy \\\\
   0 & 0 & 0 & x & y & 1 & -vx & -vy
   \\end{bmatrix} \\mathbf{h} = \\begin{bmatrix} u \\\\ v \\end{bmatrix}

Four correspondences give an 8x8 system. Raw pixel coordinates give this
system a condition number in the millions at 1080p, so Hartley isotropic
normalisation is applied first: each point set is translated to zero centroid
and scaled to mean radius :math:`\\sqrt{2}`, the system is solved in normalised
space, and the result is de-normalised as
:math:`H = T_{dst}^{-1} \\tilde{H} T_{src}`.

Pose recovery
-------------
The 45-degree acceptance criterion is a statement about the *physical* plate,
so it is checked on physical angles wherever intrinsics are available. With
:math:`K` known and the plate plane mapped to metric millimetres,
:math:`H_{metric \\to image} = \\lambda K [\\, \\mathbf{r}_1\\; \\mathbf{r}_2\\;
\\mathbf{t} \\,]`, hence

.. math::
   \\mathbf{r}_1 = \\frac{K^{-1}\\mathbf{h}_1}{\\lVert K^{-1}\\mathbf{h}_1 \\rVert},
   \\quad
   \\mathbf{r}_3 = \\mathbf{r}_1 \\times \\mathbf{r}_2

after Gram-Schmidt orthonormalisation, and the Euler angles fall out of
:math:`R`. Without intrinsics the code falls back to *edge symmetry* - the
ratio of the shorter to the longer vertical edge of the quad - which is a
monotone proxy for yaw under weak perspective.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import CameraIntrinsics, RectifyConfig
from .errors import GeometryError
from .gating import variance_of_laplacian
from .types import PlateLayout, PlatePose, PlateQuad, PlateSeries, Quad, RectifiedPlate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quadrilateral canonicalisation
# ---------------------------------------------------------------------------


def order_quad_corners(quad: np.ndarray) -> np.ndarray:
    """Return corners ordered TL, TR, BR, BL (clockwise in image axes).

    Sorting by polar angle about the centroid yields a consistent winding even
    for strongly sheared quads, where the naive ``x + y`` / ``x - y`` extremum
    rule flips corners. The winding is then normalised to clockwise and rotated
    so that the corner closest to the centroid-relative top-left leads.

    Raises:
        GeometryError: the quad is degenerate (zero area or collinear points).
    """
    points = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    if points.shape != (4, 2):
        raise GeometryError(f"Expected 4 corner points, got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise GeometryError("Quad contains non-finite coordinates")

    centre = points.mean(axis=0)
    offsets = points - centre
    if np.any(np.linalg.norm(offsets, axis=1) < 1e-6):
        raise GeometryError("Degenerate quad: a corner coincides with the centroid")

    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    order = np.argsort(angles)
    ordered = points[order]

    # Shoelace. The image y-axis points down, so a visually *clockwise* ring
    # yields a positive signed area - the opposite of the textbook convention.
    x, y = ordered[:, 0], ordered[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if abs(area2) < 1e-3:
        raise GeometryError("Degenerate quad: near-zero area")
    if area2 < 0.0:
        ordered = ordered[::-1]

    scores = ordered[:, 0] + ordered[:, 1]
    start = int(np.argmin(scores))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def quad_area(quad: np.ndarray) -> float:
    """Shoelace area of a 4-gon in pixels."""
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def is_convex(quad: np.ndarray) -> bool:
    """True when all cross products of consecutive edges share a sign."""
    signs = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(math.copysign(1.0, cross) if abs(cross) > 1e-6 else 0.0)
    positive = all(s >= 0.0 for s in signs)
    negative = all(s <= 0.0 for s in signs)
    return positive or negative


def quad_aspect_ratio(quad: np.ndarray) -> float:
    """Mean horizontal edge length divided by mean vertical edge length."""
    top = float(np.linalg.norm(quad[1] - quad[0]))
    bottom = float(np.linalg.norm(quad[2] - quad[3]))
    left = float(np.linalg.norm(quad[3] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    horizontal = 0.5 * (top + bottom)
    vertical = max(0.5 * (left + right), 1e-6)
    return horizontal / vertical


def edge_symmetry(quad: np.ndarray) -> float:
    """Ratio of shorter to longer vertical edge; 1.0 = fronto-parallel."""
    left = float(np.linalg.norm(quad[3] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    longer = max(left, right)
    if longer <= 1e-6:
        return 0.0
    return min(left, right) / longer


# ---------------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------------


def _normalisation_matrix(points: np.ndarray) -> np.ndarray:
    """Hartley isotropic normalisation transform for ``(N, 2)`` points."""
    centroid = points.mean(axis=0)
    centred = points - centroid
    mean_radius = float(np.mean(np.linalg.norm(centred, axis=1)))
    scale = math.sqrt(2.0) / mean_radius if mean_radius > 1e-9 else 1.0
    return np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _apply_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    projected = homogeneous @ matrix.T
    w = projected[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise GeometryError("Projective transform produced a point at infinity")
    return projected[:, :2] / w


def solve_homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve the 4-point normalised DLT homography ``src -> dst``.

    Args:
        src: ``(4, 2)`` source points.
        dst: ``(4, 2)`` destination points.

    Returns:
        ``(3, 3)`` float64 homography with ``H[2, 2] == 1``.

    Raises:
        GeometryError: the configuration is degenerate (three collinear points).
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise GeometryError(f"DLT requires 4 correspondences, got {src.shape} -> {dst.shape}")

    t_src = _normalisation_matrix(src)
    t_dst = _normalisation_matrix(dst)
    src_n = _apply_transform(t_src, src)
    dst_n = _apply_transform(t_dst, dst)

    a = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i in range(4):
        x, y = src_n[i]
        u, v = dst_n[i]
        a[2 * i] = (x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y)
        a[2 * i + 1] = (0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y)
        b[2 * i] = u
        b[2 * i + 1] = v

    try:
        solution = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(a, b, rcond=None)
        if not np.all(np.isfinite(solution)):
            raise GeometryError("DLT system is singular; corners are collinear") from None

    h_norm = np.append(solution, 1.0).reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    if abs(h[2, 2]) < 1e-12:
        raise GeometryError("Recovered homography is not normalisable")
    return h / h[2, 2]


def decompose_plate_pose(
    h_metric_to_image: np.ndarray, intrinsics: CameraIntrinsics
) -> Tuple[float, float, float]:
    """Recover ``(yaw_deg, pitch_deg, roll_deg)`` of the plate plane.

    Args:
        h_metric_to_image: homography mapping plate-plane millimetres to image
            pixels.
        intrinsics: pinhole intrinsics in the same pixel frame.

    Returns:
        Euler angles in degrees, ZYX convention (roll about the optical axis,
        yaw about the plate's vertical axis, pitch about its horizontal axis).
    """
    k = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    k_inv = np.linalg.inv(k)
    m = k_inv @ h_metric_to_image
    norm1 = np.linalg.norm(m[:, 0])
    norm2 = np.linalg.norm(m[:, 1])
    if norm1 < 1e-12 or norm2 < 1e-12:
        raise GeometryError("Degenerate homography: cannot recover rotation columns")

    lam = 2.0 / (norm1 + norm2)
    r1 = m[:, 0] * lam
    r2 = m[:, 1] * lam
    # Gram-Schmidt, then re-orthonormalise via the SVD projection onto SO(3).
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 - np.dot(r1, r2) * r1
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    rotation = np.stack([r1, r2, r3], axis=1)
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, 2] *= -1.0
        rotation = u @ vt

    # ZYX decomposition R = Rz(roll) Ry(yaw) Rx(pitch), where the plate frame
    # has X along the registration, Y down the plate face, Z along its normal.
    # Rotation about Y is therefore the off-axis viewing angle the 45-degree
    # acceptance criterion refers to; rotation about Z is in-plane tilt.
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    if sy > 1e-6:
        roll = math.atan2(rotation[1, 0], rotation[0, 0])
        yaw = math.atan2(-rotation[2, 0], sy)
        pitch = math.atan2(rotation[2, 1], rotation[2, 2])
    else:  # gimbal-locked
        roll = 0.0
        yaw = math.atan2(-rotation[2, 0], sy)
        pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


# ---------------------------------------------------------------------------
# Plate series (background colour) classification
# ---------------------------------------------------------------------------


def classify_plate_series(canonical_bgr: np.ndarray) -> PlateSeries:
    """Infer the plate's usage category from its background colour.

    Characters are dark on a light field (or white on black for rental plates),
    so the *background* is recovered as the modal-luminance half of the pixels
    rather than the mean, which a dense character field would bias.
    """
    if canonical_bgr is None or canonical_bgr.size == 0:
        return PlateSeries.UNKNOWN
    hsv = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()

    # Background = the brighter half, unless the plate is predominantly dark.
    median_v = float(np.median(v))
    background = v >= median_v
    if background.sum() < 16:
        return PlateSeries.UNKNOWN

    bg_h = float(np.median(h[background]))
    bg_s = float(np.median(s[background]))
    bg_v = float(np.median(v[background]))

    if bg_v < 70.0:
        return PlateSeries.RENTAL_BLACK
    if bg_s < 55.0:
        return PlateSeries.PRIVATE_WHITE
    # OpenCV hue is 0-179.
    if 18.0 <= bg_h <= 38.0:
        return PlateSeries.COMMERCIAL_YELLOW
    if 40.0 <= bg_h <= 85.0:
        return PlateSeries.ELECTRIC_GREEN
    if 90.0 <= bg_h <= 130.0 and bg_v > 120.0:
        return PlateSeries.DIPLOMATIC_LIGHT_BLUE
    return PlateSeries.PRIVATE_WHITE


# ---------------------------------------------------------------------------
# Rectifier
# ---------------------------------------------------------------------------


class PlateRectifier:
    """Warps a detected plate quad onto a canonical, fronto-parallel canvas."""

    __slots__ = ("_config",)

    def __init__(self, config: RectifyConfig) -> None:
        self._config = config

    @property
    def config(self) -> RectifyConfig:
        return self._config

    def _destination_rect(self, size: Tuple[int, int]) -> np.ndarray:
        w, h = float(size[0]), float(size[1])
        return np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], np.float64)

    def _heuristic_pose(self, quad: np.ndarray, plate_mm: Tuple[float, float]) -> PlatePose:
        """Estimate skew without intrinsics, from aspect-ratio compression.

        Edge foreshortening is a *bad* yaw estimator at ANPR standoff. For a
        plate of width :math:`W` at range :math:`d` yawed by :math:`\\theta`,
        the far and near vertical edges differ only as

        .. math::
           r = \\frac{1 - k\\sin\\theta}{1 + k\\sin\\theta},
           \\qquad k = \\frac{W}{2d}

        At :math:`W = 0.5` m and :math:`d = 10` m, :math:`k = 0.025`, so even a
        full 45-degree yaw gives :math:`r \\approx 0.97` - inside the noise of
        the keypoint regressor. Perspective is simply weak at that range.

        Aspect-ratio compression is not. The projected width scales as
        :math:`\\cos\\theta` while the height is unchanged, so

        .. math::
           \\theta = \\arccos\\!\\left(
               \\frac{\\text{AR}_{\\text{observed}}}{\\text{AR}_{\\text{nominal}}}
           \\right)

        with :math:`\\text{AR}_{\\text{nominal}} = 500/120 = 4.17` for a
        single-line plate. This also explains the two-line decision threshold:
        :math:`4.17\\cos 45^\\circ = 2.95`, and the configured 2.70 sits just
        below it, so a single-line plate is never mistaken for a two-line one
        anywhere inside the supported skew envelope.

        Roll is directly observable as the image-plane direction of the mean
        horizontal edge. Pitch is not separable from yaw without intrinsics and
        is reported as zero.
        """
        symmetry = edge_symmetry(quad)
        observed = quad_aspect_ratio(quad)
        nominal = plate_mm[0] / plate_mm[1] if plate_mm[1] > 0.0 else 1.0
        compression = float(np.clip(observed / nominal, 0.0, 1.0)) if nominal > 0.0 else 1.0
        yaw = math.degrees(math.acos(compression))

        top = quad[1] - quad[0]
        bottom = quad[2] - quad[3]
        horizontal = 0.5 * (top + bottom)
        roll = math.degrees(math.atan2(float(horizontal[1]), float(horizontal[0])))
        return PlatePose(yaw_deg=yaw, pitch_deg=0.0, roll_deg=roll, edge_symmetry=symmetry, analytic=False)

    def _estimate_pose(
        self, quad: np.ndarray, homography: np.ndarray, canvas: Tuple[int, int], plate_mm: Tuple[float, float]
    ) -> PlatePose:
        intrinsics = self._config.intrinsics
        if intrinsics is None:
            return self._heuristic_pose(quad, plate_mm)

        try:
            h_canvas_to_image = np.linalg.inv(homography)
            # Canvas pixels -> plate-plane millimetres.
            scale_mm_to_canvas = np.diag(
                [
                    (canvas[0] - 1.0) / plate_mm[0],
                    (canvas[1] - 1.0) / plate_mm[1],
                    1.0,
                ]
            )
            h_metric_to_image = h_canvas_to_image @ scale_mm_to_canvas
            yaw, pitch, roll = decompose_plate_pose(h_metric_to_image, intrinsics)
            return PlatePose(yaw, pitch, roll, edge_symmetry(quad), analytic=True)
        except (GeometryError, np.linalg.LinAlgError) as exc:
            logger.debug("analytic pose failed (%s); falling back to aspect compression", exc)
            return self._heuristic_pose(quad, plate_mm)

    def rectify(self, frame: np.ndarray, plate: PlateQuad) -> RectifiedPlate:
        """Warp ``plate`` out of ``frame`` onto the canonical canvas.

        Raises:
            GeometryError: the quad is degenerate, non-convex, too small, or
                exceeds the configured skew limit.
        """
        cfg = self._config
        quad = order_quad_corners(plate.quad)

        if not is_convex(quad):
            raise GeometryError("Plate quad is non-convex")
        area = quad_area(quad)
        if area < 1.0:
            raise GeometryError(f"Plate quad area {area:.2f}px is degenerate")

        aspect = quad_aspect_ratio(quad)
        layout = (
            PlateLayout.SINGLE_LINE
            if aspect >= cfg.two_line_aspect_threshold
            else PlateLayout.TWO_LINE
        )
        canvas = cfg.single_line_size if layout is PlateLayout.SINGLE_LINE else cfg.two_line_size
        plate_mm = cfg.single_line_mm if layout is PlateLayout.SINGLE_LINE else cfg.two_line_mm

        homography = solve_homography_dlt(quad.astype(np.float64), self._destination_rect(canvas))
        pose = self._estimate_pose(quad, homography, canvas, plate_mm)

        # Both pose paths now yield a physical angle, so the acceptance
        # criterion is applied uniformly. Edge symmetry survives only as a
        # sanity gate on wildly non-rectangular keypoint regressions.
        if pose.max_skew_deg > cfg.max_skew_deg:
            raise GeometryError(
                f"Plate skew {pose.max_skew_deg:.1f}deg exceeds {cfg.max_skew_deg:.1f}deg "
                f"({'analytic' if pose.analytic else 'aspect-compression'} estimate)"
            )
        if pose.edge_symmetry < cfg.min_edge_symmetry:
            raise GeometryError(
                f"Quad edge symmetry {pose.edge_symmetry:.2f} below {cfg.min_edge_symmetry:.2f}; "
                "corner regression is implausible"
            )

        border = cv2.BORDER_REPLICATE if cfg.border_mode_replicate else cv2.BORDER_CONSTANT
        warped = cv2.warpPerspective(
            frame,
            homography,
            canvas,
            flags=cv2.INTER_LINEAR,
            borderMode=border,
        )
        if warped is None or warped.size == 0:
            raise GeometryError("warpPerspective produced an empty canvas")

        return RectifiedPlate(
            image=warped,
            layout=layout,
            series=classify_plate_series(warped),
            pose=pose,
            homography=homography,
            focus_score=variance_of_laplacian(warped),
            source_quad=quad,
        )
