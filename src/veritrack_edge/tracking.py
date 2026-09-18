"""ByteTrack multi-object tracking.

ByteTrack's contribution is the *second association pass*: low-scoring boxes
that a conventional tracker throws away are matched against tracks left
unmatched after the first pass. In dense Indian traffic - where a two-wheeler
is routinely 60% occluded by an auto-rickshaw - this is the difference between
one continuous tracklet (which can fuse 12 frames of plate logits) and four
fragments (which each fuse three and all fail the confidence floor).

State model
-----------
Constant-velocity Kalman filter over

.. math::
   \\mathbf{x} = [c_x,\\; c_y,\\; a,\\; h,\\;
                  \\dot{c_x},\\; \\dot{c_y},\\; \\dot{a},\\; \\dot{h}]^{\\top}

with ``a = w / h``. Process and measurement noise are made proportional to the
box height ``h``, so a distant vehicle (small ``h``) is trusted to move less in
absolute pixels than a near one - the standard SORT parameterisation, and the
reason a single gate threshold works across the whole depth of field.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import TrackerConfig
from .types import BBox, Detection, TrackState

logger = logging.getLogger(__name__)

try:  # SciPy is optional on the edge image.
    from scipy.optimize import linear_sum_assignment as _lsa

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - device dependent
    _HAVE_SCIPY = False


def _greedy_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy fallback for :func:`scipy.optimize.linear_sum_assignment`.

    Not optimal, but deterministic and O(n log n); accuracy loss is confined to
    genuinely ambiguous clusters where the optimal assignment is near-degenerate.
    """
    rows, cols = [], []
    used_r, used_c = set(), set()
    flat = np.argsort(cost, axis=None)
    n_cols = cost.shape[1]
    for index in flat:
        r, c = divmod(int(index), n_cols)
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        rows.append(r)
        cols.append(c)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def linear_assignment(
    cost: np.ndarray, threshold: float, allow_greedy: bool
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Solve the assignment problem and split by ``threshold``.

    Returns ``(matches, unmatched_rows, unmatched_cols)``.
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    if _HAVE_SCIPY:
        rows, cols = _lsa(cost)
    elif allow_greedy:
        rows, cols = _greedy_assignment(cost)
    else:  # pragma: no cover - configuration choice
        raise RuntimeError("SciPy is unavailable and greedy assignment fallback is disabled")

    matches: List[Tuple[int, int]] = []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= threshold:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))
    unmatched_rows = [i for i in range(cost.shape[0]) if i not in matched_rows]
    unmatched_cols = [j for j in range(cost.shape[1]) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


def iou_distance(tracks: Sequence["STrack"], detections: Sequence[Detection]) -> np.ndarray:
    """Pairwise ``1 - IoU`` cost matrix of shape ``(len(tracks), len(detections))``."""
    cost = np.ones((len(tracks), len(detections)), dtype=np.float32)
    if not tracks or not detections:
        return cost
    track_boxes = np.array([t.tlbr for t in tracks], dtype=np.float32)
    det_boxes = np.array([d.bbox.as_tuple() for d in detections], dtype=np.float32)

    t_area = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
    d_area = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])

    lt = np.maximum(track_boxes[:, None, :2], det_boxes[None, :, :2])
    rb = np.minimum(track_boxes[:, None, 2:], det_boxes[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = t_area[:, None] + d_area[None, :] - inter
    return (1.0 - np.where(union > 0.0, inter / np.maximum(union, 1e-9), 0.0)).astype(np.float32)


class KalmanFilterXYAH:
    """8-dimensional constant-velocity filter in ``(cx, cy, a, h)`` space."""

    __slots__ = ("_motion_mat", "_update_mat", "_std_position", "_std_velocity")

    def __init__(self, std_position: float = 1.0 / 20.0, std_velocity: float = 1.0 / 160.0) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float64)
        self._std_position = std_position
        self._std_velocity = std_velocity

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.concatenate([measurement.astype(np.float64), np.zeros(4, dtype=np.float64)])
        h = float(measurement[3])
        std = np.array(
            [
                2 * self._std_position * h,
                2 * self._std_position * h,
                1e-2,
                2 * self._std_position * h,
                10 * self._std_velocity * h,
                10 * self._std_velocity * h,
                1e-5,
                10 * self._std_velocity * h,
            ],
            dtype=np.float64,
        )
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = float(mean[3])
        std_pos = np.array(
            [self._std_position * h, self._std_position * h, 1e-2, self._std_position * h]
        )
        std_vel = np.array(
            [self._std_velocity * h, self._std_velocity * h, 1e-5, self._std_velocity * h]
        )
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = float(mean[3])
        std = np.array(
            [self._std_position * h, self._std_position * h, 1e-1, self._std_position * h]
        )
        innovation_cov = np.diag(np.square(std))
        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return projected_mean, projected_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        # Solve rather than invert: numerically safer and faster for 4x4.
        kalman_gain = np.linalg.solve(
            projected_cov.T, (covariance @ self._update_mat.T).T
        ).T
        innovation = measurement.astype(np.float64) - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    def gating_distance(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> float:
        """Squared Mahalanobis distance between the prediction and a measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)
        try:
            cholesky = np.linalg.cholesky(projected_cov)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate covariance
            return float("inf")
        delta = measurement.astype(np.float64) - projected_mean
        z = np.linalg.solve(cholesky, delta)
        return float(np.dot(z, z))


def _tlbr_to_xyah(tlbr: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in tlbr)
    w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    return np.array([x1 + w * 0.5, y1 + h * 0.5, w / h, h], dtype=np.float64)


def _xyah_to_tlbr(state: np.ndarray) -> np.ndarray:
    cx, cy, a, h = (float(v) for v in state[:4])
    w = max(a * h, 1e-3)
    h = max(h, 1e-3)
    return np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dtype=np.float32)


class STrack:
    """A single tracked vehicle."""

    _next_id = 0
    __slots__ = (
        "track_id",
        "state",
        "mean",
        "covariance",
        "score",
        "class_id",
        "class_name",
        "hits",
        "age",
        "time_since_update",
        "start_frame",
        "frame_id",
        "start_timestamp",
        "timestamp",
        "embedding",
        "_kf",
        "is_activated",
        "first_tlbr",
    )

    def __init__(self, detection: Detection, kalman: KalmanFilterXYAH) -> None:
        self.track_id = -1
        self.state = TrackState.NEW
        self._kf = kalman
        measurement = _tlbr_to_xyah(detection.bbox.as_tuple())
        self.mean, self.covariance = kalman.initiate(measurement)
        self.score = detection.score
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.start_frame = 0
        self.frame_id = 0
        self.start_timestamp = 0.0
        self.timestamp = 0.0
        self.embedding: Optional[np.ndarray] = None
        self.is_activated = False
        self.first_tlbr = detection.bbox.as_array()

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._next_id = 0

    @classmethod
    def _allocate_id(cls) -> int:
        cls._next_id += 1
        return cls._next_id

    @property
    def tlbr(self) -> np.ndarray:
        return _xyah_to_tlbr(self.mean)

    @property
    def bbox(self) -> BBox:
        return BBox.from_array(self.tlbr)

    def predict(self) -> None:
        mean = self.mean.copy()
        if self.state is not TrackState.TRACKED:
            # A lost track is not assumed to keep changing size.
            mean[7] = 0.0
        self.mean, self.covariance = self._kf.predict(mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def activate(self, frame_id: int, timestamp: float, force_id: bool = True) -> None:
        if force_id or self.track_id < 0:
            self.track_id = STrack._allocate_id()
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.start_timestamp = timestamp
        self.timestamp = timestamp
        self.time_since_update = 0

    def update(self, detection: Detection, frame_id: int, timestamp: float) -> None:
        measurement = _tlbr_to_xyah(detection.bbox.as_tuple())
        self.mean, self.covariance = self._kf.update(self.mean, self.covariance, measurement)
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = detection.score
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.hits += 1
        self.time_since_update = 0
        self.frame_id = frame_id
        self.timestamp = timestamp

    def re_activate(self, detection: Detection, frame_id: int, timestamp: float) -> None:
        """Recover a lost track without minting a new identity."""
        self.update(detection, frame_id, timestamp)

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED

    def gating_distance(self, detection: Detection) -> float:
        return self._kf.gating_distance(
            self.mean, self.covariance, _tlbr_to_xyah(detection.bbox.as_tuple())
        )


class ByteTracker:
    """Two-stage association tracker."""

    __slots__ = (
        "_config",
        "_kf",
        "_tracked",
        "_lost",
        "_removed",
        "_frame_id",
        "_max_time_lost",
    )

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._kf = KalmanFilterXYAH()
        self._tracked: List[STrack] = []
        self._lost: List[STrack] = []
        self._removed: List[STrack] = []
        self._frame_id = 0
        self._max_time_lost = max(
            1, int(config.frame_rate / 30.0 * config.track_buffer_frames)
        )

    @property
    def active_tracks(self) -> List[STrack]:
        return [t for t in self._tracked if t.is_activated]

    @property
    def lost_tracks(self) -> List[STrack]:
        return list(self._lost)

    def _apply_gate(
        self, cost: np.ndarray, tracks: Sequence[STrack], detections: Sequence[Detection]
    ) -> np.ndarray:
        """Set Mahalanobis-implausible pairs to an unreachable cost."""
        if not self._config.use_mahalanobis_gate or cost.size == 0:
            return cost
        gated = cost.copy()
        for i, track in enumerate(tracks):
            # A track whose velocity has not converged will fail the gate for
            # purely kinematic reasons; leave it to the IoU cost alone.
            if track.hits < self._config.min_hits_for_gate:
                continue
            for j, detection in enumerate(detections):
                if track.gating_distance(detection) > self._config.gating_chi2_thresh:
                    gated[i, j] = np.inf
        return gated

    def update(
        self,
        high_detections: Sequence[Detection],
        low_detections: Sequence[Detection],
        timestamp: float,
    ) -> Tuple[List[STrack], List[STrack]]:
        """Advance the tracker by one frame.

        Returns ``(active_tracks, terminated_tracks)``. Terminated tracks are
        the trigger for emitting a vehicle pass downstream.
        """
        self._frame_id += 1
        cfg = self._config
        greedy_ok = cfg.allow_greedy_assignment_fallback

        unconfirmed = [t for t in self._tracked if not t.is_activated]
        confirmed = [t for t in self._tracked if t.is_activated]

        pool: List[STrack] = confirmed + self._lost
        for track in pool:
            track.predict()
        for track in unconfirmed:
            track.predict()

        # --- pass 1: high-confidence detections against tracked + lost -----
        cost = iou_distance(pool, high_detections)
        cost = self._apply_gate(cost, pool, high_detections)
        matches, u_tracks, u_dets = linear_assignment(
            cost, cfg.first_match_max_distance, greedy_ok
        )

        activated: List[STrack] = []
        refound: List[STrack] = []
        for track_idx, det_idx in matches:
            track = pool[track_idx]
            detection = high_detections[det_idx]
            if track.state is TrackState.TRACKED:
                track.update(detection, self._frame_id, timestamp)
                activated.append(track)
            else:
                track.re_activate(detection, self._frame_id, timestamp)
                refound.append(track)

        # --- pass 2: low-confidence detections against still-unmatched -----
        remaining = [pool[i] for i in u_tracks if pool[i].state is TrackState.TRACKED]
        cost2 = iou_distance(remaining, low_detections)
        # No Mahalanobis gate here: low-score boxes are precisely the occluded
        # cases where the appearance-free motion model is the only evidence.
        matches2, u_tracks2, _ = linear_assignment(
            cost2, cfg.second_match_max_distance, greedy_ok
        )
        for track_idx, det_idx in matches2:
            track = remaining[track_idx]
            track.update(low_detections[det_idx], self._frame_id, timestamp)
            activated.append(track)

        newly_lost: List[STrack] = []
        for idx in u_tracks2:
            track = remaining[idx]
            if track.state is not TrackState.LOST:
                track.mark_lost()
                newly_lost.append(track)
        for i in u_tracks:
            track = pool[i]
            if track.state is TrackState.TRACKED and track.time_since_update > 0:
                if track not in activated and track not in newly_lost and track not in remaining:
                    track.mark_lost()
                    newly_lost.append(track)

        # --- unconfirmed tracks: strict IoU, one chance ---------------------
        leftover_high = [high_detections[i] for i in u_dets]
        cost3 = iou_distance(unconfirmed, leftover_high)
        matches3, u_unconfirmed, u_dets3 = linear_assignment(
            cost3, cfg.unconfirmed_match_max_distance, greedy_ok
        )
        for track_idx, det_idx in matches3:
            track = unconfirmed[track_idx]
            track.update(leftover_high[det_idx], self._frame_id, timestamp)
            if track.hits >= cfg.min_hits:
                track.activate(self._frame_id, track.start_timestamp or timestamp)
                track.start_timestamp = timestamp if track.start_timestamp == 0.0 else track.start_timestamp
            activated.append(track)

        removed_now: List[STrack] = []
        for idx in u_unconfirmed:
            track = unconfirmed[idx]
            track.mark_removed()
            removed_now.append(track)

        # --- spawn new tracks ------------------------------------------------
        for det_idx in u_dets3:
            detection = leftover_high[det_idx]
            if detection.score < cfg.new_track_thresh:
                continue
            track = STrack(detection, self._kf)
            track.start_frame = self._frame_id
            track.frame_id = self._frame_id
            track.start_timestamp = timestamp
            track.timestamp = timestamp
            if cfg.min_hits <= 1:
                track.activate(self._frame_id, timestamp)
            self._tracked.append(track)

        # --- age out ---------------------------------------------------------
        surviving_lost: List[STrack] = []
        for track in self._lost + newly_lost:
            if track.state is TrackState.REMOVED:
                continue
            if track.time_since_update > self._max_time_lost:
                track.mark_removed()
                removed_now.append(track)
            elif track.state is TrackState.LOST:
                surviving_lost.append(track)

        recovered_ids = {id(t) for t in refound}
        self._lost = [t for t in surviving_lost if id(t) not in recovered_ids]
        self._tracked = [
            t
            for t in self._tracked
            if t.state in (TrackState.TRACKED, TrackState.NEW) and t.state is not TrackState.REMOVED
        ]
        for track in refound:
            if track not in self._tracked:
                self._tracked.append(track)

        # Promote unconfirmed tracks that have accumulated enough hits.
        for track in self._tracked:
            if not track.is_activated and track.hits >= cfg.min_hits:
                start_ts = track.start_timestamp or timestamp
                track.activate(self._frame_id, start_ts)
                track.start_timestamp = start_ts

        self._removed.extend(removed_now)
        terminated = [t for t in removed_now if t.is_activated]
        return self.active_tracks, terminated

    def flush(self) -> List[STrack]:
        """Terminate every live track, e.g. at stream shutdown."""
        live = [t for t in self._tracked + self._lost if t.is_activated]
        for track in live:
            track.mark_removed()
        self._tracked, self._lost = [], []
        return live
