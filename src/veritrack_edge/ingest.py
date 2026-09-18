"""Video stream ingestion.

A dedicated grabber thread owns the ``VideoCapture`` and pushes into a
bounded *keep-latest* queue. When the pipeline falls behind (a burst of
vehicles, a thermal throttle event), stale frames are dropped rather than
queued. For a trajectory system, a frame that is 800 ms old is worse than no
frame at all: it corrupts the ByteTrack motion model and injects a false
timestamp into the downstream HMM.

Reconnection uses exponential backoff with a cap, and the capture object is
fully released between attempts so that a wedged FFmpeg context cannot leak.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Iterator, Optional

import cv2
import numpy as np

from .config import StreamConfig
from .errors import StreamError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Frame:
    """One decoded frame with its grab-time metadata."""

    index: int
    timestamp: float
    image: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.image.shape[0]), int(self.image.shape[1])


class VideoStream:
    """Threaded, self-healing capture source."""

    __slots__ = (
        "_config",
        "_queue",
        "_thread",
        "_stop",
        "_capture",
        "_frame_index",
        "_dropped",
        "_reconnects",
        "_lock",
    )

    def __init__(self, config: StreamConfig) -> None:
        self._config = config
        self._queue: Queue[Frame] = Queue(maxsize=max(1, config.queue_size))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._dropped = 0
        self._reconnects = 0
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "VideoStream":
        if self._thread is not None:
            raise RuntimeError("VideoStream already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="veritrack-grabber", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._config.read_timeout_s + 2.0)
            self._thread = None
        self._release()

    def __enter__(self) -> "VideoStream":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- statistics --------------------------------------------------------

    @property
    def dropped_frames(self) -> int:
        return self._dropped

    @property
    def reconnect_count(self) -> int:
        return self._reconnects

    # -- consumption -------------------------------------------------------

    def read(self, timeout: Optional[float] = None) -> Frame:
        """Block for the next frame.

        Raises:
            StreamError: no frame arrived within ``timeout`` seconds.
        """
        wait = self._config.read_timeout_s if timeout is None else timeout
        try:
            return self._queue.get(timeout=wait)
        except Empty as exc:
            raise StreamError(f"No frame from {self._config.uri} within {wait:.1f}s") from exc

    def frames(self) -> Iterator[Frame]:
        """Yield frames until :meth:`stop` is called or the source dies."""
        while not self._stop.is_set():
            try:
                yield self.read()
            except StreamError as exc:
                logger.warning("stream stall: %s", exc)
                if self._stop.is_set():
                    return

    # -- internals ---------------------------------------------------------

    def _release(self) -> None:
        with self._lock:
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:  # pragma: no cover - driver-dependent
                    logger.exception("VideoCapture.release() failed")
                self._capture = None

    def _open(self) -> bool:
        self._release()
        capture = cv2.VideoCapture(self._config.uri, self._config.capture_api)
        if not capture.isOpened():
            capture.release()
            return False
        # A driver-side buffer of 1 keeps latency bounded when FFmpeg is used.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        with self._lock:
            self._capture = capture
        return True

    def _publish(self, image: np.ndarray) -> None:
        frame = Frame(index=self._frame_index, timestamp=time.time(), image=image)
        self._frame_index += 1
        try:
            self._queue.put_nowait(frame)
        except Full:
            # Keep-latest: evict the oldest, then retry once.
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except Empty:  # pragma: no cover - race with a fast consumer
                pass
            try:
                self._queue.put_nowait(frame)
            except Full:  # pragma: no cover
                self._dropped += 1

    def _run(self) -> None:
        backoff = self._config.reconnect_backoff_s
        while not self._stop.is_set():
            if not self._open():
                logger.error("cannot open %s; retrying in %.1fs", self._config.uri, backoff)
                self._reconnects += 1
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2.0, self._config.reconnect_backoff_max_s)
                continue

            logger.info("stream opened: %s", self._config.uri)
            backoff = self._config.reconnect_backoff_s
            consecutive_failures = 0

            while not self._stop.is_set():
                capture = self._capture
                if capture is None:
                    break
                ok, image = capture.read()
                if not ok or image is None or image.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        logger.warning("stream %s degraded; reconnecting", self._config.uri)
                        self._reconnects += 1
                        break
                    time.sleep(0.02)
                    continue
                consecutive_failures = 0
                self._publish(image)

            self._release()

        self._release()
        logger.info("grabber thread exiting for %s", self._config.uri)
