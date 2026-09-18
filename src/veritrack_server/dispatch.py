"""Emergency 112 CAP v1.2 formatting and asynchronous delivery worker."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Protocol, Tuple

__all__ = ["CapAlert", "DispatchAttempt", "EmergencyDispatcher"]

SenderStatus = Literal["Actual", "Exercise"]
_SENDER = "veritrack.surveillance.node.delhi_police"


def _iso_utc_millis(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("CAP sent timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class CapAlert:
    """Minimal CAP v1.2 alert with ERSS-required operational metadata."""

    event: Literal["CLONED_PLATE_DETECTED", "HOTLIST_WARRANT_HIT"]
    plate: str
    latitude: float
    longitude: float
    junction_name: str
    description: str
    resource_url: Optional[str] = None
    severity: Literal["Extreme", "Moderate"] = "Extreme"
    status: SenderStatus = "Actual"
    identifier: str = field(default_factory=lambda: str(uuid.uuid4()))
    sent_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if uuid.UUID(self.identifier).version != 4:
            raise ValueError("CAP identifier must be a UUID4")
        if not self.plate or not self.junction_name or not self.description:
            raise ValueError("plate, junction_name, and description are required")
        if not -90.0 <= self.latitude <= 90.0 or not -180.0 <= self.longitude <= 180.0:
            raise ValueError("CAP area coordinates must be WGS84")
        _iso_utc_millis(self.sent_at)

    def to_cap_json(self) -> Dict[str, Any]:
        resource: List[Dict[str, Any]] = []
        if self.resource_url:
            resource.append({"resourceDesc": "ANPR evidence crop", "uri": self.resource_url, "mimeType": "image/jpeg"})
        return {
            "identifier": self.identifier, "sender": _SENDER, "sent": _iso_utc_millis(self.sent_at),
            "status": self.status, "msgType": "Alert", "scope": "Restricted",
            "info": [{"category": ["Security"], "event": self.event, "urgency": "Immediate", "severity": self.severity,
                      "certainty": "Observed", "headline": f"{self.event}: {self.plate}", "description": self.description,
                      "area": [{"areaDesc": self.junction_name, "polygon": f"{self.latitude:.6f},{self.longitude:.6f}"}],
                      "resource": resource}],
        }


@dataclass(frozen=True, slots=True)
class DispatchAttempt:
    alert: CapAlert
    attempts: int
    error: Optional[str]
    delivered: bool


class AsyncSender(Protocol):
    async def __call__(self, payload: Dict[str, Any], timeout_s: float) -> None: ...


async def _default_sender(endpoint: str, payload: Dict[str, Any], timeout_s: float) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/cap+json"}, method="POST")
    def send() -> None:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"ERSS returned HTTP {response.status}")
    try:
        await asyncio.wait_for(asyncio.to_thread(send), timeout=timeout_s)
    except (urllib.error.URLError, OSError, asyncio.TimeoutError) as exc:
        raise RuntimeError(str(exc)) from exc


class EmergencyDispatcher:
    """Bounded async CAP dispatcher with retries and inspectable dead letters."""

    def __init__(self, endpoint: str, *, timeout_s: float = 0.5, max_retries: int = 3,
                 base_backoff_s: float = 0.05, sender: Optional[AsyncSender] = None) -> None:
        if not endpoint or not 0.0 < timeout_s <= 0.5 or max_retries < 0 or base_backoff_s <= 0.0:
            raise ValueError("invalid dispatcher settings")
        self._endpoint, self._timeout_s, self._max_retries, self._base_backoff_s = endpoint, timeout_s, max_retries, base_backoff_s
        self._sender: AsyncSender = sender or (lambda payload, timeout: _default_sender(endpoint, payload, timeout))
        self._queue: asyncio.Queue[CapAlert | None] = asyncio.Queue(maxsize=10_000)
        self._worker: Optional[asyncio.Task[None]] = None
        self._dead_letters: List[DispatchAttempt] = []
        self._delivered: List[DispatchAttempt] = []

    @property
    def dead_letters(self) -> Tuple[DispatchAttempt, ...]: return tuple(self._dead_letters)
    @property
    def delivered(self) -> Tuple[DispatchAttempt, ...]: return tuple(self._delivered)

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="veritrack-erss-dispatch")

    async def stop(self) -> None:
        if self._worker is not None:
            await self._queue.put(None)
            await self._worker
            self._worker = None

    async def submit(self, alert: CapAlert) -> None:
        await self._queue.put(alert)

    async def _run(self) -> None:
        while True:
            alert = await self._queue.get()
            try:
                if alert is None:
                    return
                await self._deliver(alert)
            finally:
                self._queue.task_done()

    async def _deliver(self, alert: CapAlert) -> None:
        error: Optional[str] = None
        for attempt in range(1, self._max_retries + 2):
            try:
                await asyncio.wait_for(self._sender(alert.to_cap_json(), self._timeout_s), timeout=self._timeout_s)
                self._delivered.append(DispatchAttempt(alert, attempt, None, True))
                return
            except Exception as exc:  # deliberate containment: alert retries must not kill worker
                error = str(exc)
                if attempt <= self._max_retries:
                    await asyncio.sleep(self._base_backoff_s * (2 ** (attempt - 1)))
        self._dead_letters.append(DispatchAttempt(alert, self._max_retries + 1, error, False))
