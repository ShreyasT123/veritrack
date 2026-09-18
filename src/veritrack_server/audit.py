"""Append-only, hash-chained operator-query audit ledger."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import List, Tuple

__all__ = ["AuditEntry", "AuditLedger"]

_GENESIS_HASH = "0" * 64


def _utc_millis(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("audit timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class AuditEntry:
    sequence: int
    officer_id: str
    plate_number: str
    warrant_reference: str
    timestamp_utc: datetime
    previous_hash: str
    entry_hash: str


class AuditLedger:
    """Thread-safe in-memory ledger suitable for a repository-backed adapter.

    Canonical field framing prevents ambiguity such as ``("ab", "c")`` versus
    ``("a", "bc")`` when hashes are verified independently.
    """

    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []
        self._lock = RLock()

    @staticmethod
    def _hash(officer_id: str, plate_number: str, warrant_reference: str, timestamp: datetime, previous_hash: str) -> str:
        payload = "\x1f".join((officer_id, plate_number, warrant_reference, _utc_millis(timestamp), previous_hash))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def append(self, officer_id: str, plate_number: str, warrant_reference: str, *, timestamp_utc: datetime | None = None) -> AuditEntry:
        if not officer_id or not plate_number or not warrant_reference:
            raise ValueError("officer_id, plate_number, and warrant_reference are required")
        timestamp = timestamp_utc or datetime.now(timezone.utc)
        _utc_millis(timestamp)
        with self._lock:
            previous = self._entries[-1].entry_hash if self._entries else _GENESIS_HASH
            entry = AuditEntry(len(self._entries) + 1, officer_id, plate_number, warrant_reference, timestamp, previous,
                               self._hash(officer_id, plate_number, warrant_reference, timestamp, previous))
            self._entries.append(entry)
            return entry

    def entries(self) -> Tuple[AuditEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def verify_audit_integrity(self) -> Tuple[bool, int]:
        """Return ``(valid, checked_entries)``; detects altered or removed links."""
        with self._lock:
            previous = _GENESIS_HASH
            for expected_sequence, entry in enumerate(self._entries, start=1):
                expected = self._hash(entry.officer_id, entry.plate_number, entry.warrant_reference, entry.timestamp_utc, previous)
                if entry.sequence != expected_sequence or entry.previous_hash != previous or entry.entry_hash != expected:
                    return False, expected_sequence
                previous = entry.entry_hash
            return True, len(self._entries)
