"""DPDP Act 2023 cryptographic core for VeriTrack Stage 2.

The Digital Personal Data Protection Act, 2023 makes a registration number
personal data when it is linkable to an individual. VeriTrack's answer is a
**dual-track** pipeline, decided per sighting:

``NOT on hotlist``
    The plate is irreversibly pseudonymised with ``HMAC-SHA256(salt_epoch,
    plate)``. Only the digest and a short prefix are persisted; the cleartext is
    dropped before the row reaches the write buffer. Because the salt rotates
    every 24 hours, two sightings of the same vehicle are linkable *within* a
    day -- which is exactly what corridor analytics needs -- and unlinkable
    *across* days once the salt has aged out. That is purpose limitation
    enforced by arithmetic rather than by policy.

``ON hotlist``
    There is a lawful basis (an active FIR/warrant), so the cleartext plate is
    retained under that warrant reference and the evidence metadata is sealed
    with AES-256-GCM envelope encryption.

Design notes worth defending:

* **Envelope, not direct, encryption.** Each record gets a fresh 256-bit Data
  Encryption Key; the DEK is wrapped by the long-lived Key Encryption Key. A KEK
  rotation therefore re-wraps a handful of small DEKs instead of re-encrypting
  the entire evidence corpus.
* **AAD binds ciphertext to its row.** The ``pass_id`` is passed as Additional
  Authenticated Data, so a ciphertext lifted from one sighting and pasted onto
  another fails authentication instead of decrypting into a plausible lie.
* **Nonces are random, never counters.** A 96-bit random nonce with a fresh DEK
  per record makes GCM nonce reuse a non-event; there is no shared counter to
  desynchronise across gateway replicas.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Final, Mapping, Optional, Protocol, Tuple

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .config import CryptoSettings

__all__ = [
    "CryptoError",
    "DecryptionError",
    "SaltMaterial",
    "SaltSource",
    "DerivedSaltSource",
    "EphemeralSaltSource",
    "RotatingSaltManager",
    "Pseudonym",
    "PlatePseudonymizer",
    "SealedEnvelope",
    "AesGcmEnvelopeEncryptor",
    "DpdpCryptoPipeline",
]

GCM_NONCE_BYTES: Final[int] = 12
DEK_BYTES: Final[int] = 32
SALT_BYTES: Final[int] = 32
_HKDF_INFO_PREFIX: Final[bytes] = b"veritrack/dpdp/pseudonym-salt/v1"
_HKDF_DEK_INFO: Final[bytes] = b"veritrack/dpdp/dek-wrap/v1"


class CryptoError(RuntimeError):
    """Base class for cryptographic faults in the Stage 2 pipeline."""


class DecryptionError(CryptoError):
    """Raised when authenticated decryption fails (tampering, wrong key, wrong AAD)."""


@dataclass(frozen=True, slots=True)
class SaltMaterial:
    """One rotation period's pseudonymisation salt."""

    epoch: int
    salt: bytes
    valid_from_epoch_s: int
    valid_until_epoch_s: int

    def covers(self, epoch_s: float, grace_s: int) -> bool:
        return self.valid_from_epoch_s <= epoch_s < (self.valid_until_epoch_s + grace_s)


class SaltSource(Protocol):
    """Strategy for materialising the salt of a given rotation epoch."""

    def materialise(self, epoch: int, rotation_seconds: int) -> SaltMaterial:
        ...


class DerivedSaltSource:
    """Derives each epoch's salt from a long-lived root pepper via HKDF-SHA256.

    Deterministic derivation is what lets a horizontally-scaled fleet of gateway
    replicas -- and a replica that restarted five minutes ago -- agree on the
    same pseudonym for the same plate on the same day, without any salt
    replication protocol or shared mutable state.

    The trade-off is honest: anyone holding the pepper can recompute any past
    epoch's salt and therefore re-derive a pseudonym from a guessed plate. The
    plate space is small enough to brute-force, so the pepper is the whole ball
    game. It lives in the gateway's secret store (HSM / KMS / sealed secret),
    never on an edge node, and never in the database that holds the digests.
    """

    __slots__ = ("_pepper",)

    def __init__(self, pepper: bytes) -> None:
        if len(pepper) < 32:
            raise CryptoError("pseudonymisation pepper must be at least 32 bytes")
        self._pepper = pepper

    def materialise(self, epoch: int, rotation_seconds: int) -> SaltMaterial:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=SALT_BYTES,
            salt=None,
            info=_HKDF_INFO_PREFIX + b"|" + str(epoch).encode("ascii"),
        )
        salt = hkdf.derive(self._pepper)
        start = epoch * rotation_seconds
        return SaltMaterial(
            epoch=epoch,
            salt=salt,
            valid_from_epoch_s=start,
            valid_until_epoch_s=start + rotation_seconds,
        )


class EphemeralSaltSource:
    """Generates a fresh random salt per epoch and keeps it only in memory.

    This is the maximally privacy-preserving mode: once the process exits, the
    salt is gone and the stored digests are unlinkable to any plate by anyone,
    including the operator. It costs horizontal scalability -- two replicas
    would disagree -- so it suits a single-instance deployment or a compliance
    demonstration rather than a city-scale fleet.
    """

    __slots__ = ("_salts", "_lock")

    def __init__(self) -> None:
        self._salts: Dict[int, bytes] = {}
        self._lock = threading.Lock()

    def materialise(self, epoch: int, rotation_seconds: int) -> SaltMaterial:
        with self._lock:
            salt = self._salts.get(epoch)
            if salt is None:
                salt = secrets.token_bytes(SALT_BYTES)
                self._salts[epoch] = salt
                # Bound growth: an epoch older than three periods can never be
                # referenced again once the grace window has closed.
                for stale in [key for key in self._salts if key < epoch - 3]:
                    del self._salts[stale]
        start = epoch * rotation_seconds
        return SaltMaterial(
            epoch=epoch,
            salt=salt,
            valid_from_epoch_s=start,
            valid_until_epoch_s=start + rotation_seconds,
        )


class RotatingSaltManager:
    """Thread-safe cache of the active and recently-retired salts.

    A sighting is pseudonymised under the salt of *its own timestamp*, not of
    wall-clock now. An edge node that buffered passes through a network outage
    and flushes them after midnight must still produce the digests that its
    same-day peers produced, or a single vehicle would fracture into two
    pseudonyms across the rotation boundary. ``grace_seconds`` bounds how far
    back that is permitted to reach.
    """

    __slots__ = ("_source", "_rotation_seconds", "_grace_seconds", "_cache", "_lock")

    def __init__(
        self,
        source: SaltSource,
        *,
        rotation_hours: int = 24,
        grace_hours: int = 2,
    ) -> None:
        if rotation_hours < 1:
            raise CryptoError("rotation_hours must be >= 1")
        self._source = source
        self._rotation_seconds = rotation_hours * 3600
        self._grace_seconds = max(0, grace_hours) * 3600
        self._cache: Dict[int, SaltMaterial] = {}
        self._lock = threading.RLock()

    @property
    def rotation_seconds(self) -> int:
        return self._rotation_seconds

    @property
    def grace_seconds(self) -> int:
        return self._grace_seconds

    def epoch_for(self, epoch_s: float) -> int:
        """Rotation index containing ``epoch_s`` (UTC seconds since the Unix epoch)."""
        return int(epoch_s // self._rotation_seconds)

    def salt_for(self, epoch_s: float) -> SaltMaterial:
        """Return the salt governing the instant ``epoch_s``."""
        epoch = self.epoch_for(epoch_s)
        with self._lock:
            material = self._cache.get(epoch)
            if material is None:
                material = self._source.materialise(epoch, self._rotation_seconds)
                self._cache[epoch] = material
                self._evict_locked(epoch)
        return material

    def current(self) -> SaltMaterial:
        return self.salt_for(time.time())

    def is_within_grace(self, epoch_s: float, *, now_s: Optional[float] = None) -> bool:
        """True when a timestamp is recent enough to pseudonymise consistently."""
        reference = time.time() if now_s is None else now_s
        material = self.salt_for(epoch_s)
        return material.covers(reference, self._grace_seconds) or epoch_s <= reference

    def _evict_locked(self, current_epoch: int) -> None:
        horizon = current_epoch - 3
        for stale in [epoch for epoch in self._cache if epoch < horizon]:
            del self._cache[stale]


@dataclass(frozen=True, slots=True)
class Pseudonym:
    """The irreversible storage form of a non-hotlist registration number."""

    digest_hex: str
    prefix: str
    salt_epoch: int

    def __repr__(self) -> str:  # pragma: no cover - defensive, avoids leaking material
        return f"Pseudonym(prefix={self.prefix!r}, salt_epoch={self.salt_epoch})"


class PlatePseudonymizer:
    """HMAC-SHA256 pseudonymiser over a rotating salt.

    HMAC rather than a bare hash because the salt is a *key*, not a public
    parameter: HMAC's key-prefix construction is what makes recovering the salt
    from observed digests infeasible, which a naive ``sha256(salt || plate)``
    would expose to length-extension structure.
    """

    __slots__ = ("_salts", "_prefix_len")

    def __init__(self, salt_manager: RotatingSaltManager, *, prefix_len: int = 12) -> None:
        if not 4 <= prefix_len <= 64:
            raise CryptoError("prefix_len must lie in [4, 64]")
        self._salts = salt_manager
        self._prefix_len = prefix_len

    @property
    def salt_manager(self) -> RotatingSaltManager:
        return self._salts

    def pseudonymize(self, plate: str, *, observed_epoch_s: float) -> Pseudonym:
        """Irreversibly map ``plate`` to a digest under the salt of its own timestamp."""
        normalised = plate.strip().upper()
        if not normalised:
            raise CryptoError("cannot pseudonymise an empty plate")
        material = self._salts.salt_for(observed_epoch_s)
        digest = hmac.new(material.salt, normalised.encode("utf-8"), hashlib.sha256).hexdigest()
        return Pseudonym(
            digest_hex=digest,
            prefix=digest[: self._prefix_len],
            salt_epoch=material.epoch,
        )

    def matches(self, plate: str, pseudonym: Pseudonym, *, observed_epoch_s: float) -> bool:
        """Constant-time check that ``plate`` produces ``pseudonym``.

        Used only by the warrant-backed re-identification path, where a court
        order supplies the cleartext plate and the operator must locate the
        corresponding pseudonymised rows.
        """
        candidate = self.pseudonymize(plate, observed_epoch_s=observed_epoch_s)
        return hmac.compare_digest(candidate.digest_hex, pseudonym.digest_hex)


@dataclass(frozen=True, slots=True)
class SealedEnvelope:
    """An AES-256-GCM envelope: wrapped DEK plus authenticated ciphertext."""

    key_version: int
    wrapped_dek_b64: str
    dek_nonce_b64: str
    ciphertext_b64: str
    nonce_b64: str
    aad: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_version": self.key_version,
            "wrapped_dek": self.wrapped_dek_b64,
            "dek_nonce": self.dek_nonce_b64,
            "ciphertext": self.ciphertext_b64,
            "nonce": self.nonce_b64,
            "aad": self.aad,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SealedEnvelope":
        try:
            return cls(
                key_version=int(payload["key_version"]),
                wrapped_dek_b64=str(payload["wrapped_dek"]),
                dek_nonce_b64=str(payload["dek_nonce"]),
                ciphertext_b64=str(payload["ciphertext"]),
                nonce_b64=str(payload["nonce"]),
                aad=str(payload["aad"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DecryptionError(f"malformed sealed envelope: {exc}") from exc

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "SealedEnvelope":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DecryptionError("sealed envelope is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise DecryptionError("sealed envelope must be a JSON object")
        return cls.from_dict(payload)


class AesGcmEnvelopeEncryptor:
    """AES-256-GCM envelope encryption for hotlist evidence metadata."""

    __slots__ = ("_kek", "_key_version")

    def __init__(self, master_key: bytes, *, key_version: int = 1) -> None:
        if len(master_key) != DEK_BYTES:
            raise CryptoError(f"master key must be exactly {DEK_BYTES} bytes for AES-256")
        if key_version < 1:
            raise CryptoError("key_version must be >= 1")
        self._kek = master_key
        self._key_version = key_version

    @property
    def key_version(self) -> int:
        return self._key_version

    def _wrapping_key(self, key_version: int) -> bytes:
        """Domain-separate the wrapping key from the raw KEK.

        Using HKDF here means the raw master secret is never itself an AES key,
        so a future need for a second purpose (say, index blinding) can derive
        an independent key from the same root without key reuse across
        algorithms.
        """
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=DEK_BYTES,
            salt=None,
            info=_HKDF_DEK_INFO + b"|" + str(key_version).encode("ascii"),
        )
        return hkdf.derive(self._kek)

    def seal(self, plaintext: bytes, *, aad: str) -> SealedEnvelope:
        """Encrypt ``plaintext`` under a fresh DEK, wrapping the DEK with the KEK."""
        if not isinstance(plaintext, (bytes, bytearray)):
            raise CryptoError("plaintext must be bytes")
        dek = secrets.token_bytes(DEK_BYTES)
        aad_bytes = aad.encode("utf-8")

        record_nonce = os.urandom(GCM_NONCE_BYTES)
        ciphertext = AESGCM(dek).encrypt(record_nonce, bytes(plaintext), aad_bytes)

        dek_nonce = os.urandom(GCM_NONCE_BYTES)
        wrapped_dek = AESGCM(self._wrapping_key(self._key_version)).encrypt(
            dek_nonce, dek, aad_bytes
        )

        return SealedEnvelope(
            key_version=self._key_version,
            wrapped_dek_b64=base64.b64encode(wrapped_dek).decode("ascii"),
            dek_nonce_b64=base64.b64encode(dek_nonce).decode("ascii"),
            ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
            nonce_b64=base64.b64encode(record_nonce).decode("ascii"),
            aad=aad,
        )

    def open(self, envelope: SealedEnvelope, *, aad: Optional[str] = None) -> bytes:
        """Authenticate and decrypt a sealed envelope.

        ``aad`` defaults to the value recorded in the envelope, but a caller
        that knows which row it fetched should pass it explicitly: that is what
        detects a ciphertext transplanted between records.
        """
        expected_aad = envelope.aad if aad is None else aad
        if not hmac.compare_digest(expected_aad, envelope.aad):
            raise DecryptionError("AAD mismatch: envelope does not belong to this record")
        aad_bytes = expected_aad.encode("utf-8")

        try:
            wrapped_dek = base64.b64decode(envelope.wrapped_dek_b64, validate=True)
            dek_nonce = base64.b64decode(envelope.dek_nonce_b64, validate=True)
            ciphertext = base64.b64decode(envelope.ciphertext_b64, validate=True)
            record_nonce = base64.b64decode(envelope.nonce_b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise DecryptionError("sealed envelope contains invalid base64") from exc

        try:
            dek = AESGCM(self._wrapping_key(envelope.key_version)).decrypt(
                dek_nonce, wrapped_dek, aad_bytes
            )
        except InvalidTag as exc:
            raise DecryptionError("DEK unwrap failed: wrong KEK version or tampered envelope") from exc

        try:
            return AESGCM(dek).decrypt(record_nonce, ciphertext, aad_bytes)
        except InvalidTag as exc:
            raise DecryptionError("ciphertext authentication failed") from exc

    def seal_json(self, payload: Mapping[str, Any], *, aad: str) -> SealedEnvelope:
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return self.seal(raw, aad=aad)

    def open_json(self, envelope: SealedEnvelope, *, aad: Optional[str] = None) -> Dict[str, Any]:
        decoded = json.loads(self.open(envelope, aad=aad).decode("utf-8"))
        if not isinstance(decoded, dict):
            raise DecryptionError("sealed payload was not a JSON object")
        return decoded


class DpdpCryptoPipeline:
    """Facade wiring the salt manager, pseudonymiser and envelope encryptor.

    One object to inject into the gateway, one object to stub in tests.
    """

    __slots__ = ("_pseudonymizer", "_encryptor", "_settings")

    def __init__(
        self,
        pseudonymizer: PlatePseudonymizer,
        encryptor: AesGcmEnvelopeEncryptor,
        settings: Optional[CryptoSettings] = None,
    ) -> None:
        self._pseudonymizer = pseudonymizer
        self._encryptor = encryptor
        self._settings = settings

    @classmethod
    def from_settings(
        cls, settings: CryptoSettings, *, ephemeral_salts: bool = False
    ) -> "DpdpCryptoPipeline":
        source: SaltSource = (
            EphemeralSaltSource() if ephemeral_salts else DerivedSaltSource(settings.pepper_bytes())
        )
        manager = RotatingSaltManager(
            source,
            rotation_hours=settings.salt_rotation_hours,
            grace_hours=settings.salt_grace_hours,
        )
        pseudonymizer = PlatePseudonymizer(manager, prefix_len=settings.pseudonym_prefix_len)
        encryptor = AesGcmEnvelopeEncryptor(
            settings.master_key_bytes(), key_version=settings.aes_key_version
        )
        return cls(pseudonymizer, encryptor, settings)

    @property
    def pseudonymizer(self) -> PlatePseudonymizer:
        return self._pseudonymizer

    @property
    def encryptor(self) -> AesGcmEnvelopeEncryptor:
        return self._encryptor

    @property
    def salt_manager(self) -> RotatingSaltManager:
        return self._pseudonymizer.salt_manager

    def pseudonymize(self, plate: str, *, observed_epoch_s: float) -> Pseudonym:
        return self._pseudonymizer.pseudonymize(plate, observed_epoch_s=observed_epoch_s)

    def seal_evidence(self, evidence: Mapping[str, Any], *, pass_id: str) -> SealedEnvelope:
        """Seal hotlist evidence metadata, bound to its own ``pass_id``."""
        return self._encryptor.seal_json(evidence, aad=pass_id)

    def open_evidence(self, envelope: SealedEnvelope, *, pass_id: str) -> Dict[str, Any]:
        return self._encryptor.open_json(envelope, aad=pass_id)

    def describe(self) -> Tuple[int, int]:
        """(current salt epoch, KEK version) -- for /healthz, never the material itself."""
        return self.salt_manager.current().epoch, self._encryptor.key_version
