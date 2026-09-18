"""Configuration for the VeriTrack Stage 2 central ingestion server.

Every knob is environment-driven (``VERITRACK_*``) so that the same image can be
promoted from developer laptop to staging to production without a rebuild.
Secrets are typed as :class:`~pydantic.SecretStr` so they never leak into a
traceback, a log line or a ``repr()``.

Unlike the Stage 1 edge package -- which deliberately uses frozen dataclasses to
keep a hot inference loop free of validation overhead -- the gateway validates
aggressively. Stage 2 is the trust boundary: every byte arriving here came from a
pole-mounted device that a determined attacker can physically reach.
"""

from __future__ import annotations

import base64
import binascii
from functools import lru_cache
from typing import Final, List, Literal, Optional

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "DatabaseSettings",
    "RedisSettings",
    "KafkaSettings",
    "CryptoSettings",
    "BloomSettings",
    "IngestSettings",
    "ServerSettings",
    "Settings",
    "get_settings",
]

AES_256_KEY_BYTES: Final[int] = 32
MIN_PEPPER_BYTES: Final[int] = 32


def _decode_b64_key(raw: str, *, expected_len: int, field_name: str) -> bytes:
    """Decode a base64 secret and assert its exact decoded length."""
    try:
        material = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field_name} must be valid base64") from exc
    if len(material) != expected_len:
        raise ValueError(
            f"{field_name} must decode to exactly {expected_len} bytes, got {len(material)}"
        )
    return material


class DatabaseSettings(BaseSettings):
    """Async TimescaleDB / PostGIS connection parameters."""

    model_config = SettingsConfigDict(env_prefix="VERITRACK_DB_", extra="ignore")

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = "veritrack"
    user: str = "veritrack"
    password: SecretStr = SecretStr("veritrack")

    min_pool_size: int = Field(default=4, ge=1, le=256)
    max_pool_size: int = Field(default=32, ge=1, le=512)
    command_timeout_s: float = Field(default=10.0, gt=0.0)
    connect_timeout_s: float = Field(default=5.0, gt=0.0)

    statement_cache_size: int = Field(default=256, ge=0)
    server_settings: dict[str, str] = Field(
        default_factory=lambda: {
            "application_name": "veritrack-gateway",
            # The ingest path is append-only and latency sensitive; a slightly
            # relaxed synchronous_commit trades <1s of durability on a hard node
            # loss for a very large write-throughput win. Hotlist hits are
            # written on a separate connection with synchronous_commit forced ON.
            "synchronous_commit": "off",
            "jit": "off",
        }
    )

    @model_validator(mode="after")
    def _pool_sizes_consistent(self) -> "DatabaseSettings":
        if self.max_pool_size < self.min_pool_size:
            raise ValueError("max_pool_size must be >= min_pool_size")
        return self

    @property
    def dsn(self) -> str:
        """Build a libpq DSN. The password is unwrapped only at connect time."""
        return (
            f"postgresql://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def safe_dsn(self) -> str:
        """A log-safe DSN with the password elided."""
        return f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.database}"


class RedisSettings(BaseSettings):
    """Redis connection parameters for the hotlist Bloom filter and metadata."""

    model_config = SettingsConfigDict(env_prefix="VERITRACK_REDIS_", extra="ignore")

    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: Optional[SecretStr] = None
    use_tls: bool = False

    socket_timeout_s: float = Field(default=0.25, gt=0.0)
    socket_connect_timeout_s: float = Field(default=1.0, gt=0.0)
    max_connections: int = Field(default=64, ge=1)

    #: When True the process refuses to start if Redis is unreachable. When
    #: False the hotlist engine degrades to its in-memory backend, which is the
    #: correct behaviour for CI and single-box demos but never for production.
    required: bool = True

    @property
    def url(self) -> str:
        scheme = "rediss" if self.use_tls else "redis"
        auth = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"

    @property
    def safe_url(self) -> str:
        scheme = "rediss" if self.use_tls else "redis"
        auth = ":***@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


class KafkaSettings(BaseSettings):
    """Downstream event-bus dispatch settings."""

    model_config = SettingsConfigDict(env_prefix="VERITRACK_KAFKA_", extra="ignore")

    enabled: bool = False
    bootstrap_servers: List[str] = Field(default_factory=lambda: ["localhost:9092"])
    sightings_topic: str = "veritrack.sightings.enriched"
    alerts_topic: str = "veritrack.alerts.hotlist"
    client_id: str = "veritrack-gateway"
    compression: Literal["none", "gzip", "lz4", "snappy", "zstd"] = "lz4"
    acks: Literal["0", "1", "all"] = "all"
    linger_ms: int = Field(default=20, ge=0, le=10_000)
    max_queue_size: int = Field(default=20_000, ge=128)

    @field_validator("bootstrap_servers", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


class CryptoSettings(BaseSettings):
    """DPDP Act 2023 cryptographic parameters.

    ``salt_pepper`` is the long-lived root secret from which each rotation
    period's pseudonymisation salt is derived via HKDF. It lives only in the
    gateway's secret store -- never on an edge node, because a pole-mounted
    device is physically reachable and compromising it must not compromise the
    pseudonymisation of the entire city.

    ``aes_master_key`` is the Key Encryption Key. Record payloads are sealed
    with a fresh per-record Data Encryption Key which is itself wrapped by the
    KEK, so a KEK rotation re-wraps a small number of DEKs rather than
    re-encrypting the whole evidence corpus.
    """

    model_config = SettingsConfigDict(env_prefix="VERITRACK_CRYPTO_", extra="ignore")

    #: base64 of 32 raw bytes.
    aes_master_key: SecretStr = SecretStr(base64.b64encode(b"\x00" * AES_256_KEY_BYTES).decode())
    aes_key_version: int = Field(default=1, ge=1)

    #: base64 of >= 32 raw bytes.
    salt_pepper: SecretStr = SecretStr(base64.b64encode(b"\x00" * MIN_PEPPER_BYTES).decode())

    salt_rotation_hours: int = Field(default=24, ge=1, le=168)
    #: How long a retired salt stays usable. Sightings buffered on an edge node
    #: across a rotation boundary must still pseudonymise to the value their
    #: timestamp implies, so the manager keeps a short grace window.
    salt_grace_hours: int = Field(default=2, ge=0, le=24)

    #: Characters of the HMAC hex digest stored alongside the full hash. The
    #: prefix supports cheap prefix-bucketed analytics without ever widening
    #: the identifiability of the full digest.
    pseudonym_prefix_len: int = Field(default=12, ge=4, le=32)

    #: Refuse to boot with the all-zero development defaults.
    allow_insecure_defaults: bool = False

    @field_validator("aes_master_key")
    @classmethod
    def _validate_master_key(cls, value: SecretStr) -> SecretStr:
        _decode_b64_key(
            value.get_secret_value(), expected_len=AES_256_KEY_BYTES, field_name="aes_master_key"
        )
        return value

    @field_validator("salt_pepper")
    @classmethod
    def _validate_pepper(cls, value: SecretStr) -> SecretStr:
        try:
            material = base64.b64decode(value.get_secret_value(), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("salt_pepper must be valid base64") from exc
        if len(material) < MIN_PEPPER_BYTES:
            raise ValueError(f"salt_pepper must decode to >= {MIN_PEPPER_BYTES} bytes")
        return value

    @model_validator(mode="after")
    def _reject_placeholder_secrets(self) -> "CryptoSettings":
        if self.allow_insecure_defaults:
            return self
        key = _decode_b64_key(
            self.aes_master_key.get_secret_value(),
            expected_len=AES_256_KEY_BYTES,
            field_name="aes_master_key",
        )
        pepper = base64.b64decode(self.salt_pepper.get_secret_value(), validate=True)
        if not any(key):
            raise ValueError(
                "aes_master_key is the all-zero development placeholder; set "
                "VERITRACK_CRYPTO_AES_MASTER_KEY or enable allow_insecure_defaults"
            )
        if not any(pepper):
            raise ValueError(
                "salt_pepper is the all-zero development placeholder; set "
                "VERITRACK_CRYPTO_SALT_PEPPER or enable allow_insecure_defaults"
            )
        return self

    def master_key_bytes(self) -> bytes:
        return _decode_b64_key(
            self.aes_master_key.get_secret_value(),
            expected_len=AES_256_KEY_BYTES,
            field_name="aes_master_key",
        )

    def pepper_bytes(self) -> bytes:
        return base64.b64decode(self.salt_pepper.get_secret_value(), validate=True)


class BloomSettings(BaseSettings):
    """Hotlist Bloom filter sizing."""

    model_config = SettingsConfigDict(env_prefix="VERITRACK_BLOOM_", extra="ignore")

    key: str = "veritrack:hotlist:bloom"
    meta_key_prefix: str = "veritrack:hotlist:meta:"
    capacity: int = Field(default=20_000_000, ge=1_000)
    error_rate: float = Field(default=1e-4, gt=0.0, lt=1.0)
    #: Ceiling on the derived bit-array size, guarding against a typo in
    #: ``capacity`` allocating a multi-gigabyte Redis string.
    max_bits: int = Field(default=1 << 32, ge=1 << 16)
    #: Fall back to the in-process bitset when Redis is unavailable.
    allow_memory_fallback: bool = True


class IngestSettings(BaseSettings):
    """Ingest pipeline shaping: batching, buffering and backpressure."""

    model_config = SettingsConfigDict(env_prefix="VERITRACK_INGEST_", extra="ignore")

    max_batch_observations: int = Field(default=512, ge=1, le=10_000)
    #: Rows accumulated before the writer flushes via COPY.
    flush_row_threshold: int = Field(default=500, ge=1, le=50_000)
    #: Wall-clock ceiling on how long a row may sit unflushed.
    flush_interval_s: float = Field(default=1.0, gt=0.0, le=60.0)
    #: Buffer depth before the gateway starts shedding with HTTP 503.
    max_buffered_rows: int = Field(default=50_000, ge=1_000)

    #: Reject observations whose timestamp is further than this into the future
    #: (clock skew on an unsynchronised edge node) or the past (replay).
    max_clock_skew_s: float = Field(default=120.0, gt=0.0)
    max_observation_age_s: float = Field(default=86_400.0, gt=0.0)

    #: Sub-50ms budget for the hotlist decision, enforced per request.
    hotlist_timeout_s: float = Field(default=0.05, gt=0.0, le=5.0)


class ServerSettings(BaseSettings):
    """HTTP surface configuration."""

    model_config = SettingsConfigDict(env_prefix="VERITRACK_SERVER_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    root_path: str = ""
    cors_origins: List[str] = Field(default_factory=list)

    #: Per-node token bucket. Edge nodes emit roughly one pass per vehicle, so a
    #: single camera on a busy arterial peaks near 5 req/s; the default leaves
    #: two orders of headroom while still stopping a looping or hostile node.
    rate_limit_per_second: float = Field(default=200.0, gt=0.0)
    rate_limit_burst: float = Field(default=800.0, gt=0.0)
    rate_limit_enabled: bool = True

    #: Shared bearer token presented by edge nodes. Empty disables auth, which
    #: is acceptable only behind mTLS or on a closed lab network.
    ingest_api_key: Optional[SecretStr] = None

    request_timeout_s: float = Field(default=15.0, gt=0.0)
    enable_docs: bool = True


class Settings(BaseSettings):
    """Root settings aggregate."""

    model_config = SettingsConfigDict(
        env_prefix="VERITRACK_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Literal["dev", "staging", "prod"] = "dev"
    node_region: str = "IN-MH"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    crypto: CryptoSettings = Field(default_factory=CryptoSettings)
    bloom: BloomSettings = Field(default_factory=BloomSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @model_validator(mode="after")
    def _production_hardening(self) -> "Settings":
        if self.environment != "prod":
            return self
        if self.crypto.allow_insecure_defaults:
            raise ValueError("allow_insecure_defaults must be False in prod")
        if self.server.ingest_api_key is None:
            raise ValueError("server.ingest_api_key is mandatory in prod")
        if not self.redis.required:
            raise ValueError("redis.required must be True in prod: hotlist cannot be in-memory")
        if self.server.enable_docs:
            raise ValueError("server.enable_docs must be False in prod")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton, safe to use as a FastAPI dependency."""
    return Settings()


def build_dev_settings() -> Settings:
    """Deterministic settings for tests and local runs.

    Uses fixed non-zero key material so that pseudonymisation is reproducible
    across a test session without tripping the placeholder guard.
    """
    return Settings(
        environment="dev",
        crypto=CryptoSettings(
            aes_master_key=SecretStr(base64.b64encode(bytes(range(32))).decode()),
            salt_pepper=SecretStr(base64.b64encode(bytes(range(64, 128))).decode()),
            allow_insecure_defaults=False,
        ),
        redis=RedisSettings(required=False),
    )
