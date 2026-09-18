-- =====================================================================
-- VeriTrack Stage 2 -- Central Storage Schema
-- TimescaleDB 2.x + PostGIS 3.x on PostgreSQL 14+
--
-- Smart India Hackathon PS 26127 (Bharat Electronics Limited)
--
-- Idempotent: safe to run repeatedly against an existing database.
--
-- Statutory posture (DPDP Act 2023):
--   * `sightings` holds NO cleartext registration number. Ever. The column
--     does not exist, so no bug, no misconfigured ORM and no ad-hoc query can
--     put one there. Plates arrive as HMAC-SHA256 digests under a salt that
--     rotates every 24 hours.
--   * `hotlist_hits` holds cleartext, but only for vehicles under an active
--     FIR/warrant, and its evidence metadata is AES-256-GCM sealed.
--   * A 30-day automated retention policy drops whole chunks of `sightings`.
--     Chunk-level DROP is a metadata operation, not a 20-million-row DELETE,
--     so purge is O(1) and leaves no dead tuples for VACUUM to chase.
-- =====================================================================

BEGIN;

-- ---------------------------------------------------------------------
-- 1. Extensions
-- ---------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- ---------------------------------------------------------------------
-- 2. Enumerated domains
-- ---------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'vehicle_class_t') THEN
        CREATE TYPE vehicle_class_t AS ENUM (
            'two_wheeler', 'three_wheeler', 'car', 'lcv',
            'truck', 'bus', 'tractor', 'emergency', 'unknown'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'plate_series_t') THEN
        CREATE TYPE plate_series_t AS ENUM (
            'private_white', 'commercial_yellow', 'ev_green',
            'government', 'diplomatic', 'unknown'
        );
    END IF;
END
$$;

-- ---------------------------------------------------------------------
-- 3. Reference tables (small, not hypertables)
-- ---------------------------------------------------------------------

-- Camera registry. `location` is geography(Point, 4326) so that distance
-- predicates are true metres on the spheroid rather than degrees, which is
-- what Stage 3's map matching and Stage 4's H3 binning both need.
CREATE TABLE IF NOT EXISTS cameras (
    camera_id           TEXT PRIMARY KEY,
    edge_device_id      TEXT NOT NULL,
    display_name        TEXT NOT NULL DEFAULT '',
    location            geography(Point, 4326) NOT NULL,
    heading_azimuth     DOUBLE PRECISION
                            CHECK (heading_azimuth IS NULL
                                   OR (heading_azimuth >= 0 AND heading_azimuth < 360)),
    corridor_id         TEXT,
    junction_id         TEXT,
    lane_count          SMALLINT CHECK (lane_count IS NULL OR lane_count > 0),
    free_flow_speed_kmh DOUBLE PRECISION
                            CHECK (free_flow_speed_kmh IS NULL OR free_flow_speed_kmh > 0),
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    installed_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_cameras_location_gist
    ON cameras USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_cameras_corridor
    ON cameras (corridor_id) WHERE corridor_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cameras_device
    ON cameras (edge_device_id);

-- Road corridors, used by the 5-minute speed aggregate and the Stage 4
-- Corridor Performance Index.
CREATE TABLE IF NOT EXISTS corridors (
    corridor_id         TEXT PRIMARY KEY,
    display_name        TEXT NOT NULL DEFAULT '',
    geometry            geography(LineString, 4326),
    length_m            DOUBLE PRECISION CHECK (length_m IS NULL OR length_m > 0),
    free_flow_speed_kmh DOUBLE PRECISION
                            CHECK (free_flow_speed_kmh IS NULL OR free_flow_speed_kmh > 0),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_corridors_geometry_gist
    ON corridors USING GIST (geometry);

-- Active warrant register. The lawful basis for retaining a cleartext plate.
CREATE TABLE IF NOT EXISTS hotlist_registry (
    plate_number        TEXT PRIMARY KEY,
    fir_number          TEXT NOT NULL,
    warrant_reference   TEXT NOT NULL,
    severity            SMALLINT NOT NULL CHECK (severity BETWEEN 1 AND 4),
    offence_category    TEXT NOT NULL DEFAULT 'UNSPECIFIED',
    issuing_authority   TEXT NOT NULL DEFAULT 'UNKNOWN',
    registered_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_hotlist_registry_active
    ON hotlist_registry (is_active, expires_at)
    WHERE is_active;
CREATE INDEX IF NOT EXISTS idx_hotlist_registry_fir
    ON hotlist_registry (fir_number);

-- ---------------------------------------------------------------------
-- 4. Sightings hypertable -- the pseudonymised bulk stream
-- ---------------------------------------------------------------------
--
-- `latitude`/`longitude` are stored as plain columns and `geom` is a STORED
-- generated column. This is deliberate: binary COPY cannot invoke a function,
-- so a generated column is what lets the high-throughput ingest path use COPY
-- while still producing a real PostGIS geography for spatial queries.
--
-- `reid_embedding` is REAL[] (float4). At 128 dimensions that is 512 bytes +
-- 24 bytes of array header. A pgvector migration is a column type change and
-- an index rebuild when approximate-nearest-neighbour search is needed at
-- Stage 3 scale; REAL[] keeps the dependency surface minimal for now.

CREATE TABLE IF NOT EXISTS sightings (
    timestamp_utc               TIMESTAMPTZ      NOT NULL,
    pass_id                     TEXT             NOT NULL,
    edge_device_id              TEXT             NOT NULL,
    camera_id                   TEXT             NOT NULL,
    tracklet_id                 TEXT             NOT NULL,

    -- DPDP: irreversible pseudonym only. No cleartext column exists.
    plate_pseudonym             TEXT             NOT NULL,
    plate_pseudonym_prefix      TEXT             NOT NULL,
    salt_epoch                  BIGINT           NOT NULL,

    vehicle_class               vehicle_class_t  NOT NULL DEFAULT 'unknown',
    plate_series                plate_series_t   NOT NULL DEFAULT 'unknown',

    plate_sequence_confidence   REAL             NOT NULL
                                    CHECK (plate_sequence_confidence BETWEEN 0 AND 1),
    min_character_confidence    REAL             NOT NULL
                                    CHECK (min_character_confidence BETWEEN 0 AND 1),
    text_entropy                REAL CHECK (text_entropy IS NULL
                                            OR text_entropy BETWEEN 0 AND 1),
    repair_cost_nats            REAL             NOT NULL DEFAULT 0
                                    CHECK (repair_cost_nats >= 0),
    is_valid_format             BOOLEAN          NOT NULL DEFAULT FALSE,
    is_dual_line_plate          BOOLEAN          NOT NULL DEFAULT FALSE,

    vehicle_speed_kmh           REAL CHECK (vehicle_speed_kmh IS NULL
                                            OR vehicle_speed_kmh BETWEEN 0 AND 400),
    travel_heading_azimuth      REAL CHECK (travel_heading_azimuth IS NULL
                                            OR (travel_heading_azimuth >= 0
                                                AND travel_heading_azimuth < 360)),

    reid_embedding              REAL[]           NOT NULL
                                    CHECK (array_length(reid_embedding, 1) = 128),

    latitude                    DOUBLE PRECISION CHECK (latitude IS NULL
                                                        OR latitude BETWEEN -90 AND 90),
    longitude                   DOUBLE PRECISION CHECK (longitude IS NULL
                                                        OR longitude BETWEEN -180 AND 180),
    geom geography(Point, 4326) GENERATED ALWAYS AS (
        CASE
            WHEN latitude IS NULL OR longitude IS NULL THEN NULL
            ELSE ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
        END
    ) STORED,

    corridor_id                 TEXT,
    hotlist_flag                BOOLEAN          NOT NULL DEFAULT FALSE,
    ingest_degraded             BOOLEAN          NOT NULL DEFAULT FALSE,
    ingested_at                 TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

-- 1-day chunk interval: with a 30-day retention window this yields 30 live
-- chunks, which keeps the chunk-exclusion planning cost negligible while
-- making the retention drop granular enough to be precise to the day.
SELECT create_hypertable(
    'sightings',
    'timestamp_utc',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE,
    migrate_data        => TRUE
);

-- Deduplication. Stage 1 derives pass_id as
-- blake2b(node | camera | track | first_ts), so an at-least-once Kafka or HTTP
-- retry produces a byte-identical key and collides here instead of duplicating.
-- The partitioning column must participate in any unique index on a hypertable.
CREATE UNIQUE INDEX IF NOT EXISTS uq_sightings_pass
    ON sightings (pass_id, timestamp_utc);

-- Compound B-tree indexes, each with timestamp_utc DESC because every analytic
-- query is "recent history for <key>".
CREATE INDEX IF NOT EXISTS idx_sightings_pseudonym_time
    ON sightings (plate_pseudonym, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_sightings_camera_time
    ON sightings (camera_id, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_sightings_device_time
    ON sightings (edge_device_id, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_sightings_corridor_time
    ON sightings (corridor_id, timestamp_utc DESC)
    WHERE corridor_id IS NOT NULL;

-- Prefix-bucketed scans for cohort analytics that must never touch the full
-- digest (the prefix is 12 hex characters = 48 bits of the HMAC).
CREATE INDEX IF NOT EXISTS idx_sightings_prefix_time
    ON sightings (plate_pseudonym_prefix, timestamp_utc DESC);

-- Spatial. GiST over the generated geography column; combined with
-- timestamp_utc via btree_gist so a bounded-time, bounded-radius query
-- (exactly what Stage 3 trajectory reconstruction issues) is one index scan.
CREATE INDEX IF NOT EXISTS idx_sightings_geom_gist
    ON sightings USING GIST (geom)
    WHERE geom IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sightings_geom_time_gist
    ON sightings USING GIST (geom, timestamp_utc)
    WHERE geom IS NOT NULL;

-- Partial index over the small hotlist-flagged minority.
CREATE INDEX IF NOT EXISTS idx_sightings_hotlist_time
    ON sightings (timestamp_utc DESC)
    WHERE hotlist_flag;

-- ---------------------------------------------------------------------
-- 5. Hotlist hits -- cleartext under warrant, encrypted evidence
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hotlist_hits (
    timestamp_utc               TIMESTAMPTZ      NOT NULL,
    pass_id                     TEXT             NOT NULL,
    plate_number                TEXT             NOT NULL,
    edge_device_id              TEXT             NOT NULL,
    camera_id                   TEXT             NOT NULL,
    tracklet_id                 TEXT             NOT NULL,
    fir_number                  TEXT             NOT NULL,
    warrant_reference           TEXT             NOT NULL,
    severity                    SMALLINT         NOT NULL CHECK (severity BETWEEN 1 AND 4),
    offence_category            TEXT             NOT NULL DEFAULT 'UNSPECIFIED',
    issuing_authority           TEXT             NOT NULL DEFAULT 'UNKNOWN',
    plate_sequence_confidence   REAL             NOT NULL
                                    CHECK (plate_sequence_confidence BETWEEN 0 AND 1),
    vehicle_class               vehicle_class_t  NOT NULL DEFAULT 'unknown',
    vehicle_speed_kmh           REAL,
    travel_heading_azimuth      REAL,
    latitude                    DOUBLE PRECISION,
    longitude                   DOUBLE PRECISION,
    geom geography(Point, 4326) GENERATED ALWAYS AS (
        CASE
            WHEN latitude IS NULL OR longitude IS NULL THEN NULL
            ELSE ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
        END
    ) STORED,
    -- AES-256-GCM envelope as JSON: wrapped DEK, nonces, ciphertext, AAD.
    evidence_envelope           TEXT,
    evidence_key_version        SMALLINT,
    acknowledged_at             TIMESTAMPTZ,
    ingested_at                 TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'hotlist_hits',
    'timestamp_utc',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE,
    migrate_data        => TRUE
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_hotlist_hits_pass
    ON hotlist_hits (pass_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_hotlist_hits_plate_time
    ON hotlist_hits (plate_number, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_hotlist_hits_fir
    ON hotlist_hits (fir_number, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_hotlist_hits_severity_time
    ON hotlist_hits (severity DESC, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_hotlist_hits_geom_gist
    ON hotlist_hits USING GIST (geom) WHERE geom IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_hotlist_hits_unacked
    ON hotlist_hits (timestamp_utc DESC) WHERE acknowledged_at IS NULL;

-- ---------------------------------------------------------------------
-- 6. Non-repudiable audit log
-- ---------------------------------------------------------------------
-- Hash-chained: each row commits to its predecessor, so a deletion or an
-- edit anywhere in the chain invalidates every digest after it. This is the
-- substrate Stage 5's non-repudiation requirement builds on.
CREATE TABLE IF NOT EXISTS audit_log (
    timestamp_utc       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    audit_id            BIGSERIAL,
    actor               TEXT        NOT NULL,
    action              TEXT        NOT NULL,
    subject             TEXT        NOT NULL,
    warrant_reference   TEXT,
    detail              JSONB       NOT NULL DEFAULT '{}'::jsonb,
    previous_digest     TEXT,
    entry_digest        TEXT        NOT NULL,
    PRIMARY KEY (audit_id, timestamp_utc)
);

SELECT create_hypertable(
    'audit_log',
    'timestamp_utc',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists       => TRUE,
    migrate_data        => TRUE
);

CREATE INDEX IF NOT EXISTS idx_audit_actor_time
    ON audit_log (actor, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_audit_warrant
    ON audit_log (warrant_reference, timestamp_utc DESC)
    WHERE warrant_reference IS NOT NULL;

COMMIT;

-- ---------------------------------------------------------------------
-- 7. Compression
-- ---------------------------------------------------------------------
-- Columnar compression after 7 days. Segmenting by camera_id groups a
-- camera's rows into the same compressed batch, so a per-camera query
-- decompresses only the batches it needs; ordering by time DESC inside the
-- segment makes the delta encoding of the timestamp column near-optimal.
ALTER TABLE sightings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'camera_id, vehicle_class',
    timescaledb.compress_orderby   = 'timestamp_utc DESC'
);

SELECT add_compression_policy('sightings', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE hotlist_hits SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'plate_number',
    timescaledb.compress_orderby   = 'timestamp_utc DESC'
);

SELECT add_compression_policy('hotlist_hits', INTERVAL '30 days', if_not_exists => TRUE);

-- ---------------------------------------------------------------------
-- 8. DPDP retention -- 30-day automated chunk drop
-- ---------------------------------------------------------------------
-- Section 8(7) of the DPDP Act requires erasure once the purpose is served.
-- Dropping whole chunks is a catalogue operation: no row-by-row DELETE, no
-- bloat, no VACUUM backlog, and no window in which a partially-purged chunk
-- still holds recoverable data.
SELECT add_retention_policy('sightings', INTERVAL '30 days', if_not_exists => TRUE);

-- Hotlist evidence is retained far longer: it is held under a warrant, and its
-- disposal is a judicial decision, not a scheduler's. Explicitly NOT given a
-- retention policy here -- deletion goes through the audited case-closure path.

-- ---------------------------------------------------------------------
-- 9. Continuous aggregate -- 5-minute corridor speeds
-- ---------------------------------------------------------------------
-- Incrementally materialised: a refresh touches only the buckets invalidated
-- since the last run, so the cost tracks new data, not table size. This is the
-- feed for Stage 4's Corridor Performance Index, whose free-flow baseline is
-- the 85th percentile of observed speed.
--
-- percentile_agg() produces a partial aggregate that Timescale can roll up
-- across buckets, so an hourly or daily percentile can be derived from these
-- 5-minute partials without re-reading the raw hypertable.

CREATE MATERIALIZED VIEW IF NOT EXISTS corridor_speed_5m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(INTERVAL '5 minutes', s.timestamp_utc) AS bucket,
    s.corridor_id,
    s.camera_id,
    count(*)                                            AS vehicle_count,
    count(*) FILTER (WHERE s.vehicle_class IN ('truck', 'bus', 'lcv'))
                                                        AS heavy_vehicle_count,
    avg(s.vehicle_speed_kmh)                            AS mean_speed_kmh,
    min(s.vehicle_speed_kmh)                            AS min_speed_kmh,
    max(s.vehicle_speed_kmh)                            AS max_speed_kmh,
    stddev_samp(s.vehicle_speed_kmh)                    AS stddev_speed_kmh,
    percentile_agg(s.vehicle_speed_kmh)                 AS speed_percentiles,
    avg(s.plate_sequence_confidence)                    AS mean_plate_confidence,
    count(*) FILTER (WHERE NOT s.is_valid_format)       AS invalid_format_count,
    count(*) FILTER (WHERE s.ingest_degraded)           AS degraded_count
FROM sightings AS s
WHERE s.corridor_id IS NOT NULL
  AND s.vehicle_speed_kmh IS NOT NULL
GROUP BY bucket, s.corridor_id, s.camera_id
WITH NO DATA;

-- Refresh every minute over a trailing window. start_offset comfortably
-- exceeds the edge buffer-and-replay horizon so that late-arriving passes
-- still land in the correct bucket; end_offset holds the materialised edge one
-- bucket back from now so a bucket is never published half-filled.
SELECT add_continuous_aggregate_policy(
    'corridor_speed_5m',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists     => TRUE
);

-- The aggregate is the durable artefact, so it outlives the raw rows that
-- produced it: 30 days of raw sightings, 2 years of 5-minute corridor history.
-- It contains no pseudonym and no per-vehicle row, so retaining it raises no
-- DPDP question.
SELECT add_retention_policy('corridor_speed_5m', INTERVAL '2 years', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_corridor_speed_5m_corridor
    ON corridor_speed_5m (corridor_id, bucket DESC);

-- ---------------------------------------------------------------------
-- 10. Operational views
-- ---------------------------------------------------------------------

-- Live edge fleet health: last contact and recent throughput per device.
CREATE OR REPLACE VIEW edge_device_status AS
SELECT
    s.edge_device_id,
    s.camera_id,
    max(s.timestamp_utc)                                    AS last_seen,
    NOW() - max(s.timestamp_utc)                            AS staleness,
    count(*)                                                AS passes_last_hour,
    avg(s.plate_sequence_confidence)                        AS mean_confidence,
    sum(CASE WHEN s.ingest_degraded THEN 1 ELSE 0 END)      AS degraded_passes
FROM sightings AS s
WHERE s.timestamp_utc > NOW() - INTERVAL '1 hour'
GROUP BY s.edge_device_id, s.camera_id;

-- Unacknowledged wanted-vehicle sightings, highest severity first. This is the
-- query Stage 5's dispatch console polls.
CREATE OR REPLACE VIEW active_hotlist_alerts AS
SELECT
    h.timestamp_utc,
    h.pass_id,
    h.plate_number,
    h.fir_number,
    h.warrant_reference,
    h.severity,
    h.offence_category,
    h.camera_id,
    c.display_name AS camera_name,
    h.geom,
    NOW() - h.timestamp_utc AS age
FROM hotlist_hits AS h
LEFT JOIN cameras AS c ON c.camera_id = h.camera_id
WHERE h.acknowledged_at IS NULL
  AND h.timestamp_utc > NOW() - INTERVAL '24 hours'
ORDER BY h.severity DESC, h.timestamp_utc DESC;
