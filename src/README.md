# VeriTrack — Stage 1: Edge Vision & Feature Extraction Pipeline

Smart India Hackathon PS **26127** · Bharat Electronics Limited
Target hardware: **RK3588 NPU** / **Jetson Orin Nano** · Budget: **<15 ms/vehicle pass**, **<5 KB JSON/pass**

Stage 1 is everything that runs on the pole. It turns an RTSP feed into a stream of
signed, deduplicable `VehiclePass` records: one per vehicle that crossed the camera's
field of view, carrying a plate reading, its per-character posteriors, a 128-D visual
Re-ID embedding, and the uncertainty metadata Stages 3–4 need to fuse readings across
cameras. Nothing here talks to a database, and nothing here holds a cryptographic salt.

---

## 1. Pipeline topology

```
RTSP ──► VideoStream (thread, keep-latest queue, backoff reconnect)
           │
           ▼
      FrameGate ────────────── MotionGate (MOG2 @ 0.25 scale)
           │                   FocusGate  (Var(Laplacian) < 65 → drop)
           ▼
   VehicleDetector (anchor-free YOLO head, dual conf tiers)
           │
           ▼
      ByteTracker (Kalman xyah + two-pass IoU association)
           │
           ├──► OsNetExtractor ──► EmbeddingAccumulator (EMA, 128-D, L2)
           │
           ▼
   PlateKeypointDetector  →  (box, score, 4 × (x,y))
           │
           ▼
      PlateRectifier ── order_quad_corners → normalised DLT homography
           │            → 48×160 canonical strip → series classification
           ▼
   PlateLineSplitter (aspect-driven; two-line plates re-flowed to one strip)
           │
           ▼
      CtcRecognizer ──► per-frame log-posteriors ──► fuse_log_probs (mixture)
           │                                      ──► ctc_prefix_beam_search
           ▼
      PlateValidator (grammar templates + posterior-priced confusion repair)
           │
           ▼
   EdgePayloadBuilder (5120 B ceiling, ordered field shedding, HMAC envelope)
           │
           ▼
      StdoutSink / KafkaSink (idempotent, keyed by deterministic pass_id)
```

Module map:

| Module | Responsibility |
|---|---|
| `config.py` | Frozen dataclass config tree + `load_config()` JSON loader |
| `errors.py` | `FatalEdgeError` (kill the node) vs `RecoverableEdgeError` (drop the pass) |
| `types.py` | All wire and intermediate dataclasses |
| `ingest.py` | Threaded RTSP reader, keep-latest bounded queue |
| `gating.py` | Motion + focus admission control |
| `runtime.py` | `InferenceBackend` ABC, ONNX Runtime and RKNN-Lite implementations |
| `detection.py` | Letterbox, anchor-free decode, NMS, vehicle + plate-keypoint heads |
| `tracking.py` | Kalman filter, `STrack`, `ByteTracker` |
| `rectify.py` | Quad ordering, DLT homography, pose, series classification |
| `splitter.py` | Ink-mask row profile, valley search, two-line re-flow |
| `ocr.py` | CTC beam search, forced alignment, multi-frame fusion |
| `validation.py` | Indian plate grammar, optical confusion repair |
| `reid.py` | OSNet embedding, EMA accumulation, int8 quantisation |
| `packaging.py` | Byte-budgeted JSON serialisation |
| `pipeline.py` | Orchestration, latency attribution, `pass_id` derivation |
| `run_node.py` | CLI entrypoint, sinks, signal handling |

---

## 2. The mathematics that actually matters

### 2.1 Normalised DLT homography

Four point correspondences $(x_i, y_i) \leftrightarrow (u_i, v_i)$ give eight equations in the
eight degrees of freedom of $H$. Each correspondence contributes two rows to $A$:

$$
\begin{bmatrix}
-x & -y & -1 & 0 & 0 & 0 & ux & uy & u \\
0 & 0 & 0 & -x & -y & -1 & vx & vy & v
\end{bmatrix}
$$

and $h = \operatorname{vec}(H)$ is the right singular vector of $A$ for the smallest
singular value.

The subtlety is conditioning. At 1080p the raw entries of $A$ span pixel coordinates
($\sim10^3$) and their products ($\sim10^6$) against a column of ones — a condition
number in the $10^{12}$ range, which eats most of float64's mantissa. Hartley
normalisation fixes this: translate each point set to zero centroid and scale so the
mean distance from the origin is $\sqrt2$, solve, then undo:

$$H = T_{\text{dst}}^{-1} \, \tilde{H} \, T_{\text{src}}$$

Measured residual after this treatment: **8.53e-14** on unit-scale quads, **3.69e-13** at
1080p magnitudes, agreeing with `cv2.getPerspectiveTransform` to the same order. We keep
our own implementation rather than calling OpenCV because the intermediate $\tilde{H}$ is
reused by the pose decomposition below.

### 2.2 Corner ordering and the winding-sign trap

`order_quad_corners` must produce (TL, TR, BR, BL) from an arbitrarily-ordered keypoint
regression, or the warp silently mirrors or rotates the plate. The standard trick is the
shoelace sign — but **image coordinates have y pointing down**, so a ring that looks
clockwise on screen yields a *positive* shoelace area, the opposite of the textbook
convention. This was a live bug caught by the test suite: the ordering was being reversed
exactly when it was already correct, which in turn made `quad_aspect_ratio` measure the
short edges and misroute single-line plates into the two-line splitter. The condition is
now `if area2 < 0.0: reverse`, and the ordering is verified invariant over all 8 windings
of the same quad.

### 2.3 Skew estimation by aspect compression

The first implementation estimated yaw from edge-length foreshortening,
$\theta \approx 2\arctan\!\frac{1-r}{1+r}$ with $r$ the ratio of the two vertical edges.
That is geometrically wrong at ANPR standoff. For a plate of width $W$ at distance $d$,
the perspective term scales as $k = W/2d$; with $W = 0.5$ m and $d = 10$ m, $k = 0.025$,
and a full **45° yaw produces $r \approx 0.965$** — comfortably inside the noise floor of
the keypoint regressor. The signal isn't there to be measured.

What *is* measurable is the projected aspect ratio. A plate rotated by $\theta$ about its
vertical axis compresses horizontally by $\cos\theta$ while its height is unchanged:

$$\theta = \arccos\!\left(\frac{\mathrm{AR}_{\text{observed}}}{\mathrm{AR}_{\text{nominal}}}\right),
\qquad \mathrm{AR}_{\text{nominal}} = \frac{500\,\text{mm}}{120\,\text{mm}} = 4.167$$

This recovers known yaw to within 1.0° at 0°/15°/30°/40°, and it *derives* the single-vs-
two-line threshold rather than leaving it as a tuned magic number: at the worst supported
skew, $4.167\cos45° = 2.95$, which is above the configured 2.70 cutoff. A single-line
plate therefore cannot be misrouted to the splitter anywhere in the supported envelope.

Roll comes from the mean direction of the two horizontal edges. Pitch is reported as 0 —
it is not separable from scale without camera intrinsics, and pretending otherwise would
feed Stage 3 a fabricated number. `min_edge_symmetry` was demoted from a 0.55 skew gate
to a 0.35 sanity gate: its real job is rejecting corner regressions that collapsed, not
measuring angle.

For the Euler decomposition the ZYX convention is used, and the labels are worth stating
explicitly because they were transposed in the first draft:

$$\text{roll} = \operatorname{atan2}(R_{10}, R_{00}), \quad
\text{yaw} = \operatorname{atan2}(-R_{20}, s_y), \quad
\text{pitch} = \operatorname{atan2}(R_{21}, R_{22})$$

### 2.4 Multi-frame CTC fusion: a mixture, not a product

A tracklet gives $F$ views of one plate. The tempting move is a log-opinion pool —
average the log-probabilities — but that is a *product* of distributions, and a product
lets any single frame veto the rest: one specular highlight assigning $p \approx 10^{-6}$
to the true character kills it no matter what eleven clean frames say.

We use a proper mixture instead, computed in log space for stability:

$$\log \bar{p}_t(c) = \operatorname{logsumexp}_f \left[\log w_f + \log p_{t,f}(c)\right],
\qquad \sum_f w_f = 1$$

This is sound **only because every crop is warped to the same canonical 48×160 strip**, so
CTC step $t$ corresponds to the same physical band of the plate in every frame. Without
the rectifier, the frames aren't aligned and the fusion is meaningless.

Frame weights combine sharpness, detector confidence, and the recogniser's own certainty:

$$w_f \;\propto\; \text{focus}_f^{\alpha} \cdot \text{conf}_f^{\beta} \cdot e^{-\lambda \bar{H}_f},
\qquad
\bar{H}_f = \frac{1}{T\ln C}\sum_t \sum_c -p_{t,f}(c)\log p_{t,f}(c)$$

Normalising entropy by $\ln C$ puts $\bar H \in [0,1]$ regardless of alphabet size.
Measured discrimination between a sharp and a blurred observation: **16.8×**.

That same $\bar H$ is exported on the wire as the `text_entropy` field — it is exactly the
quantity Stage 3's dynamic weight $w_{\text{text}}$ needs, and computing it here costs
nothing extra.

A caution learned from the test suite: a dissenting frame that is *sharper* than the
majority will **lower** mean entropy, which is why entropy alone can't detect
disagreement. The fusion test was rewritten so the dissenting frame is byte-identical to
a clean one except at a single timestep, which isolates the effect properly (0.4951 →
0.4963, and the majority reading holds).

### 2.5 Prefix beam search

Standard CTC prefix beam search with the $p_b/p_{nb}$ (blank-ending / non-blank-ending)
split, which is what makes repeated characters survive — `DL8CAF5511` decodes intact,
where a naive collapse would give `DL8CAF551`. Absolute path probabilities over ~40 CTC
steps are naturally tiny (log p ≈ −10.98), so the test asserts a *margin* over the
runner-up rather than an absolute floor: `MH12AB1234` wins by **2.92 nats** over the
spurious `MH12QAB1234`.

### 2.6 Confusion repair priced in nats

Indian plates have a strict grammar: `SS DD LL NNNN` (state, RTO district, series
letters, number), plus legacy short forms and the BH series `YY BH NNNN LL`. 39 state
codes are enumerated in `validation.py`.

When a reading fails the grammar, we do **not** apply a fixed edit distance. Optical
confusions are asymmetric and context-dependent — `8↔B`, `0↔O↔D`, `1↔I`, `5↔S`, `2↔Z`,
`6↔G` — and their real cost is the likelihood we give up by overriding the recogniser.
`ctc_forced_alignment` (Viterbi over the extended label sequence) yields a per-character
posterior $p_t(c)$, and a substitution at position $t$ costs the log-likelihood ratio:

$$\text{cost} = \log p_t(c_{\text{orig}}) - \log p_t(c_{\text{repaired}})$$

Best-first search over the confusion graph finds the cheapest grammar-satisfying string.
Because the unit is nats and not "number of edits", `repair_cost` is comparable across
plates and is exported so Stage 3 can discount repaired readings consistently.
Measured: `MH12A81234` → `MH12AB1234` at **0.318 nats**; legal plates pass through
untouched.

Plates that *cannot* be repaired are emitted with `is_valid_format=False` rather than
dropped. A damaged or obscured plate still has a vehicle attached to it, and the Re-ID
embedding is often the only thing that will link that vehicle across cameras. Dropping
the pass would be throwing away the evidence.

### 2.7 ByteTrack association, and a units bug worth remembering

`ByteTracker` runs the two-pass association: high-confidence detections first, then a
second pass that rescues low-confidence boxes (the ones a single threshold would discard
during occlusion) against the surviving tracks.

Upstream ByteTrack's `match_thresh = 0.8` is a **maximum IoU distance**, i.e. IoU ≥ 0.2 —
not a minimum IoU. The first implementation read it as the latter and computed
`1.0 - thresh`, demanding IoU ≥ 0.80, which shattered a single clean tracklet into seven
identities. The config fields are now named so the units are unmistakable:
`first_match_max_distance` (0.80), `second_match_max_distance` (0.50),
`unconfirmed_match_max_distance` (0.70).

`use_mahalanobis_gate` defaults to **False**, matching the reference implementation: a
newly-created track has zero velocity and fails the chi-squared gate for purely kinematic
reasons, not because the association is wrong. When it is enabled, `min_hits_for_gate=5`
prevents that failure mode.

Verified: one stable identity over 14 frames (0.2 px position error), preserved through a
3-frame low-confidence occlusion.

### 2.8 Re-ID quantisation

OSNet gives a 128-D L2-normalised float32 embedding. Serialised as JSON floats that is
~1.4 KB — a quarter of the entire payload budget. Symmetric int8 quantisation plus base64
brings it to **172 characters**, with measured cosine drift of **0.00179**: two orders of
magnitude below the 0.35 divergence threshold Stage 3 uses for cloned-plate detection, so
the quantisation is invisible to every downstream decision that consumes it.

Across a tracklet the embedding is accumulated by EMA with renormalisation after each
update (the mean of unit vectors is not a unit vector). After 20 noisy frames the
accumulated embedding holds cosine **0.9888** against ground truth.

---

## 3. Latency budget

Per-stage attribution is computed live by `LatencyBudget` in `pipeline.py`, which shares
frame-level costs (ingest, gating, detection, tracking) across the vehicles present in
that frame and attributes per-track costs directly. Every `VehiclePass` therefore carries
an honest `StageTimings` breakdown rather than a single opaque number.

| Stage | Field | Cost model | Indicative RK3588 |
|---|---|---|---|
| Motion + focus gating | `gating` | frame-shared, CPU, 0.25-scale MOG2 | 0.4 ms |
| Vehicle detection | `vehicle_detect` | frame-shared, NPU, 640×384 | 4.1 ms |
| ByteTrack update | `track` | frame-shared, CPU | 0.3 ms |
| Plate keypoints | `plate_detect` | per-track, NPU, 192×192 | 2.2 ms |
| Rectify + split | `rectify` | per-track, CPU, DLT + warp | 0.6 ms |
| CTC recognition | `ocr` | per-observation, NPU, 160×48 | 3.0 ms |
| OSNet embedding | `reid` | per-track, NPU, 128×256 | 2.8 ms |
| Fusion + beam + repair | `decode` | once per pass, CPU | 0.9 ms |
| JSON packaging | `package` | once per pass, CPU | 0.2 ms |
| **Total** | `timings.total` | | **≈14.5 ms** |

The figures above are the design allocation; `latency_budget_ms = 15.0` in
`PipelineConfig` is the enforced ceiling. The pipeline applies an **early-emit quota**:
when a track's accumulated cost approaches the budget, further per-observation OCR is
skipped and the pass is emitted with the observations already fused. Degradation is a
slightly less-confident reading, never a dropped vehicle or a blown frame deadline.

Payload measured end-to-end: **917 bytes** against the 5120-byte ceiling — **82%
headroom**. `EdgePayloadBuilder` still enforces the ceiling with an ordered shed sequence
(`timings` → `alternatives` → `char_confidence` → `geometry`) so that a pathological pass
degrades predictably instead of being rejected by the gateway.

---

## 4. Deliberate architectural decisions

**Plates leave the edge in plaintext.** DPDP pseudonymisation happens at the Stage 2
gateway, not here. The rolling HMAC salt must never sit on a physically-removable,
pole-mounted device in an unattended cabinet; an attacker with a ladder and a screwdriver
should get nothing that compromises the pseudonymisation of the whole city. The edge-to-
gateway link is protected by the optional device HMAC envelope in `packaging.py` plus
transport TLS.

**`pass_id` is deterministic.** `blake2b(node | camera | track | first_timestamp)` means
an at-least-once Kafka retry produces a byte-identical key, and Stage 2 deduplicates on
the primary key for free. No distributed ID coordination, no idempotency table.

**Frozen dataclasses, not Pydantic, for edge config.** Validation cost on a hot path buys
nothing when the input is a local file the operator wrote. Pydantic is the right tool at
the Stage 2 gateway, where payloads arrive from untrusted nodes — and that is exactly
where Stage 2 uses it.

**Fatal vs recoverable errors are separated in the type system.** A missing model file or
a model whose output shape violates the documented contract is a `FatalEdgeError`: the
node should die loudly at startup rather than silently emit garbage for a week. A dropped
RTSP frame, a degenerate quad, or an oversized payload is a `RecoverableEdgeError`: log,
drop the pass, keep serving.

---

## 5. Model contracts

The pipeline is model-agnostic but contract-strict. `runtime.py` validates output shapes
at load and raises `ModelContractError` on mismatch.

| Head | Input (NCHW) | Output | Notes |
|---|---|---|---|
| Vehicle detector | `(1,3,384,640)` | `(1,N,4+1+K)` or YOLOX layout | `decode_layout` config selects |
| Plate keypoints | `(1,3,192,192)` | `(1,N,13)` | box(4), score(1), 4×(x,y) |
| CTC recogniser | `(1,3,48,160)` | `(1,T,C)` | logits, blank at index 0 |
| OSNet | `(1,3,256,128)` | `(1,128)` | L2-normalised downstream |

---

## 6. Running it

```bash
pip install -r requirements.txt

# Validate the mathematics with no model artefacts present (24 checks)
python tests/test_core_math.py

# Live node, stdout sink (bench / replay)
python -m veritrack_edge.run_node \
    --config configs/node_a12.json \
    --sink stdout

# Production: Kafka sink, idempotent producer, lz4, keyed by pass_id
python -m veritrack_edge.run_node \
    --config configs/node_a12.json \
    --sink kafka \
    --brokers kafka-01:9092,kafka-02:9092 \
    --topic veritrack.passes.raw
```

The test suite is **model-free by construction**: every check synthesises its own tensors
or images, so the geometry, CTC, tracking, repair and packaging logic can be verified on a
laptop, in CI, or on a node whose NPU artefacts haven't been provisioned yet.

---

## 7. What Stage 1 hands to Stage 2

Each `VehiclePass` JSON carries:

- `pass_id`, `node_id`, `camera_id`, `track_id`, first/last timestamps
- `plate_text`, `is_valid_format`, `repair_cost` (nats), `text_entropy`
- `char_confidence[]` — per-character posteriors from forced alignment
- `alternatives[]` — runner-up beam hypotheses with scores
- `plate_series` — private_white / commercial_yellow / ev_green / government / diplomatic
- `embedding_b64` — int8-quantised 128-D OSNet vector (172 chars)
- `geometry` — quad, skew, roll, focus score
- `timings` — the full `StageTimings` breakdown
- optional `hmac` device-attestation envelope

Stage 3 consumes `text_entropy`, `repair_cost` and the embedding directly for its
$S = w_{\text{text}}S_{\text{text}} + w_{\text{vis}}S_{\text{vis}} + w_{\text{kin}}S_{\text{kin}}$
fusion. Those fields exist because Stage 3 needs them, not as diagnostics.

---

# VeriTrack — Stage 2: Central Ingestion, Storage & Hotlist Matcher

Stage 2 is the trust boundary. Stage 1 runs unattended on roadside poles, so
everything arriving here is untrusted input until validated. The package turns
that stream into two legally distinct storage tracks.

## Pipeline

```
POST /api/v1/telemetry/ingest
        │
        ▼
  Pydantic v2 validation ── extra="forbid", grammar, quad convexity,
        │                   embedding norm, no-cleartext-in-identifiers
        ▼
  Clock-skew gate ── future > 120 s rejected, older than replay horizon rejected
        │
        ▼
  Bloom hotlist check ── ONE pipelined round trip for the whole batch
        │
        ├── miss ──► HMAC-SHA256(daily salt, plate) ──► COPY buffer ──► sightings
        │                                                        └──► sightings topic
        │
        └── hit ───► Redis hash confirms (authoritative)
                        │
                        ├── cleartext under warrant ──► hotlist_hits (sync, durable)
                        ├── AES-256-GCM sealed evidence, AAD = pass_id
                        ├── pseudonymised row too, hotlist_flag = true
                        └──► alerts topic (never dropped)
```

## The dual-track argument

The privacy property is arithmetic, not policy. Because the salt rotates every
24 hours, two sightings of one vehicle are linkable **within** a day — which is
exactly what corridor analytics and O-D matrices need — and unlinkable
**across** days once the salt has aged out. Purpose limitation is enforced by
the construction rather than by a retention promise.

Cleartext is retained only where there is a lawful basis: an active FIR or
warrant. That decision is never made on probabilistic evidence. A Bloom hit
only *permits* the authoritative Redis lookup; the case record is what confirms
the match, resolves false positives and enforces warrant expiry. The downstream
consequence is a police dispatch, so it rests on a real record every time.

**No cleartext column exists in `sightings`.** Not "is left null" — the column
is absent from the DDL, so no bug, ORM misconfiguration or ad-hoc query can put
one there. The gateway additionally rejects any payload whose `pass_id`,
`tracklet_id` or `corridor_id` embeds the decoded plate, closing the one path
by which cleartext could otherwise ride into that table inside an identifier.

## Bloom sizing

For capacity *n* and target rate *p*:

$$m = \left\lceil \frac{-n \ln p}{(\ln 2)^2} \right\rceil, \qquad k = \operatorname{round}\!\left(\frac{m}{n}\ln 2\right)$$

At n = 2×10⁷ and p = 10⁻⁴ this is **m ≈ 3.83×10⁸ bits (48 MB, one Redis
string) and k = 13**, against ~1.2 GB for an equivalent Redis set. The error is
one-sided and that asymmetry is the whole design: a false positive costs one
extra hash lookup, a false negative would mean a wanted vehicle passing
unflagged — and the mathematics guarantees that cannot happen.

Bit positions use Kirsch–Mitzenmacher double hashing with a quadratic
correction, `g_i(x) = (h₁ + i·h₂ + i²) mod m`, with both halves taken from one
128-bit BLAKE2b digest. So k=13 positions cost one hash, and all 13 `GETBIT`s
go out in a single pipeline: **one round trip per membership test regardless of
k**. BLAKE2b rather than `hash()` because the latter is per-process randomised,
which would make a shared Redis filter incoherent across replicas.

## Crypto decisions

**Envelope, not direct, encryption.** Every record gets a fresh 256-bit DEK;
the DEK is wrapped by the KEK. A KEK rotation re-wraps a handful of small DEKs
instead of re-encrypting the entire evidence corpus.

**AAD binds ciphertext to its row.** The `pass_id` is the Additional
Authenticated Data, so a ciphertext lifted from one sighting and pasted onto
another fails authentication rather than decrypting into a plausible lie.

**HMAC, not a salted hash.** The salt is a key, not a public parameter. HMAC's
key-prefix construction is what makes salt recovery from observed digests
infeasible; a naive `sha256(salt ‖ plate)` would expose length-extension
structure.

**Derived salts by default, ephemeral available.** HKDF from a root pepper lets
a horizontally-scaled replica fleet agree on the same pseudonym with no salt
replication protocol. The trade-off is stated honestly in the module docstring:
the pepper is the whole ball game, so it lives in the gateway's secret store and
never on an edge node. `EphemeralSaltSource` gives the maximally-private
alternative — random per-epoch salts, memory only, unrecoverable after exit — at
the cost of horizontal scalability.

## Storage decisions

**`geom` is a STORED generated column.** Binary `COPY` cannot invoke a
function, so a generated column is what lets the high-throughput ingest path use
COPY while still producing a real PostGIS geography for spatial queries.

**Buffered writes, not write-through.** The handler appends to an in-process
deque and returns; a background task flushes via `copy_records_to_table` on a
row or time threshold. The buffer is bounded, so a database stall becomes an
explicit HTTP 503 that the edge node retries — never unbounded memory growth
followed by an OOM kill.

**Two durability classes.** Ordinary sightings ride the buffered path with
`synchronous_commit=off`; an individual anonymous sighting is statistical input
and a sub-second loss window is worth the throughput. Hotlist hits are written
inside the request with `synchronous_commit` forced ON. An alert that a wanted
vehicle passed a camera is evidence, and we do not buy performance with it.

**Chunk-drop retention.** The 30-day DPDP purge drops whole 1-day chunks — a
catalogue operation, not a 20-million-row DELETE. O(1), no bloat, no VACUUM
backlog, and no window where a partially-purged chunk still holds recoverable
data.

**The hotlist check fails open.** If Redis is slow or unreachable the sighting
is still stored, pseudonymised, and marked `ingest_degraded` for offline
re-testing. Failing closed would let a Redis blip erase traffic history, which
is worse than a delayed alert.

## Measured

62/62 tests passing, no PostgreSQL/Redis/Kafka required — collaborators are
injected, so the tests drive the *real* request path rather than a parallel mock.

| | |
|---|---|
| Ingest throughput | **5,132 obs/s**, single worker, 500-obs batches |
| Per observation | 195 µs end-to-end |
| Batch latency p50 / p95 | 10.1 / 12.8 ms per 500-observation batch |
| Hotlist check p50 / p99 | **6.9 µs / 24 µs** against a 50 ms budget |
| Filter @ 20 M capacity | 3.83×10⁸ bits, k=13, 48 MB |
| Bloom false negatives | **0 over 5,000 inserted plates** (the load-bearing guarantee) |
| DDL | 50 statements, accepted by the real PostgreSQL grammar |

## Running it

```bash
pip install -r requirements-server.txt
psql -d veritrack -f veritrack_server/sql/init_schema.sql
pytest tests/test_stage2.py -v

export VERITRACK_CRYPTO_AES_MASTER_KEY=$(openssl rand -base64 32)
export VERITRACK_CRYPTO_SALT_PEPPER=$(openssl rand -base64 64)
uvicorn veritrack_server.gateway:app --host 0.0.0.0 --port 8080
```

`environment=prod` refuses to boot without an ingest API key, with placeholder
key material, with docs enabled, or with Redis optional — misconfiguration
becomes a startup failure rather than a silent weakening.

---

# VeriTrack — Stage 3: Trajectory Reconstruction & Anomaly Engine

Stage 3 turns a scatter of discrete camera hits into continuous, map-matched
journeys with street names, and flags the kinematic and visual impossibilities
that indicate a cloned or swapped plate.

It consumes **pseudonymised** Stage 2 rows. It never needs a cleartext
registration number: same-day linkage is exactly what the rotating HMAC salt
preserves, so reconstruction works on pseudonyms alone.

## Pipeline

```
sightings ──► temporal ordering, journey splitting on idle gaps
                  │
                  ▼
        pairwise fusion   S = w_text·S_text + w_vis·S_vis + w_kin·S_kin
                  │       weights slide with OCR confidence
                  ▼
        anomaly conditions A (teleport) · B (visual divergence) · C (unreachable)
                  │
                  ▼
        Viterbi map-matching  candidates ≤50 m → log-space trellis → backtrack
                  │           blind spots filled with interpolated segments
                  ▼
        ReconstructedTrajectory + GeoJSON LineString
```

## Dynamic weight shifting

The gate `g = σ(k(C_seq − τ))` decides how far to trust the plate string. The
schedule collapses to a tidy closed form:

$$w_{kin} = 1 - w_{text} - w_{vis} = 0.35 - 0.20\,g$$

so the three weights always sum to 1 and `w_kin` stays in [0.15, 0.35] —
**never negative at any gate value**. `TrajectoryConfig` asserts that at both
extremes on construction rather than trusting it.

| C_seq | g | w_text | w_vis | w_kin |
|---|---|---|---|---|
| 0.10 | 0.001 | 0.001 | 0.649 | 0.350 |
| 0.50 | 0.142 | 0.099 | 0.579 | 0.322 |
| 0.65 | 0.500 | 0.350 | 0.400 | 0.250 |
| 0.95 | 0.973 | 0.681 | 0.163 | 0.155 |

Two deliberate properties. The gate uses the **minimum** of the pair's two
confidences, not the mean — the text channel is only as good as the weaker
read, and averaging 0.98 with 0.30 would hand most of the weight to a string
one camera barely resolved. And kinematics **never dominates**: travel speed
can veto an impossible link but must not assert identity on its own, because
thousands of vehicles traverse a corridor at a plausible speed every hour.

## Where the spec needed correcting

**The kinematic Gaussian as literally specified penalises congestion as hard as
speeding.** Centred on a 45 km/h free-flow baseline with σ=18, crawling at 15
km/h and doing 75 in a 45 zone both score 0.2494 — identical. But a real
vehicle is slower than free-flow most of the time and almost never meaningfully
faster. The implementation widens the Gaussian by 2.5× below baseline, so 15
km/h scores 0.83 while 75 km/h still scores 0.25. `symmetric_kinematic=True`
reverts to the literal spec so the two can be A/B tested rather than the change
merely asserted.

The hard 140 km/h veto is kept exactly as specified — that is what lets the
kinematic channel refute a link no matter how good the plate and appearance
evidence look, which is precisely the cloned-plate signature.

## Two numerical traps in the emission model

`max(0, cos Δθ)` is **exactly zero** at 90°, and `log 0 = −∞`. One
perpendicular candidate would annihilate an otherwise excellent path. The
heading factor is floored at 1e-4 — about −9.2 nats, still severe but finite
and recoverable.

Bearings must be compared **with wraparound**: 359° and 1° differ by 2°, not
358°. Without that, the emission model silently rejects correct candidates at
the north crossing.

Everything runs in log space. A 20-sighting trajectory multiplies ~40
probabilities each often below 1e-3, which underflows float64 long before the
path completes.

## The bug the tests caught

The first run reported **3.0 km for a journey with 2.0 km of ground truth**.

At a junction the arriving segment (`along_fraction = 1.0`) and the departing
one (`0.0`) tie *exactly* on emission — both have zero perpendicular distance
and zero heading error. The transition model measured distance **node-to-node**,
which ignores where along a segment the vehicle actually sits, so the tie broke
onto the departing segment and the matcher routed the vehicle out to that
segment's far node **and back**: a phantom detour inflating the journey by a
full block.

The fix measures travel **projection-to-projection**:

$$d = (1-f_1)\ell_1 + d_{net}(v_1^{to}, v_2^{from}) + f_2\ell_2$$

This does not make the tied candidates equal — my first docstring claimed it
did, and the test proved otherwise. It makes the transition model
*discriminate* on the right grounds: from the arriving candidate the onward
travel is the true 1000 m, from the departing one it is 2000 m including the
out-and-back. At 45 km/h over 80 s the expected travel is 1000 m, so the honest
candidate has zero discrepancy and wins. Reported distance is now exact on
straight, L-shaped and blind-spot routes.

Same measure is used for the reported journey length: summing whole matched
segments over-counts the tail of the first and the head of the last, neither of
which the vehicle was observed to traverse.

## Anomaly guards matter more than thresholds

Every flag can put a real person under suspicion, so three guards apply
throughout: a **minimum elapsed time** (dividing by clock jitter manufactures
spectacular velocities from nothing), a **minimum network distance** (over a few
metres the speed estimate is dominated by pole-survey error), and a **minimum
OCR confidence** (a clone accusation resting on a 0.3-confidence read is far
more likely a misread than a crime).

Conditions A and C are mutually exclusive by construction — A requires a finite
distance, C an infinite one — so one physical event is never reported twice.
Severity grades by margin: 145 km/h is worth a look, 400 km/h is arithmetic
proof.

## Measured

82/82 tests passing, fully offline — the road network is a synthetic grid whose
ground truth is known by construction.

| | |
|---|---|
| Graph | 1,600 nodes, 6,240 edges (40×40 grid, 250 m spacing) |
| A* cold query | 164 µs |
| Cached query | **0.18 µs**, 99.5% hit rate — the sub-millisecond path |
| 15-sighting reconstruct | **4.74 ms** p50 (was 108 ms before indexing nodes) |
| `nearest_node` | 300/300 exact vs brute force, 23× faster |
| Route recovery | straight / L-turn / blind-spot all exact to ±1 m |
| Cloned plate | 2.5 km in 45 s → 200.0 km/h, flagged CRITICAL |

`nearest_node` was the single hottest call in the stage — a linear scan issued
hundreds of times per trajectory, 716k haversine evaluations for one 15-sighting
journey. Moving it onto the KD-tree already built for candidate search cut
reconstruction 23×. The exactness test exists because an optimisation that
changes answers is not an optimisation.

## Running it

```bash
pip install -r requirements-trajectory.txt
pytest tests/test_stage3.py -v
```

```python
from veritrack_trajectory import TrajectoryEngine, build_synthetic_grid

graph = build_synthetic_grid(40, 40, spacing_m=250.0)
engine = TrajectoryEngine(graph)

sightings = TrajectoryEngine.from_stage2_rows(rows)   # asyncpg Records or dicts
for trajectory in engine.reconstruct_all(sightings):
    print(trajectory.total_distance_km, trajectory.street_names)
    geojson = trajectory.to_geojson()
```

`RoadNetworkGraph.from_osm_like()` consumes the node/edge record shape an osmnx
or Overpass export reduces to, so swapping the synthetic grid for a real city
extract is a field-mapping exercise rather than a rewrite.

---

# Stage 1 laptop demo — live webcam ANPR, no GPU required

Everything above targets an RK3588/Jetson pole camera. This section is a
different deployment target for the same Stage 1 problem: demoing VeriTrack
live, in front of judges, on a CPU-only Windows laptop with a webcam — the
hardware you actually have for the demo, not the hardware the platform ships
on. New package: `veritrack_demo`.

It is built around, and was informed by, a real measurement you already took
on the target laptop: PP-OCRv6-Tiny via `transformers`, CPU inference,
**0.098 s**, **0.947 confidence**, a correct read (`ZG7497-AH`) on a real
photographed plate. Everything about the recognizer wrapper below is a
refactor of that exact, already-working call sequence into a reusable class —
not a reimplementation of it, and not a guess at the API. I confirmed every
method name it calls (`AutoModelForTextRecognition`,
`post_process_text_recognition`) against the installed `transformers` source
directly, including that batching multiple crops through one forward pass is
genuinely supported (`post_process_text_recognition` already loops over a
batch dimension internally). What I could not do in my own sandbox is
actually run it: this environment has no route to `huggingface.co` and no
working CPU/CUDA PyTorch wheel available to it. That's fine — you already ran
the real thing, on the real machine, and got real output. Everything below was
verified against that fixed point plus 52 passing tests using dependency
injection in place of the model.

## Where each piece should run

Your machine has Docker, k8s, and WSL2 Ubuntu available, and it matters which
piece runs where:

| Piece | Recommended | Why |
|---|---|---|
| Webcam capture + recognizer | **Native Windows Python** | Docker Desktop on Windows cannot pass a USB webcam through to a Linux container without fragile workarounds (there's no `/dev/video0` on Windows to pass through). WSL2 *can* reach a USB camera via `usbipd-win`, but it's an extra moving part (attach before every session, plus a `uvcvideo`-capable kernel) for no benefit here. `cv2.VideoCapture(i, cv2.CAP_DSHOW)` on native Windows just works. |
| Stage 2 gateway + TimescaleDB + Redis | **Docker (compose)** | Pure network services — exactly what Docker on Windows is good at. No camera involved. |
| Same backend, if you want to show it on k8s | **Optional**, via `k8s/veritrack-demo.yaml` | `kind`/`minikube`/Docker Desktop's built-in Kubernetes all work; same non-goal applies — don't try to route the webcam into the cluster. |

Practical upshot: run the webcam script directly in a Windows `venv`; run
`docker compose up` (or the k8s manifest) alongside it for the backend if you
want to demonstrate the full ingest-to-DPDP-pseudonymisation loop, not just
the recognizer.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-demo.txt

# Pre-download the model weights *before* you're in the room with unreliable
# wifi. This is the single biggest failure mode for a live model demo.
huggingface-cli download PaddlePaddle/PP-OCRv6_tiny_rec_safetensors

# Confirm the camera is where you think it is (an external USB webcam does
# not reliably enumerate as index 0 once a laptop's built-in camera exists).
python -m veritrack_demo.run_webcam --list-cameras

python -m veritrack_demo.run_webcam
```

Set `HF_HUB_OFFLINE=1` once the model is cached, so a flaky venue network
fails fast at startup instead of hanging mid-demo waiting on a DNS timeout.

Keyboard controls: `q` quit · `s` snapshot · `f` toggle Stage 2 forwarding ·
`1`/`2`/`3`/`4` switch localizer (manual/haar/contour/auto) · `wasd` move the
manual ROI · `+`/`-` resize it.

## Three ways to find the plate, none of them a downloaded model

Stage 1's real vehicle and plate detectors are trained heads that need
exported weights this demo doesn't have. A hackathon demo has different
constraints than a pole camera, though — you control where the plate is held —
so all three localizer strategies run on stock OpenCV with **nothing to
download**:

- **`manual`** (default) — a fixed, on-screen ROI you hold the plate inside.
  This generalises the hardcoded `crop_box` percentages from your validated
  script into a resolution-independent, live-adjustable rectangle. It cannot
  fail to produce a candidate, which is the property that matters for the one
  take that counts in front of judges.
- **`haar`** — OpenCV's bundled plate-shaped cascade classifier. Automatic,
  coarse, not trained on Indian plates specifically, but the rectangular
  high-contrast feature it looks for transfers reasonably at close range.
- **`contour`** — classic edge-detection + rectangle search, with a
  rectangularity filter (contour area / bounding-rect area ≥ 0.55) so it
  doesn't fire on background clutter. Recovers real rotation via
  `cv2.minAreaRect`, so a plate held at a slight angle still gets an
  approximately-correct perspective correction.

`auto` chains haar → contour → manual, falling back only when the automatic
strategies find nothing — the most robust option once you trust it, not the
one to bet the first live take on.

Every candidate feeds `veritrack_edge.rectify.PlateRectifier` — reused
verbatim from Stage 1 — and falls back to a plain resized crop when the
geometry check rejects it (`GeometryError`: degenerate, too skewed, or below
the corner-symmetry sanity floor). The HF processor does its own resizing
internally and never required the exact 48×160 canonical strip Stage 1's
custom CTC head needed, so the fallback loses nothing essential.

## A real performance bug the benchmark caught

First measurement of the `haar` strategy at 1280×720: **420 ms per frame** —
about 2 fps, a slideshow, not a demo. `detectMultiScale`'s cost scales with
the number of pyramid levels it scans, which scales with input resolution; at
full webcam resolution that cost is severe.

The fix is the standard one — detect on a downscaled copy, rescale results
back to full-frame coordinates — but I measured it rather than assumed it
would help enough:

| Detection width | Time |
|---|---|
| 1280 (full) | 420 ms |
| 640 | 88 ms |
| 480 (chosen default) | **44 ms** — 9.4× faster |
| 320 | 16 ms |

480 px keeps enough resolution for a plate held close to the camera (the
realistic demo framing) while landing comfortably under the recognizer's own
~98 ms, so detection is no longer the bottleneck. The same downscale is
applied to the `contour` strategy for consistency, which dropped it from 26 ms
to 6 ms as a bonus. All 52 demo tests still pass afterward, including the ones
checking detected-box centre position and recovered rotation angle — the
downscale-then-rescale changed the cost, not the geometry.

## Measured (pipeline overhead only — recognizer excluded, see below)

52/52 tests passing, fully offline: every localizer runs against synthetic
`cv2`-drawn images, and the recognizer is a `FakeRecognizer` injected via
dependency injection, so gate → localize → rectify-or-crop → validate wiring
is verified without a camera, a GPU, `torch`, or `transformers` present.

| Strategy | Pipeline overhead @1280×720 (p50) | + your measured 98 ms recognizer |
|---|---|---|
| `manual` | 3.3 ms | ~101 ms → ~10 fps |
| `contour` | 6.2 ms | ~104 ms → ~9.6 fps |
| `haar` | 46.5 ms | ~145 ms → ~6.9 fps |
| `auto` | 53.0 ms | ~151 ms → ~6.6 fps |

The recognizer's own 98 ms — not the pipeline — is the real bottleneck at
every strategy, which is expected and fine: it is by far the most useful 98 ms
in the loop. `process_every_n_frames` (default 3) keeps the on-screen display
smooth between recognizer calls regardless of which strategy is active.

One thing this table cannot include: an actual end-to-end frame time with the
real recognizer, since that requires the model weights and a working
CPU/CUDA torch build my sandbox doesn't have access to. Add the pipeline
overhead above to whatever you measure locally with `-v` for the honest total.

## What's real, and what's clearly marked as a stand-in

Two things are reused from Stage 1 without modification, because they are
hardware-independent math:

- `veritrack_edge.rectify.PlateRectifier` — the homography warp
- `veritrack_edge.validation.PlateValidator` — Indian plate grammar and
  optical-confusion repair (`8↔B`, `0↔O`, ...). With no per-character
  posteriors from the HF recognizer, repair falls back to the documented
  uniform substitution cost, and still prefers a minimal-edit legal plate.

If you enable Stage 2 forwarding (`f` or `--forward`), be aware of what it
cannot honestly claim: Stage 2's schema requires a vehicle bounding box and a
128-D Re-ID embedding, because a real Stage 1 node runs a vehicle detector and
an OSNet extractor to produce them. This demo runs neither — it's a plate
localizer plus a text recognizer. The forwarder fills those two fields with
clearly-marked stand-ins (the plate's own box, expanded; a deterministic
pseudo-vector derived from the decoded text) and sets `"demo_stand_in": true`
on every payload, so nothing downstream can mistake them for real detections.
The payload is verified to round-trip through Stage 2's own
`EdgeObservation.from_compact_payload` parser as a cross-package test — it
isn't just shaped like the real thing, Stage 2 actually accepts it.

## Running the backend alongside it

```powershell
# docker-compose (recommended default)
cp .env.example .env    # or leave ALLOW_INSECURE_DEFAULTS=true for a quick demo
docker compose up --build
curl http://localhost:8080/healthz

# then, in the venv, in another terminal:
python -m veritrack_demo.run_webcam --forward --gateway-url http://localhost:8080
```

For k8s instead, see the usage block at the top of `k8s/veritrack-demo.yaml` —
build the gateway image, load it into your cluster (`kind load docker-image`
/ `minikube image load` / nothing extra for Docker Desktop's Kubernetes),
apply the manifest, and `kubectl port-forward svc/gateway 8080:8080`.
