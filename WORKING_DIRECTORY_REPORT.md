# VeriTrack working-directory report

Generated: 2026-09-18

## Executive status

This directory contains a substantial, multi-stage ANPR prototype rather than
a finished city-analytics product. Its most directly runnable path is the
headless/static or webcam demo in `veritrack_demo`. It uses the real cached
PP-OCRv6-Tiny model. The Stage 2 ingestion service and Stage 3 trajectory
engine contain real implementations and extensive unit tests, but no deployed
workflow currently connects stored observations to trajectory queries or a
traffic dashboard.

The working tree has no committed baseline: `git status` reports all project
files as untracked. Therefore this report describes files present on disk, not
the delta from an existing repository history.

## Tree inventory

` .git/` and `.venv/` are local Git and Python-environment internals and are
intentionally excluded below. `.env` is present and is not reproduced because
it may contain secrets.

```text
assassin/
├── .env                         local runtime settings; do not commit
├── .gitignore
├── .python-version              selected Python version
├── Dockerfile                   empty (not a usable image definition)
├── README.md                    short SIH/problem statement
├── VeriTrack_SIH2026.pdf        source/project document
├── WORKING_DIRECTORY_REPORT.md  this inventory and assessment
├── clean.ps1                    small Windows cleanup helper
├── cropped_plate.png            derived crop used in OCR experimentation
├── dry_run.py                   headless one-frame OCR/telemetry driver
├── main.py                      placeholder; prints "welp"
├── pg.yml                       local PostgreSQL-related compose config
├── pyproject.toml               root Python dependency declaration
├── samplecarimg.png             supplied static ANPR test image
├── t.py                         original one-off PP-OCR experiment
├── uv.lock                      locked dependency graph
├── veritrack.zip                archived copy of project material
└── src/
    ├── .env.example             environment-variable template
    ├── Dockerfile.gateway       Stage 2 gateway container image
    ├── README.md                detailed architecture and operating notes
    ├── docker-compose.yml       gateway + dependency local deployment
    ├── pytest.ini               test configuration
    ├── requirements.txt         common / edge dependencies
    ├── requirements-demo.txt    webcam + Hugging Face demo dependencies
    ├── requirements-server.txt  Stage 2 service dependencies
    ├── requirements-trajectory.txt
    ├── k8s/
    │   └── veritrack-demo.yaml  gateway deployment/service/config manifests
    ├── tests/
    │   ├── test_core_math.py    Edge geometry, OCR, tracker unit tests
    │   ├── test_demo.py         Webcam/localizer/demo-forwarding tests
    │   ├── test_stage2.py       Gateway, crypto, DB, hotlist tests
    │   └── test_stage3.py       map-matching, fusion, anomaly tests
    ├── veritrack_demo/
    │   ├── config.py            frozen demo configuration
    │   ├── recognizer.py        Hugging Face PP-OCRv6-Tiny wrapper
    │   ├── localizer.py         manual ROI, Haar, contour, fallback modes
    │   ├── pipeline.py          localize -> rectify -> OCR -> validate chain
    │   ├── overlay.py           OpenCV visual overlay/HUD
    │   ├── gateway_client.py    optional Stage 2 demo forwarder
    │   ├── run_webcam.py        GUI webcam entry point
    │   └── __init__.py          public imports
    ├── veritrack_edge/
    │   ├── config.py            production-node configuration / model paths
    │   ├── runtime.py           ONNX Runtime and RKNN backend adapters
    │   ├── ingest.py            video/RTSP capture and reconnect handling
    │   ├── detection.py         vehicle and plate-keypoint detector wrappers
    │   ├── tracking.py          vehicle association/tracking
    │   ├── gating.py            blur/motion admission gate
    │   ├── rectify.py           plate homography correction
    │   ├── splitter.py          two-line plate splitting
    │   ├── ocr.py               CTC OCR observation/fusion logic
    │   ├── validation.py        Indian plate format/repair logic
    │   ├── reid.py              appearance embedding wrapper
    │   ├── packaging.py         compact telemetry serialization
    │   ├── pipeline.py          full per-frame production pipeline
    │   ├── run_node.py          RTSP/video -> stdout or Kafka CLI
    │   ├── types.py             edge datatypes
    │   ├── errors.py            typed errors
    │   └── __init__.py
    ├── veritrack_server/
    │   ├── gateway.py           FastAPI Stage 2 ingest service
    │   ├── schemas.py           canonical and compact request schemas
    │   ├── db.py                async database/buffer layer
    │   ├── crypto.py            plate pseudonymisation/encryption
    │   ├── bloom.py             hotlist Bloom filter abstraction
    │   ├── config.py            Pydantic environment configuration
    │   ├── sql/init_schema.sql  PostgreSQL/Timescale schema
    │   └── __init__.py
    └── veritrack_trajectory/
        ├── graph.py             road graph and shortest-path operations
        ├── viterbi.py           map-matching trellis
        ├── fusion.py            OCR/visual/kinematic score fusion
        ├── anomaly.py           clone/teleport/unreachable detection
        ├── engine.py            journey reconstruction orchestration
        ├── config.py            trajectory thresholds/weights
        ├── types.py             trajectories, sightings, GeoJSON types
        └── __init__.py
```

## Runnable entry points

| Command / file | Current condition | Purpose |
|---|---|---|
| `.venv\\Scripts\\python.exe dry_run.py --image samplecarimg.png` | **Verified working** | Headless static-image demo; prints OCR, timings, and a local telemetry JSON packet. |
| `python -m veritrack_demo.run_webcam` | Runnable subject to camera/device setup | Interactive webcam demo with manual/automatic localizers and optional forwarding. |
| `uvicorn veritrack_server.gateway:app` | Implemented; requires configured PostgreSQL/Redis and environment secrets | Stage 2 FastAPI ingestion API. |
| `python -m veritrack_edge.run_node --config ...` | Code is assembled but **blocked by missing ONNX/RKNN model artifacts** | Intended production RTSP edge node. |
| `main.py` | Not useful | Placeholder only. |
| `t.py` | One-off experiment | Direct PP-OCR sample script; superseded by `dry_run.py`. |

## What is genuinely wired

### Stage 1 demo path

The working image path is:

```text
samplecarimg.png
  -> ManualRoiLocalizer
  -> PPOcrRecognizer (AutoImageProcessor + AutoModelForTextRecognition)
  -> decoded string/confidence
  -> printed telemetry packet
```

The last verification completed successfully with `ZG7497-AH`, 88.65% OCR
confidence, 217 x 75 crop dimensions, and CPU inference around 40 ms.
`dry_run.py` uses a local Hugging Face snapshot when it exists, preventing a
network metadata lookup from blocking the live demonstration.

The webcam path expands this to:

```text
cv2.VideoCapture -> frame gate -> localizer -> rectifier/crop -> PP-OCR
 -> Indian plate validator -> OpenCV overlay -> optional HTTP forwarder
```

The localizers are real OpenCV algorithms, but only manual ROI is a
demo-reliable detection strategy. Haar and contour are heuristics, not a
trained Indian number-plate detector.

### Stage 2 central ingestion

`veritrack_server.gateway` exposes canonical and compact ingest endpoints,
readiness/liveness, Prometheus metrics, and hotlist statistics. It validates
payloads, pseudonymises plates, checks the hotlist, and writes through the
database buffering layer. SQL schema and Docker/Kubernetes deployment material
are present.

The demo forwarder is wired to the compact ingestion endpoint. It deliberately
labels demo payloads with `demo_stand_in: true`: the vehicle bounding box and
128-D ReID embedding are fabricated because the webcam demo does not execute a
vehicle detector or visual ReID model. Those demo rows must not be used to
claim real cross-camera visual association.

### Stage 3 trajectory and anomaly code

`veritrack_trajectory` is a functional library: it accepts sightings, splits
journeys, fuses text/appearance/kinematic evidence, map matches over a road
graph, produces GeoJSON, and detects implausible movement. It is not connected
to Stage 2 as a worker, scheduled job, REST endpoint, or dashboard query.

## Missing integration and product surface

No implementation exists for the requested city dashboard features:

- browser frontend or dashboard service;
- GIS basemap, map markers, or heatmap rendering;
- camera-count / density aggregation;
- origin-destination matrix generation;
- congestion/bottleneck analytics;
- real-time aggregation stream or materialized analytics tables;
- trajectory query API backed by Stage 2 data;
- alert delivery/UI workflow.

The existing `/metrics` and `/api/v1/hotlist/stats` endpoints are operational
monitoring endpoints, not traffic analytics endpoints.

## Runtime blockers and caveats

1. No `models/` directory or configured production model files exist. The
   `veritrack_edge` defaults reference `vehicle_det.onnx`, `plate_kpt.onnx`,
   `svtr_tiny_rec.onnx`, and `osnet_x0_25.onnx`; without them the production
   edge node cannot start.
2. The current Python environment has OpenCV 5.0.0. Its `cv2` module lacks
   `CascadeClassifier`, breaking Haar localizer construction. Manual and
   contour paths are unaffected by this specific issue.
3. Hugging Face network access is denied in this environment. The real OCR
   demo works only because the required PP-OCRv6 snapshot is already cached.
4. Stage 2 and Stage 3 behavior is well unit-tested, but an actual connected
   PostgreSQL/Redis/Kafka deployment was not exercised in this directory
   review.
5. Root `Dockerfile` is empty. Use `src/Dockerfile.gateway` for the service;
   do not treat the root Dockerfile as a deployment asset.

## Test snapshot

Command executed:

```powershell
.venv\Scripts\python.exe -m pytest src\tests -q
```

Result: **215 passed, 5 failed** in 6.60 s.

- Four failures are Haar-localizer tests caused by the incompatible/broken
  OpenCV 5 installation described above.
- One failure is a one-pixel discrepancy in the synthetic two-line plate split
  test (`expected >= 36`, received `35`).

## Recommended implementation order

1. Pin/install a supported OpenCV 4.x build and repair the five failing tests.
2. Add and validate the four real edge model assets; run `veritrack_edge` on
   an actual stream before treating demo data as production evidence.
3. Deploy Stage 2 with PostgreSQL/Redis, then prove demo-to-database ingestion.
4. Add a worker that consumes Stage 2 sightings into `TrajectoryEngine` and
   exposes trajectory/GeoJSON queries.
5. Add analytics aggregates (camera counts, travel time, OD flows, congestion)
   and a versioned analytics API.
6. Build the GIS dashboard on those APIs; it should be the final product
   surface, not a frontend directly reading raw telemetry.
