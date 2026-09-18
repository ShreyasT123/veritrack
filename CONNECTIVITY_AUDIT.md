# VeriTrack connectivity audit

Audit date: 2026-09-18. Scope: all tracked workspace source, script, frontend,
configuration, documentation, and runtime assets listed below. This is based on
direct import/call/route tracing and execution, not filenames.

## Executive connectivity matrix

| Module / file | Category | Status and runtime dependencies | Factual connectivity notes |
|---|---:|---|---|
| `src/veritrack_server/gateway.py` | 1 | FastAPI, Settings, DB/Redis/Kafka as configured | `create_app()` mounts `command_console_router` and root static files; `/telemetry/ingest*` calls `process_batch()`, which enqueues `SightingRow` objects to `Database`. |
| `src/veritrack_server/{schemas,config,crypto,bloom,db}.py` | 1 | Pydantic, crypto; production path requires PostgreSQL/Redis depending on config | Imported by gateway and invoked in validation, pseudonymisation, hotlist decision, and persistence. `InMemoryDatabase` supports tests/offline injection. |
| `src/veritrack_server/{api_routes,static/index.html}.py/html` | 1 | FastAPI route state / browser | Router is included at `/api/v1`; static HTML is mounted at `/`. Read routes work, but their Stage 5 state starts empty because ingestion does not populate it. |
| `src/veritrack_demo/{config,recognizer,localizer,pipeline,overlay,gateway_client,run_webcam,__init__}.py` | 3 | OpenCV, Pillow, torch/transformers, PP-OCR cache; webcam additionally needs physical camera | Executable demo graph: capture -> localize -> rectify/crop -> PP-OCR -> validate -> overlay; optional compact POST to gateway. Static-image route verified through `dry_run.py`. |
| `dry_run.py` | 3 | `.venv`, OpenCV, cached PP-OCRv6 Tiny, `samplecarimg.png` | Directly imports demo recognizer/localizer and completed a real CPU OCR run in this workspace. |
| `seed_corridor_demo.py` | 3 | Python package imports only; optional ERSS URL | Builds an in-memory `UnifiedApiState`, feeds Stage 4, fills ticker and one alert; verified successfully. It does **not** seed gateway-owned state in a running server. |
| `frontend/{package.json,vite.config.ts,tailwind.config.js,index.html,src/main.tsx,src/App.tsx,src/index.css}` | 3 | Node modules/Vite/Leaflet; browser network optional | `main.tsx` renders `App`; `App` invokes the typed API client and renders all six dashboard components. Production Vite build passed. |
| `frontend/src/{types/telemetry.ts,data/mockData.ts,services/api.ts}` | 3 | Browser Fetch / AbortSignal | `App` invokes `api.sightings/corridors/alerts/trajectory`; `fetchWithFallback()` targets mounted FastAPI routes then returns real embedded mock data on any error/non-200. |
| `frontend/src/components/{TopNav,SightingTicker,VehicleSearch,MapView,AlertPanel,CorridorSummary}.tsx` | 3 | React, Leaflet, lucide-react | All are imported by `App`; `MapView` uses vanilla Leaflet in `useEffect`, not React-Leaflet. |
| `src/veritrack_trajectory/{types,config,graph,fusion,anomaly,viterbi,engine,__init__}.py` | 2 | Pure Python; optional SciPy for speed | Fully implemented/tested library. No gateway startup code instantiates `TrajectoryEngine`; `api_routes` has optional fields but gateway sets neither engine nor rows. |
| `src/veritrack_analytics/{types,config,spatial,od_matrix,cpi,engine,__init__}.py` | 2 | `h3` | Complete tested library. `UnifiedApiState` constructs an empty engine, but gateway ingest never calls `ingest_sighting()` or `ingest_stage2_row()`. |
| `src/veritrack_server/{audit,dispatch}.py` | 2 | Python stdlib / optional ERSS endpoint | Implemented and tested; no normal gateway call constructs `EmergencyDispatcher`, calls `record_alert`, or submits CAP alerts. |
| `src/veritrack_edge/{config,types,errors,gating,rectify,splitter,tracking,validation,packaging}.py` | 2 | NumPy/OpenCV; called only by edge pipeline/tests | Genuine tested primitives, but no runnable production consumer exists here because the aggregate edge pipeline cannot load its models. |
| `src/veritrack_edge/{runtime,detection,ocr,reid,pipeline,ingest,run_node,__init__}.py` | 4 | Requires RTSP/video and `models/*.onnx` or RKNN files | `run_node -> EdgePipeline` invokes four model wrappers. Repository contains no `models/` directory and no required model artifacts, so the production edge path fails before processing frames. |
| `src/tests/{test_core_math,test_demo,test_stage2,test_stage3,test_stage4,test_stage5}.py` | 2 | pytest / local packages | Verification-only callers, not a production runtime. Entire suite executed green: 236 passed. |
| `src/Dockerfile.gateway`, `src/docker-compose.yml`, `src/k8s/veritrack-demo.yaml`, `pg.yml`, `src/.env.example`, `src/requirements*.txt`, `pyproject.toml`, `uv.lock` | 4 | Deployment/install metadata | Useful deployment declarations but not execution paths. PostgreSQL/Redis/Kafka were not launched or verified in this audit. |
| `src/README.md`, `README.md`, `WORKING_DIRECTORY_REPORT.md`, `VeriTrack_SIH2026.pdf` | 4 | Documentation | No imports/calls. Some statements are historical and do not establish runtime wiring. |
| `main.py` | 4 | Python only | Prints `welp`; no imports or caller. |
| `t.py` | 4 | Needs `image.png`; model access/cache | Obsolete one-off OCR experiment; hardcodes `image.png`, which is absent. Superseded by `dry_run.py`. |
| root `Dockerfile` | 4 | N/A | Empty file. |
| `clean.ps1` | 4 | PowerShell | Small manual cleanup helper; not called by any script. |
| `cropped_plate.png`, `samplecarimg.png`, `sample_30fps_1440.mp4` | 3 | Local assets | `samplecarimg.png` is selected by `dry_run.py` when no `image.png` exists. The cropped image and video have no automatic consumer. |
| `frontend/{build.ts,bunfig.toml,bun-env.d.ts,components.json,bun.lock,package-lock.json,README.md,src/index.ts,src/frontend.tsx,src/APITester.tsx,src/lib/utils.ts,src/index.html,styles/globals.css,src/logo.svg,src/react.svg,src/components/ui/*.tsx}` | 4 | Legacy Bun template paths/dependencies | Not imported by Vite entry `index.html -> src/main.tsx`. `build.ts` references removed `bun-plugin-tailwind`; frontend README still describes Bun template commands. |

## Category 1 — fully wired-in modules

### Stage 2 request path

The live server chain is real:

```text
POST /api/v1/telemetry/ingest or /ingest/compact
  -> gateway.ingest()/ingest_compact()
  -> process_batch()
  -> schema validation + hotlist + DPDP crypto
  -> Database.enqueue_sighting(s)
  -> Timescale COPY buffer, or injected InMemoryDatabase
```

`gateway.py` imports and uses `schemas.py`, `bloom.py`, `crypto.py`, `db.py`,
and `config.py`. This is the only current end-to-end server data path.

The router mount is also factual: `gateway.py` imports `command_console_router`,
calls `application.include_router(command_console_router)`, initializes
`application.state.stage5 = UnifiedApiState()`, then mounts
`StaticFiles(.../static, html=True)` at `/`.

**Important limitation:** route mounting is not event wiring. `process_batch()`
contains no call to `stage5.record_telemetry`, `stage5.analytics.ingest_sighting`,
`stage5.record_alert`, or `EmergencyDispatcher`. Consequently the mounted
`/telemetry/recent`, `/alerts/active`, `/analytics/*` endpoints return their
empty initial state in a normal gateway process.

### Browser connectivity

The new Vite entry is exactly `index.html -> src/main.tsx -> App.tsx`.
`App.tsx` calls `api.sightings()`, `api.corridors()`, and `api.alerts()` every
three seconds and calls `api.trajectory(plate)` on search. `api.ts` resolves
`/api/v1` when served by a non-5173 HTTP origin, else
`http://localhost:8000/api/v1`; its 1.2-second abort and catch block returns
fallback mocks. The calls therefore occur, but live payload compatibility is
partial: FastAPI trajectories are GeoJSON only, whereas the React map requires
`route` and `waypoints`; the client deliberately substitutes `mockTrajectory`
when those fields are missing.

## Category 2 — independent / standalone libraries

### Trajectory

`TrajectoryEngine` and its graph, Viterbi, fusion, anomaly, config, and type
modules are invoked by Stage 3 tests and may be invoked by `api_routes.py`
only if someone injects both `UnifiedApiState.trajectory_engine` and
`trajectory_rows`. Gateway lifecycle code does neither. No DB query fetches
rows for `/trajectories/{plate}`. Therefore default trajectory responses use
the `_empty_trajectory` provider and return no observations.

### Macro analytics

The H3/O-D/CPI engine is only fed by `seed_corridor_demo.py` and tests. The
gateway constructs an empty `MacroAnalyticsEngine` through `UnifiedApiState`;
there is no Stage 2-to-analytics adapter in the ingest path. Thus the endpoint
exists but its normal output has zero corridors and zero completed trips.

### Audit and emergency dispatch

Trajectory requests do append to `AuditLedger`, so the ledger has a mounted
route consumer. But CAP dispatch is inactive: no normal call builds an
`EmergencyDispatcher`, and the sole non-test `record_alert()` caller is the
standalone seeder. It is a complete library, not a live response system.

### Edge primitives and tests

The listed edge primitives have robust internal imports and test callers but
their only intended production caller is `EdgePipeline`, which cannot start
without model assets. The six test modules are executable validation programs,
not deployment modules.

## Category 3 — demo-ready showcase modules

The following are verified in this workspace without camera, Docker, Postgres,
Redis, Kafka, or ONNX assets:

1. `dry_run.py --image samplecarimg.png` uses the cached PP-OCRv6 Tiny model,
   manual ROI, and prints a structured OCR/telemetry readout. It was verified
   with `ZG7497-AH` at 88.65% confidence.
2. `seed_corridor_demo.py` builds its own in-memory Stage 5 state and produces
   8 cameras, 84 telemetry records, three CPI corridors, and one CAP alert.
3. The Vite frontend is browser-visible with mock fallback. `npm run build`
   succeeded; `npm run dev` exposes the console, which tries the server then
   switches to realistic replay data if unavailable.

`run_webcam.py` is **not** placed in this risk-free subset: it needs a real
camera and its Haar option currently relies on an OpenCV capability that was
previously absent in this environment. Use manual localizer only after a
hardware preflight.

## Category 4 — inactive, placeholder, or blocked modules

The production edge runner is blocked at model construction. Configuration
defaults name `models/vehicle_det.onnx`, `models/plate_kpt.onnx`,
`models/svtr_tiny_rec.onnx`, and `models/osnet_x0_25.onnx`; the repository has
no `models/` directory and filesystem inspection found no project ONNX/RKNN
models. Only ONNX package test files exist inside `.venv`.

The PP-OCR demo is different: a local Hugging Face cache has
`config.json`, `preprocessor_config.json`, and `model.safetensors` for
`PaddlePaddle/PP-OCRv6_tiny_rec_safetensors`; the files are cache links to HF
blobs, which is why their link length displays as zero but the model loads.

Clean up or exclude before evaluator handoff: empty root `Dockerfile`,
`main.py`, obsolete `t.py`, duplicate frontend Bun template source/assets,
stale frontend README, and unused derived `cropped_plate.png` if it is not
needed as evidence. Retain deployment manifests only if their corresponding
service prerequisites are supplied.

## Test result

Executed exactly:

```powershell
.\.venv\Scripts\python.exe -m pytest src\tests -q
```

Result: **236 passed in 6.45 seconds**.

## Three-minute video execution plan

Use these no-camera commands in separate terminals:

```powershell
.\.venv\Scripts\python.exe dry_run.py --image samplecarimg.png
.\.venv\Scripts\python.exe seed_corridor_demo.py
cd frontend; npm run dev
```

The first gives real OCR. The second proves the deterministic corridor/CAP
simulation. The third launches a visible command console with automatic local
replay if no gateway is running. Do not use `--webcam` or `run_node.py` in the
recorded take without a hardware/model preflight.

## Immediate 15-minute bridge actions

1. In `gateway.process_batch()`, after each accepted pseudonymous row is
   created, call `application.state.stage5.record_telemetry(...)` and
   `analytics.ingest_stage2_row(...)` using a camera-coordinate registry.
2. Construct `TrajectoryEngine` once during gateway lifespan and add a DB
   repository method to query recent sightings by `plate_pseudonym`; populate
   `trajectory_rows` on demand instead of its current empty dictionary.
3. Convert hotlist/anomaly outcomes into `CapAlert`, start one
   `EmergencyDispatcher` in lifespan, and call `record_alert()` after durable
   alert persistence. Stop it in lifespan shutdown.
4. Return a frontend contract that includes route and waypoint DTOs, or adapt
   GeoJSON in `api.ts`; currently a successful live GeoJSON route intentionally
   falls back to the mock route.
