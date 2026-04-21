# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build all packages
go build ./...

# Build a specific demo
go build ./demos/simulated-observer

# Format code
go fmt ./...

# Run a demo (from project root)
go run ./demos/simulated-observer
go run ./demos/offline-observer
go run ./demos/benchmark
go run ./demos/tunerservice
```

Unit tests exist for `tunerservice/` (`go test ./tunerservice/...`). Demo programs in `demos/` are used for broader validation by running them and inspecting their CSV/console output.

## Architecture

The model-tuner dynamically tunes queueing model parameters (alpha, beta, gamma) for LLM inference servers using an Extended Kalman Filter (EKF). It observes server metrics, feeds them into the EKF, and exposes tuned parameters via a REST API.

**Data flow:**
```
Observer → Environment → Tuner (EKF predict + update) → GetParams() → TunerService REST API
```

### Core packages

- **`pkg/core/`** — The EKF-based tuner. `Tuner` wraps a Kalman filter and a `Configurator` (covariance matrices, initial state, bounds). `Environment` variants carry the observations fed into the filter. `SystemFuncCreator` implementations supply the nonlinear observation function (h(x)) for different serving modes.

- **`pkg/observer/`** — Four observer implementations, all returning a `core.Environment`:
  - `SimulatedObserver`: synthetic metrics with configurable noise
  - `OfflineObserver`: replays CSV traces
  - `OnlineObserver`: queries a live Prometheus server (uses `TOKEN` and `PROMETHEUS_ADDRESS` env vars)
  - `DataObserver`: processes pre-collected benchmark data

- **`pkg/config/`** — `ConfigData` struct with `FilterData` and `ModelData` sub-structs. JSON config files live in `config-data/`.

- **`pkg/metrics/`** — Thin Prometheus HTTP client used by `OnlineObserver`.

- **`tunerservice/`** — Gin-based HTTP server for control-loop integration. Accepts `POST /tune` with `[]config.ServerSpec` (ReplicaSpecs from the Collector) and returns updated `config.ModelData` (alpha/beta/gamma per model/accelerator). Also exposes `GET /getparams?model=<name>&accelerator=<acc>` for point lookups.

### Environment variants

Two concrete `Environment` implementations represent different LLM serving modes, both embedding an unexported `environmentBase` (lambda, batch size, avg queue time, max batch size):
- `EnvironmentDecode`: base + avg output tokens + avg ITL; observations = `[AvgQueueTime, AvgITL]`; state = [alpha, beta]
- `EnvironmentPrefillDecode`: base + avg input/output tokens + TTFT + ITL; observations = `[AvgTTFT, AvgITL]`; state = [alpha, beta, gamma]

The active variant determines which `SystemFuncCreator` (decode vs. prefill-decode) is used.

### Configuration

JSON configs are loaded from the directory specified by `CONFIG_DATA_DIR` (default: `config-data`). Sample configs in `config-data/`: `default-config-data.json`, `decode-config-data.json`, `prefill-decode-config-data.json`, `benchmark-config-data.json`.

Key `ModelData` fields: `initState` ([alpha, beta, gamma] initial values), `boundedState`/`minState`/`maxState` (EKF state clamping), `percentChange` (process noise scaling).

### Deployment

The primary deployment is as a sidecar container in the `inferno` pod (see `github.com/llm-inferno/control-loop/yamls/deploy/deploy-loop.yaml`), listening on port 3304 with its config ConfigMap mounted at `/etc/tuner/config`. A standalone deployment manifest is also provided in `deploy/deploy-model-tuner.yaml` for independent operation.

### Service environment variables

| Variable | Purpose | Default |
|---|---|---|
| `CONFIG_DATA_DIR` | Directory with JSON config files | `config-data` (`/etc/tuner/config` in inferno pod) |
| `TUNER_HOST` / `TUNER_PORT` | Tuner REST server address | `localhost:8081` (`localhost:3304` in inferno pod) |
| `TUNER_WARM_UP_CYCLES` | Number of accepted EKF updates during which the NIS gate is disabled | `5` |
| `TUNER_INIT_OBS` | Number of observations to collect before fitting initial parameters via Nelder-Mead | `5` |
| `TUNER_INIT_HOLD_BACK` | If `true`, report `warmingUp=true` during collection (controller skips optimize+actuate). If `false`, controller proceeds with static model-data during collection. | `true` |
| `TUNER_ESTIMATOR_MODE` | Estimation backend: `ekf` or `sliding-window` | `ekf` |
| `TUNER_WINDOW_SIZE` | (SWNM) Sliding window capacity | `10` |
| `TUNER_RESIDUAL_THRESHOLD` | (SWNM) Per-observation relative error cutoff for outlier rejection | `0.5` |
| `TUNER_INIT_FIT_THRESHOLD` | (SWNM) If `InitEstimator.Fit()` objective value exceeds this, the pair is permanently routed to EKF instead. Set to `0` to disable. | `10.0` |
| `COLLECTOR_HOST` / `COLLECTOR_PORT` | Prometheus collector address | — |
| `TOKEN` | Bearer token for Prometheus | — |
| `PROMETHEUS_ADDRESS` | Prometheus server URL | — |
