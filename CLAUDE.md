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

There are no unit tests (`*_test.go` files). Validation is done by running demo programs and inspecting their CSV/console output.

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

### Service environment variables

| Variable | Purpose | Default |
|---|---|---|
| `CONFIG_DATA_DIR` | Directory with JSON config files | `config-data` |
| `TUNER_HOST` / `TUNER_PORT` | Tuner REST server address | `localhost:8081` |
| `COLLECTOR_HOST` / `COLLECTOR_PORT` | Prometheus collector address | — |
| `TOKEN` | Bearer token for Prometheus | — |
| `PROMETHEUS_ADDRESS` | Prometheus server URL | — |
