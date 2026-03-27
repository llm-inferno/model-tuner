# TunerService — Design & Implementation

## Design rationale

The tuner service uses a push-based design: the Controller calls `/tune` with
per-replica performance data (`ReplicaSpecs`) it already has from its own Collector,
and receives tuned queueing model parameters grouped by `(model, accelerator)` pair
ready to pass to the Optimizer. There is no internal polling loop or Collector
dependency.

---

## Architecture

```
Control-Loop Controller
  │
  │  POST /tune  ──[ []ServerSpec (ReplicaSpecs) ]──►  TunerServer
  │                                                         │
  │                                                    TunerService.Tune()
  │                                                         │
  │                                           ┌────── groupByModelAccelerator()
  │                                           │             │
  │                                           │  for each (model, accelerator):
  │                                           │    buildEnvironments()
  │                                           │    createTuner()  ◄── ParameterStore
  │                                           │    for each replica env:
  │                                           │      tuner.RunWithValidation()
  │                                           │    ParameterStore.Set(results)
  │                                           └─────────────│
  │                                                         │
  │  ◄──────────────[ ModelData (alpha/beta/gamma) ]────────┘
  │
  │  SystemData.Spec.Models = ModelData
  │
  │  POST /optimizeOne(SystemData) ──►  Optimizer
  │                                     └── uses alpha/beta/gamma in queue analysis
```

### Package structure

| File | Responsibility |
|------|---------------|
| `doc.go` | Package-level godoc |
| `defaults.go` | Constants (`baseFactor`) |
| `parameters.go` | `ParameterStore`, `LearnedParameters` |
| `service.go` | `TunerService` — core tuning logic |
| `utils.go` | Grouping, environment building, `guessInitState` |
| `server.go` | Gin HTTP server setup |
| `handlers.go` | `POST /tune`, `GET /getparams` |

---

## Data Flow

### Input: `[]config.ServerSpec` (ReplicaSpecs)

Each `ServerSpec` carries:

| Field | Maps to EKF environment |
|-------|-------------------------|
| `Model` | group key |
| `CurrentAlloc.Accelerator` | group key |
| `CurrentAlloc.Load.ArrivalRate` | `Lambda` (RPM) |
| `CurrentAlloc.Load.AvgInTokens` | `AvgInputTokens` |
| `CurrentAlloc.Load.AvgOutTokens` | `AvgOutputTokens` |
| `CurrentAlloc.TTFTAverage` | `AvgTTFT` (ms) — observed |
| `CurrentAlloc.ITLAverage` | `AvgITL` (ms) — observed |
| `MaxBatchSize` / `CurrentAlloc.MaxBatch` | `MaxBatchSize` |

Replicas with `ArrivalRate <= 0` are skipped (no traffic = no useful observation).

### EKF observation function (prefill-decode mode)

State vector: `[alpha, beta, gamma]`

The observation function `h(x)` runs a full M/G/1/K queue simulation
(via `queue-analysis/pkg/analyzer`) to predict `[TTFT, ITL]` from the current state
and environment. The EKF update corrects the state to minimize the difference between
predicted and observed `[TTFT, ITL]`.

### Output: `config.ModelData`

```json
{
  "models": [
    {
      "name": "llama3-8b",
      "acc": "A100",
      "maxBatchSize": 256,
      "perfParms": { "alpha": 12.28, "beta": 0.182, "gamma": 0.00093 }
    }
  ]
}
```

One entry per `(model, accelerator)` group seen in the input.

---

## EKF Features

### 1. State continuity

On each call, `createTuner()` checks the `ParameterStore` for previously learned
parameters. If found:

- The stored `alpha/beta/gamma` are used as `InitState` (warm start).
- The stored covariance matrix `P` is restored via `core.NewTunerWithCovariance()`,
  so the filter's confidence reflects accumulated learning rather than starting over.

### 2. Initial state guessing (`guessInitState`)

On first observation (no prior state), alpha/beta/gamma are derived algebraically from
the observed TTFT and ITL using the queueing model equations from the paper:

```
TTFT = alpha + (beta + gamma) * inputTokens      (eq 12)
ITL  = alpha + beta + gamma * (inputTokens + (outputTokens+1)/2)  (eq 13)
```

Steps:
1. `alpha  = baseFactor * ITL`  (baseFactor = 0.9)
2. `beta+gamma = (TTFT - alpha) / inputTokens`
3. `gamma = (ITL - alpha - (beta+gamma)) / (inputTokens + (outputTokens+1)/2 - 1)`
4. `beta  = (beta+gamma) - gamma`

If any derived parameter is ≤ 0, the function returns nil and the config file's
`InitState` defaults are used instead.

### 3. NIS validation (`computeNIS`)

After each EKF update, the Normalized Innovation Squared is computed:

```
NIS = y^T * S^-1 * y
```

where `y` is the innovation vector and `S` is the innovation covariance matrix.

Under the correct model, NIS follows a chi-squared distribution with 2 degrees of
freedom. Updates where `NIS >= 7.378` (the 97.5th percentile) are treated as outliers:
the filter is rolled back via `Stasher.UnStash()` and the previous valid state is
returned with `ValidationFailed = true`. This prevents parameter divergence from
measurement noise spikes.

---

## Changes to `pkg/core/`

| Change | Description |
|--------|-------------|
| `stasher.go` (new) | `Stasher` — snapshot/restore of filter state `X` and covariance `P` |
| `tuner.go` | Added `TunedResults`, `RunWithValidation()`, `extractTunedResults()`, `computeNIS()`, `NewTunerWithCovariance()` |
| `configurator.go` | Added `NewConfiguratorWithCovariance()` — initializes with a provided `P` matrix |

Existing `Run()` and all other behavior are unchanged.

---

## Configuration

Filter and model parameters are loaded from JSON files in `CONFIG_DATA_DIR`
(default: `../../samples`). Lookup order:

1. `<CONFIG_DATA_DIR>/<modelName>-config-data.json`
2. `<CONFIG_DATA_DIR>/default-config-data.json` (fallback)

Key `ModelData` fields used:

| Field | Effect |
|-------|--------|
| `initState` | Starting `[alpha, beta, gamma]` (overridden by prior params or guessInitState) |
| `percentChange` | Process noise scaling per state dimension |
| `boundedState` / `minState` / `maxState` | Clamp state after each update |
| `expectedObservations` | Scale measurement noise covariance `R` |

---

## HTTP API

### `POST /tune`

```
POST /tune
Content-Type: application/json

[ <ServerSpec>, ... ]   ← ReplicaSpecs from the Collector

200 OK
{ "models": [ { "name", "acc", "maxBatchSize", "perfParms": { "alpha", "beta", "gamma" } }, ... ] }
```

### `GET /getparams`

```
GET /getparams?model=llama3-8b&accelerator=A100

200 OK
{ "model": "llama3-8b", "accelerator": "A100", "alpha": 12.28, "beta": 0.182, "gamma": 0.00093,
  "nis": 0.054, "lastUpdated": "2026-03-26T..." }

404 Not Found   ← if no tuning has been performed for this pair yet
```

---

## Running the Demo

```bash
CONFIG_DATA_DIR=./samples TUNER_PORT=8081 go run ./demos/tunerservice/
```

The demo starts the server and immediately posts synthetic ReplicaSpecs to `/tune`,
printing the returned `ModelData` and then querying `/getparams`.

---

## Dependencies

| Module | Use |
|--------|-----|
| `github.com/llm-inferno/optimizer-light` | Input/output types: `ServerSpec`, `ModelData`, `ModelAcceleratorPerfData`, `PerfParms` |
| `github.com/llm-inferno/queue-analysis` | Queue analyzer used inside the EKF observation function |
| `github.com/llm-inferno/kalman-filter` | Extended Kalman Filter implementation |
| `github.com/gin-gonic/gin` | HTTP server |
| `gonum.org/v1/gonum/mat` | Matrix operations for NIS, covariance, state vectors |

`optimizer-light` resolves to the tagged `v0.7.0` release from the public registry.
