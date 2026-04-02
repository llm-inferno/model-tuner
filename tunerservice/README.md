# tunerservice

EKF-based model parameter tuning service for integration with the llm-inferno control-loop.

## Overview

LLM inference servers are modeled by three parameters (alpha, beta, gamma) describing iteration time as a function of token workload:

```
iterationTime = alpha + beta*computedTokens + gamma*transferredTokens
```

This package continuously refines those parameters using an Extended Kalman Filter (EKF) fed by per-replica performance observations. Tuned parameters are stored in a thread-safe `ParameterStore` keyed by `model/accelerator` and returned as `optimizer-light` `ModelData`, ready for direct use by the Optimizer.

## HTTP API

### `POST /tune`

Accepts per-replica `ServerSpec` metrics from the control-loop Collector, runs EKF tuning for each `(model, accelerator)` group, and returns updated `ModelData`.

**Request body:** `[]config.ServerSpec`

```json
[
  {
    "Name": "llama3-8b/pod-0",
    "Model": "llama3-8b",
    "CurrentAlloc": {
      "Accelerator": "A100",
      "MaxBatch": 256,
      "TTFTAverage": 120.0,
      "ITLAverage": 15.0,
      "Load": { "ArrivalRate": 30.0, "Throughput": 30.0, "AvgInTokens": 512, "AvgOutTokens": 128 }
    },
    "MaxBatchSize": 256
  }
]
```

**Response:** `config.ModelData` with tuned `alpha`, `beta`, `gamma` per model/accelerator pair.

### `POST /merge`

Accepts the Controller's current `ModelData` and returns it with `PerfParms` (alpha/beta/gamma) overlaid from the `ParameterStore` for matching `(model, accelerator)` pairs. `ParameterStore` entries not present in the input are appended with tuned `PerfParms` and default non-parameter fields (`accCount=1`, `maxBatchSize=256`, `atTokens=1024`). An empty `models` array is valid input.

**Request body:** `config.ModelData`

**Response:** merged `config.ModelData`

### `GET /getparams?model=<name>&accelerator=<acc>`

Returns the most recently stored parameters for a specific model/accelerator pair without triggering a new tuning cycle.

**Response:**

```json
{
  "model": "llama3-8b",
  "accelerator": "A100",
  "alpha": 12.5,
  "beta": 0.03,
  "gamma": 0.01,
  "nis": 1.42,
  "lastUpdated": "2026-03-26T10:00:00Z"
}
```

## Control-Loop Integration

Intended usage from the control-loop `Controller`:

1. Call the Collector to obtain `ServerCollectorInfo` (includes `ReplicaSpecs`).
2. `POST` `ReplicaSpecs` to `/tune` — EKF parameters are updated in the `ParameterStore`.
3. `POST` current `ModelData` to `/merge` — receive merged `ModelData` with tuned `PerfParms` overlaid.
4. Set `SystemData.Spec.Models` to the merged `ModelData`.
5. `POST` `SystemData` to the Optimizer as usual.

## EKF Features

**State continuity** — previously tuned alpha/beta/gamma and their covariance matrix are restored at the start of each tuning cycle, so the filter converges faster over time rather than reinitializing from scratch.

**Initial state guessing** — on first observation, alpha/beta/gamma are derived algebraically from observed TTFT and ITL using the paper's queueing model equations, providing a warm start instead of cold defaults.

**NIS validation** — after each EKF update, the Normalized Innovation Squared (NIS = yᵀ S⁻¹ y) is checked against a chi-squared threshold (7.378 for 2 DOF at 97.5%). Updates that exceed the threshold are rejected and the filter is rolled back, preventing parameter divergence on outlier observations.

## Multi-Replica Tuning

Incoming `ReplicaSpecs` are grouped by `(Model, Accelerator)`. Within each group, one EKF predict+update cycle is run per replica with active traffic (`ArrivalRate > 0`), giving the filter multiple independent observations per tuning call.

## Configuration

Filter and model parameters are loaded from `default-config-data.json` in the directory specified by `CONFIG_DATA_DIR` (default: `config-data`). The tuner service always uses the `default` config type; model name does not affect which config file is loaded.

| Variable | Purpose | Default |
|---|---|---|
| `CONFIG_DATA_DIR` | Directory with JSON config files | `config-data` |
| `TUNER_HOST` | Server listen address | `localhost` |
| `TUNER_PORT` | Server listen port | `8081` |

## Running the Demo

```bash
go run ./demos/tunerservice
```

Starts the tuner HTTP server on the configured host and port, ready to accept `POST /tune`, `POST /merge`, and `GET /getparams` requests.
