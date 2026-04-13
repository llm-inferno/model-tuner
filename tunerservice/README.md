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

### `GET /warmup`

Returns whether the tuner is still in a warm-up phase (collection or EKF warm-up) for any known `(model, accelerator)` pair. The control-loop Controller polls this endpoint each cycle to decide whether to skip optimize+actuate.

**Response:**

```json
{ "warmingUp": true }
```

Returns `true` if:
- any pair is still collecting initial observations (`TUNER_INIT_HOLD_BACK=true` and fewer than `TUNER_INIT_OBS` observations accumulated), or
- any pair has fewer than `TUNER_WARM_UP_CYCLES` accepted EKF updates (NIS gate is disabled during this window).

Returns `false` once all pairs have graduated to normal operation, or if `TUNER_WARM_UP_CYCLES=0`.

## Control-Loop Integration

Intended usage from the control-loop `Controller`:

1. Call the Collector to obtain `ServerCollectorInfo` (includes `ReplicaSpecs`).
2. `POST` `ReplicaSpecs` to `/tune` — EKF parameters are updated in the `ParameterStore`.
3. `POST` current `ModelData` to `/merge` — receive merged `ModelData` with tuned `PerfParms` overlaid.
4. Set `SystemData.Spec.Models` to the merged `ModelData`.
5. `POST` `SystemData` to the Optimizer as usual.

## EKF Features

**State continuity** — previously tuned alpha/beta/gamma and their covariance matrix are restored at the start of each tuning cycle, so the filter converges faster over time rather than reinitializing from scratch.

**Initial parameter estimation** — on first use of a `(model, accelerator)` pair, the service runs a multi-observation Nelder-Mead fit before starting the EKF. It accumulates `TUNER_INIT_OBS` (default 5) operating-point snapshots across control cycles, then minimises mean squared error in TTFT and ITL across all observations jointly to find (α, β, γ) that best explain all of them. This approach is robust at any traffic level; a single-observation zero-load inversion (the previous approach) inflates α at moderate-to-high utilization where the light-traffic assumption breaks down. Token-count variation across observations resolves the three-parameter identifiability problem. Variables are scaled by the starting point before entering the optimizer so the Nelder-Mead simplex is well-conditioned across parameters that span orders of magnitude (α~5, β~0.05, γ~0.00005). If the Nelder-Mead fit fails or returns non-positive parameters, the service falls back to the single-observation algebraic inversion.

**NIS validation** — after each EKF update, the Normalized Innovation Squared (NIS = yᵀ S⁻¹ y) is checked against a chi-squared threshold (7.378 for 2 DOF at 97.5%). Updates that exceed the threshold are rejected and the filter is rolled back, preventing parameter divergence on outlier observations.

## Warm-up Phases

Each `(model, accelerator)` pair goes through three phases before reaching normal operation:

| Phase | Duration | Behavior |
|---|---|---|
| **Collection** | `TUNER_INIT_OBS` cycles (default 5) | Observations accumulated; `GET /warmup` returns `true` if `TUNER_INIT_HOLD_BACK=true`; controller skips optimize+actuate |
| **EKF warm-up** | `TUNER_WARM_UP_CYCLES` cycles (default 5) | EKF runs from Nelder-Mead fit result with NIS gate disabled; `GET /warmup` returns `true` |
| **Normal** | ongoing | NIS gate active; parameters tracked continuously; `GET /warmup` returns `false` |

Set `TUNER_INIT_HOLD_BACK=false` to let the controller proceed with static model data during collection instead of holding back.

### Interaction between TUNER_INIT_OBS and TUNER_WARM_UP_CYCLES

The two variables govern **sequential** phases — there is no overlap or interference:

- During collection, the EKF never runs and `paramStore` stays empty, so the EKF warm-up check (`UpdateCount < TUNER_WARM_UP_CYCLES`) never fires. The hold-back signal during collection comes entirely from the estimator check.
- Once collection completes and `Fit()` runs, the EKF starts and `paramStore` gains entries. The hold-back signal then transitions to the EKF warm-up check. The estimator check no longer fires (estimator is ready).
- `GET /warmup` returns `true` across both phases, presenting a single "not ready" signal to the controller.

**Total controller hold-back** when `TUNER_INIT_HOLD_BACK=true` is therefore:

```
hold-back cycles = TUNER_INIT_OBS + TUNER_WARM_UP_CYCLES  (defaults: 5 + 5 = 10)
```

At a 60-second control period this is 10 minutes before the controller first invokes the optimizer with tuned parameters. Lower one or both values for faster startup if the operating conditions are well-understood.

## Multi-Replica Tuning

Incoming `ReplicaSpecs` are grouped by `(Model, Accelerator)`. Within each group, one EKF predict+update cycle is run per replica with active traffic (`ArrivalRate > 0`), giving the filter multiple independent observations per tuning call.

## Configuration

Filter and model parameters are loaded from `default-config-data.json` in the directory specified by `CONFIG_DATA_DIR` (default: `config-data`). The tuner service always uses the `default` config type; model name does not affect which config file is loaded.

| Variable | Purpose | Default |
|---|---|---|
| `CONFIG_DATA_DIR` | Directory with JSON config files | `config-data` |
| `TUNER_HOST` | Server listen address | `localhost` |
| `TUNER_PORT` | Server listen port | `8081` |
| `TUNER_WARM_UP_CYCLES` | Accepted EKF updates during which the NIS gate is disabled | `5` |
| `TUNER_INIT_OBS` | Observations to accumulate before running the Nelder-Mead initial parameter fit; set to `1` to revert to single-observation algebraic inversion | `5` |
| `TUNER_INIT_HOLD_BACK` | If `true`, report `warmingUp=true` during collection so the controller skips optimize+actuate; if `false`, controller proceeds with static model data | `true` |

## Running the Demo

```bash
go run ./demos/tunerservice
```

Starts the tuner HTTP server on the configured host and port, ready to accept `POST /tune`, `POST /merge`, and `GET /getparams` requests.
