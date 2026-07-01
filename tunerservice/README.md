# tunerservice

Model parameter tuning service for integration with the llm-inferno control-loop. Supports two estimation backends: EKF (default) and Sliding-Window Nelder-Mead (SWNM).

## Overview

LLM inference servers are modeled by three parameters (alpha, beta, gamma) describing iteration time as a function of token workload:

```
iterationTime = alpha + beta*computedTokens + gamma*transferredTokens
```

This package continuously refines those parameters from per-replica performance observations and stores them in a thread-safe `ParameterStore` keyed by `model/accelerator`, returned as `optimizer-light` `ModelData` ready for direct use by the Optimizer.

Two estimation backends are available, selected via `TUNER_ESTIMATOR_MODE`:

- **EKF** (default) — Extended Kalman Filter with NIS gate; fast per-cycle updates and state continuity across cycles.
- **Sliding-Window Nelder-Mead (SWNM)** — re-fits [α,β,γ] via Nelder-Mead on every cycle over a fixed-size FIFO window of recent observations; no covariance matrices to tune, and includes residual-based outlier rejection. Use this when the EKF diverges or NIS-gate misfires cause bad parameter estimates.

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

Accepts the Controller's current `ModelData` and returns it with `PerfParms` (alpha/beta/gamma) overlaid from the `ParameterStore` for matching `(model, accelerator)` pairs. `ParameterStore` entries not present in the input are appended with tuned `PerfParms` and default non-parameter fields (`accCount=1`, `maxBatchSize=256`). An empty `models` array is valid input.

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
- **EKF mode:** any pair has fewer than `TUNER_WARM_UP_CYCLES` accepted EKF updates (NIS gate is disabled during this window), or
- **SWNM mode:** (no extra warmup phase; `IsReady()` returns `true` immediately after seeding with `TUNER_INIT_OBS` observations)

Returns `false` once all pairs have graduated to normal operation, or if `TUNER_WARM_UP_CYCLES=0` (EKF) / collection phase complete (SWNM).

### `GET /calibration-status`

Reports, per `(model, accelerator)` pair the tuner has begun observing, the facts the control-loop's **benchmarking-on-the-fly** trigger consumes. `needsCalibration` is `true` when the pair has collected its initial observations, the resulting fit is **ill-conditioned** (`conditionNumber > TUNER_MAX_CONDITION_NUMBER` — natural load did not span enough operating points to identify β/γ), and the pair has not been calibrated yet. Pairs not yet seen by `/tune` are absent.

**Response:**

```json
{ "statuses": [
  { "model": "llama3-8b", "accelerator": "A100",
    "storePresent": true, "calibrated": false,
    "obsCount": 3, "obsTarget": 3,
    "conditionNumber": 1.2e18, "illConditioned": true, "needsCalibration": true }
] }
```

### `POST /calibrate`

Fits `(alpha, beta, gamma)` jointly from a **batch of deliberately-diverse swept operating points** for each `(model, accelerator)` group, in one shot — the persistent-excitation cure for the single-operating-point unidentifiability the guards above only mitigate. Unlike `/tune` (one operating point per cycle), the joint multi-point fit reuses the same `InitEstimator` Nelder-Mead + condition-number and fit-quality guards. On success it stores the fit graduated (so the warm-up gate clears) and seeds the per-pair estimators from the sweep so subsequent `/tune` cycles track drift from the calibrated point. The control-loop Collector produces the sweep points; the controller drives this endpoint only when `/calibration-status` reports `needsCalibration`.

**Request body:** `[]config.ServerSpec` (the swept operating points; same shape as `/tune`).

**Response:** `config.ModelData` containing only the groups successfully calibrated in this call — a group whose fit is rejected is omitted, so parameters left in the store by a prior `/tune` or `/calibrate` never leak into the response as if freshly calibrated. A group's fit is rejected when it remains ill-conditioned (the sweep grid lacked operating-point spread) or is otherwise poor/degenerate (fit residual above `TUNER_INIT_FIT_THRESHOLD`, or Nelder-Mead fell back to a single-point guess). `422` if no group in the batch could be calibrated.

Calibration state (`calibrated` flags, `ParameterStore`) is in-memory — a pair is re-calibrated after a tuner restart.

## Control-Loop Integration

Intended usage from the control-loop `Controller`:

1. Call the Collector to obtain `ServerCollectorInfo` (includes `ReplicaSpecs`).
2. `POST` `ReplicaSpecs` to `/tune` — EKF parameters are updated in the `ParameterStore`.
3. *(optional, benchmarking-on-the-fly)* `GET /calibration-status`; for any pair with `needsCalibration`, drive a short load sweep and `POST` the points to `/calibrate` before merging, so an identifiable fit replaces the ill-conditioned one this cycle.
4. `POST` current `ModelData` to `/merge` — receive merged `ModelData` with tuned `PerfParms` overlaid.
5. Set `SystemData.Spec.Models` to the merged `ModelData`.
6. `POST` `SystemData` to the Optimizer as usual.

## Estimation Features

### Common (both modes)

**Initial parameter estimation** — on first use of a `(model, accelerator)` pair, the service accumulates `TUNER_INIT_OBS` (default 5) operating-point snapshots across control cycles, then runs a Nelder-Mead fit to find (α, β, γ) that jointly minimise mean squared error in TTFT and ITL. This approach is robust at any traffic level; a single-observation zero-load inversion inflates α at moderate-to-high utilization where the light-traffic assumption breaks down. Variables are scaled by the starting point so the simplex is well-conditioned across parameters spanning orders of magnitude (α~5, β~0.05, γ~0.00005).

### EKF mode (default, `TUNER_ESTIMATOR_MODE=ekf`)

**State continuity** — previously tuned alpha/beta/gamma and their covariance matrix are restored at the start of each tuning cycle, so the filter converges faster over time rather than reinitializing from scratch.

**NIS validation** — after each EKF update, the Normalized Innovation Squared (NIS = yᵀ S⁻¹ y) is checked against a chi-squared threshold (7.378 for 2 DOF at 97.5%). Updates that exceed the threshold are rejected and the filter is rolled back, preventing parameter divergence on outlier observations.

### Sliding-Window Nelder-Mead mode (`TUNER_ESTIMATOR_MODE=sliding-window`)

**Continuous re-fitting** — every tuning cycle, Nelder-Mead is run over the `TUNER_WINDOW_SIZE` (default 10) most recent observations. No covariance matrices to configure; convergence failure simply retains the previous estimate.

**Warm-start** — each `Fit()` call uses the previous fitted result as the Nelder-Mead starting point (`x0`), falling back to a heuristic estimate only on the very first call. This prevents Nelder-Mead from restarting from a noisy single-observation estimate each cycle.

**Residual-based outlier rejection** — after an initial fit, the observation with the highest relative squared error is dropped if its residual exceeds `TUNER_RESIDUAL_THRESHOLD` (default 0.5), then Nelder-Mead runs once more on the cleaned window. Only one observation is removed per cycle to avoid discarding good observations that appear anomalous only because the initial fit was corrupted by the outlier.

**Seeding** — the `TUNER_INIT_OBS` collection-phase observations pre-fill the sliding window and the `InitEstimator`'s fit result seeds the warm-start `x0`. `IsReady()` returns `true` immediately after seeding — no window-filling phase.

**EKF fallback on poor init fit** — when the `InitEstimator`'s Nelder-Mead objective value (`funcValue`) exceeds `TUNER_INIT_FIT_THRESHOLD` (default 10.0), the `(model, accelerator)` pair is permanently routed to EKF instead of SWNM. This handles the low-utilisation identifiability problem: when observations span a narrow RPM range, the loss surface is flat and Nelder-Mead converges to a degenerate solution that SWNM's warm-start would then propagate. A `funcValue` of 10.0 corresponds roughly to 100% average relative error across 5 observations. Set `TUNER_INIT_FIT_THRESHOLD=0` to disable the fallback and always use SWNM.

**Identifiability guard** (`TUNER_MAX_CONDITION_NUMBER`, default 1000) — a complementary, residual-independent check for the same class of problem. A degenerate fit can be both positive **and** low-residual (so the `funcValue` and outlier checks pass) yet still unidentifiable — it collapses β/γ toward zero and inflates α because the observation window does not span enough operating points (the classic case is a single-replica deployment, where every observation sits at one token point and no token spread exists). After each fit the estimator computes the condition number of the residual Jacobian, taken with respect to the *log* of each parameter so the measure is scale-invariant across α~O(10), β~O(0.01), γ~O(1e-4); a flat (unidentifiable) parameter direction yields a vanishing singular value and a very large condition number. Above the threshold the `SlidingWindowEstimator` holds its last good fit instead of adopting the degenerate solution, and the `InitEstimator` falls back to `GuessInitState` (reporting a benign funcValue so the pair stays on the guarded SWNM path rather than escalating to EKF). Healthy fits sit well below ~100; degenerate fits range from a few thousand to effectively infinite. Set `TUNER_MAX_CONDITION_NUMBER=0` to disable.

**Transient EKF excursion** (issue #19) — when the `SlidingWindowEstimator` holds a last good fit on an ill-conditioned cycle, it does not emit that stale value directly. Instead it runs one EKF predict+update seeded at the held fit using the offending observation, then resumes SWNM next cycle (no mode change). At a single operating point the Kalman gain in the unobservable β/γ direction is ≈0, so the excursion holds β/γ near the seed and nudges only the observable combination (≈α) to fit the point — emitting a feasible, point-consistent fit. Worst case (update rejected/clamped) it returns the held fit, identical to before.

**Seed-anchored cold-start guess** (issue #17) — at cold start with no prior good fit, `GuessInitState` is anchored to the config `initState` seed: it pins the unidentifiable γ to the seed's γ and solves α,β from the observation (with γ fixed, the two latency equations make α,β jointly identifiable), falling back to the full seed if that solve is degenerate. This replaces the legacy `α = 0.9·ITL` heuristic, which at a single operating point under load misattributed a batch-induced latency excess into γ — inflating it ~15–21× into a regime where the optimizer returned no feasible allocation. With no seed available, the legacy heuristic remains as the ultimate fallback.

## Warm-up Phases

### EKF mode

Each `(model, accelerator)` pair goes through three phases:

| Phase | Duration | Behavior |
|---|---|---|
| **Collection** | `TUNER_INIT_OBS` cycles (default 5) | Observations accumulated; `GET /warmup` returns `true` if `TUNER_INIT_HOLD_BACK=true`; controller skips optimize+actuate |
| **EKF warm-up** | `TUNER_WARM_UP_CYCLES` cycles (default 5) | EKF runs from Nelder-Mead fit result with NIS gate disabled; `GET /warmup` returns `true` |
| **Normal** | ongoing | NIS gate active; parameters tracked continuously; `GET /warmup` returns `false` |

**Total controller hold-back** when `TUNER_INIT_HOLD_BACK=true`:

```
hold-back cycles = TUNER_INIT_OBS + TUNER_WARM_UP_CYCLES  (defaults: 5 + 5 = 10)
```

### SWNM mode

| Phase | Duration | Behavior |
|---|---|---|
| **Collection** | `TUNER_INIT_OBS` cycles (default 5) | Observations accumulated; `GET /warmup` returns `true` if `TUNER_INIT_HOLD_BACK=true` |
| **Normal** | ongoing | Nelder-Mead re-fit every cycle; `GET /warmup` returns `false` |

The sliding window is seeded with the `TUNER_INIT_OBS` collection observations and `IsReady()` is `true` immediately — there is no window-filling phase.

Set `TUNER_INIT_HOLD_BACK=false` to let the controller proceed with static model data during collection instead of holding back.

## Multi-Replica Tuning

Incoming `ReplicaSpecs` are grouped by `(Model, Accelerator)`. Within each group, one EKF predict+update cycle is run per replica with active traffic (`ArrivalRate > 0`), giving the filter multiple independent observations per tuning call.

## Configuration

Filter and model parameters are loaded from `default-config-data.json` in the directory specified by `CONFIG_DATA_DIR` (default: `config-data`). The tuner service always uses the `default` config type; model name does not affect which config file is loaded.

| Variable | Purpose | Default |
|---|---|---|
| `CONFIG_DATA_DIR` | Directory with JSON config files | `config-data` |
| `TUNER_HOST` | Server listen address | `localhost` |
| `TUNER_PORT` | Server listen port | `8081` |
| `TUNER_WARM_UP_CYCLES` | (EKF) Accepted EKF updates during which the NIS gate is disabled | `5` |
| `TUNER_INIT_OBS` | Observations to accumulate before running the Nelder-Mead initial parameter fit | `5` |
| `TUNER_INIT_HOLD_BACK` | If `true`, report `warmingUp=true` during collection so the controller skips optimize+actuate; if `false`, controller proceeds with static model data | `true` |
| `TUNER_ESTIMATOR_MODE` | Estimation backend: `ekf` or `sliding-window` | `ekf` |
| `TUNER_WINDOW_SIZE` | (SWNM) Number of observations in the sliding window | `10` |
| `TUNER_RESIDUAL_THRESHOLD` | (SWNM) Per-observation relative error cutoff for outlier rejection | `0.5` |
| `TUNER_INIT_FIT_THRESHOLD` | (SWNM) Nelder-Mead objective threshold; if `InitEstimator.Fit()` exceeds this the pair falls back to EKF permanently. `0` disables. | `10.0` |
| `TUNER_MAX_CONDITION_NUMBER` | Identifiability guard: reject a fit whose relative-scaled Jacobian condition number exceeds this (degenerate/unidentifiable, e.g. collapsed β/γ). Holds last-good or `GuessInitState`. `0` disables. | `1000.0` |

## Running the Demo

```bash
go run ./demos/tunerservice
```

Starts the tuner HTTP server on the configured host and port, ready to accept `POST /tune`, `POST /merge`, and `GET /getparams` requests.
