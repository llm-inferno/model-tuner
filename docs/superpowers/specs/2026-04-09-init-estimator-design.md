# Multi-Observation Parameter Estimator — Design Spec

**Date**: 2026-04-09
**Repo**: `github.com/llm-inferno/model-tuner`
**Status**: Approved

---

## Problem

`guessInitState` initialises the EKF from a single observation using the zero-load queueing
equations. When the server is at moderate-to-high utilisation at the time of the first
observation, queueing delay inflates the observed TTFT and ITL, causing the algebraic inversion
to return a parameter estimate that is far from the true values. The EKF then converges to a
wrong local minimum with low NIS and drifts only slowly toward the truth.

Root cause: parameter non-identifiability at a single operating point combined with queueing
inflation of the zero-load approximation.

---

## Solution

Accumulate K observations (each with a potentially different arrival rate and token-count
distribution) before starting the EKF. Use Nelder-Mead minimisation over the **full** queueing
model forward evaluation to find the (α, β, γ) that best explains all K observations jointly.
Start the EKF from that fitted estimate instead of the single-observation guess.

Token-count variation across observations (inTokens 90–165, outTokens 670–1250 in practice)
provides enough diversity to resolve the identifiability problem even when arrival rate does not
vary much.

---

## Components

### `tunerservice/init_estimator.go` (new file)

```
InitEstimator
  observations []fitObservation   // one per Tune call
  minObs       int                // K, from TUNER_INIT_OBS env var (default 5)
  holdBack     bool               // from TUNER_INIT_HOLD_BACK env var (default true)

fitObservation
  Lambda          float64   // arrival rate, req/min
  MaxBatch        int
  AvgInputTokens  float32
  AvgOutputTokens float32
  AvgTTFT         float64   // observed, ms
  AvgITL          float64   // observed, ms
```

**Methods**:
- `AddObservation(env *core.EnvironmentPrefillDecode)` — appends to observations slice
- `IsReady() bool` — returns `len(observations) >= minObs`
- `HoldBack() bool` — returns `holdBack`
- `Fit(configData *config.ConfigData) ([]float64, error)` — runs Nelder-Mead and returns
  `[alpha, beta, gamma]`; falls back to `guessInitState` on the first observation if the fit
  fails; falls back to config `initState` if `guessInitState` also returns nil

### Objective function (inside `Fit`)

```
f(α, β, γ) = Σᵢ [ ((TTFT_model - TTFT_obs) / TTFT_obs)² +
                   ((ITL_model  - ITL_obs)  / ITL_obs)²  ]
```

where `TTFT_model` and `ITL_model` come from `queue-analysis Analyze()` evaluated at each
stored observation's (λ, maxBatch, tokens). Negative parameter values return `math.MaxFloat64`.

**Starting point**: `guessInitState` on the first observation; config `initState` if nil.
**Termination**: gonum default settings (function tolerance 1e-6, max 500 evals).

### Changes to `tunerservice/service.go`

- `TunerService` gains `estimators map[string]*InitEstimator`
- `NewTunerService` creates the map; reads `TUNER_INIT_OBS` and `TUNER_INIT_HOLD_BACK`
- `tuneGroup` calls `estimator.AddObservation(envs[0])` after `buildEnvironments` succeeds
  (i.e. after the `len(envs) == 0` early-return guard, so only valid observations are stored)
  - If `!estimator.IsReady()`: return early (no EKF this cycle)
  - If `estimator.IsReady()` and no prior paramStore entry: call `Fit()`, pass result to
    `setInitState` before `core.NewTuner`
- `IsWarmingUp()` returns `true` also when any estimator is not ready and `holdBack=true`

### Changes to `tunerservice/defaults.go`

Add:
```go
const (
    InitObsEnvName      = "TUNER_INIT_OBS"
    InitHoldBackEnvName = "TUNER_INIT_HOLD_BACK"
    DefaultInitObs      = 5
    DefaultInitHoldBack = true
)
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `TUNER_INIT_OBS` | `5` | Observations to collect before fitting |
| `TUNER_INIT_HOLD_BACK` | `true` | Report warmingUp=true during collection (Option B). Set false for Option A (proceed with static model-data during collection). |

No changes to the controller or any other component.

---

## Warm-up phases (with defaults)

```
Phase 1 — collection  (TUNER_INIT_OBS=5 cycles)
  IsWarmingUp() = true (holdBack=true)
  Controller skips optimize+actuate; uses static model-data.json

Phase 2 — EKF warm-up  (TUNER_WARM_UP_CYCLES=3 cycles)
  IsWarmingUp() = true
  EKF runs from Fit() result, NIS gate disabled

Phase 3 — normal operation
  IsWarmingUp() = false
  NIS gate enabled; controller optimizes and actuates
```

Total delay with defaults: `(5 + 3) × controlPeriod`.

---

## Error handling

| Situation | Behaviour |
|---|---|
| Nelder-Mead fails or returns non-positive params | Fall back to `guessInitState(firstObs)`; log warning |
| `guessInitState` returns nil | Fall back to config `initState`; log warning |
| `Analyze()` returns error for a trial point | Return `math.MaxFloat64` for that point |
| All observations at same operating point | Fit may be inaccurate; EKF corrects during warm-up |

---

## What is NOT changed

- Controller, collector, actuator, optimizer — no changes
- Tuner REST API (`/tune`, `/merge`, `/warmup`, `/getparams`) — no changes
- `guessInitState` — kept, used as Nelder-Mead starting point and fallback
- EKF warm-up cycles (`TUNER_WARM_UP_CYCLES`) — unchanged
- `ParameterStore` state continuity across cycles — unchanged

---

## Files touched

| File | Change |
|---|---|
| `tunerservice/init_estimator.go` | New — `InitEstimator`, `fitObservation`, objective, `Fit()` |
| `tunerservice/service.go` | Add estimator map, wire into `tuneGroup`, update `IsWarmingUp` |
| `tunerservice/defaults.go` | Add new env var names and defaults |
| `tunerservice/parameters.go` | No changes needed |
| `deploy/configmap.yaml` | No change needed (new config via env vars, not configmap) |
