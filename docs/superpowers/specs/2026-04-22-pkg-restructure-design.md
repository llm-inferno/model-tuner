# Design: Extract Estimator Logic into pkg/estimator and pkg/service

**Date:** 2026-04-22
**Issue:** #14
**Status:** Approved

## Summary

Extract the estimator orchestration logic from `tunerservice` into two new library packages so that Go consumers can use estimation primitives without depending on the HTTP service layer.

## Problem

`tunerservice` conflates two concerns:

1. **Estimator orchestration** — `TunerService`, `InitEstimator`, `SlidingWindowEstimator`, `ParameterStore`
2. **HTTP routing** — `TunerServer`, gin handlers, marshaling

This makes it impossible to import the estimation logic as a library without also pulling in gin, HTTP routing, and serialization as hard dependencies.

## Solution: Three-Layer Architecture

```
pkg/core/          ← unchanged: EKF Tuner
pkg/estimator/     ← new: pure estimation primitives (no orchestration, no HTTP)
pkg/service/       ← new: orchestration layer (imports pkg/estimator + pkg/core)
tunerservice/      ← thin HTTP adapter (imports pkg/service only)
cmd/tuner/main.go  ← unchanged except import paths
```

## Package Contents

### `pkg/estimator/` (new)

Pure estimation library. No imports from optimizer-light, gin, or any HTTP-related package.

**Files moved from `tunerservice/`:**
- `init_estimator.go` — `InitEstimator` struct and methods
- `sliding_window_estimator.go` — `SlidingWindowEstimator` struct and methods
- `init_estimator_test.go`
- `sliding_window_estimator_test.go`

**Shared type that moves here:**
- `fitObservation` (currently unexported in `tunerservice`, used by both estimators) — stays unexported

**Helper that moves here:**
- `guessInitState` → exported as `GuessInitState` — called by `InitEstimator.Fit()` (fallback) and by `pkg/service.createTuner` (initial state seeding). Must live in `pkg/estimator` to avoid an import cycle; exporting it lets `pkg/service` call it.

**New files:**
- `doc.go` — documents the estimation primitives

### `pkg/service/` (new)

Orchestration layer. Dispatches observations to `InitEstimator`, `SlidingWindowEstimator`, or `pkg/core.Tuner` depending on mode and state. Maintains per-(model, accelerator) parameter state.

**Files moved from `tunerservice/`:**
- `service.go` — `TunerService` struct and all methods
- `parameters.go` — `ParameterStore`, `LearnedParameters`, `covToSlice`
- `utils.go` — `buildEnvironments`, `groupByModelAccelerator`, `splitKey`, `maxBatchFromReplicas`, `setInitState`
- `service_sliding_test.go`
- `utils_test.go`

**Constants that move here** (split from `tunerservice/defaults.go`):
- `TUNER_WARM_UP_CYCLES`, `TUNER_INIT_OBS`, `TUNER_INIT_HOLD_BACK`
- `TUNER_ESTIMATOR_MODE`, `TUNER_WINDOW_SIZE`, `TUNER_RESIDUAL_THRESHOLD`
- `TUNER_INIT_FIT_THRESHOLD`
- `DefaultInitObs`, `DefaultInitHoldBack`, `DefaultWindowSize`, `DefaultResidualThreshold`, `DefaultInitFitThreshold`, `DefaultEstimatorMode`

**New files:**
- `doc.go` — documents TunerService orchestration

### `tunerservice/` (trimmed)

Thin HTTP adapter. No estimation logic.

**Files that stay:**
- `server.go` — `TunerServer`, gin setup, `Run`
- `handlers.go` — four handlers calling `pkg/service.TunerService`
- `doc.go` — trimmed to describe the HTTP adapter role

**Constants that stay** (trimmed `defaults.go`):
- `TUNER_HOST`, `TUNER_PORT`, `DefaultTunerHost`, `DefaultTunerPort`
- Merge defaults: `DefaultAccCount`, `DefaultMaxBatchSize`, `DefaultAtTokens`

**Helper that stays:**
- `validateKey` — used only by `handlers.go`

## `defaults.go` Split Detail

Current `tunerservice/defaults.go` (58 lines) splits into:

| Constant | Destination |
|---|---|
| `TunerHostEnvName`, `TunerPortEnvName`, `DefaultTunerHost`, `DefaultTunerPort` | `tunerservice/defaults.go` |
| `WarmUpCyclesEnvName` | `pkg/service/defaults.go` |
| `InitObsEnvName`, `InitHoldBackEnvName`, `DefaultInitObs`, `DefaultInitHoldBack` | `pkg/service/defaults.go` |
| `EstimatorModeEnvName`, `WindowSizeEnvName`, `ResidualThresholdEnvName`, `DefaultEstimatorMode`, `DefaultWindowSize`, `DefaultResidualThreshold` | `pkg/service/defaults.go` |
| `InitFitThresholdEnvName`, `DefaultInitFitThreshold` | `pkg/service/defaults.go` |
| `DefaultAccCount`, `DefaultMaxBatchSize`, `DefaultAtTokens` | `tunerservice/defaults.go` |
| `baseFactor` | `pkg/estimator/defaults.go` (used by `guessInitState`) |

## `cmd/tuner/main.go` Updates

The only consumer outside `tunerservice/`. Import paths update from `tunerservice` to `pkg/service` for:
- `NewTunerService`
- All estimator/service constants (`WarmUpCyclesEnvName`, `InitObsEnvName`, `EstimatorModeEnvName`, etc.)

HTTP constants (`TunerHostEnvName`, `TunerPortEnvName`) remain imported from `tunerservice`.

## Behavior

No behavior changes. The HTTP API surface (`POST /tune`, `GET /getparams`, `GET /warmup`, `POST /merge`) is identical. All environment variables and their defaults are unchanged.

## Tests

All tests move with their tested code. No new tests required — this is a mechanical refactor.

- `init_estimator_test.go` → `pkg/estimator/`
- `sliding_window_estimator_test.go` → `pkg/estimator/`
- `service_sliding_test.go` → `pkg/service/`
- `utils_test.go` → `pkg/service/`
