# Sliding-Window Nelder-Mead Estimator — Design Spec

**Date:** 2026-04-20
**Status:** Approved

## Problem

The EKF filter (Stages 2–3 of the tuner lifecycle) misbehaves in practice after Stage 1 produces good initial parameters. Filter divergence and NIS gate misfires cause the controller to receive bad parameter estimates, leading to ill decisions. Tuning the EKF covariance matrices is complex with no clear methodology.

## Goal

Provide an alternative estimation mode that bypasses the EKF and instead continuously re-fits parameters using a sliding window of recent observations via Nelder-Mead optimization. The EKF path remains intact; mode is selected via an environment variable.

## Three-Stage Lifecycle (unchanged)

1. **Collection:** `InitEstimator` accumulates `TUNER_INIT_OBS` observations.
2. **Seeding:** Nelder-Mead is run once over those observations to produce a good initial [α, β, γ]. In SWNM mode, these observations also pre-fill the sliding window.
3. **Steady state:** In EKF mode, the filter runs with NIS gating. In SWNM mode, N-M runs every cycle on the sliding window.

## New Component: `SlidingWindowEstimator`

**File:** `tunerservice/sliding_window_estimator.go`

```go
type SlidingWindowEstimator struct {
    window            []fitObservation
    windowSize        int
    residualThreshold float64
}
```

### `AddObservation(env *core.EnvironmentPrefillDecode)`
Appends a new `fitObservation` to the window. If `len(window) > windowSize`, the oldest entry is dropped (FIFO eviction).

### `IsReady() bool`
Returns `true` once `len(window) >= windowSize`. The tuner waits for a full window before emitting estimates.

### `Fit() ([]float64, error)`
Runs Nelder-Mead every call (no `fitDone` guard). Starting point: `guessInitState` on the most recent observation.

**Residual-based outlier rejection (one pass):**
1. Run N-M on full window → get [α, β, γ].
2. Compute per-observation relative squared error using `objective()` logic.
3. Drop observations whose error exceeds `residualThreshold`.
4. If any were dropped, run N-M once more on the cleaned window.
5. No further iterations — bounds worst-case latency.

**Error handling:** If N-M returns an error or non-positive params, `Fit()` returns an error. The caller (`tuneGroup`) leaves `paramStore` unchanged — the previous estimate is retained. No `guessInitState` fallback in steady state.

## Integration with `TunerService`

`TunerService` gains:
- `useSliding bool` — set from `TUNER_ESTIMATOR_MODE`
- `slidingEstimators map[string]*SlidingWindowEstimator` — parallel to existing `estimators` map

### `tuneGroup` modification

When `useSliding=true`:
1. `InitEstimator.AddObservation()` still runs for the collection phase.
2. Once `InitEstimator.IsReady()`, its accumulated observations are copied into `SlidingWindowEstimator` (one-time seed).
3. Each new observation is added to `SlidingWindowEstimator` via `AddObservation`.
4. `SlidingWindowEstimator.Fit()` is called; result stored in `paramStore` with `NIS=0`, `Covariance=nil`.
5. EKF path is skipped entirely.

`IsWarmingUp()` uses `SlidingWindowEstimator.IsReady()` as the readiness signal in SWNM mode (window must be full before the controller is unblocked).

## New Environment Variables

| Variable | Values / Type | Default | Purpose |
|---|---|---|---|
| `TUNER_ESTIMATOR_MODE` | `ekf` \| `sliding-window` | `ekf` | Select estimation backend |
| `TUNER_WINDOW_SIZE` | integer ≥ 1 | `10` | Sliding window capacity |
| `TUNER_RESIDUAL_THRESHOLD` | float > 0 | `0.5` | Per-observation relative error cutoff for outlier rejection |

## Data Flow (SWNM mode)

```
Observer → tuneGroup → InitEstimator (collection)
                     → SlidingWindowEstimator.AddObservation
                     → SlidingWindowEstimator.Fit()  [N-M + residual rejection]
                     → paramStore.Set(α, β, γ)
                     → buildModelData → POST /tune response
```

## Files Changed

| File | Change |
|---|---|
| `tunerservice/sliding_window_estimator.go` | New — `SlidingWindowEstimator` struct |
| `tunerservice/sliding_window_estimator_test.go` | New — unit tests |
| `tunerservice/service.go` | Add `useSliding`, `slidingEstimators`; branch in `tuneGroup` |
| `tunerservice/defaults.go` | Add new env var names and defaults |
| `tunerservice/server.go` | Read new env vars, pass to `NewTunerService` |

## Testing

Unit tests for `SlidingWindowEstimator`:
- Window capping: oldest observation is dropped when capacity exceeded
- `IsReady()`: false below `windowSize`, true at or above
- Residual rejection: one outlier observation is dropped and N-M reruns
- `Fit()` error path: non-positive params → error returned

Integration validation via `demos/tunerservice` with `TUNER_ESTIMATOR_MODE=sliding-window`.
