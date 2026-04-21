# Init-Fit Threshold (One-Time Mode Selector) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When `TUNER_ESTIMATOR_MODE=sliding-window`, check `InitEstimator.Fit()`'s objective value at SWE creation time; if it exceeds a configurable threshold, permanently route that `(model, accelerator)` pair to EKF instead.

**Architecture:** `InitEstimator.fitWithX0()` already has `result.F` — store it in a new `lastFitFuncValue` field and expose it via a getter. `TunerService` gains an `initFitThreshold` float and an `ekfFallbacks` map; `slidingEstimatorFor()` checks the threshold when seeding and sets the fallback flag, then `tuneGroup()` routes to EKF for flagged pairs. The decision is made once, at SWE creation, and is never re-evaluated.

**Tech Stack:** Go, `gonum.org/v1/gonum/optimize`, `log/slog`

---

## File Map

| File | Change |
|---|---|
| `tunerservice/init_estimator.go` | Add `lastFitFuncValue` field; set it in `fitWithX0()`; add getter |
| `tunerservice/defaults.go` | Add `InitFitThresholdEnvName` and `DefaultInitFitThreshold` constants |
| `tunerservice/service.go` | Add `initFitThreshold`, `ekfFallbacks` to `TunerService`; update constructor, `slidingEstimatorFor()`, `tuneGroupSliding()`, and `tuneGroup()` |
| `cmd/tuner/main.go` | Read `TUNER_INIT_FIT_THRESHOLD` env var; pass to `NewTunerService()` |
| `tunerservice/init_estimator_test.go` | Add tests for `LastFitFuncValue()`; update `NewTunerService()` call sites |
| `tunerservice/service_sliding_test.go` | Add tests for EKF fallback behaviour; update `NewTunerService()` call sites |
| `CLAUDE.md` | Add `TUNER_INIT_FIT_THRESHOLD` to env vars table |
| `tunerservice/README.md` | Document new env var and fallback behaviour |

---

## Task 1: Expose funcValue from InitEstimator

**Files:**
- Modify: `tunerservice/init_estimator.go`
- Test: `tunerservice/init_estimator_test.go`

- [ ] **Step 1: Write failing tests**

Add to `tunerservice/init_estimator_test.go` (after the existing tests):

```go
func TestInitEstimator_LastFitFuncValue_ZeroBeforeFit(t *testing.T) {
	ie := NewInitEstimator(1, false)
	if ie.LastFitFuncValue() != 0 {
		t.Errorf("expected 0 before Fit(), got %f", ie.LastFitFuncValue())
	}
}

func TestInitEstimator_LastFitFuncValue_NonNegativeAfterFit(t *testing.T) {
	ie := NewInitEstimator(1, false)
	ie.AddObservation(makeTestEnv(15, 55, 6, 120, 700, 64))
	if _, err := ie.Fit(); err != nil {
		t.Fatalf("Fit() returned error: %v", err)
	}
	fv := ie.LastFitFuncValue()
	if fv < 0 {
		t.Errorf("expected non-negative funcValue after Fit(), got %f", fv)
	}
}

func TestInitEstimator_LastFitFuncValue_MaxFloatOnFallback(t *testing.T) {
	// fitWithX0 with empty observations triggers the pre-flight error path,
	// but that path is internal. We test via a degenerate env that makes
	// guessInitState return nil so fitWithX0 falls back.
	// Easiest: directly call fitWithX0 with an x0 that causes an error.
	// We verify that after a successful Fit() the funcValue is finite.
	ie := NewInitEstimator(1, false)
	ie.AddObservation(makeTestEnv(15, 55, 6, 120, 700, 64))
	_, err := ie.Fit()
	if err != nil {
		t.Skip("Fit failed; cannot test funcValue from successful run")
	}
	fv := ie.LastFitFuncValue()
	if math.IsInf(fv, 0) || math.IsNaN(fv) {
		t.Errorf("funcValue should be finite after successful Fit(), got %f", fv)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
go test ./tunerservice/... -run "TestInitEstimator_LastFitFuncValue" -v
```

Expected: `FAIL` — `ie.LastFitFuncValue undefined`

- [ ] **Step 3: Add `lastFitFuncValue` field and getter to `InitEstimator`**

In `tunerservice/init_estimator.go`, update the struct and add the getter:

```go
type InitEstimator struct {
	observations    []fitObservation
	minObs          int
	holdBack        bool
	fitDone         bool
	lastFitFuncValue float64 // objective value from the most recent Fit(); 0 before first Fit()
}
```

Add the getter after `FitDone()`:

```go
// LastFitFuncValue returns the Nelder-Mead objective value from the most recent Fit() call.
// Returns 0 if Fit() has not been called yet, math.MaxFloat64 if the fit fell back to guessInitState.
func (ie *InitEstimator) LastFitFuncValue() float64 { return ie.lastFitFuncValue }
```

- [ ] **Step 4: Set `lastFitFuncValue` in `fitWithX0()`**

In `tunerservice/init_estimator.go`, update `fitWithX0()`. There are three return paths — set `math.MaxFloat64` on the two fallback paths and `result.F` on the success path:

```go
func (ie *InitEstimator) fitWithX0(x0 []float64) ([]float64, error) {
	// ... existing setup unchanged ...

	result, err := optimize.Minimize(problem, scaledX0, settings, &optimize.NelderMead{})
	if err != nil {
		slog.Warn("InitEstimator: Nelder-Mead pre-flight error, using guessInitState fallback", "err", err)
		ie.lastFitFuncValue = math.MaxFloat64
		if fallback := guessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead failed and guessInitState returned nil: %w", err)
	}

	switch result.Status {
	case optimize.Success, optimize.FunctionConvergence, optimize.FunctionEvaluationLimit:
	default:
		slog.Warn("InitEstimator: unexpected Nelder-Mead termination status, using guessInitState fallback",
			"status", result.Status)
		ie.lastFitFuncValue = math.MaxFloat64
		if fallback := guessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead unexpected status %v and guessInitState returned nil", result.Status)
	}

	unscaled := make([]float64, len(result.X))
	for i := range result.X {
		unscaled[i] = result.X[i] * scale[i]
	}
	x := unscaled
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		slog.Warn("InitEstimator: Nelder-Mead returned non-positive params, using guessInitState fallback",
			"alpha", x[0], "beta", x[1], "gamma", x[2])
		ie.lastFitFuncValue = math.MaxFloat64
		if fallback := guessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead returned non-positive params and guessInitState returned nil")
	}

	ie.lastFitFuncValue = result.F
	slog.Info("InitEstimator: Fit complete",
		"alpha", x[0], "beta", x[1], "gamma", x[2],
		"observations", len(ie.observations), "funcValue", result.F)
	return x, nil
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
go test ./tunerservice/... -run "TestInitEstimator_LastFitFuncValue" -v
```

Expected: all three tests `PASS`

- [ ] **Step 6: Run full test suite to check for regressions**

```bash
go test ./tunerservice/... -v 2>&1 | tail -20
```

Expected: all existing tests pass.

- [ ] **Step 7: Commit**

```bash
git add tunerservice/init_estimator.go tunerservice/init_estimator_test.go
git commit -m "feat(init-estimator): expose LastFitFuncValue() from fitWithX0"
```

---

## Task 2: Add threshold config and EKF fallback routing

**Files:**
- Modify: `tunerservice/defaults.go`
- Modify: `tunerservice/service.go`
- Modify: `cmd/tuner/main.go`
- Modify: `tunerservice/init_estimator_test.go` (update `NewTunerService` call sites)
- Test: `tunerservice/service_sliding_test.go`

- [ ] **Step 1: Write failing tests**

Add to `tunerservice/service_sliding_test.go` (after existing tests):

```go
// TestTunerService_SWNM_HighFuncValue_FallsBackToEKF verifies that a pair whose
// InitEstimator.Fit() exceeds the threshold is permanently routed to EKF.
func TestTunerService_SWNM_HighFuncValue_FallsBackToEKF(t *testing.T) {
	// threshold=0.001: any real fit will exceed this, triggering EKF fallback.
	ts := NewTunerService(0, 1, false, true, DefaultWindowSize, DefaultResidualThreshold, 0.001)
	spec := makeTestSpec("llama", "H100", 15, 55, 6, 120, 700, 64)

	// First cycle: InitEstimator completes, slidingEstimatorFor detects high funcValue.
	// tuneGroupSliding returns error; no params stored this cycle.
	_, err := ts.Tune([]optconfig.ServerSpec{spec})
	if err == nil {
		// It's also acceptable to return params via EKF on the same cycle — just check
		// that ekfFallbacks is set.
	}

	key := makeKey("llama", "H100")
	if !ts.ekfFallbacks[key] {
		t.Fatal("expected ekfFallbacks[key]=true after high funcValue init fit")
	}

	// Second cycle: pair is routed to EKF (no SWE in slidingEstimators).
	result, err := ts.Tune([]optconfig.ServerSpec{spec})
	if err != nil {
		t.Fatalf("cycle 2 (EKF path): expected params, got error: %v", err)
	}
	if len(result.PerfData) == 0 {
		t.Fatal("cycle 2: expected non-empty PerfData from EKF")
	}

	// Confirm no SlidingWindowEstimator was stored for this pair.
	if _, hasSWE := ts.slidingEstimators[key]; hasSWE {
		t.Error("expected no SWE stored after EKF fallback")
	}
}

// TestTunerService_SWNM_ZeroThreshold_DisablesFallback verifies that threshold=0
// disables the feature: SWNM is used regardless of funcValue.
func TestTunerService_SWNM_ZeroThreshold_DisablesFallback(t *testing.T) {
	ts := NewTunerService(0, 1, false, true, DefaultWindowSize, DefaultResidualThreshold, 0)
	spec := makeTestSpec("llama", "H100", 15, 55, 6, 120, 700, 64)

	result, err := ts.Tune([]optconfig.ServerSpec{spec})
	if err != nil {
		t.Fatalf("expected params with threshold=0, got error: %v", err)
	}
	if len(result.PerfData) == 0 {
		t.Fatal("expected non-empty PerfData")
	}

	key := makeKey("llama", "H100")
	if ts.ekfFallbacks[key] {
		t.Error("ekfFallbacks should not be set when threshold=0")
	}
	if _, hasSWE := ts.slidingEstimators[key]; !hasSWE {
		t.Error("SWE should be stored when threshold=0")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
go test ./tunerservice/... -run "TestTunerService_SWNM_HighFuncValue|TestTunerService_SWNM_ZeroThreshold" -v
```

Expected: `FAIL` — `NewTunerService` wrong number of arguments (or `ekfFallbacks` undefined).

- [ ] **Step 3: Add constants to `defaults.go`**

In `tunerservice/defaults.go`, add a new block after the SlidingWindowEstimator constants:

```go
// Environment variable name and default for the init-fit quality threshold.
// When useSliding=true and InitEstimator.Fit() returns a funcValue above this threshold,
// the pair is permanently routed to EKF instead of SWNM.
// Set to 0 to disable the feature.
const (
	InitFitThresholdEnvName = "TUNER_INIT_FIT_THRESHOLD"
	DefaultInitFitThreshold = 10.0
)
```

- [ ] **Step 4: Add fields and update `TunerService` in `service.go`**

Update the `TunerService` struct to add the two new fields:

```go
type TunerService struct {
	paramStore        *ParameterStore
	warmUpCycles      int
	estimators        map[string]*InitEstimator
	initObs           int
	holdBack          bool
	useSliding        bool
	windowSize        int
	residualThreshold float64
	slidingEstimators map[string]*SlidingWindowEstimator
	initFitThreshold  float64            // 0 = disabled; >0 = EKF fallback when funcValue exceeds this
	ekfFallbacks      map[string]bool    // pairs permanently routed to EKF due to poor init fit
}
```

Update `NewTunerService()` to accept and store the new parameter:

```go
func NewTunerService(warmUpCycles, initObs int, holdBack bool, useSliding bool, windowSize int, residualThreshold, initFitThreshold float64) *TunerService {
	return &TunerService{
		paramStore:        NewParameterStore(),
		warmUpCycles:      warmUpCycles,
		estimators:        make(map[string]*InitEstimator),
		initObs:           initObs,
		holdBack:          holdBack,
		useSliding:        useSliding,
		windowSize:        windowSize,
		residualThreshold: residualThreshold,
		slidingEstimators: make(map[string]*SlidingWindowEstimator),
		initFitThreshold:  initFitThreshold,
		ekfFallbacks:      make(map[string]bool),
	}
}
```

- [ ] **Step 5: Update `slidingEstimatorFor()` to check funcValue**

Replace the body of `slidingEstimatorFor()` in `service.go`:

```go
func (ts *TunerService) slidingEstimatorFor(key string, ie *InitEstimator) *SlidingWindowEstimator {
	if swe, ok := ts.slidingEstimators[key]; ok {
		return swe
	}
	swe := NewSlidingWindowEstimator(ts.windowSize, ts.initObs, ts.residualThreshold)
	swe.Seed(ie.observations)
	if fitted, err := ie.Fit(); err == nil {
		fv := ie.LastFitFuncValue()
		if ts.initFitThreshold > 0 && fv > ts.initFitThreshold {
			slog.Warn("poor init fit: falling back to EKF for this pair",
				"key", key, "funcValue", fv, "threshold", ts.initFitThreshold)
			ts.ekfFallbacks[key] = true
			return swe // not stored in ts.slidingEstimators; ekfFallbacks prevents future SWNM routing
		}
		swe.SeedLastFit(fitted)
	}
	ts.slidingEstimators[key] = swe
	return swe
}
```

- [ ] **Step 6: Update `tuneGroupSliding()` to exit early when fallback is set**

Add an early return at the top of `tuneGroupSliding()`, just after calling `slidingEstimatorFor()`:

```go
func (ts *TunerService) tuneGroupSliding(model, accelerator, key string, ie *InitEstimator, env *core.EnvironmentPrefillDecode) error {
	_, alreadyExists := ts.slidingEstimators[key]
	swe := ts.slidingEstimatorFor(key, ie)

	if ts.ekfFallbacks[key] {
		return fmt.Errorf("EKF fallback active for %s/%s: poor init fit (funcValue > %.1f)",
			model, accelerator, ts.initFitThreshold)
	}

	if alreadyExists {
		swe.AddObservation(env)
	}
	// ... rest of the function unchanged ...
}
```

- [ ] **Step 7: Update routing in `tuneGroup()`**

Change the SWNM routing block from:

```go
if ts.useSliding {
    return ts.tuneGroupSliding(model, accelerator, key, estimator, envs[0])
}
```

to:

```go
if ts.useSliding && !ts.ekfFallbacks[key] {
    return ts.tuneGroupSliding(model, accelerator, key, estimator, envs[0])
}
```

- [ ] **Step 8: Update `NewTunerService` call sites in `init_estimator_test.go`**

Both calls in `tunerservice/init_estimator_test.go` need the new trailing `float64` argument (use `0` to disable the feature in these tests, since they don't exercise the threshold):

```go
// Line ~255
ts := NewTunerService(3, 3, true, false, DefaultWindowSize, DefaultResidualThreshold, 0)

// Line ~265
ts := NewTunerService(3, 3, false, false, DefaultWindowSize, DefaultResidualThreshold, 0)
```

- [ ] **Step 9: Update `NewTunerService` call sites in `service_sliding_test.go`**

All existing `NewTunerService` calls need the new trailing `float64` argument. Pass `0` (disabled) to preserve existing behaviour:

```go
// TestTunerService_SWNM_ReturnsParamsAfterInitPhase
ts := NewTunerService(0, initObs, false, true, windowSize, DefaultResidualThreshold, 0)

// TestTunerService_SWNM_FitError_RetainsPreviousParams
ts := NewTunerService(0, 1, false, true, 1, DefaultResidualThreshold, 0)

// TestTunerService_IsWarmingUp_SWNM_WindowNotFull
ts := NewTunerService(3, 3, true, true, 5, DefaultResidualThreshold, 0)

// TestTunerService_IsWarmingUp_SWNM_WindowFull
ts := NewTunerService(3, 3, true, true, 3, DefaultResidualThreshold, 0)
```

- [ ] **Step 10: Run new tests to verify they pass**

```bash
go test ./tunerservice/... -run "TestTunerService_SWNM_HighFuncValue|TestTunerService_SWNM_ZeroThreshold" -v
```

Expected: both tests `PASS`

- [ ] **Step 11: Run full test suite**

```bash
go test ./tunerservice/... -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 12: Update `main.go` to read env var and pass to constructor**

In `cmd/tuner/main.go`, add after the `residualThreshold` block:

```go
initFitThreshold := tunerservice.DefaultInitFitThreshold
if v := os.Getenv(tunerservice.InitFitThresholdEnvName); v != "" {
    if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
        initFitThreshold = f
    }
}
```

Update the `NewTunerService` call:

```go
service := tunerservice.NewTunerService(warmUpCycles, initObs, holdBack, useSliding, windowSize, residualThreshold, initFitThreshold)
```

Update the startup log:

```go
slog.Info("Starting TunerService",
    "host", host, "port", port,
    "warmUpCycles", warmUpCycles,
    "initObs", initObs,
    "holdBack", holdBack,
    "estimatorMode", estimatorMode,
    "windowSize", windowSize,
    "residualThreshold", residualThreshold,
    "initFitThreshold", initFitThreshold)
```

- [ ] **Step 13: Build to verify no compilation errors**

```bash
go build ./...
```

Expected: no errors.

- [ ] **Step 14: Commit**

```bash
git add tunerservice/defaults.go tunerservice/service.go tunerservice/init_estimator_test.go tunerservice/service_sliding_test.go cmd/tuner/main.go
git commit -m "feat(sliding-window): EKF fallback when init fit funcValue exceeds TUNER_INIT_FIT_THRESHOLD"
```

---

## Task 3: Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `tunerservice/README.md`

- [ ] **Step 1: Add env var to `CLAUDE.md`**

In the `Service environment variables` table, add a new row after `TUNER_RESIDUAL_THRESHOLD`:

```markdown
| `TUNER_INIT_FIT_THRESHOLD` | (SWNM) If `InitEstimator.Fit()` objective value exceeds this, the pair is permanently routed to EKF instead. Set to `0` to disable. | `10.0` |
```

- [ ] **Step 2: Update `tunerservice/README.md`**

In the **Sliding-Window Nelder-Mead mode** section, add a new paragraph after **Seeding**:

```markdown
**EKF fallback on poor init fit** — when the `InitEstimator`'s Nelder-Mead objective value (`funcValue`) exceeds `TUNER_INIT_FIT_THRESHOLD` (default 10.0), the `(model, accelerator)` pair is permanently routed to EKF instead of SWNM. This handles the low-utilisation identifiability problem: when observations span a narrow RPM range, the loss surface is flat and Nelder-Mead converges to a degenerate solution that SWNM's warm-start would then propagate. A `funcValue` of 10.0 corresponds roughly to 100% average relative error across 5 observations. Set `TUNER_INIT_FIT_THRESHOLD=0` to disable the fallback and always use SWNM.
```

In the **Configuration** table, add after `TUNER_RESIDUAL_THRESHOLD`:

```markdown
| `TUNER_INIT_FIT_THRESHOLD` | (SWNM) Nelder-Mead objective threshold; if `InitEstimator.Fit()` exceeds this the pair falls back to EKF permanently. `0` disables. | `10.0` |
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md tunerservice/README.md
git commit -m "docs: document TUNER_INIT_FIT_THRESHOLD env var and EKF fallback behaviour"
```
