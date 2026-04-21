# Sliding-Window Nelder-Mead Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `SlidingWindowEstimator` as a drop-in EKF alternative, selected via `TUNER_ESTIMATOR_MODE=sliding-window`, that continuously re-fits [α,β,γ] using a fixed-size observation window with Nelder-Mead and residual-based outlier rejection.

**Architecture:** A new `SlidingWindowEstimator` struct lives in `tunerservice/` and is activated by a new `useSliding` flag on `TunerService`. After `InitEstimator` completes its collection phase, `tuneGroup` branches: the EKF path is unchanged; the SWNM path calls `SlidingWindowEstimator.Fit()` each cycle and stores results directly in `paramStore` (NIS=0, Covariance=nil). The `InitEstimator` observations seed the SWE window so the first estimate is available as soon as the window is full.

**Tech Stack:** Go, `gonum.org/v1/gonum/optimize` (Nelder-Mead), `github.com/llm-inferno/queue-analysis/pkg/analyzer` (queue model).

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Create | `tunerservice/sliding_window_estimator.go` | `SlidingWindowEstimator` struct, `Fit`, residual rejection |
| Create | `tunerservice/sliding_window_estimator_test.go` | Unit tests for `SlidingWindowEstimator` |
| Create | `tunerservice/service_sliding_test.go` | Integration tests for SWNM path in `TunerService` |
| Modify | `tunerservice/defaults.go` | Three new env var constants and defaults |
| Modify | `tunerservice/service.go` | New fields, updated `NewTunerService`, `tuneGroup` branch, `tuneGroupSliding`, `slidingEstimatorFor`, updated `IsWarmingUp` |
| Modify | `tunerservice/init_estimator_test.go` | Update `NewTunerService(3, 3, ...)` calls to new signature |
| Modify | `cmd/tuner/main.go` | Read three new env vars, pass to `NewTunerService` |

---

## Task 1: Add env var constants to `defaults.go`

**Files:**
- Modify: `tunerservice/defaults.go`

- [ ] **Step 1: Add three constants**

Open `tunerservice/defaults.go` and append after the `InitEstimator` block:

```go
// Environment variable names and defaults for the SlidingWindowEstimator.
const (
	EstimatorModeEnvName     = "TUNER_ESTIMATOR_MODE"
	WindowSizeEnvName        = "TUNER_WINDOW_SIZE"
	ResidualThresholdEnvName = "TUNER_RESIDUAL_THRESHOLD"

	DefaultEstimatorMode     = "ekf"
	DefaultWindowSize        = 10
	DefaultResidualThreshold = 0.5
)
```

- [ ] **Step 2: Build to confirm no syntax errors**

```bash
go build ./tunerservice/...
```

Expected: no output (success).

- [ ] **Step 3: Commit**

```bash
git add tunerservice/defaults.go
git commit -m "feat: add TUNER_ESTIMATOR_MODE/WINDOW_SIZE/RESIDUAL_THRESHOLD constants"
```

---

## Task 2: Write failing tests for `SlidingWindowEstimator` core behaviour

**Files:**
- Create: `tunerservice/sliding_window_estimator_test.go`

- [ ] **Step 1: Create the test file**

```go
package tunerservice

import (
	"testing"

	"github.com/llm-inferno/model-tuner/pkg/core"
)

// TestSlidingWindowEstimator_IsReady verifies that IsReady requires a full window.
func TestSlidingWindowEstimator_IsReady(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 0.5)
	env := makeTestEnv(10, 50, 5, 100, 500, 64)

	if swe.IsReady() {
		t.Fatal("should not be ready with 0 observations")
	}
	swe.AddObservation(env)
	if swe.IsReady() {
		t.Fatal("should not be ready with 1 observation")
	}
	swe.AddObservation(env)
	swe.AddObservation(env)
	if !swe.IsReady() {
		t.Fatal("should be ready with 3 observations (windowSize=3)")
	}
}

// TestSlidingWindowEstimator_AddObservation_Caps verifies FIFO eviction at window capacity.
func TestSlidingWindowEstimator_AddObservation_Caps(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 0.5)
	e1 := makeTestEnv(10, 50, 5, 100, 500, 64)
	e2 := makeTestEnv(20, 60, 6, 110, 510, 64)
	e3 := makeTestEnv(30, 70, 7, 120, 520, 64)
	e4 := makeTestEnv(40, 80, 8, 130, 530, 64) // should evict e1

	swe.AddObservation(e1)
	swe.AddObservation(e2)
	swe.AddObservation(e3)
	swe.AddObservation(e4)

	if len(swe.window) != 3 {
		t.Fatalf("expected window size 3, got %d", len(swe.window))
	}
	// e1 evicted; e4 (lambda=40) should be in window
	if swe.window[0].Lambda != float64(e2.Lambda) {
		t.Errorf("expected oldest entry lambda=20, got %v", swe.window[0].Lambda)
	}
	if swe.window[2].Lambda != float64(e4.Lambda) {
		t.Errorf("expected newest entry lambda=40, got %v", swe.window[2].Lambda)
	}
}

// TestSlidingWindowEstimator_AddObservation_IgnoresNilAndInvalid mirrors the InitEstimator test.
func TestSlidingWindowEstimator_AddObservation_IgnoresNilAndInvalid(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 0.5)
	swe.AddObservation(nil)
	invalid := core.NewEnvironmentPrefillDecode(10, 0, 0, 64, 100, 500, 0, 0)
	swe.AddObservation(invalid)
	if len(swe.window) != 0 {
		t.Fatalf("expected 0 observations stored, got %d", len(swe.window))
	}
}

// TestSlidingWindowEstimator_Seed verifies that seeding caps at windowSize.
func TestSlidingWindowEstimator_Seed(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 0.5)
	obs := []fitObservation{
		{Lambda: 10}, {Lambda: 20}, {Lambda: 30}, {Lambda: 40}, {Lambda: 50},
	}
	swe.Seed(obs)
	if len(swe.window) != 3 {
		t.Fatalf("expected window capped at 3, got %d", len(swe.window))
	}
	// oldest evicted — window should hold lambda=30,40,50
	if swe.window[0].Lambda != 30 {
		t.Errorf("expected lambda=30 at index 0, got %v", swe.window[0].Lambda)
	}
}

// TestSlidingWindowEstimator_Fit_EmptyError verifies Fit() returns an error on empty window.
func TestSlidingWindowEstimator_Fit_EmptyError(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 0.5)
	_, err := swe.Fit()
	if err == nil {
		t.Fatal("expected error from Fit() on empty window")
	}
}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
go test ./tunerservice/... -run TestSlidingWindow -v 2>&1 | head -30
```

Expected: compile error — `NewSlidingWindowEstimator` undefined.

- [ ] **Step 3: Commit the test file**

```bash
git add tunerservice/sliding_window_estimator_test.go
git commit -m "test: add failing tests for SlidingWindowEstimator core behaviour"
```

---

## Task 3: Implement `SlidingWindowEstimator` — struct, `AddObservation`, `IsReady`, `Seed`

**Files:**
- Create: `tunerservice/sliding_window_estimator.go`

- [ ] **Step 1: Create the file with struct, constructor, and core methods**

```go
package tunerservice

import (
	"fmt"
	"log/slog"
	"math"

	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
	"gonum.org/v1/gonum/optimize"

	"github.com/llm-inferno/model-tuner/pkg/core"
)

// SlidingWindowEstimator maintains a fixed-capacity circular buffer of recent observations
// and re-runs Nelder-Mead on every Fit() call to produce fresh [α,β,γ] estimates.
type SlidingWindowEstimator struct {
	window            []fitObservation
	windowSize        int
	residualThreshold float64
}

// NewSlidingWindowEstimator creates a SlidingWindowEstimator with the given window capacity
// and residual outlier rejection threshold (relative error, e.g. 0.5 = 50%).
func NewSlidingWindowEstimator(windowSize int, residualThreshold float64) *SlidingWindowEstimator {
	if windowSize < 1 {
		windowSize = 1
	}
	return &SlidingWindowEstimator{
		windowSize:        windowSize,
		residualThreshold: residualThreshold,
	}
}

// Seed pre-fills the window with observations from the InitEstimator collection phase.
// Oldest entries are evicted when the seed exceeds windowSize.
func (swe *SlidingWindowEstimator) Seed(obs []fitObservation) {
	for _, o := range obs {
		swe.window = append(swe.window, o)
		if len(swe.window) > swe.windowSize {
			swe.window = swe.window[1:]
		}
	}
}

// AddObservation appends a new operating-point observation.
// Oldest entry is evicted when the window is at capacity.
// Nil and invalid environments are silently ignored.
func (swe *SlidingWindowEstimator) AddObservation(env *core.EnvironmentPrefillDecode) {
	if env == nil || !env.Valid() {
		return
	}
	obs := fitObservation{
		Lambda:          float64(env.Lambda),
		MaxBatch:        env.MaxBatchSize,
		MaxQueueSize:    env.MaxQueueSize,
		AvgInputTokens:  env.AvgInputTokens,
		AvgOutputTokens: env.AvgOutputTokens,
		AvgTTFT:         float64(env.AvgTTFT),
		AvgITL:          float64(env.AvgITL),
	}
	swe.window = append(swe.window, obs)
	if len(swe.window) > swe.windowSize {
		swe.window = swe.window[1:]
	}
}

// IsReady returns true once the window holds at least windowSize observations.
func (swe *SlidingWindowEstimator) IsReady() bool {
	return len(swe.window) >= swe.windowSize
}

// Fit runs Nelder-Mead on the current window, performs one residual-based outlier
// rejection pass, and refits if any observations were dropped.
// Returns [alpha, beta, gamma] or an error if fitting fails.
func (swe *SlidingWindowEstimator) Fit() ([]float64, error) {
	if len(swe.window) == 0 {
		return nil, fmt.Errorf("no observations in window")
	}

	x0 := guessInitState(swe.window[len(swe.window)-1].toEnv())
	if x0 == nil {
		x0 = []float64{5.0, 0.05, 0.0005}
	}

	fitted, err := swe.fitWithX0(x0, swe.window)
	if err != nil {
		return nil, err
	}

	cleaned := swe.filterOutliers(swe.window, fitted)
	if len(cleaned) < len(swe.window) {
		slog.Info("SlidingWindowEstimator: outliers removed, refitting",
			"total", len(swe.window), "kept", len(cleaned))
		fitted, err = swe.fitWithX0(x0, cleaned)
		if err != nil {
			return nil, err
		}
	}

	return fitted, nil
}

// filterOutliers returns the subset of obs whose residual is within swe.residualThreshold.
// If all observations would be dropped (e.g. threshold=0), returns obs unchanged.
func (swe *SlidingWindowEstimator) filterOutliers(obs []fitObservation, x []float64) []fitObservation {
	var kept []fitObservation
	for _, o := range obs {
		if swe.residual(o, x) <= swe.residualThreshold {
			kept = append(kept, o)
		}
	}
	if len(kept) == 0 {
		return obs
	}
	return kept
}

// residual returns sqrt(dTTFT² + dITL²) for one observation evaluated at params x=[α,β,γ].
// Returns math.MaxFloat64 if the model evaluation fails.
func (swe *SlidingWindowEstimator) residual(obs fitObservation, x []float64) float64 {
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		return math.MaxFloat64
	}
	qConfig := &analyzer.Configuration{
		MaxBatchSize: obs.MaxBatch,
		MaxQueueSize: obs.MaxQueueSize,
		ServiceParms: &analyzer.ServiceParms{
			Alpha: float32(x[0]),
			Beta:  float32(x[1]),
			Gamma: float32(x[2]),
		},
	}
	requestSize := &analyzer.RequestSize{
		AvgInputTokens:  obs.AvgInputTokens,
		AvgOutputTokens: obs.AvgOutputTokens,
	}
	qa, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
	if err != nil {
		return math.MaxFloat64
	}
	metrics, err := qa.Analyze(float32(obs.Lambda / 60))
	if err != nil {
		return math.MaxFloat64
	}
	ttftModel := float64(metrics.AvgTTFT)
	itlModel := float64(metrics.AvgTokenTime)
	if obs.AvgTTFT <= 0 || obs.AvgITL <= 0 || ttftModel <= 0 || itlModel <= 0 {
		return math.MaxFloat64
	}
	dTTFT := (ttftModel - obs.AvgTTFT) / obs.AvgTTFT
	dITL := (itlModel - obs.AvgITL) / obs.AvgITL
	return math.Sqrt(dTTFT*dTTFT + dITL*dITL)
}

// fitWithX0 runs Nelder-Mead on obs starting from x0.
// Variables are scaled by x0 so the optimizer sees O(1) quantities in all dimensions.
func (swe *SlidingWindowEstimator) fitWithX0(x0 []float64, obs []fitObservation) ([]float64, error) {
	if len(obs) == 0 {
		return nil, fmt.Errorf("no observations to fit")
	}

	scale := make([]float64, len(x0))
	scaledX0 := make([]float64, len(x0))
	for i, v := range x0 {
		scale[i] = v
		scaledX0[i] = 1.0
	}

	scaledObjective := func(p []float64) float64 {
		unscaled := make([]float64, len(p))
		for i := range p {
			unscaled[i] = p[i] * scale[i]
		}
		return swe.objective(obs, unscaled)
	}

	problem := optimize.Problem{Func: scaledObjective}
	settings := &optimize.Settings{FuncEvaluations: 500}
	result, err := optimize.Minimize(problem, scaledX0, settings, &optimize.NelderMead{})
	if err != nil {
		slog.Warn("SlidingWindowEstimator: Nelder-Mead pre-flight error, using guessInitState fallback", "err", err)
		if fallback := guessInitState(obs[len(obs)-1].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead failed and guessInitState returned nil: %w", err)
	}

	switch result.Status {
	case optimize.Success, optimize.FunctionConvergence, optimize.FunctionEvaluationLimit:
	default:
		slog.Warn("SlidingWindowEstimator: unexpected Nelder-Mead termination", "status", result.Status)
		if fallback := guessInitState(obs[len(obs)-1].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead unexpected status %v", result.Status)
	}

	unscaled := make([]float64, len(result.X))
	for i := range result.X {
		unscaled[i] = result.X[i] * scale[i]
	}
	x := unscaled
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		slog.Warn("SlidingWindowEstimator: non-positive params, using guessInitState fallback",
			"alpha", x[0], "beta", x[1], "gamma", x[2])
		if fallback := guessInitState(obs[len(obs)-1].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead returned non-positive params")
	}

	slog.Info("SlidingWindowEstimator: Fit complete",
		"alpha", x[0], "beta", x[1], "gamma", x[2],
		"observations", len(obs), "funcValue", result.F)
	return x, nil
}

// objective returns sum of relative squared errors in TTFT and ITL across obs for params x=[α,β,γ].
func (swe *SlidingWindowEstimator) objective(obs []fitObservation, x []float64) float64 {
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		return math.MaxFloat64 / 2
	}
	var total float64
	for _, o := range obs {
		qConfig := &analyzer.Configuration{
			MaxBatchSize: o.MaxBatch,
			MaxQueueSize: o.MaxQueueSize,
			ServiceParms: &analyzer.ServiceParms{
				Alpha: float32(x[0]),
				Beta:  float32(x[1]),
				Gamma: float32(x[2]),
			},
		}
		requestSize := &analyzer.RequestSize{
			AvgInputTokens:  o.AvgInputTokens,
			AvgOutputTokens: o.AvgOutputTokens,
		}
		qa, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
		if err != nil {
			return math.MaxFloat64 / 2
		}
		metrics, err := qa.Analyze(float32(o.Lambda / 60))
		if err != nil {
			return math.MaxFloat64 / 2
		}
		ttftModel := float64(metrics.AvgTTFT)
		itlModel := float64(metrics.AvgTokenTime)
		ttftObs := o.AvgTTFT
		itlObs := o.AvgITL
		if ttftObs <= 0 || itlObs <= 0 || ttftModel <= 0 || itlModel <= 0 {
			return math.MaxFloat64 / 2
		}
		dTTFT := (ttftModel - ttftObs) / ttftObs
		dITL := (itlModel - itlObs) / itlObs
		total += dTTFT*dTTFT + dITL*dITL
	}
	return total
}
```

- [ ] **Step 2: Run core tests**

```bash
go test ./tunerservice/... -run TestSlidingWindowEstimator_IsReady -v
go test ./tunerservice/... -run TestSlidingWindowEstimator_AddObservation_Caps -v
go test ./tunerservice/... -run TestSlidingWindowEstimator_AddObservation_IgnoresNilAndInvalid -v
go test ./tunerservice/... -run TestSlidingWindowEstimator_Seed -v
go test ./tunerservice/... -run TestSlidingWindowEstimator_Fit_EmptyError -v
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tunerservice/sliding_window_estimator.go
git commit -m "feat: implement SlidingWindowEstimator with AddObservation, IsReady, Seed, Fit"
```

---

## Task 4: Add Fit parameter-recovery and residual-rejection tests

**Files:**
- Modify: `tunerservice/sliding_window_estimator_test.go`

- [ ] **Step 1: Add the two tests**

Append to `tunerservice/sliding_window_estimator_test.go`:

```go
import (
	"math"
	"testing"

	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
)
```

Replace the existing import block at the top of the file with the above (add `"math"` and the `analyzer` import).

Then append the following two test functions:

```go
// TestSlidingWindowEstimator_Fit_ParameterRecovery verifies that Fit() recovers true
// parameters within 10% when the window contains noiseless synthetic observations.
func TestSlidingWindowEstimator_Fit_ParameterRecovery(t *testing.T) {
	trueAlpha := float32(16.78)
	trueBeta := float32(0.073)
	trueGamma := float32(0.00228)
	maxBatch := 64

	type opPoint struct {
		lambda float32
		inTok  float32
		outTok float32
	}
	ops := []opPoint{
		{lambda: 10, inTok: 90, outTok: 670},
		{lambda: 15, inTok: 120, outTok: 900},
		{lambda: 20, inTok: 150, outTok: 1100},
		{lambda: 12, inTok: 100, outTok: 750},
	}

	swe := NewSlidingWindowEstimator(len(ops), 0.5)
	for _, op := range ops {
		qConfig := &analyzer.Configuration{
			MaxBatchSize: maxBatch,
			ServiceParms: &analyzer.ServiceParms{Alpha: trueAlpha, Beta: trueBeta, Gamma: trueGamma},
		}
		requestSize := &analyzer.RequestSize{AvgInputTokens: op.inTok, AvgOutputTokens: op.outTok}
		qa, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
		if err != nil {
			t.Fatalf("failed to create analyzer: %v", err)
		}
		metrics, err := qa.Analyze(op.lambda / 60)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}
		env := core.NewEnvironmentPrefillDecode(op.lambda, 0, 0, maxBatch, op.inTok, op.outTok,
			metrics.AvgTTFT, metrics.AvgTokenTime)
		swe.AddObservation(env)
	}

	if !swe.IsReady() {
		t.Fatal("expected window to be ready")
	}
	fitted, err := swe.Fit()
	if err != nil {
		t.Fatalf("Fit() returned error: %v", err)
	}

	tolerance := 0.10
	checkParam := func(name string, got, want float64) {
		t.Helper()
		relErr := math.Abs(got-want) / want
		if relErr > tolerance {
			t.Errorf("param %s: got %.6f, want %.6f (%.1f%% > %.0f%%)",
				name, got, want, relErr*100, tolerance*100)
		}
	}
	checkParam("alpha", fitted[0], float64(trueAlpha))
	checkParam("beta", fitted[1], float64(trueBeta))
	checkParam("gamma", fitted[2], float64(trueGamma))
}

// TestSlidingWindowEstimator_Fit_ResidualRejection verifies that one anomalous observation
// (TTFT 10× too high) is detected and removed so that true parameters are recovered.
func TestSlidingWindowEstimator_Fit_ResidualRejection(t *testing.T) {
	trueAlpha := float32(16.78)
	trueBeta := float32(0.073)
	trueGamma := float32(0.00228)
	maxBatch := 64

	type opPoint struct {
		lambda float32
		inTok  float32
		outTok float32
	}
	cleanOps := []opPoint{
		{lambda: 10, inTok: 90, outTok: 670},
		{lambda: 15, inTok: 120, outTok: 900},
		{lambda: 20, inTok: 150, outTok: 1100},
		{lambda: 12, inTok: 100, outTok: 750},
	}

	swe := NewSlidingWindowEstimator(5, 0.5) // window=5, 4 clean + 1 outlier

	addSynthetic := func(op opPoint) {
		qConfig := &analyzer.Configuration{
			MaxBatchSize: maxBatch,
			ServiceParms: &analyzer.ServiceParms{Alpha: trueAlpha, Beta: trueBeta, Gamma: trueGamma},
		}
		requestSize := &analyzer.RequestSize{AvgInputTokens: op.inTok, AvgOutputTokens: op.outTok}
		qa, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
		if err != nil {
			t.Fatalf("failed to create analyzer: %v", err)
		}
		metrics, err := qa.Analyze(op.lambda / 60)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}
		env := core.NewEnvironmentPrefillDecode(op.lambda, 0, 0, maxBatch, op.inTok, op.outTok,
			metrics.AvgTTFT, metrics.AvgTokenTime)
		swe.AddObservation(env)
	}

	for _, op := range cleanOps {
		addSynthetic(op)
	}
	// Outlier: TTFT 10× higher than normal — should be rejected
	outlier := core.NewEnvironmentPrefillDecode(10, 0, 0, maxBatch, 90, 670, 500, 5)
	swe.AddObservation(outlier)

	if !swe.IsReady() {
		t.Fatal("expected window to be ready")
	}
	fitted, err := swe.Fit()
	if err != nil {
		t.Fatalf("Fit() with outlier returned error: %v", err)
	}

	tolerance := 0.15 // slightly looser — outlier may shift fit before rejection
	checkParam := func(name string, got, want float64) {
		t.Helper()
		relErr := math.Abs(got-want) / want
		if relErr > tolerance {
			t.Errorf("param %s: got %.6f, want %.6f (%.1f%% > %.0f%%) — outlier not rejected?",
				name, got, want, relErr*100, tolerance*100)
		}
	}
	checkParam("alpha", fitted[0], float64(trueAlpha))
	checkParam("beta", fitted[1], float64(trueBeta))
	checkParam("gamma", fitted[2], float64(trueGamma))
}

// TestSlidingWindowEstimator_FilterOutliers_KeepsAllWhenThresholdZero verifies the safety
// guard: if every observation exceeds the threshold, all are kept to avoid an empty refit.
func TestSlidingWindowEstimator_FilterOutliers_KeepsAllWhenThresholdZero(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 0) // threshold=0 — every obs is an "outlier"
	obs := []fitObservation{
		{Lambda: 10, AvgTTFT: 50, AvgITL: 5, AvgInputTokens: 100, AvgOutputTokens: 500, MaxBatch: 64},
		{Lambda: 20, AvgTTFT: 60, AvgITL: 6, AvgInputTokens: 110, AvgOutputTokens: 510, MaxBatch: 64},
	}
	x := []float64{1.0, 0.01, 0.001} // arbitrary params
	kept := swe.filterOutliers(obs, x)
	if len(kept) != len(obs) {
		t.Errorf("expected all %d kept when threshold=0, got %d", len(obs), len(kept))
	}
}
```

- [ ] **Step 2: Run the new tests**

```bash
go test ./tunerservice/... -run TestSlidingWindowEstimator_Fit -v
go test ./tunerservice/... -run TestSlidingWindowEstimator_FilterOutliers -v
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tunerservice/sliding_window_estimator_test.go
git commit -m "test: add parameter recovery and residual rejection tests for SlidingWindowEstimator"
```

---

## Task 5: Update `TunerService` — new fields, `NewTunerService` signature, SWNM integration

**Files:**
- Modify: `tunerservice/service.go`

- [ ] **Step 1: Update `TunerService` struct**

In `tunerservice/service.go`, replace the struct definition:

```go
// Before:
type TunerService struct {
	paramStore   *ParameterStore
	warmUpCycles int
	estimators   map[string]*InitEstimator
	initObs      int
	holdBack     bool
}
```

with:

```go
// After:
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
}
```

- [ ] **Step 2: Update `NewTunerService`**

Replace:

```go
func NewTunerService(warmUpCycles, initObs int, holdBack bool) *TunerService {
	return &TunerService{
		paramStore:   NewParameterStore(),
		warmUpCycles: warmUpCycles,
		estimators:   make(map[string]*InitEstimator),
		initObs:      initObs,
		holdBack:     holdBack,
	}
}
```

with:

```go
func NewTunerService(warmUpCycles, initObs int, holdBack bool, useSliding bool, windowSize int, residualThreshold float64) *TunerService {
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
	}
}
```

- [ ] **Step 3: Add `slidingEstimatorFor` and `tuneGroupSliding`**

Add these two methods to `service.go` (after `estimatorFor`):

```go
// slidingEstimatorFor returns the SlidingWindowEstimator for the given key, creating and
// seeding it from ie.observations if it does not yet exist.
func (ts *TunerService) slidingEstimatorFor(key string, ie *InitEstimator) *SlidingWindowEstimator {
	if swe, ok := ts.slidingEstimators[key]; ok {
		return swe
	}
	swe := NewSlidingWindowEstimator(ts.windowSize, ts.residualThreshold)
	swe.Seed(ie.observations)
	ts.slidingEstimators[key] = swe
	return swe
}

// tuneGroupSliding performs parameter estimation for one (model, accelerator) group using
// the SlidingWindowEstimator. Called by tuneGroup when useSliding is true and the
// InitEstimator has completed its collection phase.
func (ts *TunerService) tuneGroupSliding(model, accelerator, key string, ie *InitEstimator, env *core.EnvironmentPrefillDecode) error {
	_, alreadyExists := ts.slidingEstimators[key]
	swe := ts.slidingEstimatorFor(key, ie)
	if alreadyExists {
		// SWE already existed: add the current observation (it was not part of the seed).
		swe.AddObservation(env)
	}
	// If SWE was just created: ie.observations (which includes env) was used as the seed.

	if !swe.IsReady() {
		slog.Info("sliding window filling",
			"model", model, "accelerator", accelerator,
			"count", len(swe.window), "windowSize", swe.windowSize)
		return fmt.Errorf("sliding window filling for %s/%s (%d/%d)",
			model, accelerator, len(swe.window), swe.windowSize)
	}

	fitted, err := swe.Fit()
	if err != nil {
		return fmt.Errorf("SlidingWindowEstimator.Fit for %s/%s: %w", model, accelerator, err)
	}

	updateCount := 0
	if existing := ts.paramStore.Get(model, accelerator); existing != nil {
		updateCount = existing.UpdateCount
	}
	ts.paramStore.Set(model, accelerator, &LearnedParameters{
		Alpha:       float32(fitted[0]),
		Beta:        float32(fitted[1]),
		Gamma:       float32(fitted[2]),
		NIS:         0,
		UpdateCount: updateCount + 1,
		LastUpdated: time.Now(),
	})
	slog.Info("sliding-window tuned parameters",
		"model", model, "accelerator", accelerator,
		"alpha", fitted[0], "beta", fitted[1], "gamma", fitted[2],
		"updateCount", updateCount+1)
	return nil
}
```

- [ ] **Step 4: Branch `tuneGroup` on `useSliding`**

In `tuneGroup`, after the `!estimator.IsReady()` guard (around line 103), add the SWNM branch immediately after:

```go
	if !estimator.IsReady() {
		slog.Info("collecting initial observations",
			"model", model, "accelerator", accelerator,
			"count", estimator.ObsCount(), "minObs", estimator.MinObs())
		return fmt.Errorf("collecting initial observations for %s/%s (%d/%d)",
			model, accelerator, estimator.ObsCount(), estimator.MinObs())
	}

	// SWNM path: bypass the EKF and use the sliding-window estimator instead.
	if ts.useSliding {
		return ts.tuneGroupSliding(model, accelerator, key, estimator, envs[0])
	}

	// Existing EKF path continues below...
```

- [ ] **Step 5: Build**

```bash
go build ./...
```

Expected: compile error about `NewTunerService` call sites with wrong arity. Fix in next steps.

- [ ] **Step 6: Update existing `NewTunerService` call sites in tests**

In `tunerservice/init_estimator_test.go`, update two calls:

Replace:
```go
ts := NewTunerService(3, 3, true)
```
with:
```go
ts := NewTunerService(3, 3, true, false, DefaultWindowSize, DefaultResidualThreshold)
```

(Two occurrences: `TestTunerService_IsWarmingUp_DuringCollection` and `TestTunerService_IsWarmingUp_HoldBackFalse` — the second uses `false` as the third arg but the pattern is the same.)

Replace:
```go
ts := NewTunerService(3, 3, false)
```
with:
```go
ts := NewTunerService(3, 3, false, false, DefaultWindowSize, DefaultResidualThreshold)
```

- [ ] **Step 7: Build again**

```bash
go build ./...
```

Expected: success (remaining `cmd/tuner/main.go` is fixed in Task 7).

- [ ] **Step 8: Run existing tests to confirm nothing regressed**

```bash
go test ./tunerservice/... -v 2>&1 | tail -20
```

Expected: all existing tests PASS.

- [ ] **Step 9: Commit**

```bash
git add tunerservice/service.go tunerservice/init_estimator_test.go
git commit -m "feat: add SWNM fields to TunerService and tuneGroupSliding integration"
```

---

## Task 6: Write and pass integration tests for the SWNM path

**Files:**
- Create: `tunerservice/service_sliding_test.go`

- [ ] **Step 1: Create the file**

```go
package tunerservice

import (
	"testing"

	optconfig "github.com/llm-inferno/optimizer-light/pkg/config"
)

// makeTestSpec returns a minimal ServerSpec with sufficient fields for buildEnvironments.
func makeTestSpec(model, acc string, lambda, ttft, itl, inTok, outTok float32, maxBatch int) optconfig.ServerSpec {
	return optconfig.ServerSpec{
		Model:        model,
		MaxBatchSize: maxBatch,
		CurrentAlloc: optconfig.AllocationData{
			Accelerator: acc,
			MaxBatch:    maxBatch,
			TTFTAverage: ttft,
			ITLAverage:  itl,
			Load: optconfig.ServerLoadSpec{
				ArrivalRate:  lambda,
				AvgInTokens:  float64(inTok),
				AvgOutTokens: float64(outTok),
			},
		},
	}
}

// TestTunerService_SWNM_ReturnsParamsAfterWindowFills verifies that Tune() returns
// parameters once the sliding window is full and returns an error while it is filling.
func TestTunerService_SWNM_ReturnsParamsAfterWindowFills(t *testing.T) {
	initObs := 3
	windowSize := 5
	ts := NewTunerService(0, initObs, false, true, windowSize, DefaultResidualThreshold)

	spec := makeTestSpec("llama", "H100", 15, 55, 6, 120, 700, 64)

	// Cycles 1–3: collection phase (InitEstimator filling).
	for i := range initObs {
		_, err := ts.Tune([]optconfig.ServerSpec{spec})
		if err == nil {
			t.Fatalf("cycle %d: expected error during collection, got nil", i+1)
		}
	}

	// Cycle 4: SWE seeded with 3 obs, adds 1 more (total=4 < windowSize=5) — still filling.
	_, err := ts.Tune([]optconfig.ServerSpec{spec})
	if err == nil {
		t.Fatal("cycle 4: expected error while sliding window filling")
	}

	// Cycle 5: window reaches 5 — should now return parameters.
	result, err := ts.Tune([]optconfig.ServerSpec{spec})
	if err != nil {
		t.Fatalf("cycle 5: expected parameters, got error: %v", err)
	}
	if len(result.PerfData) == 0 {
		t.Fatal("cycle 5: expected non-empty PerfData")
	}
	p := result.PerfData[0]
	if p.PerfParms.Alpha <= 0 || p.PerfParms.Beta <= 0 || p.PerfParms.Gamma <= 0 {
		t.Errorf("expected positive params, got alpha=%.4f beta=%.4f gamma=%.6f",
			p.PerfParms.Alpha, p.PerfParms.Beta, p.PerfParms.Gamma)
	}
}

// TestTunerService_IsWarmingUp_SWNM_WindowNotFull verifies that IsWarmingUp returns
// true while the sliding window is being filled.
func TestTunerService_IsWarmingUp_SWNM_WindowNotFull(t *testing.T) {
	ts := NewTunerService(3, 3, true, true, 5, DefaultResidualThreshold)
	key := makeKey("mymodel", "myacc")
	ts.estimators[key] = NewInitEstimator(3, true)

	// InitEstimator not ready + holdBack=true → warmingUp
	if !ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=true when estimator not ready and holdBack=true")
	}

	// InitEstimator ready but no SWE yet → still warmingUp
	ie := NewInitEstimator(3, true)
	ie.observations = []fitObservation{
		{Lambda: 10, AvgTTFT: 50, AvgITL: 5, AvgInputTokens: 100, AvgOutputTokens: 500, MaxBatch: 64},
		{Lambda: 10, AvgTTFT: 50, AvgITL: 5, AvgInputTokens: 100, AvgOutputTokens: 500, MaxBatch: 64},
		{Lambda: 10, AvgTTFT: 50, AvgITL: 5, AvgInputTokens: 100, AvgOutputTokens: 500, MaxBatch: 64},
	}
	ts.estimators[key] = ie
	if !ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=true when InitEstimator ready but SWE not yet created")
	}

	// SWE created but not full → still warmingUp
	swe := NewSlidingWindowEstimator(5, DefaultResidualThreshold)
	swe.Seed(ie.observations) // 3 obs, windowSize=5 → not ready
	ts.slidingEstimators[key] = swe
	if !ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=true when SWE window not full")
	}
}

// TestTunerService_IsWarmingUp_SWNM_WindowFull verifies that IsWarmingUp returns false
// once the sliding window is full.
func TestTunerService_IsWarmingUp_SWNM_WindowFull(t *testing.T) {
	ts := NewTunerService(3, 3, true, true, 3, DefaultResidualThreshold)
	key := makeKey("mymodel", "myacc")

	ie := NewInitEstimator(3, true)
	obs := fitObservation{Lambda: 10, AvgTTFT: 50, AvgITL: 5, AvgInputTokens: 100, AvgOutputTokens: 500, MaxBatch: 64}
	ie.observations = []fitObservation{obs, obs, obs}
	ts.estimators[key] = ie

	swe := NewSlidingWindowEstimator(3, DefaultResidualThreshold)
	swe.Seed(ie.observations) // 3 obs, windowSize=3 → ready
	ts.slidingEstimators[key] = swe

	if ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=false when SWE window is full")
	}
}
```

- [ ] **Step 2: Run these tests to confirm they fail**

```bash
go test ./tunerservice/... -run TestTunerService_SWNM -v
go test ./tunerservice/... -run TestTunerService_IsWarmingUp_SWNM -v
```

Expected: compile errors about `IsWarmingUp` not handling SWNM, and the SWNM integration returning wrong results.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tunerservice/service_sliding_test.go
git commit -m "test: add failing SWNM integration tests for TunerService"
```

---

## Task 7: Update `IsWarmingUp` for SWNM mode

**Files:**
- Modify: `tunerservice/service.go`

- [ ] **Step 1: Replace `IsWarmingUp`**

Replace the entire `IsWarmingUp` method with:

```go
func (ts *TunerService) IsWarmingUp() bool {
	// Phase 1: InitEstimator collection (applies to both modes when holdBack=true).
	for _, ie := range ts.estimators {
		if !ie.IsReady() && ie.HoldBack() {
			return true
		}
	}

	if ts.useSliding {
		// SWNM mode: warm-up continues until every ready InitEstimator has a full SWE window.
		for key, ie := range ts.estimators {
			if !ie.IsReady() {
				continue // holdBack=false; collection doesn't gate controller
			}
			swe, ok := ts.slidingEstimators[key]
			if !ok || !swe.IsReady() {
				return true
			}
		}
		return false
	}

	// EKF mode: warm-up based on accepted update count.
	if ts.warmUpCycles == 0 {
		return false
	}
	for _, params := range ts.paramStore.GetAll() {
		if params.UpdateCount < ts.warmUpCycles {
			return true
		}
	}
	return false
}
```

- [ ] **Step 2: Run all tests**

```bash
go test ./tunerservice/... -v 2>&1 | tail -30
```

Expected: all tests PASS, including `TestTunerService_SWNM_ReturnsParamsAfterWindowFills` and the `IsWarmingUp_SWNM_*` tests.

- [ ] **Step 3: Commit**

```bash
git add tunerservice/service.go
git commit -m "feat: update IsWarmingUp to use SWE readiness as gate in SWNM mode"
```

---

## Task 8: Wire new env vars in `cmd/tuner/main.go`

**Files:**
- Modify: `cmd/tuner/main.go`

- [ ] **Step 1: Add imports**

`strconv` is already imported. Verify `cmd/tuner/main.go` imports section includes it (it does).

- [ ] **Step 2: Add env var reading and update `NewTunerService` call**

Replace:

```go
	service := tunerservice.NewTunerService(warmUpCycles, initObs, holdBack)
```

with:

```go
	useSliding := os.Getenv(tunerservice.EstimatorModeEnvName) == "sliding-window"

	windowSize := tunerservice.DefaultWindowSize
	if v := os.Getenv(tunerservice.WindowSizeEnvName); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			windowSize = n
		}
	}

	residualThreshold := tunerservice.DefaultResidualThreshold
	if v := os.Getenv(tunerservice.ResidualThresholdEnvName); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
			residualThreshold = f
		}
	}

	service := tunerservice.NewTunerService(warmUpCycles, initObs, holdBack, useSliding, windowSize, residualThreshold)
```

Also update the `slog.Info` call to include the new fields:

```go
	slog.Info("Starting TunerService",
		"host", host, "port", port,
		"warmUpCycles", warmUpCycles,
		"initObs", initObs,
		"holdBack", holdBack,
		"estimatorMode", os.Getenv(tunerservice.EstimatorModeEnvName),
		"windowSize", windowSize,
		"residualThreshold", residualThreshold)
```

- [ ] **Step 3: Build everything**

```bash
go build ./...
```

Expected: success with no output.

- [ ] **Step 4: Run full test suite**

```bash
go test ./... -v 2>&1 | grep -E "^(=== RUN|--- PASS|--- FAIL|FAIL|ok)" | head -60
```

Expected: all tests PASS, no FAIL lines.

- [ ] **Step 5: Commit**

```bash
git add cmd/tuner/main.go
git commit -m "feat: wire TUNER_ESTIMATOR_MODE, TUNER_WINDOW_SIZE, TUNER_RESIDUAL_THRESHOLD env vars"
```

---

## Task 9: Final verification

- [ ] **Step 1: Full build**

```bash
go build ./...
```

Expected: success.

- [ ] **Step 2: Full test suite**

```bash
go test ./... -count=1
```

Expected: all packages PASS.

- [ ] **Step 3: Push branch**

```bash
git push -u origin feat/sliding-window-estimator
```

- [ ] **Step 4: Create PR**

```bash
gh pr create \
  --title "feat: sliding-window Nelder-Mead estimator as EKF alternative" \
  --body "$(cat <<'EOF'
## Summary

- Add `SlidingWindowEstimator` that re-runs Nelder-Mead on a fixed-size circular buffer of recent observations each cycle, bypassing the EKF entirely
- Residual-based outlier rejection: post-fit, observations with relative error > `TUNER_RESIDUAL_THRESHOLD` are dropped and the fit is rerun once
- `InitEstimator` observations seed the sliding window, so the first estimate arrives after `windowSize` total observations (not `initObs + windowSize`)
- `IsWarmingUp` uses SWE readiness (window full) as the controller gate in SWNM mode
- EKF path is unchanged; mode is selected via `TUNER_ESTIMATOR_MODE=sliding-window`

Closes #8

## New env vars

| Variable | Default | Purpose |
|---|---|---|
| `TUNER_ESTIMATOR_MODE` | `ekf` | `ekf` \| `sliding-window` |
| `TUNER_WINDOW_SIZE` | `10` | Sliding window capacity |
| `TUNER_RESIDUAL_THRESHOLD` | `0.5` | Per-observation relative error cutoff |

## Test plan

- [ ] `go test ./tunerservice/... -v` — all unit + integration tests pass
- [ ] `go build ./...` — clean build
- [ ] Manually run `TUNER_ESTIMATOR_MODE=sliding-window go run ./cmd/tuner` and verify startup log shows correct mode
- [ ] Test with `TUNER_ESTIMATOR_MODE=ekf` (default) to confirm EKF path is unaffected

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---
