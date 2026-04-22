# pkg/estimator + pkg/service Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `InitEstimator`, `SlidingWindowEstimator`, and `TunerService` out of `tunerservice/` into `pkg/estimator/` and `pkg/service/` so consumers can import estimation logic without taking on an HTTP dependency.

**Architecture:** Three layers — `pkg/estimator` (pure estimation primitives), `pkg/service` (orchestration: TunerService + ParameterStore), `tunerservice` (thin HTTP adapter). No behavior changes; HTTP API surface is identical.

**Tech Stack:** Go 1.25, gin (HTTP only), gonum/optimize (Nelder-Mead), queue-analysis analyzer

---

## File Map

### Create
- `pkg/estimator/defaults.go` — `baseFactor` constant
- `pkg/estimator/fit_observation.go` — unexported `fitObservation` struct + `toEnv()`
- `pkg/estimator/guess.go` — exported `GuessInitState` (renamed from `guessInitState`)
- `pkg/estimator/init_estimator.go` — `InitEstimator` (moved from tunerservice, package updated)
- `pkg/estimator/sliding_window_estimator.go` — `SlidingWindowEstimator` (moved, add `SeedFromEstimator`)
- `pkg/estimator/doc.go`
- `pkg/estimator/init_estimator_test.go` — InitEstimator tests + `makeTestEnv`/`makeTestEnvWithQueue` helpers
- `pkg/estimator/sliding_window_estimator_test.go` — SlidingWindowEstimator tests
- `pkg/service/defaults.go` — estimator/service env-var names and defaults
- `pkg/service/doc.go`
- `pkg/service/parameters.go` — `LearnedParameters`, `ParameterStore`, `makeKey`, `splitKey`, `covToSlice`
- `pkg/service/service.go` — `TunerService` (moved, imports `pkg/estimator`)
- `pkg/service/utils.go` — `buildEnvironments`, `groupByModelAccelerator`, `maxBatchFromReplicas`, `setInitState`
- `pkg/service/service_test.go` — TunerService warm-up tests (split from old `init_estimator_test.go`)
- `pkg/service/service_sliding_test.go` — SWNM tests (moved, API-only access)
- `pkg/service/utils_test.go` — `buildEnvironments` tests

### Modify
- `tunerservice/server.go` — import `pkg/service`, update `*TunerService` type
- `tunerservice/defaults.go` — trim to HTTP-only constants
- `tunerservice/doc.go` — trim to HTTP adapter description
- `tunerservice/utils.go` — keep only `validateKey`
- `cmd/tuner/main.go` — import `pkg/service` for constants + `NewTunerService`

### Delete
- `tunerservice/init_estimator.go`
- `tunerservice/sliding_window_estimator.go`
- `tunerservice/service.go`
- `tunerservice/parameters.go`
- `tunerservice/handlers.go` — **no change needed** (accesses `ts.service.*` without naming pkgsvc types)
- `tunerservice/init_estimator_test.go`
- `tunerservice/sliding_window_estimator_test.go`
- `tunerservice/service_sliding_test.go`
- `tunerservice/utils_test.go`

---

## Task 1: Create Feature Branch

- [ ] **Create and switch to branch**

```bash
git checkout -b issue-14-pkg-restructure
```

Expected: `Switched to a new branch 'issue-14-pkg-restructure'`

---

## Task 2: Create pkg/estimator Foundation Files

**Files:** Create `pkg/estimator/defaults.go`, `pkg/estimator/fit_observation.go`, `pkg/estimator/guess.go`, `pkg/estimator/doc.go`

- [ ] **Create pkg/estimator/defaults.go**

```go
package estimator

// baseFactor is the fraction of ITL assumed to be baseline iteration overhead (alpha).
// Used in GuessInitState to derive an initial estimate of alpha from observed ITL.
const baseFactor = 0.9
```

- [ ] **Create pkg/estimator/fit_observation.go**

```go
package estimator

import "github.com/llm-inferno/model-tuner/pkg/core"

// fitObservation holds a single operating-point snapshot used in Nelder-Mead objectives.
type fitObservation struct {
	Lambda          float64
	MaxBatch        int
	MaxQueueSize    int
	AvgInputTokens  float32
	AvgOutputTokens float32
	AvgTTFT         float64
	AvgITL          float64
}

func (fo *fitObservation) toEnv() *core.EnvironmentPrefillDecode {
	env := core.NewEnvironmentPrefillDecode(
		float32(fo.Lambda),
		0,
		0,
		fo.MaxBatch,
		fo.AvgInputTokens,
		fo.AvgOutputTokens,
		float32(fo.AvgTTFT),
		float32(fo.AvgITL),
	)
	env.MaxQueueSize = fo.MaxQueueSize
	return env
}
```

- [ ] **Create pkg/estimator/guess.go**

```go
package estimator

import "github.com/llm-inferno/model-tuner/pkg/core"

// GuessInitState derives initial alpha, beta, gamma from observed TTFT and ITL using the
// queueing model equations from the paper:
//
//	TTFT = alpha + (beta + gamma) * inputTokens           (eq 12)
//	ITL  = alpha + beta + gamma * (inputTokens + (outputTokens+1)/2)  (eq 13)
//
// Returns nil if the derivation yields non-positive parameters.
func GuessInitState(env *core.EnvironmentPrefillDecode) []float64 {
	if env == nil || !env.Valid() {
		return nil
	}
	ttft := float64(env.AvgTTFT)
	itl := float64(env.AvgITL)
	inputToks := float64(env.AvgInputTokens)
	outputToks := float64(env.AvgOutputTokens)

	if ttft <= 0 || itl <= 0 || inputToks <= 0 || outputToks <= 0 {
		return nil
	}

	alpha := baseFactor * itl
	sumBetaGamma := (ttft - alpha) / inputToks
	if sumBetaGamma < 0 {
		return nil
	}

	denominator := inputToks + (outputToks+1)/2 - 1
	if denominator <= 0 {
		return nil
	}
	gamma := ((itl - alpha) - sumBetaGamma) / denominator
	beta := sumBetaGamma - gamma

	if alpha <= 0 || beta <= 0 || gamma <= 0 {
		return nil
	}
	return []float64{alpha, beta, gamma}
}
```

- [ ] **Create pkg/estimator/doc.go**

```go
// Package estimator provides pure estimation primitives for LLM inference parameter tuning.
//
// It contains two estimators and the shared types they depend on:
//
//   - [InitEstimator]: collects K initial observations then runs Nelder-Mead to fit
//     an initial (alpha, beta, gamma) before handing off to the EKF or SWNM.
//
//   - [SlidingWindowEstimator]: maintains a fixed-capacity circular buffer of recent
//     observations and re-runs Nelder-Mead on every Fit() call.
//
// [GuessInitState] provides an algebraic cold-start estimate from a single observation.
// Both estimators use it as a Nelder-Mead warm-start and fallback.
//
// This package has no dependency on HTTP routing or the optimizer-light config types.
// It depends only on pkg/core (for EnvironmentPrefillDecode) and the queue-analysis
// analyzer (for the queueing model objective function).
package estimator
```

---

## Task 3: Create pkg/estimator/init_estimator.go

**Files:** Create `pkg/estimator/init_estimator.go`

- [ ] **Create pkg/estimator/init_estimator.go**

This is a move of `tunerservice/init_estimator.go` with three changes: package declaration, removal of `fitObservation` (now in `fit_observation.go`), and `guessInitState` → `GuessInitState`.

```go
package estimator

import (
	"fmt"
	"log/slog"
	"math"

	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
	"gonum.org/v1/gonum/optimize"

	"github.com/llm-inferno/model-tuner/pkg/core"
)

// InitEstimator accumulates observations before the EKF starts and fits initial parameters.
type InitEstimator struct {
	observations     []fitObservation
	minObs           int
	holdBack         bool
	fitDone          bool
	lastFitFuncValue float64
}

// NewInitEstimator creates an InitEstimator with the given minimum observation count and hold-back flag.
func NewInitEstimator(minObs int, holdBack bool) *InitEstimator {
	if minObs < 1 {
		minObs = 1
	}
	return &InitEstimator{
		minObs:   minObs,
		holdBack: holdBack,
	}
}

// AddObservation records a new operating-point observation from an environment.
func (ie *InitEstimator) AddObservation(env *core.EnvironmentPrefillDecode) {
	if env == nil || !env.Valid() {
		return
	}
	ie.observations = append(ie.observations, fitObservation{
		Lambda:          float64(env.Lambda),
		MaxBatch:        env.MaxBatchSize,
		MaxQueueSize:    env.MaxQueueSize,
		AvgInputTokens:  env.AvgInputTokens,
		AvgOutputTokens: env.AvgOutputTokens,
		AvgTTFT:         float64(env.AvgTTFT),
		AvgITL:          float64(env.AvgITL),
	})
}

// IsReady returns true once at least minObs observations have been collected.
func (ie *InitEstimator) IsReady() bool {
	return len(ie.observations) >= ie.minObs
}

// HoldBack returns true if the controller should report warmingUp=true during collection.
func (ie *InitEstimator) HoldBack() bool {
	return ie.holdBack
}

// ObsCount returns the number of observations accumulated so far.
func (ie *InitEstimator) ObsCount() int { return len(ie.observations) }

// MinObs returns the minimum number of observations required before Fit() can run.
func (ie *InitEstimator) MinObs() int { return ie.minObs }

// FitDone returns true if Fit() has already been called (regardless of success or failure).
func (ie *InitEstimator) FitDone() bool { return ie.fitDone }

// LastFitFuncValue returns the Nelder-Mead objective value from the most recent Fit() call.
// Returns 0 if Fit() has not been called yet, math.MaxFloat64 if the fit fell back to GuessInitState.
func (ie *InitEstimator) LastFitFuncValue() float64 { return ie.lastFitFuncValue }

// Fit runs Nelder-Mead minimisation over all accumulated observations to find the
// (alpha, beta, gamma) that best explains all K observations jointly via the full
// queueing model. Returns [alpha, beta, gamma] or an error.
// Falls back to GuessInitState on the first observation if the fit fails.
func (ie *InitEstimator) Fit() ([]float64, error) {
	if len(ie.observations) == 0 {
		return nil, fmt.Errorf("no observations to fit")
	}

	x0 := GuessInitState(ie.observations[0].toEnv())
	if x0 == nil {
		x0 = []float64{5.0, 0.05, 0.0005}
	}

	result, err := ie.fitWithX0(x0)
	ie.fitDone = true
	return result, err
}

// fitWithX0 runs the Nelder-Mead optimisation starting from the given x0.
func (ie *InitEstimator) fitWithX0(x0 []float64) ([]float64, error) {
	if len(ie.observations) == 0 {
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
		return ie.objective(unscaled)
	}

	problem := optimize.Problem{Func: scaledObjective}
	settings := &optimize.Settings{FuncEvaluations: 500}
	result, err := optimize.Minimize(problem, scaledX0, settings, &optimize.NelderMead{})
	if err != nil {
		ie.lastFitFuncValue = math.MaxFloat64
		slog.Warn("InitEstimator: Nelder-Mead pre-flight error, using GuessInitState fallback", "err", err)
		if fallback := GuessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead failed and GuessInitState returned nil: %w", err)
	}

	switch result.Status {
	case optimize.Success, optimize.FunctionConvergence, optimize.FunctionEvaluationLimit:
	default:
		ie.lastFitFuncValue = math.MaxFloat64
		slog.Warn("InitEstimator: unexpected Nelder-Mead termination status, using GuessInitState fallback",
			"status", result.Status)
		if fallback := GuessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead unexpected status %v and GuessInitState returned nil", result.Status)
	}

	unscaled := make([]float64, len(result.X))
	for i := range result.X {
		unscaled[i] = result.X[i] * scale[i]
	}
	x := unscaled
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		ie.lastFitFuncValue = math.MaxFloat64
		slog.Warn("InitEstimator: Nelder-Mead returned non-positive params, using GuessInitState fallback",
			"alpha", x[0], "beta", x[1], "gamma", x[2])
		if fallback := GuessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead returned non-positive params and GuessInitState returned nil")
	}

	ie.lastFitFuncValue = result.F
	slog.Info("InitEstimator: Fit complete",
		"alpha", x[0], "beta", x[1], "gamma", x[2],
		"observations", len(ie.observations), "funcValue", result.F)
	return x, nil
}

// objective computes the sum of relative squared errors in TTFT and ITL across all
// stored observations, evaluated using the full queueing model for the trial params x=[α,β,γ].
func (ie *InitEstimator) objective(x []float64) float64 {
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		return math.MaxFloat64 / 2
	}

	var total float64
	for _, obs := range ie.observations {
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
			return math.MaxFloat64 / 2
		}
		metrics, err := qa.Analyze(float32(obs.Lambda / 60))
		if err != nil {
			return math.MaxFloat64 / 2
		}

		ttftModel := float64(metrics.AvgTTFT)
		itlModel := float64(metrics.AvgTokenTime)
		ttftObs := obs.AvgTTFT
		itlObs := obs.AvgITL

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

---

## Task 4: Create pkg/estimator/sliding_window_estimator.go

**Files:** Create `pkg/estimator/sliding_window_estimator.go`

Move of `tunerservice/sliding_window_estimator.go` with: package declaration updated, `guessInitState` → `GuessInitState`, and new `SeedFromEstimator` method added.

- [ ] **Create pkg/estimator/sliding_window_estimator.go**

```go
package estimator

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
	minObs            int
	residualThreshold float64
	lastFit           []float64
}

// NewSlidingWindowEstimator creates a SlidingWindowEstimator with the given window capacity,
// minimum observations before IsReady(), and residual outlier rejection threshold.
func NewSlidingWindowEstimator(windowSize, minObs int, residualThreshold float64) *SlidingWindowEstimator {
	if windowSize < 1 {
		windowSize = 1
	}
	if minObs < 1 {
		minObs = 1
	}
	if minObs > windowSize {
		minObs = windowSize
	}
	return &SlidingWindowEstimator{
		windowSize:        windowSize,
		minObs:            minObs,
		residualThreshold: residualThreshold,
	}
}

// SeedLastFit sets the warm-start x0 for the first Fit() call, e.g. from InitEstimator.Fit().
func (swe *SlidingWindowEstimator) SeedLastFit(x []float64) {
	if len(x) > 0 {
		swe.lastFit = x
	}
}

// Seed pre-fills the window with raw observations. Oldest entries are evicted when the
// seed exceeds windowSize.
func (swe *SlidingWindowEstimator) Seed(obs []fitObservation) {
	for _, o := range obs {
		swe.window = append(swe.window, o)
		if len(swe.window) > swe.windowSize {
			swe.window = swe.window[1:]
		}
	}
}

// SeedFromEstimator pre-fills the window from an InitEstimator's collected observations.
// Callers outside pkg/estimator use this instead of Seed to avoid accessing the unexported
// fitObservation type directly.
func (swe *SlidingWindowEstimator) SeedFromEstimator(ie *InitEstimator) {
	swe.Seed(ie.observations)
}

// AddObservation appends a new operating-point observation.
// Oldest entry is evicted when the window is at capacity.
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

// IsReady returns true once the window holds at least minObs observations.
func (swe *SlidingWindowEstimator) IsReady() bool {
	return len(swe.window) >= swe.minObs
}

// Fit runs Nelder-Mead on the current window, performs one residual-based outlier
// rejection pass, and refits if any observations were dropped.
func (swe *SlidingWindowEstimator) Fit() ([]float64, error) {
	if len(swe.window) == 0 {
		return nil, fmt.Errorf("no observations in window")
	}

	x0 := swe.lastFit
	if x0 == nil {
		x0 = GuessInitState(swe.window[len(swe.window)-1].toEnv())
	}
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
		fitted, err = swe.fitWithX0(fitted, cleaned)
		if err != nil {
			return nil, err
		}
	}

	swe.lastFit = fitted
	return fitted, nil
}

// filterOutliers removes the single observation with the largest residual if that residual
// exceeds swe.residualThreshold.
func (swe *SlidingWindowEstimator) filterOutliers(obs []fitObservation, x []float64) []fitObservation {
	if swe.residualThreshold <= 0 {
		return obs
	}
	worstIdx := -1
	worstResidual := swe.residualThreshold
	for i, o := range obs {
		r := swe.residual(o, x)
		if r > worstResidual {
			worstResidual = r
			worstIdx = i
		}
	}
	if worstIdx < 0 {
		return obs
	}
	kept := make([]fitObservation, 0, len(obs)-1)
	kept = append(kept, obs[:worstIdx]...)
	kept = append(kept, obs[worstIdx+1:]...)
	return kept
}

// residual returns sqrt(dTTFT² + dITL²) for one observation evaluated at params x=[α,β,γ].
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
		slog.Warn("SlidingWindowEstimator: Nelder-Mead pre-flight error, using GuessInitState fallback", "err", err)
		if fallback := GuessInitState(obs[len(obs)-1].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead failed and GuessInitState returned nil: %w", err)
	}

	switch result.Status {
	case optimize.Success, optimize.FunctionConvergence, optimize.FunctionEvaluationLimit:
	default:
		slog.Warn("SlidingWindowEstimator: unexpected Nelder-Mead termination", "status", result.Status)
		if fallback := GuessInitState(obs[len(obs)-1].toEnv()); fallback != nil {
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
		slog.Warn("SlidingWindowEstimator: non-positive params, using GuessInitState fallback",
			"alpha", x[0], "beta", x[1], "gamma", x[2])
		if fallback := GuessInitState(obs[len(obs)-1].toEnv()); fallback != nil {
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

---

## Task 5: Create pkg/estimator Test Files

**Files:** Create `pkg/estimator/init_estimator_test.go`, `pkg/estimator/sliding_window_estimator_test.go`

These are direct moves with only the package declaration changed. The `TunerService` warm-up tests that were in the original `init_estimator_test.go` are NOT included here — they move to `pkg/service/service_test.go` in Task 11.

- [ ] **Create pkg/estimator/init_estimator_test.go**

```go
package estimator

import (
	"math"
	"testing"

	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
)

func makeTestEnv(lambda, ttft, itl float32, inTok, outTok float32, maxBatch int) *core.EnvironmentPrefillDecode {
	return core.NewEnvironmentPrefillDecode(
		lambda, 0, 0, maxBatch, inTok, outTok, ttft, itl,
	)
}

func makeTestEnvWithQueue(lambda, ttft, itl float32, inTok, outTok float32, maxBatch, maxQueue int) *core.EnvironmentPrefillDecode {
	env := core.NewEnvironmentPrefillDecode(lambda, 0, 0, maxBatch, inTok, outTok, ttft, itl)
	env.MaxQueueSize = maxQueue
	return env
}

func TestInitEstimator_HoldBack(t *testing.T) {
	ie := NewInitEstimator(3, true)
	if !ie.HoldBack() {
		t.Fatal("expected HoldBack=true")
	}
	ie2 := NewInitEstimator(3, false)
	if ie2.HoldBack() {
		t.Fatal("expected HoldBack=false")
	}
}

func TestInitEstimator_IsReady(t *testing.T) {
	ie := NewInitEstimator(3, true)
	env := makeTestEnv(10, 50, 5, 100, 500, 64)

	if ie.IsReady() {
		t.Fatal("should not be ready with 0 observations")
	}
	ie.AddObservation(env)
	if ie.IsReady() {
		t.Fatal("should not be ready with 1 observation")
	}
	ie.AddObservation(env)
	if ie.IsReady() {
		t.Fatal("should not be ready with 2 observations")
	}
	ie.AddObservation(env)
	if !ie.IsReady() {
		t.Fatal("should be ready with 3 observations (minObs=3)")
	}
}

func TestInitEstimator_AddObservation_IgnoresNilAndInvalid(t *testing.T) {
	ie := NewInitEstimator(3, true)
	ie.AddObservation(nil)
	invalid := core.NewEnvironmentPrefillDecode(10, 0, 0, 64, 100, 500, 0, 0)
	ie.AddObservation(invalid)
	if len(ie.observations) != 0 {
		t.Fatalf("expected 0 observations stored, got %d", len(ie.observations))
	}
}

func TestInitEstimator_AddObservation_StoresFields(t *testing.T) {
	ie := NewInitEstimator(1, false)
	env := makeTestEnvWithQueue(30, 45.5, 6.2, 120, 700, 64, 128)
	ie.AddObservation(env)
	if len(ie.observations) != 1 {
		t.Fatalf("expected 1 observation, got %d", len(ie.observations))
	}
	obs := ie.observations[0]
	if obs.Lambda != float64(env.Lambda) {
		t.Errorf("Lambda mismatch: got %v want %v", obs.Lambda, env.Lambda)
	}
	if obs.MaxBatch != env.MaxBatchSize {
		t.Errorf("MaxBatch mismatch: got %v want %v", obs.MaxBatch, env.MaxBatchSize)
	}
	if obs.MaxQueueSize != env.MaxQueueSize {
		t.Errorf("MaxQueueSize mismatch: got %v want %v", obs.MaxQueueSize, env.MaxQueueSize)
	}
	if obs.AvgTTFT != float64(env.AvgTTFT) {
		t.Errorf("AvgTTFT mismatch: got %v want %v", obs.AvgTTFT, env.AvgTTFT)
	}
}

func TestInitEstimator_Fit_ParameterRecovery(t *testing.T) {
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
		{lambda: 18, inTok: 165, outTok: 1250},
	}

	ie := NewInitEstimator(len(ops), false)

	for _, op := range ops {
		qConfig := &analyzer.Configuration{
			MaxBatchSize: maxBatch,
			MaxQueueSize: 0,
			ServiceParms: &analyzer.ServiceParms{
				Alpha: trueAlpha,
				Beta:  trueBeta,
				Gamma: trueGamma,
			},
		}
		requestSize := &analyzer.RequestSize{
			AvgInputTokens:  op.inTok,
			AvgOutputTokens: op.outTok,
		}
		qa, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
		if err != nil {
			t.Fatalf("failed to create analyzer: %v", err)
		}
		metrics, err := qa.Analyze(op.lambda / 60)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}

		env := core.NewEnvironmentPrefillDecode(
			op.lambda, 0, 0, maxBatch,
			op.inTok, op.outTok,
			metrics.AvgTTFT, metrics.AvgTokenTime,
		)
		ie.AddObservation(env)
	}

	if !ie.IsReady() {
		t.Fatal("estimator should be ready after adding all observations")
	}

	fitted, err := ie.Fit()
	if err != nil {
		t.Fatalf("Fit() returned error: %v", err)
	}
	if len(fitted) != 3 {
		t.Fatalf("expected 3 params, got %d", len(fitted))
	}

	tolerance := 0.10
	checkParam := func(name string, got, want float64) {
		t.Helper()
		relErr := math.Abs(got-want) / want
		if relErr > tolerance {
			t.Errorf("param %s: got %.6f, want %.6f (relative error %.1f%% > %.0f%%)",
				name, got, want, relErr*100, tolerance*100)
		}
	}
	checkParam("alpha", fitted[0], float64(trueAlpha))
	checkParam("beta", fitted[1], float64(trueBeta))
	checkParam("gamma", fitted[2], float64(trueGamma))
}

func TestInitEstimator_Fit_PoorStartingPoint(t *testing.T) {
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
		{lambda: 18, inTok: 165, outTok: 1250},
	}

	ie := NewInitEstimator(len(ops), false)

	for _, op := range ops {
		qConfig := &analyzer.Configuration{
			MaxBatchSize: maxBatch,
			MaxQueueSize: 0,
			ServiceParms: &analyzer.ServiceParms{
				Alpha: trueAlpha,
				Beta:  trueBeta,
				Gamma: trueGamma,
			},
		}
		requestSize := &analyzer.RequestSize{
			AvgInputTokens:  op.inTok,
			AvgOutputTokens: op.outTok,
		}
		qa, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
		if err != nil {
			t.Fatalf("failed to create analyzer: %v", err)
		}
		metrics, err := qa.Analyze(op.lambda / 60)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}
		env := core.NewEnvironmentPrefillDecode(
			op.lambda, 0, 0, maxBatch,
			op.inTok, op.outTok,
			metrics.AvgTTFT, metrics.AvgTokenTime,
		)
		ie.AddObservation(env)
	}

	poorX0 := []float64{1.0, 0.01, 0.0001}
	fitted, err := ie.fitWithX0(poorX0)
	if err != nil {
		t.Fatalf("fitWithX0() returned error: %v", err)
	}
	if len(fitted) != 3 {
		t.Fatalf("expected 3 params, got %d", len(fitted))
	}

	tolerance := 0.10
	checkParam := func(name string, got, want float64) {
		t.Helper()
		relErr := math.Abs(got-want) / want
		if relErr > tolerance {
			t.Errorf("param %s: got %.6f, want %.6f (relative error %.1f%% > %.0f%%)",
				name, got, want, relErr*100, tolerance*100)
		}
	}
	checkParam("alpha", fitted[0], float64(trueAlpha))
	checkParam("beta", fitted[1], float64(trueBeta))
	checkParam("gamma", fitted[2], float64(trueGamma))
}

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
	ie := NewInitEstimator(1, false)
	ie.AddObservation(makeTestEnv(15, 55, 6, 120, 700, 64))
	ie.fitWithX0([]float64{0, 0, 0})
	if ie.LastFitFuncValue() != math.MaxFloat64 {
		t.Errorf("expected math.MaxFloat64 after fallback, got %f", ie.LastFitFuncValue())
	}
}
```

- [ ] **Create pkg/estimator/sliding_window_estimator_test.go**

```go
package estimator

import (
	"math"
	"testing"

	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
)

func TestSlidingWindowEstimator_IsReady(t *testing.T) {
	swe := NewSlidingWindowEstimator(5, 3, 0.5)
	env := makeTestEnv(10, 50, 5, 100, 500, 64)

	if swe.IsReady() {
		t.Fatal("should not be ready with 0 observations")
	}
	swe.AddObservation(env)
	if swe.IsReady() {
		t.Fatal("should not be ready with 1 observation (minObs=3)")
	}
	swe.AddObservation(env)
	swe.AddObservation(env)
	if !swe.IsReady() {
		t.Fatal("should be ready with 3 observations (minObs=3, windowSize=5)")
	}
}

func TestSlidingWindowEstimator_AddObservation_Caps(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 1, 0.5)
	e1 := makeTestEnv(10, 50, 5, 100, 500, 64)
	e2 := makeTestEnv(20, 60, 6, 110, 510, 64)
	e3 := makeTestEnv(30, 70, 7, 120, 520, 64)
	e4 := makeTestEnv(40, 80, 8, 130, 530, 64)

	swe.AddObservation(e1)
	swe.AddObservation(e2)
	swe.AddObservation(e3)
	swe.AddObservation(e4)

	if len(swe.window) != 3 {
		t.Fatalf("expected window size 3, got %d", len(swe.window))
	}
	if swe.window[0].Lambda != float64(e2.Lambda) {
		t.Errorf("expected oldest entry lambda=20, got %v", swe.window[0].Lambda)
	}
	if swe.window[2].Lambda != float64(e4.Lambda) {
		t.Errorf("expected newest entry lambda=40, got %v", swe.window[2].Lambda)
	}
}

func TestSlidingWindowEstimator_AddObservation_IgnoresNilAndInvalid(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 1, 0.5)
	swe.AddObservation(nil)
	invalid := core.NewEnvironmentPrefillDecode(10, 0, 0, 64, 100, 500, 0, 0)
	swe.AddObservation(invalid)
	if len(swe.window) != 0 {
		t.Fatalf("expected 0 observations stored, got %d", len(swe.window))
	}
}

func TestSlidingWindowEstimator_Seed(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 1, 0.5)
	obs := []fitObservation{
		{Lambda: 10}, {Lambda: 20}, {Lambda: 30}, {Lambda: 40}, {Lambda: 50},
	}
	swe.Seed(obs)
	if len(swe.window) != 3 {
		t.Fatalf("expected window capped at 3, got %d", len(swe.window))
	}
	if swe.window[0].Lambda != 30 {
		t.Errorf("expected lambda=30 at index 0, got %v", swe.window[0].Lambda)
	}
}

func TestSlidingWindowEstimator_Fit_EmptyError(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 1, 0.5)
	_, err := swe.Fit()
	if err == nil {
		t.Fatal("expected error from Fit() on empty window")
	}
}

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

	swe := NewSlidingWindowEstimator(len(ops), 1, 0.5)
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

	swe := NewSlidingWindowEstimator(5, 1, 0.5)

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
	outlier := core.NewEnvironmentPrefillDecode(10, 0, 0, maxBatch, 90, 670, 500, 5)
	swe.AddObservation(outlier)

	if !swe.IsReady() {
		t.Fatal("expected window to be ready")
	}
	fitted, err := swe.Fit()
	if err != nil {
		t.Fatalf("Fit() with outlier returned error: %v", err)
	}

	tolerance := 0.15
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

func TestSlidingWindowEstimator_FilterOutliers_DisabledWhenThresholdZero(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 1, 0)
	obs := []fitObservation{
		{Lambda: 10, AvgTTFT: 50, AvgITL: 5, AvgInputTokens: 100, AvgOutputTokens: 500, MaxBatch: 64},
		{Lambda: 20, AvgTTFT: 60, AvgITL: 6, AvgInputTokens: 110, AvgOutputTokens: 510, MaxBatch: 64},
	}
	x := []float64{1.0, 0.01, 0.001}
	kept := swe.filterOutliers(obs, x)
	if len(kept) != len(obs) {
		t.Errorf("expected all %d kept when threshold=0, got %d", len(obs), len(kept))
	}
}
```

- [ ] **Verify pkg/estimator compiles and tests pass**

```bash
go build ./pkg/estimator/... && go test ./pkg/estimator/...
```

Expected: all tests pass (same tests as before, just moved)

- [ ] **Commit**

```bash
git add pkg/estimator/
git commit -m "feat(pkg/estimator): add pure estimation library extracted from tunerservice"
```

---

## Task 6: Create pkg/service Foundation and Parameters

**Files:** Create `pkg/service/defaults.go`, `pkg/service/doc.go`, `pkg/service/parameters.go`

- [ ] **Create pkg/service/defaults.go**

```go
package service

// Environment variable names and defaults for tuner behaviour.
const (
	WarmUpCyclesEnvName = "TUNER_WARM_UP_CYCLES"
)

// Environment variable names and defaults for the InitEstimator.
const (
	InitObsEnvName      = "TUNER_INIT_OBS"
	InitHoldBackEnvName = "TUNER_INIT_HOLD_BACK"

	DefaultInitObs      = 5
	DefaultInitHoldBack = true
)

// Environment variable names and defaults for the SlidingWindowEstimator.
const (
	EstimatorModeEnvName     = "TUNER_ESTIMATOR_MODE"
	WindowSizeEnvName        = "TUNER_WINDOW_SIZE"
	ResidualThresholdEnvName = "TUNER_RESIDUAL_THRESHOLD"

	DefaultEstimatorMode     = "ekf"
	DefaultWindowSize        = 10
	DefaultResidualThreshold = 0.5
)

// Environment variable name and default for the init-fit quality threshold.
const (
	InitFitThresholdEnvName = "TUNER_INIT_FIT_THRESHOLD"
	DefaultInitFitThreshold = 10.0
)

// Default field values used when the ParameterStore has a model/accelerator entry
// that is not present in the Controller's current ModelData.
const (
	DefaultAccCount     = 1
	DefaultMaxBatchSize = 256
	DefaultAtTokens     = 1024
)
```

- [ ] **Create pkg/service/doc.go**

```go
// Package service provides TunerService, the orchestration layer for LLM inference
// parameter tuning. It sits between the estimation primitives in pkg/estimator and the
// HTTP adapter in tunerservice.
//
// TunerService groups per-replica ServerSpecs by (model, accelerator), runs an
// InitEstimator cold-start phase for each new pair, then dispatches subsequent
// observations to either a SlidingWindowEstimator (SWNM mode) or the EKF Tuner
// (pkg/core). Tuned parameters are stored in a ParameterStore for state continuity
// across tuning cycles.
//
// Consumers that want estimation without the HTTP layer can import this package directly
// and call [TunerService.Tune], [TunerService.GetParams], and [TunerService.Merge]
// without running a gin server.
package service
```

- [ ] **Create pkg/service/parameters.go**

Move of `tunerservice/parameters.go` with package declaration and the addition of `splitKey` (moved from `tunerservice/utils.go`).

```go
package service

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

// LearnedParameters holds the tuned parameters for one model/accelerator pair.
type LearnedParameters struct {
	Alpha       float32
	Beta        float32
	Gamma       float32
	NIS         float64
	UpdateCount int
	Covariance  [][]float64
	LastUpdated time.Time
}

// CovarianceMatrix converts the stored slice representation back to a mat.Dense.
func (lp *LearnedParameters) CovarianceMatrix() *mat.Dense {
	n := len(lp.Covariance)
	if n == 0 {
		return nil
	}
	data := make([]float64, n*n)
	for i, row := range lp.Covariance {
		copy(data[i*n:], row)
	}
	return mat.NewDense(n, n, data)
}

// ParameterStore is a thread-safe in-memory store of LearnedParameters keyed by "modelName/accelerator".
type ParameterStore struct {
	mu     sync.RWMutex
	params map[string]*LearnedParameters
}

// NewParameterStore creates an empty ParameterStore.
func NewParameterStore() *ParameterStore {
	return &ParameterStore{params: make(map[string]*LearnedParameters)}
}

func makeKey(model, accelerator string) string {
	return fmt.Sprintf("%s/%s", model, accelerator)
}

// splitKey splits a "model/accelerator" key back into its components.
// If the model name itself contains slashes, only the last slash is used.
func splitKey(key string) (model, accelerator string) {
	idx := strings.LastIndex(key, "/")
	if idx < 0 {
		return key, ""
	}
	return key[:idx], key[idx+1:]
}

// Get returns the stored parameters for a model/accelerator pair, or nil if not found.
func (ps *ParameterStore) Get(model, accelerator string) *LearnedParameters {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	return ps.params[makeKey(model, accelerator)]
}

// Set stores parameters for a model/accelerator pair.
func (ps *ParameterStore) Set(model, accelerator string, params *LearnedParameters) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.params[makeKey(model, accelerator)] = params
}

// GetAll returns a snapshot of all stored parameters.
func (ps *ParameterStore) GetAll() map[string]*LearnedParameters {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	out := make(map[string]*LearnedParameters, len(ps.params))
	for k, v := range ps.params {
		out[k] = v
	}
	return out
}

func covToSlice(p *mat.Dense) [][]float64 {
	if p == nil {
		return nil
	}
	n, _ := p.Dims()
	out := make([][]float64, n)
	for i := range n {
		out[i] = make([]float64, n)
		for j := range n {
			out[i][j] = p.At(i, j)
		}
	}
	return out
}
```

---

## Task 7: Create pkg/service/utils.go

**Files:** Create `pkg/service/utils.go`

Move of the orchestration helpers from `tunerservice/utils.go`. Excludes `guessInitState` (now in `pkg/estimator`) and `validateKey` (stays in `tunerservice`). `splitKey` moved to `parameters.go` in Task 6.

- [ ] **Create pkg/service/utils.go**

```go
package service

import (
	optconfig "github.com/llm-inferno/optimizer-light/pkg/config"

	"github.com/llm-inferno/model-tuner/pkg/core"
)

// groupByModelAccelerator groups ServerSpecs by "model/accelerator" key.
// Only replicas with traffic (ArrivalRate > 0) are included.
func groupByModelAccelerator(replicas []optconfig.ServerSpec) map[string][]optconfig.ServerSpec {
	groups := make(map[string][]optconfig.ServerSpec)
	for _, r := range replicas {
		if r.CurrentAlloc.Load.ArrivalRate <= 0 {
			continue
		}
		key := makeKey(r.Model, r.CurrentAlloc.Accelerator)
		groups[key] = append(groups[key], r)
	}
	return groups
}

// buildEnvironments creates EnvironmentPrefillDecode instances from replica specs.
// Replicas with zero tokens or latency are skipped.
func buildEnvironments(replicas []optconfig.ServerSpec) []*core.EnvironmentPrefillDecode {
	var envs []*core.EnvironmentPrefillDecode
	for _, r := range replicas {
		a := r.CurrentAlloc
		if a.Load.AvgInTokens <= 0 || a.Load.AvgOutTokens <= 0 || a.TTFTAverage <= 0 || a.ITLAverage <= 0 {
			continue
		}
		maxBatch := r.MaxBatchSize
		if maxBatch <= 0 {
			maxBatch = a.MaxBatch
		}
		if maxBatch <= 0 {
			continue
		}
		env := core.NewEnvironmentPrefillDecode(
			a.Load.ArrivalRate,
			0,
			0,
			maxBatch,
			float32(a.Load.AvgInTokens),
			float32(a.Load.AvgOutTokens),
			a.TTFTAverage,
			a.ITLAverage,
		)
		env.MaxQueueSize = r.MaxQueueSize
		envs = append(envs, env)
	}
	return envs
}

// maxBatchFromReplicas returns the largest MaxBatchSize seen across replicas.
func maxBatchFromReplicas(replicas []optconfig.ServerSpec) int {
	result := 0
	for _, r := range replicas {
		b := r.MaxBatchSize
		if b <= 0 {
			b = r.CurrentAlloc.MaxBatch
		}
		if b > result {
			result = b
		}
	}
	return result
}
```

---

## Task 8: Create pkg/service/service.go

**Files:** Create `pkg/service/service.go`

Move of `tunerservice/service.go`. Key changes: import `pkg/estimator`, replace `*InitEstimator`/`*SlidingWindowEstimator` with `*estimator.InitEstimator`/`*estimator.SlidingWindowEstimator`, replace `guessInitState` with `estimator.GuessInitState`, replace `swe.Seed(ie.observations)` with `swe.SeedFromEstimator(ie)`. `setInitState` stays in this file.

- [ ] **Create pkg/service/service.go**

```go
package service

import (
	"fmt"
	"log/slog"
	"math"
	"time"

	optconfig "github.com/llm-inferno/optimizer-light/pkg/config"

	"github.com/llm-inferno/model-tuner/pkg/config"
	"github.com/llm-inferno/model-tuner/pkg/core"
	estimator "github.com/llm-inferno/model-tuner/pkg/estimator"
	"github.com/llm-inferno/model-tuner/pkg/utils"
)

// TunerService groups replica metrics by (model, accelerator), runs EKF tuning per group,
// maintains a ParameterStore for state continuity, and returns updated ModelData.
type TunerService struct {
	paramStore        *ParameterStore
	warmUpCycles      int
	estimators        map[string]*estimator.InitEstimator
	initObs           int
	holdBack          bool
	useSliding        bool
	windowSize        int
	residualThreshold float64
	slidingEstimators map[string]*estimator.SlidingWindowEstimator
	initFitThreshold  float64
	ekfFallbacks      map[string]bool
}

// NewTunerService creates a TunerService with an empty ParameterStore.
func NewTunerService(warmUpCycles, initObs int, holdBack bool, useSliding bool, windowSize int, residualThreshold, initFitThreshold float64) *TunerService {
	return &TunerService{
		paramStore:        NewParameterStore(),
		warmUpCycles:      warmUpCycles,
		estimators:        make(map[string]*estimator.InitEstimator),
		initObs:           initObs,
		holdBack:          holdBack,
		useSliding:        useSliding,
		windowSize:        windowSize,
		residualThreshold: residualThreshold,
		slidingEstimators: make(map[string]*estimator.SlidingWindowEstimator),
		initFitThreshold:  initFitThreshold,
		ekfFallbacks:      make(map[string]bool),
	}
}

func (ts *TunerService) estimatorFor(key string) *estimator.InitEstimator {
	if ie, ok := ts.estimators[key]; ok {
		return ie
	}
	ie := estimator.NewInitEstimator(ts.initObs, ts.holdBack)
	ts.estimators[key] = ie
	return ie
}

func (ts *TunerService) slidingEstimatorFor(key string, ie *estimator.InitEstimator) *estimator.SlidingWindowEstimator {
	if swe, ok := ts.slidingEstimators[key]; ok {
		return swe
	}
	swe := estimator.NewSlidingWindowEstimator(ts.windowSize, ts.initObs, ts.residualThreshold)
	swe.SeedFromEstimator(ie)
	if fitted, err := ie.Fit(); err == nil {
		fv := ie.LastFitFuncValue()
		if ts.initFitThreshold > 0 && fv > ts.initFitThreshold {
			slog.Warn("poor init fit: falling back to EKF for this pair",
				"key", key, "funcValue", fv, "threshold", ts.initFitThreshold)
			ts.ekfFallbacks[key] = true
			return swe
		}
		swe.SeedLastFit(fitted)
	} else if ts.initFitThreshold > 0 {
		slog.Warn("init fit error: falling back to EKF for this pair", "key", key, "err", err)
		ts.ekfFallbacks[key] = true
		return swe
	}
	ts.slidingEstimators[key] = swe
	return swe
}

func (ts *TunerService) tuneGroupSliding(model, accelerator, key string, ie *estimator.InitEstimator, env *core.EnvironmentPrefillDecode) error {
	_, alreadyExists := ts.slidingEstimators[key]
	swe := ts.slidingEstimatorFor(key, ie)

	if ts.ekfFallbacks[key] {
		return fmt.Errorf("EKF fallback active for %s/%s: poor init fit (funcValue > %.1f)",
			model, accelerator, ts.initFitThreshold)
	}

	if alreadyExists {
		swe.AddObservation(env)
	}

	if !swe.IsReady() {
		slog.Info("sliding window filling",
			"model", model, "accelerator", accelerator,
			"count", len(swe.WindowSnapshot()), "windowSize", ts.windowSize)
		return fmt.Errorf("sliding window filling for %s/%s (%d/%d)",
			model, accelerator, len(swe.WindowSnapshot()), ts.windowSize)
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

// Tune accepts per-replica ServerSpecs, runs EKF or SWNM tuning for each
// (model, accelerator) group, and returns updated ModelData with tuned alpha/beta/gamma.
func (ts *TunerService) Tune(replicaSpecs []optconfig.ServerSpec) (*optconfig.ModelData, error) {
	groups := groupByModelAccelerator(replicaSpecs)
	if len(groups) == 0 {
		return nil, fmt.Errorf("no replicas with active traffic in request")
	}

	for key, replicas := range groups {
		model, accelerator := splitKey(key)
		if err := ts.tuneGroup(model, accelerator, replicas); err != nil {
			slog.Warn("tuning failed for group", "key", key, "err", err)
		}
	}

	modelData := ts.buildModelData(groups)
	if len(modelData.PerfData) == 0 {
		return nil, fmt.Errorf("tuning produced no results for any model/accelerator group")
	}
	return modelData, nil
}

func (ts *TunerService) tuneGroup(model, accelerator string, replicas []optconfig.ServerSpec) error {
	envs := buildEnvironments(replicas)
	if len(envs) == 0 {
		return fmt.Errorf("no valid environments for %s/%s", model, accelerator)
	}

	for i, env := range envs {
		slog.Info("replica environment",
			"model", model,
			"accelerator", accelerator,
			"replica", i,
			"arrivalRateRPM", env.Lambda,
			"maxBatch", env.MaxBatchSize,
			"avgInTokens", env.AvgInputTokens,
			"avgOutTokens", env.AvgOutputTokens,
			"avgTTFT", env.AvgTTFT,
			"avgITL", env.AvgITL,
		)
	}

	key := makeKey(model, accelerator)
	ie := ts.estimatorFor(key)
	ie.AddObservation(envs[0])

	if !ie.IsReady() {
		slog.Info("collecting initial observations",
			"model", model, "accelerator", accelerator,
			"count", ie.ObsCount(), "minObs", ie.MinObs())
		return fmt.Errorf("collecting initial observations for %s/%s (%d/%d)",
			model, accelerator, ie.ObsCount(), ie.MinObs())
	}

	if ts.useSliding && !ts.ekfFallbacks[key] {
		return ts.tuneGroupSliding(model, accelerator, key, ie, envs[0])
	}

	var fitInitState []float64
	if ts.paramStore.Get(model, accelerator) == nil && !ie.FitDone() {
		var fitErr error
		fitInitState, fitErr = ie.Fit()
		if fitErr != nil {
			slog.Warn("InitEstimator Fit failed, EKF will use guessInitState", "err", fitErr)
		}
	}

	tuner, err := ts.createTuner(model, accelerator, envs[0], fitInitState)
	if err != nil {
		return fmt.Errorf("create tuner for %s/%s: %w", model, accelerator, err)
	}

	updateCount := 0
	if existing := ts.paramStore.Get(model, accelerator); existing != nil {
		updateCount = existing.UpdateCount
	}
	skipNIS := updateCount < ts.warmUpCycles

	var accepted *core.TunedResults
	for _, env := range envs {
		results, runErr := tuner.RunWithValidation(env, skipNIS)
		if runErr != nil {
			slog.Warn("EKF run error", "model", model, "accelerator", accelerator, "err", runErr)
			continue
		}
		if results.ValidationFailed {
			if results.NIS > 0 {
				slog.Info("EKF update rejected: NIS gate", "model", model, "accelerator", accelerator, "NIS", results.NIS)
			} else {
				slog.Info("EKF update rejected: state validation", "model", model, "accelerator", accelerator)
			}
			continue
		}
		accepted = results
	}

	if accepted == nil {
		return fmt.Errorf("no accepted results for %s/%s", model, accelerator)
	}

	ts.paramStore.Set(model, accelerator, &LearnedParameters{
		Alpha:       accepted.ServiceParms.Alpha,
		Beta:        accepted.ServiceParms.Beta,
		Gamma:       accepted.ServiceParms.Gamma,
		NIS:         accepted.NIS,
		UpdateCount: updateCount + 1,
		Covariance:  covToSlice(accepted.Covariance),
		LastUpdated: time.Now(),
	})
	slog.Info("tuned parameters",
		"model", model,
		"accelerator", accelerator,
		"alpha", accepted.ServiceParms.Alpha,
		"beta", accepted.ServiceParms.Beta,
		"gamma", accepted.ServiceParms.Gamma,
		"NIS", accepted.NIS,
		"updateCount", updateCount+1,
		"warmUp", skipNIS,
	)
	return nil
}

func (ts *TunerService) createTuner(model, accelerator string, firstEnv *core.EnvironmentPrefillDecode, fitInitState []float64) (*core.Tuner, error) {
	existing := ts.paramStore.Get(model, accelerator)

	configData, err := utils.LoadConfigForServer(config.DefaultConfigType)
	if err != nil {
		return nil, fmt.Errorf("load config for %s: %w", model, err)
	}

	if existing != nil {
		setInitState(&configData.ModelData, []float64{
			float64(existing.Alpha),
			float64(existing.Beta),
			float64(existing.Gamma),
		})
		if cov := existing.CovarianceMatrix(); cov != nil {
			tuner, err := core.NewTunerWithCovariance(configData, firstEnv, cov)
			if err != nil {
				return nil, err
			}
			if err := tuner.SetObservationFunc(core.NewQueueModelSystemFuncCreatorPrefillDecode(tuner)); err != nil {
				return nil, err
			}
			return tuner, nil
		}
	} else {
		if fitInitState != nil {
			setInitState(&configData.ModelData, fitInitState)
		} else if initState := estimator.GuessInitState(firstEnv); initState != nil {
			setInitState(&configData.ModelData, initState)
		}
	}

	tuner, err := core.NewTuner(configData, firstEnv)
	if err != nil {
		return nil, err
	}
	if err := tuner.SetObservationFunc(core.NewQueueModelSystemFuncCreatorPrefillDecode(tuner)); err != nil {
		return nil, err
	}
	return tuner, nil
}

func setInitState(md *config.ModelData, initState []float64) {
	md.InitState = initState
	md.MinState = make([]float64, len(initState))
	md.MaxState = make([]float64, len(initState))
	for i, v := range initState {
		md.MinState[i] = math.Max(v/config.DefaultInitStateFactor, config.DefaultInitStateMinEpsilon)
		md.MaxState[i] = v * config.DefaultInitStateFactor
	}
}

func (ts *TunerService) buildModelData(groups map[string][]optconfig.ServerSpec) *optconfig.ModelData {
	var entries []optconfig.ModelAcceleratorPerfData
	for key, replicas := range groups {
		model, accelerator := splitKey(key)
		params := ts.paramStore.Get(model, accelerator)
		if params == nil {
			continue
		}
		maxBatch := maxBatchFromReplicas(replicas)
		entries = append(entries, optconfig.ModelAcceleratorPerfData{
			Name:         model,
			Acc:          accelerator,
			MaxBatchSize: maxBatch,
			PerfParms: optconfig.PerfParms{
				Alpha: params.Alpha,
				Beta:  params.Beta,
				Gamma: params.Gamma,
			},
		})
	}
	return &optconfig.ModelData{PerfData: entries}
}

// GetParams returns the most recently tuned parameters for a model/accelerator pair,
// or nil if no tuning has been performed for that pair yet.
func (ts *TunerService) GetParams(model, accelerator string) *LearnedParameters {
	return ts.paramStore.Get(model, accelerator)
}

// IsWarmingUp returns true if any known pair has not yet completed its init or warm-up phase.
func (ts *TunerService) IsWarmingUp() bool {
	for _, ie := range ts.estimators {
		if !ie.IsReady() && ie.HoldBack() {
			return true
		}
	}
	if ts.useSliding {
		for key, ie := range ts.estimators {
			if !ie.IsReady() {
				continue
			}
			if ts.ekfFallbacks[key] {
				continue
			}
			swe, ok := ts.slidingEstimators[key]
			if !ok || !swe.IsReady() {
				return true
			}
		}
		if ts.warmUpCycles > 0 {
			for _, params := range ts.paramStore.GetAll() {
				if params.UpdateCount < ts.warmUpCycles {
					return true
				}
			}
		}
		return false
	}
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

// Merge accepts the Controller's current ModelData and returns it with PerfParms overlaid
// from the ParameterStore for any matching (name, accelerator) pairs.
func (ts *TunerService) Merge(modelData *optconfig.ModelData) *optconfig.ModelData {
	if modelData == nil {
		modelData = &optconfig.ModelData{}
	}

	allParams := ts.paramStore.GetAll()
	matched := make(map[string]bool, len(allParams))

	result := make([]optconfig.ModelAcceleratorPerfData, len(modelData.PerfData))
	for i, entry := range modelData.PerfData {
		result[i] = entry
		key := makeKey(entry.Name, entry.Acc)
		if params, ok := allParams[key]; ok {
			result[i].PerfParms = optconfig.PerfParms{
				Alpha: params.Alpha,
				Beta:  params.Beta,
				Gamma: params.Gamma,
			}
			if matched[key] {
				slog.Warn("duplicate model/accelerator key in input ModelData", "key", key)
			} else {
				matched[key] = true
			}
		}
	}

	for key, params := range allParams {
		if matched[key] {
			continue
		}
		model, acc := splitKey(key)
		result = append(result, optconfig.ModelAcceleratorPerfData{
			Name:         model,
			Acc:          acc,
			AccCount:     DefaultAccCount,
			MaxBatchSize: DefaultMaxBatchSize,
			AtTokens:     DefaultAtTokens,
			PerfParms: optconfig.PerfParms{
				Alpha: params.Alpha,
				Beta:  params.Beta,
				Gamma: params.Gamma,
			},
		})
	}

	return &optconfig.ModelData{PerfData: result}
}
```

**Note:** `tuneGroupSliding` uses `swe.WindowSnapshot()` to get the window length for the log message. Add this method to `SlidingWindowEstimator` in `pkg/estimator/sliding_window_estimator.go`:

- [ ] **Add WindowSnapshot to pkg/estimator/sliding_window_estimator.go**

Append to `pkg/estimator/sliding_window_estimator.go`:

```go
// WindowSnapshot returns a copy of the current observation window (for logging/diagnostics).
func (swe *SlidingWindowEstimator) WindowSnapshot() []struct{} {
	out := make([]struct{}, len(swe.window))
	return out
}
```

Wait — actually the original code uses `len(swe.window)` directly from within the same package. Since `service.go` is now in a *different* package, it can't access `swe.window`. The log line in `tuneGroupSliding` is:

```go
slog.Info("sliding window filling", "count", len(swe.window), "windowSize", swe.windowSize)
```

The cleanest fix: add a `Len() int` method to `SlidingWindowEstimator` that returns `len(swe.window)`.

- [ ] **Replace WindowSnapshot with Len() in pkg/estimator/sliding_window_estimator.go**

Append to `pkg/estimator/sliding_window_estimator.go` (replace the WindowSnapshot stub above):

```go
// Len returns the number of observations currently in the window.
func (swe *SlidingWindowEstimator) Len() int {
	return len(swe.window)
}
```

Then in `pkg/service/service.go`, update the log line in `tuneGroupSliding`:

```go
slog.Info("sliding window filling",
    "model", model, "accelerator", accelerator,
    "count", swe.Len(), "windowSize", ts.windowSize)
return fmt.Errorf("sliding window filling for %s/%s (%d/%d)",
    model, accelerator, swe.Len(), ts.windowSize)
```

- [ ] **Update tuneGroupSliding in pkg/service/service.go to use swe.Len()**

In `pkg/service/service.go`, replace the two lines that used `len(swe.window)`:

```go
	if !swe.IsReady() {
		slog.Info("sliding window filling",
			"model", model, "accelerator", accelerator,
			"count", swe.Len(), "windowSize", ts.windowSize)
		return fmt.Errorf("sliding window filling for %s/%s (%d/%d)",
			model, accelerator, swe.Len(), ts.windowSize)
	}
```

(Remove the `swe.WindowSnapshot()` stub; this is the correct version.)

- [ ] **Verify pkg/service compiles**

```bash
go build ./pkg/service/...
```

Expected: compiles cleanly

---

## Task 9: Create pkg/service Test Files

**Files:** Create `pkg/service/service_test.go`, `pkg/service/service_sliding_test.go`, `pkg/service/utils_test.go`

**Key differences from originals:**
- `service_sliding_test.go`: `fitObservation` direct-field access replaced with `AddObservation` calls; `swe.Seed(ie.observations)` replaced with `swe.SeedFromEstimator(ie)`; `swe.windowSize = 0` hack replaced with a SWE that has `minObs=2` so it's not ready after one `AddObservation`; `makeTestEnv` helper added locally; `NewInitEstimator`/`NewSlidingWindowEstimator` qualified with `estimator.`
- `service_test.go`: new file for the two warm-up tests that lived in `tunerservice/init_estimator_test.go`

- [ ] **Create pkg/service/service_test.go**

```go
package service

import (
	"testing"

	estimator "github.com/llm-inferno/model-tuner/pkg/estimator"
)

func TestTunerService_IsWarmingUp_DuringCollection(t *testing.T) {
	ts := NewTunerService(3, 3, true, false, DefaultWindowSize, DefaultResidualThreshold, 0)
	key := makeKey("mymodel", "myacc")
	ts.estimators[key] = estimator.NewInitEstimator(3, true)
	if !ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=true when estimator not ready and holdBack=true")
	}
}

func TestTunerService_IsWarmingUp_HoldBackFalse(t *testing.T) {
	ts := NewTunerService(3, 3, false, false, DefaultWindowSize, DefaultResidualThreshold, 0)
	key := makeKey("mymodel", "myacc")
	ts.estimators[key] = estimator.NewInitEstimator(3, false)
	if ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=false when holdBack=false")
	}
}
```

- [ ] **Create pkg/service/service_sliding_test.go**

```go
package service

import (
	"testing"

	optconfig "github.com/llm-inferno/optimizer-light/pkg/config"

	estimator "github.com/llm-inferno/model-tuner/pkg/estimator"
	"github.com/llm-inferno/model-tuner/pkg/core"
)

func makeTestEnv(lambda, ttft, itl float32, inTok, outTok float32, maxBatch int) *core.EnvironmentPrefillDecode {
	return core.NewEnvironmentPrefillDecode(
		lambda, 0, 0, maxBatch, inTok, outTok, ttft, itl,
	)
}

func makeTestSpec(model, acc string, lambda, ttft, itl float32, inTok, outTok, maxBatch int) optconfig.ServerSpec {
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
				AvgInTokens:  inTok,
				AvgOutTokens: outTok,
			},
		},
	}
}

func TestTunerService_SWNM_ReturnsParamsAfterInitPhase(t *testing.T) {
	initObs := 3
	windowSize := 5
	ts := NewTunerService(0, initObs, false, true, windowSize, DefaultResidualThreshold, 0)

	spec := makeTestSpec("llama", "H100", 15, 55, 6, 120, 700, 64)

	for i := range initObs - 1 {
		_, err := ts.Tune([]optconfig.ServerSpec{spec})
		if err == nil {
			t.Fatalf("cycle %d: expected error during init collection, got nil", i+1)
		}
	}

	result, err := ts.Tune([]optconfig.ServerSpec{spec})
	if err != nil {
		t.Fatalf("cycle 3: expected parameters after init phase, got error: %v", err)
	}
	if len(result.PerfData) == 0 {
		t.Fatal("cycle 3: expected non-empty PerfData")
	}
	p := result.PerfData[0]
	if p.PerfParms.Alpha <= 0 || p.PerfParms.Beta <= 0 || p.PerfParms.Gamma <= 0 {
		t.Errorf("expected positive params, got alpha=%.4f beta=%.4f gamma=%.6f",
			p.PerfParms.Alpha, p.PerfParms.Beta, p.PerfParms.Gamma)
	}
}

// TestTunerService_SWNM_SWENotReady_RetainsPreviousParams verifies that when
// tuneGroupSliding returns an error (SWE not yet ready), the paramStore entry is unchanged.
func TestTunerService_SWNM_SWENotReady_RetainsPreviousParams(t *testing.T) {
	ts := NewTunerService(0, 1, false, true, 5, DefaultResidualThreshold, 0)
	model, acc := "llama", "H100"
	key := makeKey(model, acc)

	ts.paramStore.Set(model, acc, &LearnedParameters{Alpha: 10.0, Beta: 0.05, Gamma: 0.001, UpdateCount: 1})

	ie := estimator.NewInitEstimator(1, false)
	ie.AddObservation(makeTestEnv(15, 55, 6, 120, 700, 64))
	ts.estimators[key] = ie

	// SWE with minObs=2: after one AddObservation it will have 1 entry < minObs → not ready.
	swe := estimator.NewSlidingWindowEstimator(5, 2, DefaultResidualThreshold)
	ts.slidingEstimators[key] = swe

	env := makeTestEnv(15, 55, 6, 120, 700, 64)
	err := ts.tuneGroupSliding(model, acc, key, ie, env)
	if err == nil {
		t.Fatal("expected error when SWE not ready, got nil")
	}

	params := ts.paramStore.Get(model, acc)
	if params == nil {
		t.Fatal("paramStore entry missing after SWE-not-ready error")
	}
	if params.Alpha != 10.0 || params.Beta != 0.05 || params.Gamma != 0.001 {
		t.Errorf("paramStore was modified on error: alpha=%.4f beta=%.4f gamma=%.6f",
			params.Alpha, params.Beta, params.Gamma)
	}
}

func TestTunerService_IsWarmingUp_SWNM_WindowNotFull(t *testing.T) {
	ts := NewTunerService(3, 3, true, true, 5, DefaultResidualThreshold, 0)
	key := makeKey("mymodel", "myacc")
	ts.estimators[key] = estimator.NewInitEstimator(3, true)

	if !ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=true when estimator not ready and holdBack=true")
	}

	ie := estimator.NewInitEstimator(3, true)
	env := makeTestEnv(10, 50, 5, 100, 500, 64)
	ie.AddObservation(env)
	ie.AddObservation(env)
	ie.AddObservation(env)
	ts.estimators[key] = ie
	if !ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=true when InitEstimator ready but SWE not yet created")
	}

	// SWE with minObs=4 seeded from ie (3 obs) → not ready
	swe := estimator.NewSlidingWindowEstimator(5, 4, DefaultResidualThreshold)
	swe.SeedFromEstimator(ie)
	ts.slidingEstimators[key] = swe
	if !ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=true when SWE has fewer than minObs observations")
	}
}

func TestTunerService_IsWarmingUp_SWNM_WindowFull(t *testing.T) {
	ts := NewTunerService(3, 3, true, true, 3, DefaultResidualThreshold, 0)
	key := makeKey("mymodel", "myacc")

	ie := estimator.NewInitEstimator(3, true)
	env := makeTestEnv(10, 50, 5, 100, 500, 64)
	ie.AddObservation(env)
	ie.AddObservation(env)
	ie.AddObservation(env)
	ts.estimators[key] = ie

	swe := estimator.NewSlidingWindowEstimator(3, 1, DefaultResidualThreshold)
	swe.SeedFromEstimator(ie)
	ts.slidingEstimators[key] = swe

	if ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=false when SWE window is full")
	}
}

func TestTunerService_SWNM_HighFuncValue_FallsBackToEKF(t *testing.T) {
	ts := NewTunerService(0, 2, false, true, DefaultWindowSize, DefaultResidualThreshold, 0.0001)
	spec1 := makeTestSpec("llama", "H100", 15, 55, 6, 120, 700, 64)
	spec2 := makeTestSpec("llama", "H100", 30, 120, 12, 200, 1500, 64)

	ts.Tune([]optconfig.ServerSpec{spec1})
	ts.Tune([]optconfig.ServerSpec{spec2})

	key := makeKey("llama", "H100")
	if !ts.ekfFallbacks[key] {
		t.Fatal("expected ekfFallbacks[key]=true after high funcValue init fit")
	}

	if _, hasSWE := ts.slidingEstimators[key]; hasSWE {
		t.Error("expected no SWE stored after EKF fallback")
	}

	ts.Tune([]optconfig.ServerSpec{spec1})
	if _, hasSWE := ts.slidingEstimators[key]; hasSWE {
		t.Error("SWE should still not be stored on subsequent cycles after EKF fallback")
	}
}

func TestTunerService_IsWarmingUp_SWNM_EKFFallbackPair(t *testing.T) {
	ts := NewTunerService(0, 1, false, true, DefaultWindowSize, DefaultResidualThreshold, 0)
	key := makeKey("llama", "H100")

	ie := estimator.NewInitEstimator(1, false)
	ie.AddObservation(makeTestEnv(15, 55, 6, 120, 700, 64))
	ts.estimators[key] = ie
	ts.ekfFallbacks[key] = true

	if ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=false for EKF-fallback pair with warmUpCycles=0")
	}
}

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

- [ ] **Create pkg/service/utils_test.go**

```go
package service

import (
	"testing"

	optconfig "github.com/llm-inferno/optimizer-light/pkg/config"
)

func TestBuildEnvironments_MaxQueueSizeFromSpec(t *testing.T) {
	specs := []optconfig.ServerSpec{
		{
			Model:        "granite_8b",
			MaxQueueSize: 128,
			CurrentAlloc: optconfig.AllocationData{
				Accelerator: "H100",
				MaxBatch:    64,
				ITLAverage:  8.5,
				TTFTAverage: 45.0,
				Load: optconfig.ServerLoadSpec{
					ArrivalRate:  60,
					AvgInTokens:  2048,
					AvgOutTokens: 1024,
				},
			},
		},
	}

	envs := buildEnvironments(specs)
	if len(envs) != 1 {
		t.Fatalf("expected 1 environment, got %d", len(envs))
	}
	if envs[0].MaxQueueSize != 128 {
		t.Errorf("MaxQueueSize = %d, want 128", envs[0].MaxQueueSize)
	}
}

func TestBuildEnvironments_ZeroMaxQueueSizeWhenUnset(t *testing.T) {
	specs := []optconfig.ServerSpec{
		{
			Model:        "llama_13b",
			MaxQueueSize: 0,
			CurrentAlloc: optconfig.AllocationData{
				Accelerator: "H100",
				MaxBatch:    64,
				ITLAverage:  12.0,
				TTFTAverage: 60.0,
				Load: optconfig.ServerLoadSpec{
					ArrivalRate:  30,
					AvgInTokens:  768,
					AvgOutTokens: 768,
				},
			},
		},
	}

	envs := buildEnvironments(specs)
	if len(envs) != 1 {
		t.Fatalf("expected 1 environment, got %d", len(envs))
	}
	if envs[0].MaxQueueSize != 0 {
		t.Errorf("MaxQueueSize = %d, want 0 (no external queue)", envs[0].MaxQueueSize)
	}
}
```

- [ ] **Verify pkg/service tests pass**

```bash
go test ./pkg/service/...
```

Expected: all tests pass

- [ ] **Commit**

```bash
git add pkg/service/
git commit -m "feat(pkg/service): add orchestration layer extracted from tunerservice"
```

---

## Task 10: Gut tunerservice/

Replace all non-HTTP files; keep `server.go`, `handlers.go` (no change needed), and a slim `utils.go`.

- [ ] **Overwrite tunerservice/defaults.go (HTTP constants only)**

```go
package tunerservice

// Environment variable names and defaults for the tuner REST server.
const (
	TunerHostEnvName = "TUNER_HOST"
	TunerPortEnvName = "TUNER_PORT"

	DefaultTunerHost = "localhost"
	DefaultTunerPort = "8081"
)
```

- [ ] **Update tunerservice/server.go to import pkg/service**

```go
package tunerservice

import (
	"fmt"
	"log/slog"

	"github.com/gin-gonic/gin"
	pkgsvc "github.com/llm-inferno/model-tuner/pkg/service"
)

// TunerServer is the HTTP layer that wraps TunerService and exposes its functionality
// over a Gin REST API.
type TunerServer struct {
	service *pkgsvc.TunerService
	router  *gin.Engine
}

// NewTunerServer creates a TunerServer with the given service and registers all routes.
func NewTunerServer(service *pkgsvc.TunerService) *TunerServer {
	router := gin.Default()
	ts := &TunerServer{service: service, router: router}
	router.POST("/tune", ts.handleTune)
	router.GET("/getparams", ts.handleGetParams)
	router.GET("/warmup", ts.handleWarmUp)
	router.POST("/merge", ts.handleMerge)
	return ts
}

// Run starts the HTTP server on host:port (blocks until the server stops).
func (ts *TunerServer) Run(host, port string) error {
	addr := fmt.Sprintf("%s:%s", host, port)
	slog.Info("starting TunerServer", "addr", addr)
	return ts.router.Run(addr)
}
```

- [ ] **Overwrite tunerservice/utils.go (validateKey only)**

```go
package tunerservice

import "fmt"

// validateKey is used in handler input validation.
func validateKey(model, accelerator string) error {
	if model == "" {
		return fmt.Errorf("model is required")
	}
	if accelerator == "" {
		return fmt.Errorf("accelerator is required")
	}
	return nil
}
```

- [ ] **Overwrite tunerservice/doc.go**

```go
// Package tunerservice is a thin HTTP adapter over [pkg/service.TunerService].
//
// It exposes four endpoints:
//
//	POST /tune
//	  Body:     []config.ServerSpec   (ReplicaSpecs from the Collector)
//	  Response: config.ModelData      (updated alpha/beta/gamma per model/accelerator)
//
//	GET /getparams?model=<name>&accelerator=<acc>
//	  Response: JSON with alpha, beta, gamma, NIS, updateCount, lastUpdated
//
//	GET /warmup
//	  Response: {"warmingUp": bool}
//
//	POST /merge
//	  Body:     config.ModelData
//	  Response: config.ModelData with PerfParms overlaid from the parameter store
//
// All estimation logic lives in pkg/estimator and pkg/service.
package tunerservice
```

- [ ] **Delete moved source files from tunerservice/**

```bash
rm tunerservice/init_estimator.go \
   tunerservice/sliding_window_estimator.go \
   tunerservice/service.go \
   tunerservice/parameters.go
```

- [ ] **Delete moved test files from tunerservice/**

```bash
rm tunerservice/init_estimator_test.go \
   tunerservice/sliding_window_estimator_test.go \
   tunerservice/service_sliding_test.go \
   tunerservice/utils_test.go
```

- [ ] **Verify tunerservice still builds**

```bash
go build ./tunerservice/...
```

Expected: compiles cleanly

---

## Task 11: Update cmd/tuner/main.go

- [ ] **Overwrite cmd/tuner/main.go**

```go
package main

import (
	"log"
	"log/slog"
	"os"
	"strconv"

	pkgconfig "github.com/llm-inferno/model-tuner/pkg/config"
	pkgsvc "github.com/llm-inferno/model-tuner/pkg/service"
	"github.com/llm-inferno/model-tuner/tunerservice"
)

func main() {
	host := os.Getenv(tunerservice.TunerHostEnvName)
	if host == "" {
		host = tunerservice.DefaultTunerHost
	}
	port := os.Getenv(tunerservice.TunerPortEnvName)
	if port == "" {
		port = tunerservice.DefaultTunerPort
	}

	warmUpCycles := pkgconfig.DefaultWarmUpCycles
	if v := os.Getenv(pkgsvc.WarmUpCyclesEnvName); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			warmUpCycles = n
		}
	}

	initObs := pkgsvc.DefaultInitObs
	if v := os.Getenv(pkgsvc.InitObsEnvName); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			initObs = n
		}
	}

	holdBack := pkgsvc.DefaultInitHoldBack
	if v := os.Getenv(pkgsvc.InitHoldBackEnvName); v != "" {
		holdBack = v == "true" || v == "1"
	}

	useSliding := os.Getenv(pkgsvc.EstimatorModeEnvName) == "sliding-window"

	windowSize := pkgsvc.DefaultWindowSize
	if v := os.Getenv(pkgsvc.WindowSizeEnvName); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			windowSize = n
		}
	}

	residualThreshold := pkgsvc.DefaultResidualThreshold
	if v := os.Getenv(pkgsvc.ResidualThresholdEnvName); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
			residualThreshold = f
		}
	}

	initFitThreshold := pkgsvc.DefaultInitFitThreshold
	if v := os.Getenv(pkgsvc.InitFitThresholdEnvName); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
			initFitThreshold = f
		}
	}

	service := pkgsvc.NewTunerService(warmUpCycles, initObs, holdBack, useSliding, windowSize, residualThreshold, initFitThreshold)
	server := tunerservice.NewTunerServer(service)

	estimatorMode := pkgsvc.DefaultEstimatorMode
	if useSliding {
		estimatorMode = "sliding-window"
	}
	slog.Info("Starting TunerService",
		"host", host, "port", port,
		"warmUpCycles", warmUpCycles,
		"initObs", initObs,
		"holdBack", holdBack,
		"estimatorMode", estimatorMode,
		"windowSize", windowSize,
		"residualThreshold", residualThreshold,
		"initFitThreshold", initFitThreshold)
	if err := server.Run(host, port); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
```

---

## Task 12: Final Verification

- [ ] **Full build**

```bash
go build ./...
```

Expected: no errors

- [ ] **Format check**

```bash
go fmt ./...
```

Expected: no output (already formatted)

- [ ] **All tests pass**

```bash
go test ./...
```

Expected: all tests pass, zero failures

---

## Task 13: Commit and Create PR

- [ ] **Stage all remaining changes**

```bash
git add tunerservice/defaults.go tunerservice/server.go tunerservice/utils.go tunerservice/doc.go \
        cmd/tuner/main.go
```

- [ ] **Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor: extract estimators and TunerService into pkg/estimator and pkg/service (#14)

- pkg/estimator: InitEstimator, SlidingWindowEstimator, GuessInitState — pure
  estimation library with no HTTP dependency
- pkg/service: TunerService, ParameterStore, LearnedParameters — orchestration
  layer; importable without gin
- tunerservice: trimmed to TunerServer + handlers + HTTP constants only
- cmd/tuner: updated to import pkg/service for constants and NewTunerService

No behavior changes. HTTP API surface and all env vars are identical.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Push branch**

```bash
git push -u origin issue-14-pkg-restructure
```

- [ ] **Create PR**

```bash
gh pr create \
  --title "refactor: extract estimators and TunerService into pkg/estimator and pkg/service (#14)" \
  --body "$(cat <<'EOF'
## Summary

- Adds `pkg/estimator`: pure estimation library (`InitEstimator`, `SlidingWindowEstimator`, `GuessInitState`) — no HTTP or optimizer-light dependency
- Adds `pkg/service`: orchestration layer (`TunerService`, `ParameterStore`, `LearnedParameters`) — importable without gin
- Trims `tunerservice` to a thin HTTP adapter (`TunerServer` + handlers + HTTP constants)
- Updates `cmd/tuner/main.go` to import `pkg/service` for constants and `NewTunerService`

Closes #14.

## No behavior changes
HTTP API surface (`POST /tune`, `GET /getparams`, `GET /warmup`, `POST /merge`) and all environment variables are identical. All existing tests pass in their new packages.

## Test plan
- [ ] `go build ./...` — clean build
- [ ] `go test ./pkg/estimator/...` — all estimator unit tests pass
- [ ] `go test ./pkg/service/...` — all service/SWNM/utils tests pass
- [ ] `go test ./...` — full suite green

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- ✅ `pkg/estimator`: InitEstimator, SlidingWindowEstimator, fitObservation, GuessInitState
- ✅ `pkg/service`: TunerService, ParameterStore, LearnedParameters, utils
- ✅ `tunerservice`: trimmed to TunerServer, handlers, HTTP defaults, validateKey
- ✅ `cmd/tuner/main.go` import updates
- ✅ All test files moved with their code
- ✅ `defaults.go` split: HTTP constants in tunerservice, service constants in pkg/service, baseFactor in pkg/estimator

**Placeholder scan:** None found — all steps have complete code.

**Type consistency:**
- `*estimator.InitEstimator` used consistently in `pkg/service/service.go` and `pkg/service/service_sliding_test.go`
- `*estimator.SlidingWindowEstimator` used consistently
- `swe.SeedFromEstimator(ie)` defined in Task 4 and called in Tasks 8 and 9
- `swe.Len()` defined in Task 8 and called in Task 8
- `estimator.GuessInitState` defined in Task 2 and called in Tasks 3, 4, 8
