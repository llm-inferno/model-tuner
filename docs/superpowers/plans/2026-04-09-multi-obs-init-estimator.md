# Multi-Observation Parameter Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-observation `guessInitState` with a Nelder-Mead fit over K accumulated observations to resolve EKF parameter non-identifiability.

**Architecture:** A new `InitEstimator` per (model, accelerator) key accumulates `fitObservation` structs across `Tune` calls. Once K observations are collected, `Fit()` minimises a relative-squared-error objective over the full queueing model (queue-analysis `Analyze()`) using Nelder-Mead. `TunerService` holds a map of estimators; `tuneGroup` returns early during collection if `holdBack=true`, then seeds `createTuner` from `Fit()` once ready.

**Tech Stack:** Go 1.25, `gonum.org/v1/gonum/optimize` (NelderMead already available via go.mod), `github.com/llm-inferno/queue-analysis/pkg/analyzer` (existing dep).

---

### Task 1: Add new constants to defaults.go

**Files:**
- Modify: `tunerservice/defaults.go`

- [ ] **Step 1: Add InitEstimator constants**

Append to `tunerservice/defaults.go`:

```go
// Environment variable names and defaults for the InitEstimator.
const (
	InitObsEnvName      = "TUNER_INIT_OBS"
	InitHoldBackEnvName = "TUNER_INIT_HOLD_BACK"

	DefaultInitObs      = 5
	DefaultInitHoldBack = true
)
```

- [ ] **Step 2: Build to verify no syntax errors**

Run: `go build ./...` from `model-tuner/`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add tunerservice/defaults.go
git commit -m "feat: add InitEstimator env var names and defaults"
```

---

### Task 2: Create InitEstimator struct and observation methods

**Files:**
- Create: `tunerservice/init_estimator.go`

- [ ] **Step 1: Create the file with structs and basic methods**

Create `tunerservice/init_estimator.go`:

```go
package tunerservice

import (
	"github.com/llm-inferno/model-tuner/pkg/core"
)

// fitObservation holds a single operating-point snapshot for use in the Fit objective.
type fitObservation struct {
	Lambda          float64 // arrival rate, req/min
	MaxBatch        int
	AvgInputTokens  float32
	AvgOutputTokens float32
	AvgTTFT         float64 // ms
	AvgITL          float64 // ms
}

// toEnv converts a fitObservation to an EnvironmentPrefillDecode for use with guessInitState.
func (fo *fitObservation) toEnv() *core.EnvironmentPrefillDecode {
	return core.NewEnvironmentPrefillDecode(
		float32(fo.Lambda),
		0, // BatchSize not used
		0, // AvgQueueTime not available
		fo.MaxBatch,
		fo.AvgInputTokens,
		fo.AvgOutputTokens,
		float32(fo.AvgTTFT),
		float32(fo.AvgITL),
	)
}

// InitEstimator accumulates observations before the EKF starts and fits initial parameters.
type InitEstimator struct {
	observations []fitObservation
	minObs       int
	holdBack     bool
}

// NewInitEstimator creates an InitEstimator with the given minimum observation count and hold-back flag.
func NewInitEstimator(minObs int, holdBack bool) *InitEstimator {
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
```

- [ ] **Step 2: Build**

Run: `go build ./...` from `model-tuner/`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add tunerservice/init_estimator.go
git commit -m "feat: add InitEstimator struct and observation accumulation"
```

---

### Task 3: Tests for InitEstimator struct and observation methods

**Files:**
- Create: `tunerservice/init_estimator_test.go`

- [ ] **Step 1: Write the failing tests**

Create `tunerservice/init_estimator_test.go`:

```go
package tunerservice

import (
	"testing"

	"github.com/llm-inferno/model-tuner/pkg/core"
)

func makeTestEnv(lambda, ttft, itl float32, inTok, outTok float32, maxBatch int) *core.EnvironmentPrefillDecode {
	return core.NewEnvironmentPrefillDecode(
		lambda, 0, 0, maxBatch, inTok, outTok, ttft, itl,
	)
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
	// invalid env: zero TTFT
	invalid := core.NewEnvironmentPrefillDecode(10, 0, 0, 64, 100, 500, 0, 5)
	ie.AddObservation(invalid)
	if len(ie.observations) != 0 {
		t.Fatalf("expected 0 observations stored, got %d", len(ie.observations))
	}
}

func TestInitEstimator_AddObservation_StoresFields(t *testing.T) {
	ie := NewInitEstimator(1, false)
	env := makeTestEnv(30, 45.5, 6.2, 120, 700, 64)
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
	if obs.AvgTTFT != float64(env.AvgTTFT) {
		t.Errorf("AvgTTFT mismatch: got %v want %v", obs.AvgTTFT, env.AvgTTFT)
	}
}
```

- [ ] **Step 2: Run tests (should pass — no Fit yet, just struct tests)**

Run: `go test ./tunerservice/ -run TestInitEstimator -v`
Expected: all 4 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tunerservice/init_estimator_test.go
git commit -m "test: add InitEstimator struct and accumulation tests"
```

---

### Task 4: Add Fit() and objective() to init_estimator.go

**Files:**
- Modify: `tunerservice/init_estimator.go`

- [ ] **Step 1: Add imports and Fit/objective**

Rewrite the import block and add the two functions at the bottom of `tunerservice/init_estimator.go`:

Replace:
```go
import (
	"github.com/llm-inferno/model-tuner/pkg/core"
)
```

With:
```go
import (
	"fmt"
	"log/slog"
	"math"

	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
	"gonum.org/v1/gonum/optimize"

	"github.com/llm-inferno/model-tuner/pkg/core"
)
```

Append to `tunerservice/init_estimator.go`:

```go
// Fit runs Nelder-Mead minimisation over all accumulated observations to find the
// (alpha, beta, gamma) that best explains all K observations jointly via the full
// queueing model. Returns [alpha, beta, gamma] or an error.
// Falls back to guessInitState on the first observation if the fit fails.
func (ie *InitEstimator) Fit() ([]float64, error) {
	if len(ie.observations) == 0 {
		return nil, fmt.Errorf("no observations to fit")
	}

	// Starting point: guessInitState on first observation; fall back to a safe default.
	x0 := guessInitState(ie.observations[0].toEnv())
	if x0 == nil {
		x0 = []float64{5.0, 0.05, 0.0005}
	}

	problem := optimize.Problem{Func: ie.objective}
	settings := &optimize.Settings{FuncEvaluations: 500}
	result, err := optimize.Minimize(problem, x0, settings, &optimize.NelderMead{})
	if err != nil && result == nil {
		slog.Warn("InitEstimator: Nelder-Mead failed, using guessInitState fallback", "err", err)
		if fallback := guessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead failed and guessInitState returned nil: %w", err)
	}

	x := result.X
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		slog.Warn("InitEstimator: Nelder-Mead returned non-positive params, using guessInitState fallback",
			"alpha", x[0], "beta", x[1], "gamma", x[2])
		if fallback := guessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead returned non-positive params and guessInitState returned nil")
	}

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
			MaxQueueSize: 10 * obs.MaxBatch,
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
		metrics, err := qa.Analyze(float32(obs.Lambda / 60)) // RPM → req/sec
		if err != nil {
			return math.MaxFloat64 / 2
		}

		ttftModel := float64(metrics.AvgTTFT)
		itlModel := float64(metrics.AvgTokenTime) // AvgTokenTime = ITL
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

- [ ] **Step 2: Build**

Run: `go build ./...` from `model-tuner/`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add tunerservice/init_estimator.go
git commit -m "feat: add Fit() and objective() to InitEstimator using Nelder-Mead"
```

---

### Task 5: Tests for Fit() parameter recovery

**Files:**
- Modify: `tunerservice/init_estimator_test.go`

- [ ] **Step 1: Write a failing test for parameter recovery**

Append to `tunerservice/init_estimator_test.go`:

```go
import (
	// add these to the existing import block:
	"math"

	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
)
```

> **Note**: The test file already imports `testing` and `github.com/llm-inferno/model-tuner/pkg/core`. Merge `math` and the `analyzer` import into the existing import block — do not add a second `import` block.

Append the following test function to `tunerservice/init_estimator_test.go`:

```go
// TestInitEstimator_Fit_ParameterRecovery generates synthetic observations from known
// true parameters using the queue analyzer, feeds them to the estimator, and verifies
// that Fit() recovers the true values within 10% relative error.
func TestInitEstimator_Fit_ParameterRecovery(t *testing.T) {
	// True parameters (granite_13b/G2 values from model-data.json)
	trueAlpha := float32(16.78)
	trueBeta := float32(0.073)
	trueGamma := float32(0.00228)
	maxBatch := 64

	// Operating points: vary token counts and arrival rates to overcome identifiability
	type opPoint struct {
		lambda   float32 // RPM
		inTok    float32
		outTok   float32
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
			MaxQueueSize: 10 * maxBatch,
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

	tolerance := 0.10 // 10% relative error
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
```

- [ ] **Step 2: Run the failing test to verify it compiles and the assertion is reachable**

Run: `go test ./tunerservice/ -run TestInitEstimator_Fit_ParameterRecovery -v`
Expected: PASS (synthetic observations are noise-free, Nelder-Mead should recover within 10%)

- [ ] **Step 3: Run all InitEstimator tests together**

Run: `go test ./tunerservice/ -run TestInitEstimator -v`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add tunerservice/init_estimator_test.go
git commit -m "test: add Fit() parameter recovery test for InitEstimator"
```

---

### Task 6: Wire InitEstimator into TunerService

**Files:**
- Modify: `tunerservice/service.go`

- [ ] **Step 1: Write tests first — verify holdBack blocks optimize during collection**

Append to `tunerservice/init_estimator_test.go`:

```go
func TestTunerService_IsWarmingUp_DuringCollection(t *testing.T) {
	// holdBack=true, minObs=3: IsWarmingUp should return true before any Tune calls
	// when the paramStore is empty and estimator is not ready
	ts := NewTunerService(3, 3, true)
	// Inject a fake estimator that is not ready yet
	key := makeKey("mymodel", "myacc")
	ts.estimators[key] = NewInitEstimator(3, true)
	if !ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=true when estimator not ready and holdBack=true")
	}
}

func TestTunerService_IsWarmingUp_HoldBackFalse(t *testing.T) {
	ts := NewTunerService(3, 3, false)
	key := makeKey("mymodel", "myacc")
	ts.estimators[key] = NewInitEstimator(3, false)
	// holdBack=false: collection phase does NOT count as warm-up
	if ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=false when holdBack=false")
	}
}
```

- [ ] **Step 2: Run the new tests (should fail — TunerService not yet updated)**

Run: `go test ./tunerservice/ -run TestTunerService_IsWarmingUp -v`
Expected: compile error (NewTunerService signature mismatch, `ts.estimators` not a field)

- [ ] **Step 3: Update TunerService struct and NewTunerService**

In `tunerservice/service.go`, replace:

```go
// TunerService groups replica metrics by (model, accelerator), runs EKF tuning per group,
// maintains a ParameterStore for state continuity, and returns updated ModelData.
type TunerService struct {
	paramStore   *ParameterStore
	warmUpCycles int
}

// NewTunerService creates a TunerService with an empty ParameterStore.
func NewTunerService(warmUpCycles int) *TunerService {
	return &TunerService{
		paramStore:   NewParameterStore(),
		warmUpCycles: warmUpCycles,
	}
}
```

With:

```go
// TunerService groups replica metrics by (model, accelerator), runs EKF tuning per group,
// maintains a ParameterStore for state continuity, and returns updated ModelData.
type TunerService struct {
	paramStore   *ParameterStore
	warmUpCycles int
	estimators   map[string]*InitEstimator
	initObs      int
	holdBack     bool
}

// NewTunerService creates a TunerService with an empty ParameterStore and estimator map.
func NewTunerService(warmUpCycles, initObs int, holdBack bool) *TunerService {
	return &TunerService{
		paramStore:   NewParameterStore(),
		warmUpCycles: warmUpCycles,
		estimators:   make(map[string]*InitEstimator),
		initObs:      initObs,
		holdBack:     holdBack,
	}
}

// estimatorFor returns the InitEstimator for the given key, creating it if needed.
func (ts *TunerService) estimatorFor(key string) *InitEstimator {
	if ie, ok := ts.estimators[key]; ok {
		return ie
	}
	ie := NewInitEstimator(ts.initObs, ts.holdBack)
	ts.estimators[key] = ie
	return ie
}
```

- [ ] **Step 4: Update tuneGroup to add observation and return early during collection**

In `tunerservice/service.go`, replace `tuneGroup`:

```go
// tuneGroup runs EKF tuning for all replicas in a single (model, accelerator) group.
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
	estimator := ts.estimatorFor(key)
	estimator.AddObservation(envs[0])

	if !estimator.IsReady() {
		slog.Info("collecting initial observations",
			"model", model, "accelerator", accelerator,
			"count", len(estimator.observations), "minObs", estimator.minObs)
		return fmt.Errorf("collecting initial observations for %s/%s (%d/%d)",
			model, accelerator, len(estimator.observations), estimator.minObs)
	}

	// Fit once when we have no prior paramStore entry (first EKF initialisation).
	var fitInitState []float64
	if ts.paramStore.Get(model, accelerator) == nil {
		var fitErr error
		fitInitState, fitErr = estimator.Fit()
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
```

- [ ] **Step 5: Update createTuner to accept fitInitState**

In `tunerservice/service.go`, replace `createTuner` signature and the `else` branch:

```go
// createTuner creates a Tuner for the given model/accelerator. If fitInitState is non-nil,
// it is used as the initial state instead of guessInitState.
func (ts *TunerService) createTuner(model, accelerator string, firstEnv *core.EnvironmentPrefillDecode, fitInitState []float64) (*core.Tuner, error) {
	existing := ts.paramStore.Get(model, accelerator)

	configData, err := utils.LoadConfigForServer(config.DefaultConfigType)
	if err != nil {
		return nil, fmt.Errorf("load config for %s: %w", model, err)
	}

	if existing != nil {
		// Restore previous alpha/beta/gamma as initial state
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
		// Use fitted initial state if available, otherwise guess from the first observation.
		if fitInitState != nil {
			setInitState(&configData.ModelData, fitInitState)
		} else if initState := guessInitState(firstEnv); initState != nil {
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
```

- [ ] **Step 6: Update IsWarmingUp to check estimators**

In `tunerservice/service.go`, replace `IsWarmingUp`:

```go
// IsWarmingUp returns true if the tuner has not yet completed warmUpCycles accepted EKF
// updates for at least one known pair, or if holdBack=true and any estimator is still
// collecting its initial observations.
// Returns false when warmUpCycles is zero and no estimator is holding back.
func (ts *TunerService) IsWarmingUp() bool {
	// Check estimators in collection phase (holdBack=true only)
	for _, ie := range ts.estimators {
		if !ie.IsReady() && ie.HoldBack() {
			return true
		}
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
```

- [ ] **Step 7: Build**

Run: `go build ./...` from `model-tuner/`
Expected: no errors

- [ ] **Step 8: Run all tests including the new IsWarmingUp tests**

Run: `go test ./tunerservice/ -v`
Expected: all tests PASS

- [ ] **Step 9: Commit**

```bash
git add tunerservice/service.go tunerservice/init_estimator_test.go
git commit -m "feat: wire InitEstimator into TunerService; update IsWarmingUp"
```

---

### Task 7: Update cmd/tuner/main.go to read new env vars

**Files:**
- Modify: `cmd/tuner/main.go`

- [ ] **Step 1: Write the test (smoke — just build and run with defaults)**

This is a main package; there are no unit tests for it. The test is: the binary starts without errors.

- [ ] **Step 2: Update main.go**

In `cmd/tuner/main.go`, replace the entire file contents:

```go
package main

import (
	"log"
	"log/slog"
	"os"
	"strconv"

	pkgconfig "github.com/llm-inferno/model-tuner/pkg/config"
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
	if v := os.Getenv(tunerservice.WarmUpCyclesEnvName); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			warmUpCycles = n
		}
	}

	initObs := tunerservice.DefaultInitObs
	if v := os.Getenv(tunerservice.InitObsEnvName); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			initObs = n
		}
	}

	holdBack := tunerservice.DefaultInitHoldBack
	if v := os.Getenv(tunerservice.InitHoldBackEnvName); v != "" {
		holdBack = v == "true" || v == "1"
	}

	service := tunerservice.NewTunerService(warmUpCycles, initObs, holdBack)
	server := tunerservice.NewTunerServer(service)

	slog.Info("Starting TunerService",
		"host", host, "port", port,
		"warmUpCycles", warmUpCycles,
		"initObs", initObs,
		"holdBack", holdBack)
	if err := server.Run(host, port); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
```

- [ ] **Step 3: Build**

Run: `go build ./...` from `model-tuner/`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add cmd/tuner/main.go
git commit -m "feat: read TUNER_INIT_OBS and TUNER_INIT_HOLD_BACK in tuner main"
```

---

### Task 8: Update CLAUDE.md documentation

**Files:**
- Modify: `docs/CLAUDE.md` (model-tuner repo — check if it exists, else update the root `CLAUDE.md`)
- Modify: `/Users/tantawi/Projects/llm-inferno/control-loop/CLAUDE.md`

- [ ] **Step 1: Check for model-tuner CLAUDE.md**

Run: `ls /Users/tantawi/Projects/llm-inferno/model-tuner/CLAUDE.md 2>/dev/null || echo "not found"`

If it exists, add to its environment variables table:

```
| `TUNER_INIT_OBS`       | `5`    | Number of observations to collect before fitting initial EKF parameters |
| `TUNER_INIT_HOLD_BACK` | `true` | If true, report warmingUp=true during collection phase (Option B). Set false to proceed with static model-data during collection (Option A). |
```

If it does not exist, skip this step.

- [ ] **Step 2: Update control-loop CLAUDE.md**

In `/Users/tantawi/Projects/llm-inferno/control-loop/CLAUDE.md`, in the **Environment Variables** table for the Tuner block, append two new rows after the `TUNER_WARM_UP_CYCLES` row:

```
| `TUNER_INIT_OBS` | `5` | Observations to accumulate before the multi-observation Nelder-Mead fit; set to `1` to revert to single-observation `guessInitState` behaviour |
| `TUNER_INIT_HOLD_BACK` | `true` | If `true`, the tuner reports `warmingUp=true` during the collection phase so the controller skips optimize+actuate (Option B). Set `false` to let the controller proceed with static model-data during collection (Option A). |
```

- [ ] **Step 3: Commit**

```bash
# From control-loop/
git add CLAUDE.md
git commit -m "docs: document TUNER_INIT_OBS and TUNER_INIT_HOLD_BACK env vars"

# From model-tuner/ (only if CLAUDE.md exists there)
git add CLAUDE.md
git commit -m "docs: document TUNER_INIT_OBS and TUNER_INIT_HOLD_BACK env vars"
```

---

### Task 9: Build image, reload into kind, and verify collection phase

**Files:** None (build + deploy)

- [ ] **Step 1: Build the inferno-tuner image**

Run from `model-tuner/`:
```bash
docker build -t quay.io/atantawi/inferno-tuner:latest .
```
Expected: build succeeds

- [ ] **Step 2: Load image into kind**

Run from `control-loop/`:
```bash
kind load docker-image quay.io/atantawi/inferno-tuner:latest --name kind-cluster
```
Expected: Image loaded

- [ ] **Step 3: Restart the tuner container**

```bash
kubectl rollout restart deployment/inferno -n inferno
kubectl rollout status  deployment/inferno -n inferno --timeout=60s
```

- [ ] **Step 4: Watch tuner logs and verify collection phase**

```bash
kubectl logs -f -n inferno deployment/inferno -c tuner
```

Expected log lines during the first 5 cycles (TUNER_INIT_OBS=5 default):
```
level=INFO msg="collecting initial observations" model=granite_13b accelerator=G2 count=1 minObs=5
level=INFO msg="collecting initial observations" model=granite_13b accelerator=G2 count=2 minObs=5
...
level=INFO msg="InitEstimator: Fit complete" alpha=16.78... beta=0.073... gamma=0.00228... observations=5
level=INFO msg="tuned parameters" model=granite_13b accelerator=G2 alpha=16.78... warmUp=true
```

- [ ] **Step 5: Watch controller logs and verify warmingUp=true during collection**

```bash
kubectl logs -f -n inferno deployment/inferno -c controller
```

Expected during collection (cycles 1–5): controller skips optimize+actuate (prints warm-up message or skips timing lines).
Expected after collection (cycle 6+): normal timing line with optimize+actuate.

- [ ] **Step 6: Verify EKF convergence for both workloads**

After at least 10 cycles (5 collection + 5 EKF warm-up), alpha for both models should be near the model-data.json values (~16.78 for both llama_13b and granite_13b on G2).

```bash
kubectl logs -n inferno deployment/inferno -c tuner | grep "tuned parameters" | tail -20
```

Expected: alpha ≈ 16.78, not drifting to 3 or 35.

- [ ] **Step 7: Commit any fixes found during testing**

If any issues are found and fixed, commit with:
```bash
git commit -m "fix: <describe fix>"
```
