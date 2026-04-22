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
