package tunerservice

import (
	"testing"

	optconfig "github.com/llm-inferno/optimizer-light/pkg/config"
)

// makeTestSpec returns a minimal ServerSpec with sufficient fields for buildEnvironments.
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

// TestTunerService_SWNM_ReturnsParamsAfterInitPhase verifies that Tune() returns
// parameters as soon as the init phase completes (minObs=initObs, no extra filling wait).
func TestTunerService_SWNM_ReturnsParamsAfterInitPhase(t *testing.T) {
	initObs := 3
	windowSize := 5
	ts := NewTunerService(0, initObs, false, true, windowSize, DefaultResidualThreshold)

	spec := makeTestSpec("llama", "H100", 15, 55, 6, 120, 700, 64)

	// Cycles 1–2: InitEstimator still collecting — errors expected.
	for i := range initObs - 1 {
		_, err := ts.Tune([]optconfig.ServerSpec{spec})
		if err == nil {
			t.Fatalf("cycle %d: expected error during init collection, got nil", i+1)
		}
	}

	// Cycle 3: InitEstimator completes, SWE seeded with initObs observations and immediately
	// ready (minObs=initObs=3) — should return parameters on this very cycle.
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

// TestTunerService_SWNM_FitError_RetainsPreviousParams verifies that when Fit() returns
// an error the paramStore entry for that key is left unchanged (previous estimate retained).
func TestTunerService_SWNM_FitError_RetainsPreviousParams(t *testing.T) {
	ts := NewTunerService(0, 1, false, true, 1, DefaultResidualThreshold)
	model, acc := "llama", "H100"
	key := makeKey(model, acc)

	// Pre-seed paramStore with a known estimate.
	ts.paramStore.Set(model, acc, &LearnedParameters{Alpha: 10.0, Beta: 0.05, Gamma: 0.001, UpdateCount: 1})

	// Ready InitEstimator with one valid observation.
	ie := NewInitEstimator(1, false)
	ie.AddObservation(makeTestEnv(15, 55, 6, 120, 700, 64))
	ts.estimators[key] = ie

	// SWE with windowSize=0: IsReady() returns true (0>=0) but AddObservation immediately
	// evicts every entry (len>0), so Fit() always sees an empty window and returns an error.
	swe := NewSlidingWindowEstimator(1, 1, DefaultResidualThreshold)
	swe.windowSize = 0
	ts.slidingEstimators[key] = swe

	env := makeTestEnv(15, 55, 6, 120, 700, 64)
	err := ts.tuneGroupSliding(model, acc, key, ie, env)
	if err == nil {
		t.Fatal("expected Fit error, got nil")
	}

	params := ts.paramStore.Get(model, acc)
	if params == nil {
		t.Fatal("paramStore entry missing after Fit error")
	}
	if params.Alpha != 10.0 || params.Beta != 0.05 || params.Gamma != 0.001 {
		t.Errorf("paramStore was modified after Fit error: alpha=%.4f beta=%.4f gamma=%.6f",
			params.Alpha, params.Beta, params.Gamma)
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

	// SWE created but minObs not reached → still warmingUp
	swe := NewSlidingWindowEstimator(5, 4, DefaultResidualThreshold) // minObs=4 > 3 seeded
	swe.Seed(ie.observations) // 3 obs < minObs=4 → not ready
	ts.slidingEstimators[key] = swe
	if !ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=true when SWE has fewer than minObs observations")
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

	swe := NewSlidingWindowEstimator(3, 1, DefaultResidualThreshold)
	swe.Seed(ie.observations) // 3 obs, windowSize=3 → ready
	ts.slidingEstimators[key] = swe

	if ts.IsWarmingUp() {
		t.Fatal("expected IsWarmingUp=false when SWE window is full")
	}
}
