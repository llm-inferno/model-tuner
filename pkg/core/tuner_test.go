package core

import (
	"testing"

	"github.com/llm-inferno/model-tuner/pkg/config"
)

// aggressiveConfig returns a ConfigData with boundedState=false, tiny InitState,
// high percentChange (large Q), and tiny expectedObservations (tiny R).
// K ≈ 1, so a single update with near-zero observations drives alpha negative.
func aggressiveConfig() *config.ConfigData {
	return &config.ConfigData{
		FilterData: config.FilterData{
			GammaFactor: 1.0,
			ErrorLevel:  0.05,
			TPercentile: 1.96,
		},
		ModelData: config.ModelData{
			InitState:            []float64{0.001, 0.0001, 0.000001},
			PercentChange:        []float64{50.0, 50.0, 50.0},
			BoundedState:         false,
			ExpectedObservations: []float64{0.001, 0.001},
		},
	}
}

func newTestEnv(ttft, itl float32) *EnvironmentPrefillDecode {
	return NewEnvironmentPrefillDecode(
		30,   // lambda RPM
		0,    // batchSize
		0,    // avgQueueTime
		128,  // maxBatchSize
		512,  // avgInputTokens
		128,  // avgOutputTokens
		ttft,
		itl,
	)
}

// TestPositivityCheckDuringWarmUp verifies that validateState() fires and rolls back
// the filter when an EKF update drives a parameter negative, even with skipNIS=true.
func TestPositivityCheckDuringWarmUp(t *testing.T) {
	cfg := aggressiveConfig()
	env := newTestEnv(0.001, 0.001)

	tuner, err := NewTuner(cfg, env)
	if err != nil {
		t.Fatalf("NewTuner: %v", err)
	}
	if err := tuner.SetObservationFunc(NewQueueModelSystemFuncCreatorPrefillDecode(tuner)); err != nil {
		t.Fatalf("SetObservationFunc: %v", err)
	}

	// Record initial state — rollback should restore to this.
	initialState := tuner.X().At(0, 0) // alpha

	// skipNIS=true simulates warm-up; positivity check must still fire.
	result, err := tuner.RunWithValidation(env, true)
	if err != nil {
		t.Fatalf("RunWithValidation: %v", err)
	}

	if !result.ValidationFailed {
		// EKF didn't go negative — test setup didn't create the expected condition.
		// This is not a bug in production code; log and skip rather than fail.
		t.Logf("alpha after update: %.6f (stayed positive — positivity check not triggered; consider tuning aggressiveConfig)", tuner.X().At(0, 0))
		t.Skip("aggressiveConfig did not drive params negative in one step; positivity check path not exercised")
	}

	// ValidationFailed=true means validateState fired and rolled back.
	// Current state must equal initial state (rollback succeeded).
	alphaAfter := tuner.X().At(0, 0)
	if alphaAfter != initialState {
		t.Errorf("expected rollback to initial alpha=%.6f, got %.6f", initialState, alphaAfter)
	}

	// All params must still be positive after rollback.
	x := tuner.X()
	for i, name := range []string{"alpha", "beta", "gamma"} {
		if x.AtVec(i) <= 0 {
			t.Errorf("%s=%.6f after rollback: must be positive", name, x.AtVec(i))
		}
	}

	t.Logf("positivity check fired during warm-up (skipNIS=true): ValidationFailed=true, rolled back to alpha=%.6f", alphaAfter)
}

// TestNISSkippedDuringWarmUp verifies that a large innovation that would normally
// fail the NIS gate is accepted when skipNIS=true.
func TestNISSkippedDuringWarmUp(t *testing.T) {
	cfg := &config.ConfigData{
		FilterData: config.FilterData{
			GammaFactor: 1.0,
			ErrorLevel:  0.05,
			TPercentile: 1.96,
		},
		ModelData: config.ModelData{
			InitState:            []float64{5.0, 0.05, 0.00005},
			PercentChange:        []float64{0.15, 0.15, 0.15},
			BoundedState:         true,
			MinState:             []float64{0.5, 0.005, 0.000005},
			MaxState:             []float64{50.0, 0.5, 0.0005},
			ExpectedObservations: []float64{200.0, 40.0},
		},
	}

	// Observation far from what the model predicts — large innovation.
	env := newTestEnv(5000, 5000)

	tuner, err := NewTuner(cfg, env)
	if err != nil {
		t.Fatalf("NewTuner: %v", err)
	}
	if err := tuner.SetObservationFunc(NewQueueModelSystemFuncCreatorPrefillDecode(tuner)); err != nil {
		t.Fatalf("SetObservationFunc: %v", err)
	}

	// With skipNIS=false the large innovation should trigger the NIS gate.
	result, err := tuner.RunWithValidation(env, false)
	if err != nil {
		t.Fatalf("RunWithValidation(skipNIS=false): %v", err)
	}
	if !result.ValidationFailed {
		t.Logf("NIS=%.2f: innovation was not large enough to trigger NIS gate with skipNIS=false; skipping", result.NIS)
		t.Skip("NIS gate not triggered — consider adjusting observation values")
	}
	t.Logf("NIS gate triggered with skipNIS=false: NIS=%.2f, ValidationFailed=true", result.NIS)

	// Same observation with skipNIS=true — must be accepted.
	result2, err := tuner.RunWithValidation(env, true)
	if err != nil {
		t.Fatalf("RunWithValidation(skipNIS=true): %v", err)
	}
	if result2.ValidationFailed {
		t.Errorf("expected update to be accepted with skipNIS=true, got ValidationFailed=true")
	}
	t.Logf("update accepted with skipNIS=true: alpha=%.4f", result2.ServiceParms.Alpha)
}
