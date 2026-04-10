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
	// invalid env: zero AvgITL (not allowed by Valid())
	invalid := core.NewEnvironmentPrefillDecode(10, 0, 0, 64, 100, 500, 0, 0)
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
