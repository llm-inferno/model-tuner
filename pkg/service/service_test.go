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
