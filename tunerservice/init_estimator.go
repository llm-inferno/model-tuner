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
