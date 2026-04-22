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
