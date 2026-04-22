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
