package tunerservice2

import (
	"fmt"
	"strings"

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

// splitKey splits a "model/accelerator" key back into its components.
// If the model name itself contains slashes (e.g. a path), only the last slash is used.
func splitKey(key string) (model, accelerator string) {
	idx := strings.LastIndex(key, "/")
	if idx < 0 {
		return key, ""
	}
	return key[:idx], key[idx+1:]
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
			0, // BatchSize not provided by collector; not used by observation func
			0, // AvgQueueTime not available from collector
			maxBatch,
			float32(a.Load.AvgInTokens),
			float32(a.Load.AvgOutTokens),
			a.TTFTAverage,
			a.ITLAverage,
		)
		// env.Valid() is always true here: ArrivalRate > 0 (filtered by caller),
		// tokens > 0, latency > 0, and maxBatch > 0 are all checked above.
		envs = append(envs, env)
	}
	return envs
}

// guessInitState derives initial alpha, beta, gamma from observed TTFT and ITL using the
// queueing model equations from the paper:
//
//	TTFT = alpha + (beta + gamma) * inputTokens           (eq 12)
//	ITL  = alpha + beta + gamma * (inputTokens + (outputTokens+1)/2)  (eq 13)
//
// Returns nil if the derivation yields non-positive parameters.
func guessInitState(env *core.EnvironmentPrefillDecode) []float64 {
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
