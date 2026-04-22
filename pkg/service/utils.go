package service

import (
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
			0,
			0,
			maxBatch,
			float32(a.Load.AvgInTokens),
			float32(a.Load.AvgOutTokens),
			a.TTFTAverage,
			a.ITLAverage,
		)
		env.MaxQueueSize = r.MaxQueueSize
		envs = append(envs, env)
	}
	return envs
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
