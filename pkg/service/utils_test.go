package service

import (
	"testing"

	optconfig "github.com/llm-inferno/optimizer-light/pkg/config"
)

func TestBuildEnvironments_MaxQueueSizeFromSpec(t *testing.T) {
	specs := []optconfig.ServerSpec{
		{
			Model:        "granite_8b",
			MaxQueueSize: 128,
			CurrentAlloc: optconfig.AllocationData{
				Accelerator: "H100",
				MaxBatch:    64,
				ITLAverage:  8.5,
				TTFTAverage: 45.0,
				Load: optconfig.ServerLoadSpec{
					ArrivalRate:  60,
					AvgInTokens:  2048,
					AvgOutTokens: 1024,
				},
			},
		},
	}

	envs := buildEnvironments(specs)
	if len(envs) != 1 {
		t.Fatalf("expected 1 environment, got %d", len(envs))
	}
	if envs[0].MaxQueueSize != 128 {
		t.Errorf("MaxQueueSize = %d, want 128", envs[0].MaxQueueSize)
	}
}

func TestBuildEnvironments_ZeroMaxQueueSizeWhenUnset(t *testing.T) {
	specs := []optconfig.ServerSpec{
		{
			Model:        "llama_13b",
			MaxQueueSize: 0,
			CurrentAlloc: optconfig.AllocationData{
				Accelerator: "H100",
				MaxBatch:    64,
				ITLAverage:  12.0,
				TTFTAverage: 60.0,
				Load: optconfig.ServerLoadSpec{
					ArrivalRate:  30,
					AvgInTokens:  768,
					AvgOutTokens: 768,
				},
			},
		},
	}

	envs := buildEnvironments(specs)
	if len(envs) != 1 {
		t.Fatalf("expected 1 environment, got %d", len(envs))
	}
	if envs[0].MaxQueueSize != 0 {
		t.Errorf("MaxQueueSize = %d, want 0 (no external queue)", envs[0].MaxQueueSize)
	}
}
