package tunerservice2

import (
	"fmt"
	"log/slog"
	"time"

	optconfig "github.com/llm-inferno/optimizer-light/pkg/config"

	"github.com/llm-inferno/model-tuner/pkg/config"
	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/model-tuner/pkg/utils"
)

// TunerService groups replica metrics by (model, accelerator), runs EKF tuning per group,
// maintains a ParameterStore for state continuity, and returns updated ModelData.
type TunerService struct {
	paramStore *ParameterStore
}

// NewTunerService creates a TunerService with an empty ParameterStore.
func NewTunerService() *TunerService {
	return &TunerService{
		paramStore: NewParameterStore(),
	}
}

// Tune accepts per-replica ServerSpecs from the control-loop Collector, runs EKF tuning for each
// (model, accelerator) group, and returns updated ModelData with tuned alpha/beta/gamma.
func (ts *TunerService) Tune(replicaSpecs []optconfig.ServerSpec) (*optconfig.ModelData, error) {
	groups := groupByModelAccelerator(replicaSpecs)

	for key, replicas := range groups {
		model, accelerator := splitKey(key)
		if err := ts.tuneGroup(model, accelerator, replicas); err != nil {
			slog.Warn("tuning failed for group", "key", key, "err", err)
		}
	}

	return ts.buildModelData(groups), nil
}

// tuneGroup runs EKF tuning for all replicas in a single (model, accelerator) group.
func (ts *TunerService) tuneGroup(model, accelerator string, replicas []optconfig.ServerSpec) error {
	envs := buildEnvironments(replicas)
	if len(envs) == 0 {
		return fmt.Errorf("no valid environments for %s/%s", model, accelerator)
	}

	tuner, err := ts.createTuner(model, accelerator, envs[0])
	if err != nil {
		return fmt.Errorf("create tuner for %s/%s: %w", model, accelerator, err)
	}

	var lastResults *core.TunedResults
	for _, env := range envs {
		results, runErr := tuner.RunWithValidation(env)
		if runErr != nil {
			slog.Warn("EKF run error", "model", model, "accelerator", accelerator, "err", runErr)
			continue
		}
		if results.ValidationFailed {
			slog.Debug("EKF update rejected (NIS)", "model", model, "accelerator", accelerator, "NIS", results.NIS)
		}
		lastResults = results
	}

	if lastResults == nil || lastResults.ServiceParms == nil {
		return fmt.Errorf("no valid results for %s/%s", model, accelerator)
	}

	ts.paramStore.Set(model, accelerator, &LearnedParameters{
		Alpha:       lastResults.ServiceParms.Alpha,
		Beta:        lastResults.ServiceParms.Beta,
		Gamma:       lastResults.ServiceParms.Gamma,
		NIS:         lastResults.NIS,
		Covariance:  covToSlice(lastResults.Covariance),
		LastUpdated: time.Now(),
	})
	return nil
}

// createTuner creates a Tuner for the given model/accelerator, restoring state from the
// ParameterStore if available, or guessing initial state from the first environment.
func (ts *TunerService) createTuner(model, accelerator string, firstEnv *core.EnvironmentPrefillDecode) (*core.Tuner, error) {
	existing := ts.paramStore.Get(model, accelerator)

	var configData *config.ConfigData
	var err error

	// LoadConfigForServer falls back to default if no model-specific config exists.
	configData, err = utils.LoadConfigForServer(model)
	if err != nil {
		return nil, fmt.Errorf("load config for %s: %w", model, err)
	}

	if existing != nil {
		// Restore previous alpha/beta/gamma as initial state
		configData.ModelData.InitState = []float64{
			float64(existing.Alpha),
			float64(existing.Beta),
			float64(existing.Gamma),
		}
		if cov := existing.CovarianceMatrix(); cov != nil {
			tuner, err := core.NewTunerWithCovariance(configData, firstEnv, cov)
			if err != nil {
				return nil, err
			}
			if err := tuner.SetObservationFunc(core.NewQueueModelSystemFuncCreatorPrefillDecode(tuner)); err != nil {
				return nil, err
			}
			return tuner, nil
		}
	} else {
		// Guess initial state from observations if possible
		if initState := guessInitState(firstEnv); initState != nil {
			configData.ModelData.InitState = initState
		}
	}

	tuner, err := core.NewTuner(configData, firstEnv)
	if err != nil {
		return nil, err
	}
	if err := tuner.SetObservationFunc(core.NewQueueModelSystemFuncCreatorPrefillDecode(tuner)); err != nil {
		return nil, err
	}
	return tuner, nil
}

// buildModelData constructs ModelData from the ParameterStore for all observed model/accelerator groups.
// It uses replica data to fill in MaxBatchSize.
func (ts *TunerService) buildModelData(groups map[string][]optconfig.ServerSpec) *optconfig.ModelData {
	var entries []optconfig.ModelAcceleratorPerfData
	for key, replicas := range groups {
		model, accelerator := splitKey(key)
		params := ts.paramStore.Get(model, accelerator)
		if params == nil {
			continue
		}
		maxBatch := maxBatchFromReplicas(replicas)
		entries = append(entries, optconfig.ModelAcceleratorPerfData{
			Name:         model,
			Acc:          accelerator,
			MaxBatchSize: maxBatch,
			PerfParms: optconfig.PerfParms{
				Alpha: params.Alpha,
				Beta:  params.Beta,
				Gamma: params.Gamma,
			},
		})
	}
	return &optconfig.ModelData{PerfData: entries}
}

// GetParams returns the most recently tuned parameters for a model/accelerator pair,
// or nil if no tuning has been performed for that pair yet.
func (ts *TunerService) GetParams(model, accelerator string) *LearnedParameters {
	return ts.paramStore.Get(model, accelerator)
}
