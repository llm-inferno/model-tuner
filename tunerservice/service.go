package tunerservice

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
// Returns an error if no parameters could be produced (all replicas idle or all groups failed).
func (ts *TunerService) Tune(replicaSpecs []optconfig.ServerSpec) (*optconfig.ModelData, error) {
	groups := groupByModelAccelerator(replicaSpecs)
	if len(groups) == 0 {
		return nil, fmt.Errorf("no replicas with active traffic in request")
	}

	for key, replicas := range groups {
		model, accelerator := splitKey(key)
		if err := ts.tuneGroup(model, accelerator, replicas); err != nil {
			slog.Warn("tuning failed for group", "key", key, "err", err)
		}
	}

	modelData := ts.buildModelData(groups)
	if len(modelData.PerfData) == 0 {
		return nil, fmt.Errorf("tuning produced no results for any model/accelerator group")
	}
	return modelData, nil
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
			slog.Debug("EKF update rejected", "model", model, "accelerator", accelerator, "NIS", results.NIS)
		}
		lastResults = results
	}

	if lastResults == nil || lastResults.ServiceParms == nil {
		return fmt.Errorf("no valid results for %s/%s", model, accelerator)
	}

	// Preserve the previously stored NIS when the last update was rolled back, so the
	// stored NIS always reflects a valid (accepted) filter update, not a rejected outlier.
	nisToStore := lastResults.NIS
	if lastResults.ValidationFailed {
		if prev := ts.paramStore.Get(model, accelerator); prev != nil {
			nisToStore = prev.NIS
		} else {
			nisToStore = 0
		}
	}

	ts.paramStore.Set(model, accelerator, &LearnedParameters{
		Alpha:       lastResults.ServiceParms.Alpha,
		Beta:        lastResults.ServiceParms.Beta,
		Gamma:       lastResults.ServiceParms.Gamma,
		NIS:         nisToStore,
		Covariance:  covToSlice(lastResults.Covariance),
		LastUpdated: time.Now(),
	})
	slog.Info("tuned parameters",
		"model", model,
		"accelerator", accelerator,
		"alpha", lastResults.ServiceParms.Alpha,
		"beta", lastResults.ServiceParms.Beta,
		"gamma", lastResults.ServiceParms.Gamma,
		"NIS", nisToStore,
		"rejected", lastResults.ValidationFailed,
	)
	return nil
}

// createTuner creates a Tuner for the given model/accelerator, restoring state from the
// ParameterStore if available, or guessing initial state from the first environment.
func (ts *TunerService) createTuner(model, accelerator string, firstEnv *core.EnvironmentPrefillDecode) (*core.Tuner, error) {
	existing := ts.paramStore.Get(model, accelerator)

	var configData *config.ConfigData
	var err error

	configData, err = utils.LoadConfigForServer(config.DefaultConfigType)
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

// Merge accepts the Controller's current ModelData and returns it with PerfParms overlaid
// from the ParameterStore for any matching (name, accelerator) pairs. Entries in the
// ParameterStore that have no match in the input are appended with default non-parameter fields.
func (ts *TunerService) Merge(modelData *optconfig.ModelData) *optconfig.ModelData {
	if modelData == nil {
		modelData = &optconfig.ModelData{}
	}

	allParams := ts.paramStore.GetAll()
	matched := make(map[string]bool, len(allParams))

	// Phase 1: overlay tuned PerfParms onto existing entries.
	result := make([]optconfig.ModelAcceleratorPerfData, len(modelData.PerfData))
	for i, entry := range modelData.PerfData {
		result[i] = entry
		key := makeKey(entry.Name, entry.Acc)
		if params, ok := allParams[key]; ok {
			result[i].PerfParms = optconfig.PerfParms{
				Alpha: params.Alpha,
				Beta:  params.Beta,
				Gamma: params.Gamma,
			}
			if matched[key] {
				slog.Warn("duplicate model/accelerator key in input ModelData", "key", key)
			} else {
				matched[key] = true
			}
		}
	}

	// Phase 2: append ParameterStore entries not present in the input.
	for key, params := range allParams {
		if matched[key] {
			continue
		}
		model, acc := splitKey(key)
		result = append(result, optconfig.ModelAcceleratorPerfData{
			Name:         model,
			Acc:          acc,
			AccCount:     DefaultAccCount,
			MaxBatchSize: DefaultMaxBatchSize,
			AtTokens:     DefaultAtTokens,
			PerfParms: optconfig.PerfParms{
				Alpha: params.Alpha,
				Beta:  params.Beta,
				Gamma: params.Gamma,
			},
		})
	}

	return &optconfig.ModelData{PerfData: result}
}
