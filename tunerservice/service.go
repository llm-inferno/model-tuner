package tunerservice

import (
	"fmt"
	"log/slog"
	"math"
	"time"

	optconfig "github.com/llm-inferno/optimizer-light/pkg/config"

	"github.com/llm-inferno/model-tuner/pkg/config"
	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/model-tuner/pkg/utils"
)

// TunerService groups replica metrics by (model, accelerator), runs EKF tuning per group,
// maintains a ParameterStore for state continuity, and returns updated ModelData.
type TunerService struct {
	paramStore   *ParameterStore
	warmUpCycles int
	estimators   map[string]*InitEstimator
	initObs      int
	holdBack     bool
}

// NewTunerService creates a TunerService with an empty ParameterStore.
func NewTunerService(warmUpCycles, initObs int, holdBack bool) *TunerService {
	return &TunerService{
		paramStore:   NewParameterStore(),
		warmUpCycles: warmUpCycles,
		estimators:   make(map[string]*InitEstimator),
		initObs:      initObs,
		holdBack:     holdBack,
	}
}

// estimatorFor returns the InitEstimator for the given key, creating it if needed.
func (ts *TunerService) estimatorFor(key string) *InitEstimator {
	if ie, ok := ts.estimators[key]; ok {
		return ie
	}
	ie := NewInitEstimator(ts.initObs, ts.holdBack)
	ts.estimators[key] = ie
	return ie
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

	for i, env := range envs {
		slog.Info("replica environment",
			"model", model,
			"accelerator", accelerator,
			"replica", i,
			"arrivalRateRPM", env.Lambda,
			"maxBatch", env.MaxBatchSize,
			"avgInTokens", env.AvgInputTokens,
			"avgOutTokens", env.AvgOutputTokens,
			"avgTTFT", env.AvgTTFT,
			"avgITL", env.AvgITL,
		)
	}

	key := makeKey(model, accelerator)
	estimator := ts.estimatorFor(key)
	estimator.AddObservation(envs[0])

	if !estimator.IsReady() {
		slog.Info("collecting initial observations",
			"model", model, "accelerator", accelerator,
			"count", len(estimator.observations), "minObs", estimator.minObs)
		return fmt.Errorf("collecting initial observations for %s/%s (%d/%d)",
			model, accelerator, len(estimator.observations), estimator.minObs)
	}

	// Fit once when we have no prior paramStore entry (first EKF initialisation).
	var fitInitState []float64
	if ts.paramStore.Get(model, accelerator) == nil {
		var fitErr error
		fitInitState, fitErr = estimator.Fit()
		if fitErr != nil {
			slog.Warn("InitEstimator Fit failed, EKF will use guessInitState", "err", fitErr)
		}
	}

	tuner, err := ts.createTuner(model, accelerator, envs[0], fitInitState)
	if err != nil {
		return fmt.Errorf("create tuner for %s/%s: %w", model, accelerator, err)
	}

	updateCount := 0
	if existing := ts.paramStore.Get(model, accelerator); existing != nil {
		updateCount = existing.UpdateCount
	}
	skipNIS := updateCount < ts.warmUpCycles

	var accepted *core.TunedResults
	for _, env := range envs {
		results, runErr := tuner.RunWithValidation(env, skipNIS)
		if runErr != nil {
			slog.Warn("EKF run error", "model", model, "accelerator", accelerator, "err", runErr)
			continue
		}
		if results.ValidationFailed {
			if results.NIS > 0 {
				slog.Info("EKF update rejected: NIS gate", "model", model, "accelerator", accelerator, "NIS", results.NIS)
			} else {
				slog.Info("EKF update rejected: state validation", "model", model, "accelerator", accelerator)
			}
			continue
		}
		accepted = results
	}

	if accepted == nil {
		return fmt.Errorf("no accepted results for %s/%s", model, accelerator)
	}

	ts.paramStore.Set(model, accelerator, &LearnedParameters{
		Alpha:       accepted.ServiceParms.Alpha,
		Beta:        accepted.ServiceParms.Beta,
		Gamma:       accepted.ServiceParms.Gamma,
		NIS:         accepted.NIS,
		UpdateCount: updateCount + 1,
		Covariance:  covToSlice(accepted.Covariance),
		LastUpdated: time.Now(),
	})
	slog.Info("tuned parameters",
		"model", model,
		"accelerator", accelerator,
		"alpha", accepted.ServiceParms.Alpha,
		"beta", accepted.ServiceParms.Beta,
		"gamma", accepted.ServiceParms.Gamma,
		"NIS", accepted.NIS,
		"updateCount", updateCount+1,
		"warmUp", skipNIS,
	)
	return nil
}

// createTuner creates a Tuner for the given model/accelerator, restoring state from the
// ParameterStore if available, or guessing initial state from the first environment.
func (ts *TunerService) createTuner(model, accelerator string, firstEnv *core.EnvironmentPrefillDecode, fitInitState []float64) (*core.Tuner, error) {
	existing := ts.paramStore.Get(model, accelerator)

	var configData *config.ConfigData
	var err error

	configData, err = utils.LoadConfigForServer(config.DefaultConfigType)
	if err != nil {
		return nil, fmt.Errorf("load config for %s: %w", model, err)
	}

	if existing != nil {
		// Restore previous alpha/beta/gamma as initial state
		setInitState(&configData.ModelData, []float64{
			float64(existing.Alpha),
			float64(existing.Beta),
			float64(existing.Gamma),
		})
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
		// Use fitted initial state if available, otherwise guess from the first observation.
		if fitInitState != nil {
			setInitState(&configData.ModelData, fitInitState)
		} else if initState := guessInitState(firstEnv); initState != nil {
			setInitState(&configData.ModelData, initState)
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

// setInitState sets InitState and recomputes MinState/MaxState using a log-symmetric factor:
// Min = max(init[i]/factor, epsilon), Max = init[i]*factor.
// This keeps the three fields consistent whenever the initial state changes.
func setInitState(md *config.ModelData, initState []float64) {
	md.InitState = initState
	md.MinState = make([]float64, len(initState))
	md.MaxState = make([]float64, len(initState))
	for i, v := range initState {
		md.MinState[i] = math.Max(v/config.DefaultInitStateFactor, config.DefaultInitStateMinEpsilon)
		md.MaxState[i] = v * config.DefaultInitStateFactor
	}
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

// IsWarmingUp returns true if the tuner has not yet completed warmUpCycles accepted EKF
// updates for at least one known (model, accelerator) pair.
// Returns false when warmUpCycles is zero or when all pairs have graduated.
func (ts *TunerService) IsWarmingUp() bool {
	// Check estimators in collection phase (holdBack=true only)
	for _, ie := range ts.estimators {
		if !ie.IsReady() && ie.HoldBack() {
			return true
		}
	}
	if ts.warmUpCycles == 0 {
		return false
	}
	for _, params := range ts.paramStore.GetAll() {
		if params.UpdateCount < ts.warmUpCycles {
			return true
		}
	}
	return false
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
