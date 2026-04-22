package service

import (
	"fmt"
	"log/slog"
	"math"
	"time"

	optconfig "github.com/llm-inferno/optimizer-light/pkg/config"

	"github.com/llm-inferno/model-tuner/pkg/config"
	"github.com/llm-inferno/model-tuner/pkg/core"
	estimator "github.com/llm-inferno/model-tuner/pkg/estimator"
	"github.com/llm-inferno/model-tuner/pkg/utils"
)

// TunerService groups replica metrics by (model, accelerator), runs EKF tuning per group,
// maintains a ParameterStore for state continuity, and returns updated ModelData.
type TunerService struct {
	paramStore        *ParameterStore
	warmUpCycles      int
	estimators        map[string]*estimator.InitEstimator
	initObs           int
	holdBack          bool
	useSliding        bool
	windowSize        int
	residualThreshold float64
	slidingEstimators map[string]*estimator.SlidingWindowEstimator
	initFitThreshold  float64
	ekfFallbacks      map[string]bool
}

// NewTunerService creates a TunerService with an empty ParameterStore.
func NewTunerService(warmUpCycles, initObs int, holdBack bool, useSliding bool, windowSize int, residualThreshold, initFitThreshold float64) *TunerService {
	return &TunerService{
		paramStore:        NewParameterStore(),
		warmUpCycles:      warmUpCycles,
		estimators:        make(map[string]*estimator.InitEstimator),
		initObs:           initObs,
		holdBack:          holdBack,
		useSliding:        useSliding,
		windowSize:        windowSize,
		residualThreshold: residualThreshold,
		slidingEstimators: make(map[string]*estimator.SlidingWindowEstimator),
		initFitThreshold:  initFitThreshold,
		ekfFallbacks:      make(map[string]bool),
	}
}

func (ts *TunerService) estimatorFor(key string) *estimator.InitEstimator {
	if ie, ok := ts.estimators[key]; ok {
		return ie
	}
	ie := estimator.NewInitEstimator(ts.initObs, ts.holdBack)
	ts.estimators[key] = ie
	return ie
}

func (ts *TunerService) slidingEstimatorFor(key string, ie *estimator.InitEstimator) *estimator.SlidingWindowEstimator {
	if swe, ok := ts.slidingEstimators[key]; ok {
		return swe
	}
	swe := estimator.NewSlidingWindowEstimator(ts.windowSize, ts.initObs, ts.residualThreshold)
	swe.SeedFromEstimator(ie)
	if fitted, err := ie.Fit(); err == nil {
		fv := ie.LastFitFuncValue()
		if ts.initFitThreshold > 0 && fv > ts.initFitThreshold {
			slog.Warn("poor init fit: falling back to EKF for this pair",
				"key", key, "funcValue", fv, "threshold", ts.initFitThreshold)
			ts.ekfFallbacks[key] = true
			return swe
		}
		swe.SeedLastFit(fitted)
	} else if ts.initFitThreshold > 0 {
		slog.Warn("init fit error: falling back to EKF for this pair", "key", key, "err", err)
		ts.ekfFallbacks[key] = true
		return swe
	}
	ts.slidingEstimators[key] = swe
	return swe
}

func (ts *TunerService) tuneGroupSliding(model, accelerator, key string, ie *estimator.InitEstimator, env *core.EnvironmentPrefillDecode) error {
	_, alreadyExists := ts.slidingEstimators[key]
	swe := ts.slidingEstimatorFor(key, ie)

	if ts.ekfFallbacks[key] {
		return fmt.Errorf("EKF fallback active for %s/%s: poor init fit (funcValue > %.1f)",
			model, accelerator, ts.initFitThreshold)
	}

	if alreadyExists {
		swe.AddObservation(env)
	}

	if !swe.IsReady() {
		slog.Info("sliding window filling",
			"model", model, "accelerator", accelerator,
			"count", swe.Len(), "windowSize", ts.windowSize)
		return fmt.Errorf("sliding window filling for %s/%s (%d/%d)",
			model, accelerator, swe.Len(), ts.windowSize)
	}

	fitted, err := swe.Fit()
	if err != nil {
		return fmt.Errorf("SlidingWindowEstimator.Fit for %s/%s: %w", model, accelerator, err)
	}

	updateCount := 0
	if existing := ts.paramStore.Get(model, accelerator); existing != nil {
		updateCount = existing.UpdateCount
	}
	ts.paramStore.Set(model, accelerator, &LearnedParameters{
		Alpha:       float32(fitted[0]),
		Beta:        float32(fitted[1]),
		Gamma:       float32(fitted[2]),
		NIS:         0,
		UpdateCount: updateCount + 1,
		LastUpdated: time.Now(),
	})
	slog.Info("sliding-window tuned parameters",
		"model", model, "accelerator", accelerator,
		"alpha", fitted[0], "beta", fitted[1], "gamma", fitted[2],
		"updateCount", updateCount+1)
	return nil
}

// Tune accepts per-replica ServerSpecs, runs EKF or SWNM tuning for each
// (model, accelerator) group, and returns updated ModelData with tuned alpha/beta/gamma.
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
	ie := ts.estimatorFor(key)
	ie.AddObservation(envs[0])

	if !ie.IsReady() {
		slog.Info("collecting initial observations",
			"model", model, "accelerator", accelerator,
			"count", ie.ObsCount(), "minObs", ie.MinObs())
		return fmt.Errorf("collecting initial observations for %s/%s (%d/%d)",
			model, accelerator, ie.ObsCount(), ie.MinObs())
	}

	if ts.useSliding && !ts.ekfFallbacks[key] {
		return ts.tuneGroupSliding(model, accelerator, key, ie, envs[0])
	}

	var fitInitState []float64
	if ts.paramStore.Get(model, accelerator) == nil && !ie.FitDone() {
		var fitErr error
		fitInitState, fitErr = ie.Fit()
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

func (ts *TunerService) createTuner(model, accelerator string, firstEnv *core.EnvironmentPrefillDecode, fitInitState []float64) (*core.Tuner, error) {
	existing := ts.paramStore.Get(model, accelerator)

	configData, err := utils.LoadConfigForServer(config.DefaultConfigType)
	if err != nil {
		return nil, fmt.Errorf("load config for %s: %w", model, err)
	}

	if existing != nil {
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
		if fitInitState != nil {
			setInitState(&configData.ModelData, fitInitState)
		} else if initState := estimator.GuessInitState(firstEnv); initState != nil {
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

func setInitState(md *config.ModelData, initState []float64) {
	md.InitState = initState
	md.MinState = make([]float64, len(initState))
	md.MaxState = make([]float64, len(initState))
	for i, v := range initState {
		md.MinState[i] = math.Max(v/config.DefaultInitStateFactor, config.DefaultInitStateMinEpsilon)
		md.MaxState[i] = v * config.DefaultInitStateFactor
	}
}

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

// IsWarmingUp returns true if any known pair has not yet completed its init or warm-up phase.
func (ts *TunerService) IsWarmingUp() bool {
	for _, ie := range ts.estimators {
		if !ie.IsReady() && ie.HoldBack() {
			return true
		}
	}
	if ts.useSliding {
		for key, ie := range ts.estimators {
			if !ie.IsReady() {
				continue
			}
			if ts.ekfFallbacks[key] {
				continue
			}
			swe, ok := ts.slidingEstimators[key]
			if !ok || !swe.IsReady() {
				return true
			}
		}
		if ts.warmUpCycles > 0 {
			for _, params := range ts.paramStore.GetAll() {
				if params.UpdateCount < ts.warmUpCycles {
					return true
				}
			}
		}
		return false
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
// from the ParameterStore for any matching (name, accelerator) pairs.
func (ts *TunerService) Merge(modelData *optconfig.ModelData) *optconfig.ModelData {
	if modelData == nil {
		modelData = &optconfig.ModelData{}
	}

	allParams := ts.paramStore.GetAll()
	matched := make(map[string]bool, len(allParams))

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
