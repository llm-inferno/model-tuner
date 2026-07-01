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
	paramStore         *ParameterStore
	warmUpCycles       int
	estimators         map[string]*estimator.InitEstimator
	initObs            int
	holdBack           bool
	useSliding         bool
	windowSize         int
	residualThreshold  float64
	slidingEstimators  map[string]*estimator.SlidingWindowEstimator
	initFitThreshold   float64
	maxConditionNumber float64
	ekfFallbacks       map[string]bool
	calibrated         map[string]bool
	coldSeed           []float64
	coldSeedLoaded     bool
}

// coldStartSeed returns the cold-start anchor [alpha, beta, gamma] used by the estimators'
// GuessInitState fallback (issue #17): the config initState. Loaded once and cached; nil on
// load failure (estimators then keep their legacy heuristic).
func (ts *TunerService) coldStartSeed() []float64 {
	if ts.coldSeedLoaded {
		return ts.coldSeed
	}
	ts.coldSeedLoaded = true
	if configData, err := utils.LoadConfigForServer(config.DefaultConfigType); err == nil {
		ts.coldSeed = configData.ModelData.InitState
	} else {
		slog.Warn("cold-start seed unavailable: config load failed, estimators use legacy guess", "err", err)
	}
	return ts.coldSeed
}

// SetMaxConditionNumber sets the identifiability guard threshold applied to every estimator
// created thereafter (> 0 enables; <= 0 disables). Wire this from configuration before the
// service handles any tune requests.
func (ts *TunerService) SetMaxConditionNumber(k float64) {
	ts.maxConditionNumber = k
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
		calibrated:        make(map[string]bool),
	}
}

func (ts *TunerService) estimatorFor(key string) *estimator.InitEstimator {
	if ie, ok := ts.estimators[key]; ok {
		return ie
	}
	ie := estimator.NewInitEstimator(ts.initObs, ts.holdBack)
	ie.SetMaxConditionNumber(ts.maxConditionNumber)
	ie.SetSeed(ts.coldStartSeed())
	ts.estimators[key] = ie
	return ie
}

func (ts *TunerService) slidingEstimatorFor(key string, ie *estimator.InitEstimator) *estimator.SlidingWindowEstimator {
	if swe, ok := ts.slidingEstimators[key]; ok {
		return swe
	}
	swe := estimator.NewSlidingWindowEstimator(ts.windowSize, ts.initObs, ts.residualThreshold)
	swe.SetMaxConditionNumber(ts.maxConditionNumber)
	swe.SetSeed(ts.coldStartSeed())
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

	// Identifiability gap (issue #19): the fit was ill-conditioned and SWNM held the last good
	// fit (`fitted`). Rather than emit that stale value, run one transient EKF predict+update
	// seeded at the held good fit. The EKF regularizes via its prior — in the unobservable
	// beta/gamma direction the Kalman gain is ~0, so it holds them near the seed and only nudges
	// the observable combination to fit the offending point. Worst case it returns the seed
	// (== today's hold); it never emits collapsed or inflated params. The tuner is discarded and
	// SWNM resumes next cycle.
	if swe.HeldLastGoodFit() {
		if excursed := ts.ekfExcursion(model, accelerator, fitted, env); excursed != nil {
			fitted = excursed
		}
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

// ekfExcursion runs a single seeded EKF predict+update for the transient SWNM->EKF excursion
// (issue #19), seeded at the held good SWNM fit. It returns the EKF-updated [alpha,beta,gamma]
// on success, or nil to signal the caller to keep the held fit (the safe fallback). skipNIS is
// true so the offending collinear point is absorbed; the seed and seed-derived bounds anchor
// the result, and the unobservable beta/gamma direction is held by the near-zero Kalman gain.
func (ts *TunerService) ekfExcursion(model, accelerator string, seed []float64, env *core.EnvironmentPrefillDecode) []float64 {
	tuner, err := ts.newSeededTuner(seed, env)
	if err != nil {
		slog.Warn("EKF excursion: tuner construction failed, holding SWNM fit",
			"model", model, "accelerator", accelerator, "err", err)
		return nil
	}
	results, err := tuner.RunWithValidation(env, true)
	if err != nil {
		slog.Warn("EKF excursion: run error, holding SWNM fit",
			"model", model, "accelerator", accelerator, "err", err)
		return nil
	}
	if results.ValidationFailed {
		slog.Info("EKF excursion: update rejected, holding SWNM fit",
			"model", model, "accelerator", accelerator)
		return nil
	}
	excursed := []float64{
		float64(results.ServiceParms.Alpha),
		float64(results.ServiceParms.Beta),
		float64(results.ServiceParms.Gamma),
	}
	slog.Info("EKF excursion: ill-conditioned SWNM fit, emitting seeded EKF update",
		"model", model, "accelerator", accelerator,
		"seedAlpha", seed[0], "seedBeta", seed[1], "seedGamma", seed[2],
		"alpha", excursed[0], "beta", excursed[1], "gamma", excursed[2])
	return excursed
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
			return attachQueueModelObsFunc(tuner)
		}
	} else {
		if fitInitState != nil {
			setInitState(&configData.ModelData, fitInitState)
		} else if initState := estimator.GuessInitState(firstEnv, configData.ModelData.InitState); initState != nil {
			setInitState(&configData.ModelData, initState)
		}
	}

	tuner, err := core.NewTuner(configData, firstEnv)
	if err != nil {
		return nil, err
	}
	return attachQueueModelObsFunc(tuner)
}

// attachQueueModelObsFunc wires the prefill-decode queue-model observation function h(x) onto
// a freshly constructed Tuner and returns it.
func attachQueueModelObsFunc(tuner *core.Tuner) (*core.Tuner, error) {
	if err := tuner.SetObservationFunc(core.NewQueueModelSystemFuncCreatorPrefillDecode(tuner)); err != nil {
		return nil, err
	}
	return tuner, nil
}

// newSeededTuner builds a fresh, single-use EKF Tuner whose state mean is seeded at the given
// [alpha,beta,gamma] values, with the queue-model observation function attached. setInitState
// also derives MinState/MaxState from the seed, so the filter is bounded around the seed. Used
// for the transient SWNM->EKF excursion (issue #19): one predict+update from a known-good state.
func (ts *TunerService) newSeededTuner(seed []float64, env *core.EnvironmentPrefillDecode) (*core.Tuner, error) {
	configData, err := utils.LoadConfigForServer(config.DefaultConfigType)
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	if seed != nil {
		// Copy: the EKF state aliases ModelData.InitState and is mutated in place by the
		// predict/update, so passing the caller's slice (e.g. SWNM's held lastFit) directly
		// would corrupt the frozen prior the excursion is meant to re-anchor to.
		setInitState(&configData.ModelData, append([]float64(nil), seed...))
	}
	tuner, err := core.NewTuner(configData, env)
	if err != nil {
		return nil, err
	}
	return attachQueueModelObsFunc(tuner)
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
			PerfParms: optconfig.PerfParms{
				Alpha: params.Alpha,
				Beta:  params.Beta,
				Gamma: params.Gamma,
			},
		})
	}

	return &optconfig.ModelData{PerfData: result}
}

// CalibrationStatus reports, for one (model, accelerator) pair the tuner has seen, the facts the
// controller's calibration trigger needs: whether warm-up observations have been collected, the
// identifiability (Jacobian condition number) of the most recent fit, whether a calibration has
// already succeeded for this pair, and the derived NeedsCalibration decision. NeedsCalibration is
// true when warm-up collected enough observations to attempt a fit, that fit was ill-conditioned
// (natural excitation insufficient — beta/gamma unidentifiable), and the pair has not yet been
// calibrated. In-memory: calibration state resets on tuner restart (see project design).
type CalibrationStatus struct {
	Model            string  `json:"model"`
	Accelerator      string  `json:"accelerator"`
	StorePresent     bool    `json:"storePresent"`
	Calibrated       bool    `json:"calibrated"`
	ObsCount         int     `json:"obsCount"`
	ObsTarget        int     `json:"obsTarget"`
	ConditionNumber  float64 `json:"conditionNumber"`
	IllConditioned   bool    `json:"illConditioned"`
	NeedsCalibration bool    `json:"needsCalibration"`
}

// CalibrationStatuses returns the calibration status of every (model, accelerator) pair the tuner
// has begun collecting observations for. Pairs not yet seen by /tune are absent (the controller
// cannot judge excitation before any observation exists).
func (ts *TunerService) CalibrationStatuses() []CalibrationStatus {
	out := make([]CalibrationStatus, 0, len(ts.estimators))
	for key, ie := range ts.estimators {
		model, accelerator := splitKey(key)
		kappa := ie.LastConditionNumber()
		illConditioned := ts.maxConditionNumber > 0 && kappa > ts.maxConditionNumber
		calibrated := ts.calibrated[key]
		// A fit must have run (IsReady ⇒ init observations collected and a fit attempted) before
		// kappa is meaningful; only then can excitation be judged insufficient.
		needs := ie.IsReady() && ie.FitDone() && illConditioned && !calibrated
		out = append(out, CalibrationStatus{
			Model:            model,
			Accelerator:      accelerator,
			StorePresent:     ts.paramStore.Get(model, accelerator) != nil,
			Calibrated:       calibrated,
			ObsCount:         ie.ObsCount(),
			ObsTarget:        ie.MinObs(),
			ConditionNumber:  kappa,
			IllConditioned:   illConditioned,
			NeedsCalibration: needs,
		})
	}
	return out
}

// Calibrate fits (alpha, beta, gamma) from a batch of deliberately-diverse sweep observations
// (benchmarking-on-the-fly) for each (model, accelerator) group, in one shot. Unlike Tune — which
// folds one operating point per cycle into the EKF/sliding estimators — Calibrate fits jointly over
// all swept operating points, so the parameters are identifiable by construction. On success it
// stores the fit (graduated, so warm-up no longer blocks the pair) and seeds the per-pair estimators
// from the sweep, so subsequent Tune cycles track drift from the calibrated point rather than
// re-warming. Reuses the same InitEstimator multi-point Nelder-Mead fit and condition-number guard
// as the normal path. Returns the calibrated ModelData; errors if no group could be calibrated.
func (ts *TunerService) Calibrate(specs []optconfig.ServerSpec) (*optconfig.ModelData, error) {
	groups := groupByModelAccelerator(specs)
	if len(groups) == 0 {
		return nil, fmt.Errorf("no calibration points with active traffic in request")
	}

	// Build the response from only the groups calibrated in THIS call. buildModelData reads params
	// from the store, so passing groups that failed calibration would leak stale params (from an
	// earlier /tune or /calibrate) into the response as if freshly calibrated.
	calibratedGroups := make(map[string][]optconfig.ServerSpec)
	for key, replicas := range groups {
		model, accelerator := splitKey(key)
		if err := ts.calibrateGroup(model, accelerator, key, replicas); err != nil {
			slog.Warn("calibration failed for group", "key", key, "err", err)
			continue
		}
		calibratedGroups[key] = replicas
	}
	if len(calibratedGroups) == 0 {
		return nil, fmt.Errorf("calibration produced no results for any model/accelerator group")
	}

	modelData := ts.buildModelData(calibratedGroups)
	if len(modelData.PerfData) == 0 {
		return nil, fmt.Errorf("calibration produced no results for any model/accelerator group")
	}
	return modelData, nil
}

// calibrateGroup runs a single-shot multi-point fit over one group's sweep observations, stores the
// result, and seeds the per-pair estimators so the normal Tune path continues from the calibrated
// fit. A still-ill-conditioned fit (the sweep grid lacked operating-point spread) is rejected
// rather than stored.
func (ts *TunerService) calibrateGroup(model, accelerator, key string, replicas []optconfig.ServerSpec) error {
	envs := buildEnvironments(replicas)
	if len(envs) < 2 {
		return fmt.Errorf("need >= 2 calibration points for %s/%s, got %d", model, accelerator, len(envs))
	}

	ie := estimator.NewInitEstimator(len(envs), false)
	ie.SetMaxConditionNumber(ts.maxConditionNumber)
	ie.SetSeed(ts.coldStartSeed())
	for _, env := range envs {
		ie.AddObservation(env)
	}

	fitted, err := ie.Fit()
	if err != nil {
		return fmt.Errorf("calibration fit for %s/%s: %w", model, accelerator, err)
	}
	if ts.maxConditionNumber > 0 && ie.LastConditionNumber() > ts.maxConditionNumber {
		return fmt.Errorf("calibration fit for %s/%s ill-conditioned (kappa=%.3g > %.3g): sweep grid lacks operating-point spread",
			model, accelerator, ie.LastConditionNumber(), ts.maxConditionNumber)
	}
	// Reject a poor fit rather than storing it. Fit() reports lastFitFuncValue = math.MaxFloat64 when it
	// fell back to the single-point GuessInitState (Nelder-Mead pre-flight error, unexpected termination,
	// or non-positive params) — those paths leave lastConditionNumber unset, so the condition-number guard
	// above cannot catch them. The MaxFloat64 sentinel is rejected unconditionally; a converged-but-poor
	// fit is rejected against initFitThreshold, matching the sliding-window init path.
	if fv := ie.LastFitFuncValue(); fv == math.MaxFloat64 || (ts.initFitThreshold > 0 && fv > ts.initFitThreshold) {
		return fmt.Errorf("calibration fit for %s/%s poor (funcValue=%.3g): rejecting rather than storing a degenerate calibration",
			model, accelerator, fv)
	}

	// Store graduated so the warm-up gate no longer blocks this pair (UpdateCount >= warmUpCycles).
	ts.paramStore.Set(model, accelerator, &LearnedParameters{
		Alpha:       float32(fitted[0]),
		Beta:        float32(fitted[1]),
		Gamma:       float32(fitted[2]),
		UpdateCount: ts.warmUpCycles,
		LastUpdated: time.Now(),
	})

	// Seed the per-pair estimators from the sweep so subsequent Tune cycles track drift from the
	// calibrated fit (rich warm-up in one shot) rather than re-collecting init observations.
	ts.estimators[key] = ie
	delete(ts.ekfFallbacks, key)
	if ts.useSliding {
		swe := estimator.NewSlidingWindowEstimator(ts.windowSize, ts.initObs, ts.residualThreshold)
		swe.SetMaxConditionNumber(ts.maxConditionNumber)
		swe.SetSeed(ts.coldStartSeed())
		swe.SeedFromEstimator(ie)
		swe.SeedLastFit(fitted)
		ts.slidingEstimators[key] = swe
	}
	ts.calibrated[key] = true

	slog.Info("calibrated parameters (benchmarking-on-the-fly)",
		"model", model, "accelerator", accelerator,
		"alpha", fitted[0], "beta", fitted[1], "gamma", fitted[2],
		"points", len(envs), "conditionNumber", ie.LastConditionNumber())
	return nil
}
