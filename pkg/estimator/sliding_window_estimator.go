package estimator

import (
	"fmt"
	"log/slog"
	"math"

	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
	"gonum.org/v1/gonum/optimize"

	"github.com/llm-inferno/model-tuner/pkg/core"
)

// SlidingWindowEstimator maintains a fixed-capacity circular buffer of recent observations
// and re-runs Nelder-Mead on every Fit() call to produce fresh [α,β,γ] estimates.
type SlidingWindowEstimator struct {
	window            []fitObservation
	windowSize        int
	minObs            int
	residualThreshold float64
	lastFit           []float64
}

// NewSlidingWindowEstimator creates a SlidingWindowEstimator with the given window capacity,
// minimum observations before IsReady(), and residual outlier rejection threshold.
func NewSlidingWindowEstimator(windowSize, minObs int, residualThreshold float64) *SlidingWindowEstimator {
	if windowSize < 1 {
		windowSize = 1
	}
	if minObs < 1 {
		minObs = 1
	}
	if minObs > windowSize {
		minObs = windowSize
	}
	return &SlidingWindowEstimator{
		windowSize:        windowSize,
		minObs:            minObs,
		residualThreshold: residualThreshold,
	}
}

// SeedLastFit sets the warm-start x0 for the first Fit() call, e.g. from InitEstimator.Fit().
func (swe *SlidingWindowEstimator) SeedLastFit(x []float64) {
	if len(x) > 0 {
		swe.lastFit = x
	}
}

// Seed pre-fills the window with raw observations. Oldest entries are evicted when the
// seed exceeds windowSize.
func (swe *SlidingWindowEstimator) Seed(obs []fitObservation) {
	for _, o := range obs {
		swe.window = append(swe.window, o)
		if len(swe.window) > swe.windowSize {
			swe.window = swe.window[1:]
		}
	}
}

// SeedFromEstimator pre-fills the window from an InitEstimator's collected observations.
// Callers outside pkg/estimator use this instead of Seed to avoid accessing the unexported
// fitObservation type directly.
func (swe *SlidingWindowEstimator) SeedFromEstimator(ie *InitEstimator) {
	swe.Seed(ie.observations)
}

// AddObservation appends a new operating-point observation.
// Oldest entry is evicted when the window is at capacity.
func (swe *SlidingWindowEstimator) AddObservation(env *core.EnvironmentPrefillDecode) {
	if env == nil || !env.Valid() {
		return
	}
	obs := fitObservation{
		Lambda:          float64(env.Lambda),
		MaxBatch:        env.MaxBatchSize,
		MaxQueueSize:    env.MaxQueueSize,
		AvgInputTokens:  env.AvgInputTokens,
		AvgOutputTokens: env.AvgOutputTokens,
		AvgTTFT:         float64(env.AvgTTFT),
		AvgITL:          float64(env.AvgITL),
	}
	swe.window = append(swe.window, obs)
	if len(swe.window) > swe.windowSize {
		swe.window = swe.window[1:]
	}
}

// IsReady returns true once the window holds at least minObs observations.
func (swe *SlidingWindowEstimator) IsReady() bool {
	return len(swe.window) >= swe.minObs
}

// Len returns the number of observations currently in the window.
func (swe *SlidingWindowEstimator) Len() int {
	return len(swe.window)
}

// Fit runs Nelder-Mead on the current window, performs one residual-based outlier
// rejection pass, and refits if any observations were dropped.
func (swe *SlidingWindowEstimator) Fit() ([]float64, error) {
	if len(swe.window) == 0 {
		return nil, fmt.Errorf("no observations in window")
	}

	x0 := swe.lastFit
	if x0 == nil {
		x0 = GuessInitState(swe.window[len(swe.window)-1].toEnv())
	}
	if x0 == nil {
		x0 = []float64{5.0, 0.05, 0.0005}
	}

	fitted, err := swe.fitWithX0(x0, swe.window)
	if err != nil {
		return nil, err
	}

	cleaned := swe.filterOutliers(swe.window, fitted)
	if len(cleaned) < len(swe.window) {
		slog.Info("SlidingWindowEstimator: outliers removed, refitting",
			"total", len(swe.window), "kept", len(cleaned))
		fitted, err = swe.fitWithX0(fitted, cleaned)
		if err != nil {
			return nil, err
		}
	}

	swe.lastFit = fitted
	return fitted, nil
}

// filterOutliers removes the single observation with the largest residual if that residual
// exceeds swe.residualThreshold.
func (swe *SlidingWindowEstimator) filterOutliers(obs []fitObservation, x []float64) []fitObservation {
	if swe.residualThreshold <= 0 {
		return obs
	}
	worstIdx := -1
	worstResidual := swe.residualThreshold
	for i, o := range obs {
		r := swe.residual(o, x)
		if r > worstResidual {
			worstResidual = r
			worstIdx = i
		}
	}
	if worstIdx < 0 {
		return obs
	}
	kept := make([]fitObservation, 0, len(obs)-1)
	kept = append(kept, obs[:worstIdx]...)
	kept = append(kept, obs[worstIdx+1:]...)
	return kept
}

// residual returns sqrt(dTTFT² + dITL²) for one observation evaluated at params x=[α,β,γ].
func (swe *SlidingWindowEstimator) residual(obs fitObservation, x []float64) float64 {
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		return math.MaxFloat64
	}
	qConfig := &analyzer.Configuration{
		MaxBatchSize: obs.MaxBatch,
		MaxQueueSize: obs.MaxQueueSize,
		ServiceParms: &analyzer.ServiceParms{
			Alpha: float32(x[0]),
			Beta:  float32(x[1]),
			Gamma: float32(x[2]),
		},
	}
	requestSize := &analyzer.RequestSize{
		AvgInputTokens:  obs.AvgInputTokens,
		AvgOutputTokens: obs.AvgOutputTokens,
	}
	qa, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
	if err != nil {
		return math.MaxFloat64
	}
	metrics, err := qa.Analyze(float32(obs.Lambda / 60))
	if err != nil {
		return math.MaxFloat64
	}
	ttftModel := float64(metrics.AvgTTFT)
	itlModel := float64(metrics.AvgTokenTime)
	if obs.AvgTTFT <= 0 || obs.AvgITL <= 0 || ttftModel <= 0 || itlModel <= 0 {
		return math.MaxFloat64
	}
	dTTFT := (ttftModel - obs.AvgTTFT) / obs.AvgTTFT
	dITL := (itlModel - obs.AvgITL) / obs.AvgITL
	return math.Sqrt(dTTFT*dTTFT + dITL*dITL)
}

// fitWithX0 runs Nelder-Mead on obs starting from x0.
func (swe *SlidingWindowEstimator) fitWithX0(x0 []float64, obs []fitObservation) ([]float64, error) {
	if len(obs) == 0 {
		return nil, fmt.Errorf("no observations to fit")
	}

	scale := make([]float64, len(x0))
	scaledX0 := make([]float64, len(x0))
	for i, v := range x0 {
		scale[i] = v
		scaledX0[i] = 1.0
	}

	scaledObjective := func(p []float64) float64 {
		unscaled := make([]float64, len(p))
		for i := range p {
			unscaled[i] = p[i] * scale[i]
		}
		return swe.objective(obs, unscaled)
	}

	problem := optimize.Problem{Func: scaledObjective}
	settings := &optimize.Settings{FuncEvaluations: 500}
	result, err := optimize.Minimize(problem, scaledX0, settings, &optimize.NelderMead{})
	if err != nil {
		slog.Warn("SlidingWindowEstimator: Nelder-Mead pre-flight error, using GuessInitState fallback", "err", err)
		if fallback := GuessInitState(obs[len(obs)-1].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead failed and GuessInitState returned nil: %w", err)
	}

	switch result.Status {
	case optimize.Success, optimize.FunctionConvergence, optimize.FunctionEvaluationLimit:
	default:
		slog.Warn("SlidingWindowEstimator: unexpected Nelder-Mead termination", "status", result.Status)
		if fallback := GuessInitState(obs[len(obs)-1].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead unexpected status %v", result.Status)
	}

	unscaled := make([]float64, len(result.X))
	for i := range result.X {
		unscaled[i] = result.X[i] * scale[i]
	}
	x := unscaled
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		slog.Warn("SlidingWindowEstimator: non-positive params, using GuessInitState fallback",
			"alpha", x[0], "beta", x[1], "gamma", x[2])
		if fallback := GuessInitState(obs[len(obs)-1].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead returned non-positive params")
	}

	slog.Info("SlidingWindowEstimator: Fit complete",
		"alpha", x[0], "beta", x[1], "gamma", x[2],
		"observations", len(obs), "funcValue", result.F)
	return x, nil
}

// objective returns sum of relative squared errors in TTFT and ITL across obs for params x=[α,β,γ].
func (swe *SlidingWindowEstimator) objective(obs []fitObservation, x []float64) float64 {
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		return math.MaxFloat64 / 2
	}
	var total float64
	for _, o := range obs {
		qConfig := &analyzer.Configuration{
			MaxBatchSize: o.MaxBatch,
			MaxQueueSize: o.MaxQueueSize,
			ServiceParms: &analyzer.ServiceParms{
				Alpha: float32(x[0]),
				Beta:  float32(x[1]),
				Gamma: float32(x[2]),
			},
		}
		requestSize := &analyzer.RequestSize{
			AvgInputTokens:  o.AvgInputTokens,
			AvgOutputTokens: o.AvgOutputTokens,
		}
		qa, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
		if err != nil {
			return math.MaxFloat64 / 2
		}
		metrics, err := qa.Analyze(float32(o.Lambda / 60))
		if err != nil {
			return math.MaxFloat64 / 2
		}
		ttftModel := float64(metrics.AvgTTFT)
		itlModel := float64(metrics.AvgTokenTime)
		ttftObs := o.AvgTTFT
		itlObs := o.AvgITL
		if ttftObs <= 0 || itlObs <= 0 || ttftModel <= 0 || itlModel <= 0 {
			return math.MaxFloat64 / 2
		}
		dTTFT := (ttftModel - ttftObs) / ttftObs
		dITL := (itlModel - itlObs) / itlObs
		total += dTTFT*dTTFT + dITL*dITL
	}
	return total
}
