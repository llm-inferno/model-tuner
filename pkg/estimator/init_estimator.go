package estimator

import (
	"fmt"
	"log/slog"
	"math"

	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
	"gonum.org/v1/gonum/optimize"

	"github.com/llm-inferno/model-tuner/pkg/core"
)

// InitEstimator accumulates observations before the EKF starts and fits initial parameters.
type InitEstimator struct {
	observations     []fitObservation
	minObs           int
	holdBack         bool
	fitDone          bool
	lastFitFuncValue float64
}

// NewInitEstimator creates an InitEstimator with the given minimum observation count and hold-back flag.
func NewInitEstimator(minObs int, holdBack bool) *InitEstimator {
	if minObs < 1 {
		minObs = 1
	}
	return &InitEstimator{
		minObs:   minObs,
		holdBack: holdBack,
	}
}

// AddObservation records a new operating-point observation from an environment.
func (ie *InitEstimator) AddObservation(env *core.EnvironmentPrefillDecode) {
	if env == nil || !env.Valid() {
		return
	}
	ie.observations = append(ie.observations, fitObservation{
		Lambda:          float64(env.Lambda),
		MaxBatch:        env.MaxBatchSize,
		MaxQueueSize:    env.MaxQueueSize,
		AvgInputTokens:  env.AvgInputTokens,
		AvgOutputTokens: env.AvgOutputTokens,
		AvgTTFT:         float64(env.AvgTTFT),
		AvgITL:          float64(env.AvgITL),
	})
}

// IsReady returns true once at least minObs observations have been collected.
func (ie *InitEstimator) IsReady() bool {
	return len(ie.observations) >= ie.minObs
}

// HoldBack returns true if the controller should report warmingUp=true during collection.
func (ie *InitEstimator) HoldBack() bool {
	return ie.holdBack
}

// ObsCount returns the number of observations accumulated so far.
func (ie *InitEstimator) ObsCount() int { return len(ie.observations) }

// MinObs returns the minimum number of observations required before Fit() can run.
func (ie *InitEstimator) MinObs() int { return ie.minObs }

// FitDone returns true if Fit() has already been called (regardless of success or failure).
func (ie *InitEstimator) FitDone() bool { return ie.fitDone }

// LastFitFuncValue returns the Nelder-Mead objective value from the most recent Fit() call.
// Returns 0 if Fit() has not been called yet, math.MaxFloat64 if the fit fell back to GuessInitState.
func (ie *InitEstimator) LastFitFuncValue() float64 { return ie.lastFitFuncValue }

// Fit runs Nelder-Mead minimisation over all accumulated observations to find the
// (alpha, beta, gamma) that best explains all K observations jointly via the full
// queueing model. Returns [alpha, beta, gamma] or an error.
// Falls back to GuessInitState on the first observation if the fit fails.
func (ie *InitEstimator) Fit() ([]float64, error) {
	if len(ie.observations) == 0 {
		return nil, fmt.Errorf("no observations to fit")
	}

	x0 := GuessInitState(ie.observations[0].toEnv())
	if x0 == nil {
		x0 = []float64{5.0, 0.05, 0.0005}
	}

	result, err := ie.fitWithX0(x0)
	ie.fitDone = true
	return result, err
}

// fitWithX0 runs the Nelder-Mead optimisation starting from the given x0.
func (ie *InitEstimator) fitWithX0(x0 []float64) ([]float64, error) {
	if len(ie.observations) == 0 {
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
		return ie.objective(unscaled)
	}

	problem := optimize.Problem{Func: scaledObjective}
	settings := &optimize.Settings{FuncEvaluations: 500}
	result, err := optimize.Minimize(problem, scaledX0, settings, &optimize.NelderMead{})
	if err != nil {
		ie.lastFitFuncValue = math.MaxFloat64
		slog.Warn("InitEstimator: Nelder-Mead pre-flight error, using GuessInitState fallback", "err", err)
		if fallback := GuessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead failed and GuessInitState returned nil: %w", err)
	}

	switch result.Status {
	case optimize.Success, optimize.FunctionConvergence, optimize.FunctionEvaluationLimit:
	default:
		ie.lastFitFuncValue = math.MaxFloat64
		slog.Warn("InitEstimator: unexpected Nelder-Mead termination status, using GuessInitState fallback",
			"status", result.Status)
		if fallback := GuessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead unexpected status %v and GuessInitState returned nil", result.Status)
	}

	unscaled := make([]float64, len(result.X))
	for i := range result.X {
		unscaled[i] = result.X[i] * scale[i]
	}
	x := unscaled
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		ie.lastFitFuncValue = math.MaxFloat64
		slog.Warn("InitEstimator: Nelder-Mead returned non-positive params, using GuessInitState fallback",
			"alpha", x[0], "beta", x[1], "gamma", x[2])
		if fallback := GuessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead returned non-positive params and GuessInitState returned nil")
	}

	ie.lastFitFuncValue = result.F
	slog.Info("InitEstimator: Fit complete",
		"alpha", x[0], "beta", x[1], "gamma", x[2],
		"observations", len(ie.observations), "funcValue", result.F)
	return x, nil
}

// objective computes the sum of relative squared errors in TTFT and ITL across all
// stored observations, evaluated using the full queueing model for the trial params x=[α,β,γ].
func (ie *InitEstimator) objective(x []float64) float64 {
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		return math.MaxFloat64 / 2
	}

	var total float64
	for _, obs := range ie.observations {
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
			return math.MaxFloat64 / 2
		}
		metrics, err := qa.Analyze(float32(obs.Lambda / 60))
		if err != nil {
			return math.MaxFloat64 / 2
		}

		ttftModel := float64(metrics.AvgTTFT)
		itlModel := float64(metrics.AvgTokenTime)
		ttftObs := obs.AvgTTFT
		itlObs := obs.AvgITL

		if ttftObs <= 0 || itlObs <= 0 || ttftModel <= 0 || itlModel <= 0 {
			return math.MaxFloat64 / 2
		}

		dTTFT := (ttftModel - ttftObs) / ttftObs
		dITL := (itlModel - itlObs) / itlObs
		total += dTTFT*dTTFT + dITL*dITL
	}
	return total
}
