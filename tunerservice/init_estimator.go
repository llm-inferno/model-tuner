package tunerservice

import (
	"fmt"
	"log/slog"
	"math"

	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
	"gonum.org/v1/gonum/optimize"

	"github.com/llm-inferno/model-tuner/pkg/core"
)

// fitObservation holds a single operating-point snapshot for use in the Fit objective.
type fitObservation struct {
	Lambda          float64 // arrival rate, req/min
	MaxBatch        int
	AvgInputTokens  float32
	AvgOutputTokens float32
	AvgTTFT         float64 // ms
	AvgITL          float64 // ms
}

// toEnv converts a fitObservation to an EnvironmentPrefillDecode for use with guessInitState.
func (fo *fitObservation) toEnv() *core.EnvironmentPrefillDecode {
	return core.NewEnvironmentPrefillDecode(
		float32(fo.Lambda),
		0, // BatchSize not used
		0, // AvgQueueTime not available
		fo.MaxBatch,
		fo.AvgInputTokens,
		fo.AvgOutputTokens,
		float32(fo.AvgTTFT),
		float32(fo.AvgITL),
	)
}

// InitEstimator accumulates observations before the EKF starts and fits initial parameters.
type InitEstimator struct {
	observations []fitObservation
	minObs       int  // minimum K before Fit(); K<3 may underconstraint the 3-param fit
	holdBack     bool
	fitDone      bool // set after the first Fit() call to prevent repeated fitting
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

// Fit runs Nelder-Mead minimisation over all accumulated observations to find the
// (alpha, beta, gamma) that best explains all K observations jointly via the full
// queueing model. Returns [alpha, beta, gamma] or an error.
// Falls back to guessInitState on the first observation if the fit fails.
func (ie *InitEstimator) Fit() ([]float64, error) {
	if len(ie.observations) == 0 {
		return nil, fmt.Errorf("no observations to fit")
	}

	// Starting point: guessInitState on first observation; fall back to a safe default.
	x0 := guessInitState(ie.observations[0].toEnv())
	if x0 == nil {
		x0 = []float64{5.0, 0.05, 0.0005}
	}

	// Scale variables by x0 so the optimizer sees O(1) quantities.
	// Nelder-Mead builds its initial simplex with a fixed absolute offset (SimplexSize=0.05)
	// per dimension. Without scaling, gamma (~0.00005) gets a 100,000% perturbation while
	// alpha (~5) gets only 1% — a degenerate simplex. x0 is always positive here.
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
	// 500 evaluations bounds worst-case latency (~30ms for K=5 obs) while allowing convergence
	// for a 3-parameter problem. Nelder-Mead typically terminates via FunctionEvaluationLimit.
	settings := &optimize.Settings{FuncEvaluations: 500}
	result, err := optimize.Minimize(problem, scaledX0, settings, &optimize.NelderMead{})
	ie.fitDone = true
	if err != nil {
		// Genuine pre-flight failure (ill-formed problem or settings); result may be nil.
		slog.Warn("InitEstimator: Nelder-Mead pre-flight error, using guessInitState fallback", "err", err)
		if fallback := guessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead failed and guessInitState returned nil: %w", err)
	}

	// gonum always returns a non-nil result when err==nil; verify termination was acceptable.
	switch result.Status {
	case optimize.Success, optimize.FunctionConvergence, optimize.FunctionEvaluationLimit:
		// acceptable termination
	default:
		slog.Warn("InitEstimator: unexpected Nelder-Mead termination status, using guessInitState fallback",
			"status", result.Status)
		if fallback := guessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead unexpected status %v and guessInitState returned nil", result.Status)
	}

	// Unscale the result back to raw parameter space.
	unscaled := make([]float64, len(result.X))
	for i := range result.X {
		unscaled[i] = result.X[i] * scale[i]
	}
	x := unscaled
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		slog.Warn("InitEstimator: Nelder-Mead returned non-positive params, using guessInitState fallback",
			"alpha", x[0], "beta", x[1], "gamma", x[2])
		if fallback := guessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead returned non-positive params and guessInitState returned nil")
	}

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
			MaxQueueSize: 10 * obs.MaxBatch,
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
		metrics, err := qa.Analyze(float32(obs.Lambda / 60)) // RPM → req/sec
		if err != nil {
			return math.MaxFloat64 / 2
		}

		ttftModel := float64(metrics.AvgTTFT)
		itlModel := float64(metrics.AvgTokenTime) // AvgTokenTime = ITL
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
