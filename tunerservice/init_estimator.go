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

	problem := optimize.Problem{Func: ie.objective}
	settings := &optimize.Settings{FuncEvaluations: 500}
	result, err := optimize.Minimize(problem, x0, settings, &optimize.NelderMead{})
	if err != nil && result == nil {
		slog.Warn("InitEstimator: Nelder-Mead failed, using guessInitState fallback", "err", err)
		if fallback := guessInitState(ie.observations[0].toEnv()); fallback != nil {
			return fallback, nil
		}
		return nil, fmt.Errorf("Nelder-Mead failed and guessInitState returned nil: %w", err)
	}

	x := result.X
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
