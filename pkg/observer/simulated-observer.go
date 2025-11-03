/*
Simulated Observer simulates randomness in wait and token time
*/

package observer

import (
	"math"
	"math/rand/v2"

	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
)

type SimulatedObserver struct {
	BaseObserver

	rpm          []float32
	inputTokens  []float32
	outputTokens []float32
	alpha, beta  []float32
	gamma, delta []float32
	percentNoise []float32
	maxBatchSize []int

	timeStep int
	maxStep  int
}

func NewSimulatedObserver(rpm,
	inputTokens, outputTokens,
	alpha, beta, gamma, delta,
	percentNoise []float32, maxBatchSize []int) *SimulatedObserver {

	maxStep := max(len(rpm), len(inputTokens), len(outputTokens),
		len(alpha), len(beta), len(gamma), len(delta),
		len(percentNoise), len(maxBatchSize)) - 1
	if maxStep < 0 {
		return nil
	}
	return &SimulatedObserver{
		BaseObserver: BaseObserver{},

		rpm:          rpm,
		inputTokens:  inputTokens,
		outputTokens: outputTokens,
		alpha:        alpha,
		beta:         beta,
		gamma:        gamma,
		delta:        delta,
		maxBatchSize: maxBatchSize,
		percentNoise: percentNoise,
		timeStep:     0,
		maxStep:      maxStep,
	}
}

func (obs *SimulatedObserver) GetEnvironment() core.Environment {

	// get parameters at current time step
	i := min(obs.timeStep, obs.maxStep)
	rpm := obs.rpm[i]
	inputTokens := obs.inputTokens[i]
	outputTokens := obs.outputTokens[i]
	alpha := obs.alpha[i]
	beta := obs.beta[i]
	gamma := obs.gamma[i]
	delta := obs.delta[i]
	pctNoise := obs.percentNoise[i]
	maxBatchSize := obs.maxBatchSize[i]

	// create queueing model
	qConfig := &analyzer.Configuration{
		MaxBatchSize: maxBatchSize,
		MaxQueueSize: 10 * maxBatchSize,
		ServiceParms: &analyzer.ServiceParms{
			Prefill: &analyzer.PrefillParms{
				Gamma: gamma,
				Delta: delta,
			},
			Decode: &analyzer.DecodeParms{
				Alpha: alpha,
				Beta:  beta,
			},
		},
	}

	requestSize := &analyzer.RequestSize{
		AvgInputTokens:  int(math.Ceil(float64(inputTokens))),
		AvgOutputTokens: int(math.Ceil(float64(outputTokens))),
	}
	queueAnalyzer, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
	if err != nil {
		return nil
	}

	// request per msec

	if rpm *= 1 + pctNoise*(2*rand.Float32()-1); rpm < 0 {
		rpm = 0
	}

	metrics, err := queueAnalyzer.Analyze(rpm / 60)
	if err != nil {
		return nil
	}

	// fmt.Println("sim-queueAnalyzer: " + queueAnalyzer.String())
	// fmt.Println("sim-metrics: " + metrics.String())

	avgWaitTime := metrics.AvgWaitTime
	avgPrefillTime := metrics.AvgPrefillTime
	avgTokenDecodeTime := metrics.AvgTokenTime

	avgConcurrency := metrics.AvgNumInServ

	// add noise
	if avgWaitTime *= 1 + pctNoise*(2*rand.Float32()-1); avgWaitTime < 0 {
		avgWaitTime = 0
	}
	if avgPrefillTime *= 1 + pctNoise*(2*rand.Float32()-1); avgPrefillTime < 0 {
		avgPrefillTime = 0
	}
	if avgTokenDecodeTime *= 1 + pctNoise*(2*rand.Float32()-1); avgTokenDecodeTime < 0 {
		avgTokenDecodeTime = 0
	}

	obs.timeStep++

	// create environment with input parameters
	env := core.NewEnvironmentPrefillDecode(rpm, avgConcurrency, avgWaitTime, maxBatchSize,
		inputTokens, outputTokens, avgPrefillTime, avgTokenDecodeTime)
	return env
}
