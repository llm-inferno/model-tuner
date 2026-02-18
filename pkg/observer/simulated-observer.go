/*
Simulated Observer simulates randomness in wait and token time
*/

package observer

import (
	"fmt"
	"math/rand/v2"

	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
)

type SimulatedObserver struct {
	BaseObserver

	rpm                       []float32
	inputTokens, outputTokens []float32
	alpha, beta, gamma        []float32
	percentNoise              []float32
	maxBatchSize              []int

	timeStep int
	maxStep  int
}

func NewSimulatedObserver(rpm,
	inputTokens, outputTokens,
	alpha, beta, gamma, percentNoise []float32,
	maxBatchSize []int) *SimulatedObserver {

	maxStep := max(len(rpm), len(inputTokens), len(outputTokens),
		len(alpha), len(beta), len(gamma),
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
	pctNoise := obs.percentNoise[i]
	maxBatchSize := obs.maxBatchSize[i]

	// create queueing model
	qConfig := &analyzer.Configuration{
		MaxBatchSize: maxBatchSize,
		MaxQueueSize: 10 * maxBatchSize,
		ServiceParms: &analyzer.ServiceParms{
			Alpha: alpha,
			Beta:  beta,
			Gamma: gamma,
		},
	}

	requestSize := &analyzer.RequestSize{
		AvgInputTokens:  inputTokens,
		AvgOutputTokens: outputTokens,
	}
	queueAnalyzer, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
	if err != nil {
		fmt.Println("failed to create queue analyzer: " + err.Error())
		return nil
	}

	// request per msec

	if rpm *= 1 + pctNoise*(2*rand.Float32()-1); rpm < 0 {
		rpm = 0
	}

	metrics, err := queueAnalyzer.Analyze(rpm / 60)
	if err != nil {
		fmt.Println("failed to analyze queueing model: " + err.Error())
		return nil
	}

	// fmt.Println("sim-queueAnalyzer: " + queueAnalyzer.String())
	// fmt.Println("sim-metrics: " + metrics.String())

	avgWaitTime := metrics.AvgWaitTime
	avgTTFT := metrics.AvgTTFT
	avgITL := metrics.AvgTokenTime

	avgConcurrency := metrics.AvgNumInServ

	// add noise
	if avgWaitTime *= 1 + pctNoise*(2*rand.Float32()-1); avgWaitTime < 0 {
		avgWaitTime = 0
	}
	if avgTTFT *= 1 + pctNoise*(2*rand.Float32()-1); avgTTFT < 0 {
		avgTTFT = 0
	}
	if avgITL *= 1 + pctNoise*(2*rand.Float32()-1); avgITL < 0 {
		avgITL = 0
	}

	obs.timeStep++

	// create environment with input parameters
	env := core.NewEnvironmentPrefillDecode(rpm, avgConcurrency, avgWaitTime, maxBatchSize,
		inputTokens, outputTokens, avgTTFT, avgITL)
	return env
}
