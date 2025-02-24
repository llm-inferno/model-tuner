package core

import (
	"math/rand/v2"

	"github.ibm.com/modeling-analysis/queue-analysis/pkg/queue"
)

type SimulatedObserver struct {
	BaseObserver

	rpm          []float32
	avgNumTokens []float32
	alpha, beta  []float32
	percentNoise []float32
	maxBatchSize []int

	timeStep int
	maxStep  int
}

func NewSimulatedObserver(rpm, avgNumTokens, alpha, beta, percentNoise []float32, maxBatchSize []int) *SimulatedObserver {
	maxStep := max(len(rpm), len(avgNumTokens), len(alpha), len(beta), len(percentNoise), len(maxBatchSize)) - 1
	if maxStep < 0 {
		return nil
	}
	return &SimulatedObserver{
		BaseObserver: BaseObserver{},

		rpm:          rpm,
		avgNumTokens: avgNumTokens,
		alpha:        alpha,
		beta:         beta,
		maxBatchSize: maxBatchSize,
		percentNoise: percentNoise,
		timeStep:     0,
		maxStep:      maxStep,
	}
}

func (obs *SimulatedObserver) GetEnvironment() *Environment {

	// get parameters at current time step
	i := min(obs.timeStep, obs.maxStep)
	rpm := obs.rpm[i]
	avgNumTokens := obs.avgNumTokens[i]
	alpha := obs.alpha[i]
	beta := obs.beta[i]
	pctNoise := obs.percentNoise[i]
	maxBatchSize := obs.maxBatchSize[i]

	// calculate state-dependent service rate
	servRate := make([]float32, maxBatchSize)
	for n := 1; n <= maxBatchSize; n++ {
		servRate[n-1] = float32(n) / (alpha + beta*float32(n)) / float32(avgNumTokens)
	}

	// create queueing model
	maxQueueSize := 100 * maxBatchSize
	model := queue.NewMM1ModelStateDependent(maxQueueSize, servRate)

	// request per msec
	lambda := (rpm / 1000 / 60)
	model.Solve(lambda, 1)
	avgWaitTime := model.GetAvgWaitTime()
	avgTokenTime := model.GetAvgServTime() / float32(avgNumTokens)

	// add noise
	if avgWaitTime *= 1 + pctNoise*(2*rand.Float32()-1); avgWaitTime < 0 {
		avgWaitTime = 0
	}
	if avgTokenTime *= 1 + pctNoise*(2*rand.Float32()-1); avgTokenTime < 0 {
		avgTokenTime = 0
	}

	obs.timeStep++
	return &Environment{
		Lambda:              rpm,
		MaxBatchSize:        maxBatchSize,
		AvgTokensPerRequest: avgNumTokens,
		AvgQueueTime:        avgWaitTime,
		AvgTokenTime:        avgTokenTime,
	}
}
