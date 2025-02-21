package core

import (
	"math/rand/v2"

	"github.ibm.com/modeling-analysis/queue-analysis/pkg/queue"
)

type SimulatedMonitor struct {
	BaseMonitor

	rpm          float32
	avgNumTokens float32
	alpha, beta  float32
	percentNoise float32
	maxBatchSize int
}

func NewSimulatedMonitor(rpm, avgNumTokens, alpha, beta, percentNoise float32, maxBatchSize int) *SimulatedMonitor {
	return &SimulatedMonitor{
		BaseMonitor: BaseMonitor{},

		rpm:          rpm,
		avgNumTokens: avgNumTokens,
		alpha:        alpha,
		beta:         beta,
		maxBatchSize: maxBatchSize,
		percentNoise: percentNoise,
	}
}

func (m *SimulatedMonitor) GetEnvironment() *Environment {

	// calculate state-dependent service rate
	servRate := make([]float32, m.maxBatchSize)
	for n := 1; n <= m.maxBatchSize; n++ {
		servRate[n-1] = float32(n) / (m.alpha + m.beta*float32(n)) / float32(m.avgNumTokens)
	}

	// create queueing model
	maxQueueSize := 100 * m.maxBatchSize
	model := queue.NewMM1ModelStateDependent(maxQueueSize, servRate)

	// request per msec
	lambda := (m.rpm / 1000 / 60)
	model.Solve(lambda, 1)
	avgWaitTime := model.GetAvgWaitTime()
	avgTokenTime := model.GetAvgServTime() / float32(m.avgNumTokens)

	// add noise
	if avgWaitTime *= 1 + m.percentNoise*(2*rand.Float32()-1); avgWaitTime < 0 {
		avgWaitTime = 0
	}
	if avgTokenTime *= 1 + m.percentNoise*(2*rand.Float32()-1); avgTokenTime < 0 {
		avgTokenTime = 0
	}

	return &Environment{
		Lambda:              m.rpm,
		MaxBatchSize:        m.maxBatchSize,
		AvgTokensPerRequest: m.avgNumTokens,
		AvgQueueTime:        avgWaitTime,
		AvgTokenTime:        avgTokenTime,
	}
}
