/*
Simulated Observer simulates randomness in wait and token time
*/

package observer

import "github.com/llm-inferno/model-tuner/pkg/core"

type DataObserver struct {
	BaseObserver

	rpm          []float32
	inputTokens  []float32
	outputTokens []float32
	itl          []float32
	ttft         []float32
	avgBatchSize []float32
	maxBatchSize []int

	timeStep int
	maxStep  int
}

func NewDataObserver(rpmTotal, inputTokens, outputTokens, itl, ttft, avgBatchSize []float32,
	numReplicas, maxBatchSize []int) *DataObserver {

	maxStep := max(len(rpmTotal), len(inputTokens), len(outputTokens),
		len(itl), len(ttft), len(avgBatchSize), len(numReplicas), len(maxBatchSize)) - 1
	if maxStep < 0 {
		return nil
	}

	rpm := make([]float32, len(rpmTotal))
	for i := range rpmTotal {
		rpm[i] = rpmTotal[i] / float32(numReplicas[i])
	}

	return &DataObserver{
		BaseObserver: BaseObserver{},

		rpm:          rpm,
		inputTokens:  inputTokens,
		outputTokens: outputTokens,
		itl:          itl,
		ttft:         ttft,
		avgBatchSize: avgBatchSize,
		maxBatchSize: maxBatchSize,
		timeStep:     0,
		maxStep:      maxStep,
	}
}

func (dbs *DataObserver) GetEnvironment() core.Environment {

	// get parameters at current time step
	i := min(dbs.timeStep, dbs.maxStep)
	rpm := dbs.rpm[i]
	inputTokens := dbs.inputTokens[i]
	outputTokens := dbs.outputTokens[i]
	itl := dbs.itl[i]
	ttft := dbs.ttft[i]
	avgBatchSize := dbs.avgBatchSize[i]
	maxBatchSize := dbs.maxBatchSize[i]

	dbs.timeStep++

	// create environment with input parameters
	env := core.NewEnvironmentPrefillDecode(rpm, avgBatchSize, 0, maxBatchSize, inputTokens, outputTokens, ttft, itl)
	return env
}

func (dbs *DataObserver) Reset() {
	dbs.timeStep = 0
}
