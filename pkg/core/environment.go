package core

import (
	"bytes"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type Environment struct {
	Lambda              float32 // request arrival rate (per minute)
	AvgTokensPerRequest float32 // average number of tokens per request
	MaxBatchSize        int     // maximum batch size
	BatchSize           float32 // batch size
	AvgQueueTime        float32 // average request queueing time (msec)
	AvgTokenTime        float32 // average inter token latency (msec)
}

func (e *Environment) Valid() bool {
	return e.Lambda > 0 && e.AvgTokensPerRequest > 0 && e.MaxBatchSize > 0
}

func (e *Environment) GetObservations() *mat.VecDense {
	return mat.NewVecDense(2, []float64{float64(e.AvgQueueTime), float64(e.AvgTokenTime)})
}

func (e *Environment) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "Environment: ")
	fmt.Fprintf(&b, "rpm=%5.2f; avgTokens=%6.2f; maxBatch=%d; batchSize=%6.2f; avgWait=%10.6f; avgITL=%10.6f",
		e.Lambda, e.AvgTokensPerRequest, e.MaxBatchSize, e.BatchSize, e.AvgQueueTime, e.AvgTokenTime)
	return b.String()
}
