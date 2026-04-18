package core

import (
	"bytes"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// representation of the environment in which the system operates
type environmentBase struct {
	Lambda       float32 // request arrival rate (per minute)
	BatchSize    float32 // batch size
	AvgQueueTime float32 // average request queueing time (msec)
	MaxBatchSize int     // maximum batch size
	MaxQueueSize int     // maximum external queue depth (0 = no external queue)
}

type EnvironmentDecode struct {
	environmentBase
	AvgOutputTokens float32 // average number of output tokens per request
	AvgITL          float32 // average inter token latency (msec)
}

type EnvironmentPrefillDecode struct {
	environmentBase
	AvgInputTokens  float32 // average number of input tokens per request
	AvgOutputTokens float32 // average number of output tokens per request
	AvgTTFT         float32 // average time to first token (msec)
	AvgITL          float32 // average inter token latency (msec)
}

type Environment interface {
	Valid() bool                    // validate the environment parameters
	GetObservations() *mat.VecDense // observation vector, consumed by the filter, from the environment
	String() string                 // string representation of the environment
}

func newEnvironmentBase(lambda, batchSize, avgQueueTime float32, maxBatchSize int) environmentBase {
	return environmentBase{
		Lambda:       lambda,
		BatchSize:    batchSize,
		AvgQueueTime: avgQueueTime,
		MaxBatchSize: maxBatchSize,
	}
}

func NewEnvironmentDecode(lambda, batchSize, avgQueueTime float32, maxBatchSize int,
	avgOutputTokens, avgITL float32) *EnvironmentDecode {
	return &EnvironmentDecode{
		environmentBase: newEnvironmentBase(lambda, batchSize, avgQueueTime, maxBatchSize),
		AvgOutputTokens: avgOutputTokens,
		AvgITL:          avgITL,
	}
}

func NewEnvironmentPrefillDecode(lambda, batchSize, avgQueueTime float32, maxBatchSize int,
	avgInputTokens, avgOutputTokens, avgTTFT, avgITL float32) *EnvironmentPrefillDecode {
	return &EnvironmentPrefillDecode{
		environmentBase: newEnvironmentBase(lambda, batchSize, avgQueueTime, maxBatchSize),
		AvgInputTokens:  avgInputTokens,
		AvgOutputTokens: avgOutputTokens,
		AvgTTFT:         avgTTFT,
		AvgITL:          avgITL,
	}
}

func (e *environmentBase) Valid() bool {
	return e.Lambda > 0 && e.BatchSize >= 0 && e.AvgQueueTime >= 0 && e.MaxBatchSize > 0
}

func (e *EnvironmentDecode) Valid() bool {
	return e.environmentBase.Valid() && e.AvgOutputTokens > 0 && e.AvgITL > 0
}

func (e *EnvironmentPrefillDecode) Valid() bool {
	return e.environmentBase.Valid() && e.AvgInputTokens >= 0 && e.AvgOutputTokens > 0 &&
		e.AvgTTFT >= 0 && e.AvgITL > 0
}

func (e *EnvironmentDecode) GetObservations() *mat.VecDense {
	return mat.NewVecDense(2, []float64{float64(e.AvgQueueTime), float64(e.AvgITL)})
}

func (e *EnvironmentPrefillDecode) GetObservations() *mat.VecDense {
	return mat.NewVecDense(2, []float64{float64(e.AvgTTFT), float64(e.AvgITL)})
}

func (e *EnvironmentDecode) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "Environment: ")
	fmt.Fprintf(&b, "arrivalRateRPM=%5.2f; avgOutTokens=%6.2f; batchSize=%6.2f; maxBatch=%d; avgWait=%10.6f; avgITL=%10.6f",
		e.Lambda, e.AvgOutputTokens, e.BatchSize, e.MaxBatchSize, e.AvgQueueTime, e.AvgITL)
	return b.String()
}

func (e *EnvironmentPrefillDecode) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "Environment: ")
	fmt.Fprintf(&b, "arrivalRateRPM=%5.2f; avgInTokens=%6.2f; avgOutTokens=%6.2f; batchSize=%6.2f; maxBatch=%d; avgWait=%10.6f; avgTTFT=%10.6f; avgITL=%10.6f",
		e.Lambda, e.AvgInputTokens, e.AvgOutputTokens, e.BatchSize, e.MaxBatchSize,
		e.AvgQueueTime, e.AvgTTFT, e.AvgITL)
	return b.String()
}
