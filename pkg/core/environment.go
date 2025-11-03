package core

import (
	"bytes"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// representation of the environment in which the system operates
type EnvironmentBase struct {
	Lambda       float32 // request arrival rate (per minute)
	BatchSize    float32 // batch size
	AvgQueueTime float32 // average request queueing time (msec)
	MaxBatchSize int     // maximum batch size
}

type EnvironmentDecode struct {
	EnvironmentBase
	AvgOutputTokens    float32 // average number of output tokens per request
	AvgTokenDecodeTime float32 // average inter token latency (msec)
}

type EnvironmentPrefillDecode struct {
	EnvironmentBase
	AvgInputTokens     float32 // average number of input tokens per request
	AvgOutputTokens    float32 // average number of output tokens per request
	AvgPrefillTime     float32 // average prefill time (msec)
	AvgTokenDecodeTime float32 // average inter token latency (msec)
}

type Environment interface {
	Valid() bool                    // validate the environment parameters
	GetObservations() *mat.VecDense // observation vector, consumed by the filter, from the environment
	String() string                 // string representation of the environment
}

func NewEnvironmentBase(lambda, batchSize, avgQueueTime float32, maxBatchSize int) *EnvironmentBase {
	return &EnvironmentBase{
		Lambda:       lambda,
		BatchSize:    batchSize,
		AvgQueueTime: avgQueueTime,
		MaxBatchSize: maxBatchSize,
	}
}

func NewEnvironmentDecode(lambda, batchSize, avgQueueTime float32, maxBatchSize int,
	avgOutputTokens, avgTokenDecodeTime float32) *EnvironmentDecode {
	return &EnvironmentDecode{
		EnvironmentBase:    *NewEnvironmentBase(lambda, batchSize, avgQueueTime, maxBatchSize),
		AvgOutputTokens:    avgOutputTokens,
		AvgTokenDecodeTime: avgTokenDecodeTime,
	}
}

func NewEnvironmentPrefillDecode(lambda, batchSize, avgQueueTime float32, maxBatchSize int,
	avgInputTokens, avgOutputTokens, avgPrefillTime, avgTokenDecodeTime float32) *EnvironmentPrefillDecode {
	return &EnvironmentPrefillDecode{
		EnvironmentBase:    *NewEnvironmentBase(lambda, batchSize, avgQueueTime, maxBatchSize),
		AvgInputTokens:     avgInputTokens,
		AvgOutputTokens:    avgOutputTokens,
		AvgPrefillTime:     avgPrefillTime,
		AvgTokenDecodeTime: avgTokenDecodeTime,
	}
}

func (e *EnvironmentBase) Valid() bool {
	return e.Lambda > 0 && e.BatchSize >= 0 && e.AvgQueueTime >= 0 && e.MaxBatchSize > 0
}

func (e *EnvironmentDecode) Valid() bool {
	return e.EnvironmentBase.Valid() && e.AvgOutputTokens > 0 && e.AvgTokenDecodeTime > 0
}

func (e *EnvironmentPrefillDecode) Valid() bool {
	return e.EnvironmentBase.Valid() && e.AvgInputTokens >= 0 && e.AvgOutputTokens > 0 &&
		e.AvgPrefillTime >= 0 && e.AvgTokenDecodeTime > 0
}

func (e *EnvironmentDecode) GetObservations() *mat.VecDense {
	return mat.NewVecDense(2, []float64{float64(e.AvgQueueTime), float64(e.AvgTokenDecodeTime)})
}

func (e *EnvironmentPrefillDecode) GetObservations() *mat.VecDense {
	return mat.NewVecDense(2, []float64{float64(e.AvgPrefillTime), float64(e.AvgTokenDecodeTime)})
}

func (e *EnvironmentDecode) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "Environment: ")
	fmt.Fprintf(&b, "rpm=%5.2f; avgOutTokens=%6.2f; batchSize=%6.2f; maxBatch=%d; avgWait=%10.6f; avgITL=%10.6f",
		e.Lambda, e.AvgOutputTokens, e.BatchSize, e.MaxBatchSize, e.AvgQueueTime, e.AvgTokenDecodeTime)
	return b.String()
}

func (e *EnvironmentPrefillDecode) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "Environment: ")
	fmt.Fprintf(&b, "rpm=%5.2f; avgInTokens=%6.2f; avgOutTokens=%6.2f; batchSize=%6.2f; maxBatch=%d; avgWait=%10.6f; avgPrefill=%10.6f; avgITL=%10.6f",
		e.Lambda, e.AvgInputTokens, e.AvgOutputTokens, e.BatchSize, e.MaxBatchSize,
		e.AvgQueueTime, e.AvgPrefillTime, e.AvgTokenDecodeTime)
	return b.String()
}
