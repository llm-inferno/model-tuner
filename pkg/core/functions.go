package core

import (
	"fmt"
	"math"

	"github.com/llm-inferno/model-tuner/pkg/config"
	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
	"gonum.org/v1/gonum/mat"
)

// The system function relates system state (and the environment) to observations.

// Interface for creating system functions
type SystemFuncCreator interface {
	Create() func(x *mat.VecDense) *mat.VecDense
}

// QueueModelSystemFuncCreator creates a system function based on a queueing model
type QueueModelSystemFuncCreator struct {
	tuner *Tuner // the system function uses the tuner to access the current environment
}

type QueueModelSystemFuncCreatorDecode struct {
	QueueModelSystemFuncCreator
}

type QueueModelSystemFuncCreatorPrefillDecode struct {
	QueueModelSystemFuncCreator
}

func NewQueueModelSystemFuncCreatorDecode(tuner *Tuner) *QueueModelSystemFuncCreatorDecode {
	return &QueueModelSystemFuncCreatorDecode{
		QueueModelSystemFuncCreator: QueueModelSystemFuncCreator{tuner: tuner}}
}

func NewQueueModelSystemFuncCreatorPrefillDecode(tuner *Tuner) *QueueModelSystemFuncCreatorPrefillDecode {
	return &QueueModelSystemFuncCreatorPrefillDecode{
		QueueModelSystemFuncCreator: QueueModelSystemFuncCreator{tuner: tuner}}
}

// Create a system function based on the queueing model. The function maps the state vector
// and the environment (from the tuner) to observations.
func (c *QueueModelSystemFuncCreatorPrefillDecode) Create() func(x *mat.VecDense) *mat.VecDense {
	tuner := c.tuner
	return func(x *mat.VecDense) *mat.VecDense {
		if !tuner.env.Valid() || x.Len() != 4 {
			return nil
		}

		envData := tuner.env.(*EnvironmentPrefillDecode)
		maxBatchSize := envData.MaxBatchSize
		avgInputTokens := envData.AvgInputTokens
		avgOutputTokens := envData.AvgOutputTokens

		alpha := float32(x.AtVec(0))
		beta := float32(x.AtVec(1))
		gamma := float32(x.AtVec(2))
		delta := float32(x.AtVec(3))

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
			AvgInputTokens:  int(math.Ceil(float64(avgInputTokens))),
			AvgOutputTokens: int(math.Ceil(float64(avgOutputTokens))),
		}
		queueAnalyzer, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
		if err != nil {
			return nil
		}

		// request per msec
		rpm := envData.Lambda
		metrics, err := queueAnalyzer.Analyze(rpm / 60)
		if err != nil {
			return nil
		}

		// get metrics from queue analyzer
		// avgWaitTime := float64(metrics.AvgWaitTime)
		avgPrefillTime := float64(metrics.AvgPrefillTime)
		avgTokenDecodeTime := float64(metrics.AvgTokenTime)

		return mat.NewVecDense(2, []float64{avgPrefillTime, avgTokenDecodeTime})
	}
}

func (c *QueueModelSystemFuncCreatorDecode) Create() func(x *mat.VecDense) *mat.VecDense {
	tuner := c.tuner
	return func(x *mat.VecDense) *mat.VecDense {
		if !tuner.env.Valid() || x.Len() != 2 {
			return nil
		}

		envData := tuner.env.(*EnvironmentDecode)
		maxBatchSize := envData.MaxBatchSize
		avgOutputTokens := envData.AvgOutputTokens

		alpha := float32(x.AtVec(0))
		beta := float32(x.AtVec(1))

		// create queueing model
		qConfig := &analyzer.Configuration{
			MaxBatchSize: maxBatchSize,
			MaxQueueSize: 10 * maxBatchSize,
			ServiceParms: &analyzer.ServiceParms{
				Prefill: &analyzer.PrefillParms{},
				Decode: &analyzer.DecodeParms{
					Alpha: alpha,
					Beta:  beta,
				},
			},
		}

		requestSize := &analyzer.RequestSize{
			AvgInputTokens:  0,
			AvgOutputTokens: int(math.Ceil(float64(avgOutputTokens))),
		}
		queueAnalyzer, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
		if err != nil {
			return nil
		}

		// request per msec
		rpm := envData.Lambda
		metrics, err := queueAnalyzer.Analyze(rpm / 60)
		if err != nil {
			return nil
		}

		// get metrics from queue analyzer
		avgWaitTime := float64(metrics.AvgWaitTime)
		avgTokenDecodeTime := float64(metrics.AvgTokenTime)

		return mat.NewVecDense(2, []float64{avgWaitTime, avgTokenDecodeTime})
	}
}

// helper to set up a tuner with a queueing model system function
func SetupTunerForQueueingModel(configData *config.ConfigData, env Environment, kind string) (tuner *Tuner, observationFuncCreator SystemFuncCreator, err error) {
	tuner, err = NewTuner(configData, env)
	if err != nil {
		return nil, nil, err
	}
	switch kind {
	case "decode":
		observationFuncCreator = NewQueueModelSystemFuncCreatorDecode(tuner)
	case "prefill-decode":
		observationFuncCreator = NewQueueModelSystemFuncCreatorPrefillDecode(tuner)
	default:
		return nil, nil, fmt.Errorf("unknown queueing model system function kind: %s", kind)
	}

	if err := tuner.SetObservationFunc(observationFuncCreator); err != nil {
		return nil, nil, err
	}
	return tuner, observationFuncCreator, nil
}
