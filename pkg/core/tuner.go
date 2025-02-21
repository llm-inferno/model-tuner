package core

import (
	"bytes"
	"fmt"

	kalman "github.ibm.com/modeling-analysis/kalman-filter/pkg/core"
	"github.ibm.com/modeling-analysis/model-tuner/pkg/config"
	"github.ibm.com/modeling-analysis/model-tuner/pkg/utils"
	"github.ibm.com/modeling-analysis/queue-analysis/pkg/queue"
	"gonum.org/v1/gonum/mat"
)

type Tuner struct {
	configurator *Configurator
	filter       *kalman.ExtendedKalmanFilter
	monitor      Monitor
}

var env *Environment

func NewTuner(configData *config.ConfigData, monitor Monitor) (tuner *Tuner, err error) {
	var c *Configurator
	var f *kalman.ExtendedKalmanFilter

	// get environment
	env = monitor.GetEnvironment()

	// create configurator
	if c, err = NewConfigurator(configData); err != nil {
		return nil, err
	}

	//create filter
	f, err = kalman.NewExtendedKalmanFilter(c.NumStates(), c.NumObservations(), c.X0, c.P)
	if err != nil {
		return nil, err
	}
	if err := f.SetQ(c.Q); err != nil {
		return nil, err
	}
	if err := f.SetR(c.R); err != nil {
		return nil, err
	}
	if err := f.SetfF(c.fFunc); err != nil {
		return nil, err
	}
	if c.Xbounded {
		if err := f.SetStateLimiter(c.Xmin, c.Xmax); err != nil {
			return nil, err
		}
	}
	if err := f.SethH(observationFunc); err != nil {
		return nil, err
	}

	// create tuner
	return &Tuner{
		configurator: c,
		filter:       f,
		monitor:      monitor,
	}, nil
}

func (t *Tuner) Run(numSteps int) {
	for i := 0; i < numSteps; i++ {
		// get environment
		env = t.monitor.GetEnvironment()

		// predict
		X := t.filter.State()
		Q, err := t.configurator.GetStateCov(X)
		if err != nil {
			fmt.Println(err)
			continue
		}
		if err := t.filter.Predict(Q); err != nil {
			fmt.Println(err)
			continue
		}

		// correct
		Z := env.GetObservations()
		if err := t.filter.Update(Z, t.configurator.R); err != nil {
			fmt.Println(err)
			continue
		}

		// print state
		fmt.Printf("%s;   %s;   %s\n",
			utils.VecString("X", t.filter.State()),
			utils.VecString("Delta", t.filter.Innovation()),
			utils.MatString("P", t.filter.P))
	}
}

func observationFunc(x *mat.VecDense) *mat.VecDense {
	if !env.Valid() || x.Len() != 2 {
		return mat.NewVecDense(2, nil)
	}
	maxBatchSize := env.MaxBatchSize
	avgNumTokens := env.AvgTokensPerRequest
	alpha := float32(x.AtVec(0))
	beta := float32(x.AtVec(1))

	// calculate state-dependent service rate
	servRate := make([]float32, maxBatchSize)
	for n := 1; n <= maxBatchSize; n++ {
		servRate[n-1] = float32(n) / (alpha + beta*float32(n)) / float32(avgNumTokens)
	}

	// create queueing model
	maxQueueSize := 100 * maxBatchSize
	queue := queue.NewMM1ModelStateDependent(maxQueueSize, servRate)

	// request per msec
	rpm := env.Lambda
	lambda := (rpm / 1000 / 60)
	queue.Solve(lambda, 1)
	avgWaitTime := float64(queue.GetAvgWaitTime())
	avgTokenTime := float64(queue.GetAvgServTime() / float32(avgNumTokens))

	return mat.NewVecDense(2, []float64{avgWaitTime, avgTokenTime})
}

func (t *Tuner) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "Tuner: \n")
	fmt.Fprintf(&b, "%v\n", t.configurator)
	fmt.Fprintf(&b, "%v\n", env)
	return b.String()
}
