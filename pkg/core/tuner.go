package core

import (
	"bytes"
	"fmt"

	kalman "github.com/llm-inferno/kalman-filter/pkg/core"
	"github.com/llm-inferno/model-tuner/pkg/config"

	"gonum.org/v1/gonum/mat"
)

type Tuner struct {
	configurator *Configurator
	filter       *kalman.ExtendedKalmanFilter
	env          Environment
}

func NewTuner(configData *config.ConfigData, env Environment) (tuner *Tuner, err error) {
	var c *Configurator
	var f *kalman.ExtendedKalmanFilter

	// create configurator
	if c, err = NewConfigurator(configData); err != nil {
		return nil, err
	}

	//create filter
	f, err = kalman.NewExtendedKalmanFilter(c.NumStates(), c.NumObservations(), c.X0, c.P)
	if err != nil {
		return nil, err
	}
	if err = f.SetQ(c.Q); err != nil {
		return nil, err
	}
	if err = f.SetR(c.R); err != nil {
		return nil, err
	}
	if err = f.SetfF(c.fFunc); err != nil {
		return nil, err
	}
	if c.Xbounded {
		if err = f.SetStateLimiter(c.Xmin, c.Xmax); err != nil {
			return nil, err
		}
	}

	// create tuner
	return &Tuner{
		configurator: c,
		filter:       f,
		env:          env,
	}, nil
}

func (t *Tuner) SetObservationFunc(systemFuncCreator SystemFuncCreator) error {
	obsFunc := systemFuncCreator.Create()
	if obsFunc == nil {
		return fmt.Errorf("observation function is nil")
	}
	if err := t.filter.SethH(obsFunc); err != nil {
		return err
	}
	return nil
}

func (t *Tuner) Run(env Environment) (err error) {
	t.UpdateEnvironment(env)

	Q := t.filter.Q
	if err = t.filter.Predict(Q); err != nil {
		return err
	}

	// correct
	Z := env.GetObservations()
	if err = t.filter.Update(Z, t.configurator.R); err != nil {
		return err
	}
	return nil
}

func (t *Tuner) UpdateEnvironment(env Environment) {
	t.env = env
}

func (t *Tuner) Environment() Environment {
	return t.env
}

func (t *Tuner) X() *mat.VecDense {
	return t.filter.State()
}

func (t *Tuner) Innovation() *mat.VecDense {
	return t.filter.Innovation()
}

func (t *Tuner) P() *mat.Dense {
	return t.filter.P
}

func (t *Tuner) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "Tuner: \n")
	fmt.Fprintf(&b, "%v\n", t.configurator)
	return b.String()
}

func (t *Tuner) GetParams() *mat.VecDense {
	// TODO: intelligent state return
	return t.X()
}
