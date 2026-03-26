package core

import (
	"bytes"
	"fmt"

	kalman "github.com/llm-inferno/kalman-filter/pkg/core"
	"github.com/llm-inferno/model-tuner/pkg/config"
	"github.com/llm-inferno/queue-analysis/pkg/analyzer"

	"gonum.org/v1/gonum/mat"
)

const defaultMaxNIS = 7.378

// TunedResults holds the outcome of one EKF predict+update cycle.
type TunedResults struct {
	ServiceParms     *analyzer.ServiceParms
	Innovation       *mat.VecDense
	Covariance       *mat.Dense
	NIS              float64
	ValidationFailed bool
}

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

// NewTunerWithCovariance creates a Tuner restoring a previously saved covariance matrix.
// This provides state continuity across tuning cycles.
func NewTunerWithCovariance(configData *config.ConfigData, env Environment, covariance *mat.Dense) (tuner *Tuner, err error) {
	var c *Configurator
	var f *kalman.ExtendedKalmanFilter

	if c, err = NewConfiguratorWithCovariance(configData, covariance); err != nil {
		return nil, err
	}

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

// RunWithValidation runs one EKF predict+update cycle with NIS validation and rollback on failure.
// On validation failure the filter is rolled back to its previous state and TunedResults.ValidationFailed is set.
func (t *Tuner) RunWithValidation(env Environment) (*TunedResults, error) {
	t.UpdateEnvironment(env)

	stasher, err := NewStasher(t.filter)
	if err != nil {
		return nil, fmt.Errorf("failed to create stasher: %w", err)
	}
	if err := stasher.Stash(); err != nil {
		return nil, fmt.Errorf("failed to stash filter state: %w", err)
	}

	if err := t.filter.Predict(t.filter.Q); err != nil {
		return nil, fmt.Errorf("failed to predict: %w", err)
	}

	Z := env.GetObservations()
	if err := t.filter.Update(Z, t.configurator.R); err != nil {
		return nil, fmt.Errorf("failed to update filter: %w", err)
	}

	nis, valErr := t.computeNIS()
	if valErr != nil {
		if err := stasher.UnStash(); err != nil {
			return nil, fmt.Errorf("failed to unstash after validation failure: %w", err)
		}
		prev, err := t.extractTunedResults()
		if err != nil {
			return nil, fmt.Errorf("validation failed and previous state extraction failed: %w", err)
		}
		prev.ValidationFailed = true
		prev.NIS = nis
		return prev, nil
	}

	results, err := t.extractTunedResults()
	if err != nil {
		return nil, err
	}
	results.NIS = nis
	return results, nil
}

func (t *Tuner) extractTunedResults() (*TunedResults, error) {
	x := t.filter.State()
	if x == nil || x.Len() < 3 {
		return nil, fmt.Errorf("state vector too short (len=%d, need 3)", x.Len())
	}
	return &TunedResults{
		ServiceParms: &analyzer.ServiceParms{
			Alpha: float32(x.AtVec(0)),
			Beta:  float32(x.AtVec(1)),
			Gamma: float32(x.AtVec(2)),
		},
		Innovation: mat.VecDenseCopyOf(t.filter.Innovation()),
		Covariance: mat.DenseCopyOf(t.filter.P),
	}, nil
}

// computeNIS validates the EKF update result using Normalized Innovation Squared.
// Returns (NIS, nil) on success, (NIS, error) if validation fails.
func (t *Tuner) computeNIS() (float64, error) {
	x := t.filter.State()
	if x == nil {
		return -1, fmt.Errorf("nil state vector")
	}
	if x.Len() >= 1 && x.AtVec(0) <= 0 {
		return -1, fmt.Errorf("alpha must be positive: %f", x.AtVec(0))
	}
	if x.Len() >= 2 && x.AtVec(1) <= 0 {
		return -1, fmt.Errorf("beta must be positive: %f", x.AtVec(1))
	}
	if x.Len() >= 3 && x.AtVec(2) <= 0 {
		return -1, fmt.Errorf("gamma must be positive: %f", x.AtVec(2))
	}

	y := mat.VecDenseCopyOf(t.filter.Innovation())
	S := mat.DenseCopyOf(t.filter.InnovationCov())

	Sinv := mat.NewDense(S.RawMatrix().Rows, S.RawMatrix().Cols, nil)
	if err := Sinv.Inverse(S); err != nil {
		return -1, fmt.Errorf("singular innovation covariance: %w", err)
	}

	tmp := mat.NewVecDense(Sinv.RawMatrix().Rows, nil)
	tmp.MulVec(Sinv, y)
	nis := mat.Dot(y, tmp)

	if nis >= defaultMaxNIS {
		return nis, fmt.Errorf("NIS=%.2f exceeds threshold %.2f", nis, defaultMaxNIS)
	}
	return nis, nil
}
