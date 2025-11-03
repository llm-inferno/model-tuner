package core

import (
	"bytes"
	"fmt"
	"math"

	"github.com/llm-inferno/model-tuner/pkg/config"
	"gonum.org/v1/gonum/mat"
)

type Configurator struct {
	// dimensions
	nX int
	nZ int

	// matrices
	X0 *mat.VecDense
	P  *mat.Dense
	Q  *mat.Dense
	R  *mat.Dense

	// functions
	fFunc func(*mat.VecDense) *mat.VecDense

	// other
	percentChange []float64
	Xbounded      bool
	Xmin          []float64
	Xmax          []float64
}

func NewConfigurator(configData *config.ConfigData) (c *Configurator, err error) {
	if !validConfigData(configData) {
		return nil, fmt.Errorf("invalid config data")
	}

	md := configData.ModelData
	n := len(md.InitState)
	X0 := mat.NewVecDense(n, md.InitState)

	fd := configData.FilterData
	m := len(md.ExpectedObservations)
	obsCOV := make([]float64, m)
	factor := math.Pow(fd.ErrorLevel/fd.TPercentile, 2) / fd.GammaFactor
	for j := range m {
		obsCOV[j] = factor * math.Pow(md.ExpectedObservations[j], 2)
	}
	R := mat.DenseCopyOf(mat.NewDiagDense(m, obsCOV))

	c = &Configurator{
		nX:            n,
		nZ:            m,
		X0:            X0,
		P:             nil,
		Q:             nil,
		R:             R,
		fFunc:         nil,
		percentChange: md.PercentChange,
		Xbounded:      md.BoundedState,
		Xmin:          md.MinState,
		Xmax:          md.MaxState,
	}

	if c.P, err = c.GetStateCov(X0); err != nil {
		return nil, err
	}
	if c.Q, err = c.GetStateCov(X0); err != nil {
		return nil, err
	}
	c.fFunc = stateTransitionFunc
	return c, nil
}

func (c *Configurator) GetStateCov(x *mat.VecDense) (*mat.Dense, error) {
	if x.Len() != c.nX {
		return nil, mat.ErrNormOrder
	}
	changeCov := make([]float64, c.nX)
	for i := 0; i < c.nX; i++ {
		changeCov[i] = math.Pow(c.percentChange[i]*x.AtVec(i), 2)
	}
	return mat.DenseCopyOf(mat.NewDiagDense(c.nX, changeCov)), nil
}

func (c *Configurator) NumStates() int {
	return c.nX
}

func (c *Configurator) NumObservations() int {
	return c.nZ
}

func validConfigData(cd *config.ConfigData) bool {
	if cd == nil {
		return false
	}
	md := cd.ModelData
	n := len(md.InitState)
	if n == 0 || len(md.PercentChange) != n ||
		md.BoundedState && (len(md.MinState) != n || len(md.MaxState) != n) {
		return false
	}
	if len(md.ExpectedObservations) == 0 {
		return false
	}
	return true
}

func stateTransitionFunc(x *mat.VecDense) *mat.VecDense {
	return x
}

func (c *Configurator) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "Configurator: ")
	fmt.Fprintf(&b, "nX=%d; nZ=%d; ", c.nX, c.nZ)
	fmt.Fprintf(&b, "X0=%v; ", c.X0.RawVector().Data)
	fmt.Fprintf(&b, "Xbounded=%v; ", c.Xbounded)
	if c.Xbounded {
		fmt.Fprintf(&b, "Xmin=%v; ", c.Xmin)
		fmt.Fprintf(&b, "Xmax=%v; ", c.Xmax)
	}
	fmt.Fprintf(&b, "P=%v; ", c.P.RawMatrix().Data)
	fmt.Fprintf(&b, "Q=%v; ", c.Q.RawMatrix().Data)
	fmt.Fprintf(&b, "R=%v; ", c.R.RawMatrix().Data)
	fmt.Fprintf(&b, "change=%v; ", c.percentChange)
	return b.String()
}
