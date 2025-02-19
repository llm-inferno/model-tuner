package config

// Tuner configuration data
type ConfigData struct {
	FilterData FilterData `json:"filterData"` // filter data
	ModelData  ModelData  `json:"modelData"`  // model data
}

// Filter configuration data
type FilterData struct {
	GammaFactor       float32 `json:"gammaFactor"`       // gamma factor
	ErrorLevel        float32 `json:"errorLevel"`        // error level percentile
	StudentPercentile float32 `json:"studentPercentile"` // tail of student distribution
	PercentChange     float32 `json:"percentChange"`     // percent change in state
	StepSize          float32 `json:"stepSize"`          // relative step size
}

// Model configuration data
type ModelData struct {
	InitState            []float32 `json:"initState"`            // initial state of model parameters
	BoundedState         bool      `json:"boundedState"`         // are the state values bounded
	MinState             []float32 `json:"minState"`             // lower bound on state
	MaxState             []float32 `json:"maxState"`             // upper bound on state
	ExpectedObservations []float32 `json:"expectedObservations"` // expected values of observations
}
