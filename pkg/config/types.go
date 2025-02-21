package config

// Tuner configuration data
type ConfigData struct {
	FilterData FilterData `json:"filterData"` // filter data
	ModelData  ModelData  `json:"modelData"`  // model data
}

// Filter configuration data
type FilterData struct {
	GammaFactor float64 `json:"gammaFactor"` // gamma factor
	ErrorLevel  float64 `json:"errorLevel"`  // error level percentile
	TPercentile float64 `json:"tPercentile"` // tail of student distribution
}

// Model configuration data
type ModelData struct {
	InitState            []float64 `json:"initState"`            // initial state of model parameters
	PercentChange        []float64 `json:"percentChange"`        // percent change in state
	BoundedState         bool      `json:"boundedState"`         // are the state values bounded
	MinState             []float64 `json:"minState"`             // lower bound on state
	MaxState             []float64 `json:"maxState"`             // upper bound on state
	ExpectedObservations []float64 `json:"expectedObservations"` // expected values of observations
}
