package config

/**
 * Default filter parameters
 */

// gamma factor in observation noise variance calculation
const DefaultGammaFactor = float32(1.0)

// percentile level
const DefaultErrorLevel = float32(0.05)

// value of tail of student distribution at percentile
const DefaultStudentPercentile = float32(1.96)

// predicted percent change in state values
const DefaultPercentChange = float32(5.0)

// DefaultConfigType is the config-data type used when no specific type is requested.
// Valid types correspond to config-data filenames: "default", "decode", "prefill-decode", "benchmark".
const DefaultConfigType = "default"

// DefaultInitStateFactor is the multiplicative factor used to derive MinState and MaxState
// from InitState: Min = Init/factor, Max = Init*factor (symmetric in log space).
const DefaultInitStateFactor = float64(10.0)

// DefaultInitStateMinEpsilon is the lower floor applied to MinState to avoid zero or
// near-zero lower bounds for parameters that are physically positive but may be very small.
const DefaultInitStateMinEpsilon = float64(1e-9)
