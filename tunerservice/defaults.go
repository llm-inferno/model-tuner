package tunerservice

const (
	// baseFactor is the fraction of ITL assumed to be the baseline iteration overhead (alpha).
	// Used in guessInitState to derive an initial estimate of alpha from observed ITL.
	baseFactor = 0.9
)

// Environment variable names and defaults for the tuner REST server.
const (
	TunerHostEnvName = "TUNER_HOST"
	TunerPortEnvName = "TUNER_PORT"

	DefaultTunerHost = "localhost"
	DefaultTunerPort = "8081"
)

// Default field values used when the ParameterStore has a model/accelerator entry
// that is not present in the Controller's current ModelData.
const (
	DefaultAccCount     = 1
	DefaultMaxBatchSize = 256
	DefaultAtTokens     = 1024
)
