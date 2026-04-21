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

// Environment variable names and defaults for tuner behaviour.
const (
	WarmUpCyclesEnvName = "TUNER_WARM_UP_CYCLES"
)

// Default field values used when the ParameterStore has a model/accelerator entry
// that is not present in the Controller's current ModelData.
const (
	DefaultAccCount     = 1
	DefaultMaxBatchSize = 256
	DefaultAtTokens     = 1024
)

// Environment variable names and defaults for the InitEstimator.
const (
	InitObsEnvName      = "TUNER_INIT_OBS"
	InitHoldBackEnvName = "TUNER_INIT_HOLD_BACK"

	DefaultInitObs      = 5
	DefaultInitHoldBack = true
)

// Environment variable names and defaults for the SlidingWindowEstimator.
const (
	EstimatorModeEnvName     = "TUNER_ESTIMATOR_MODE"
	WindowSizeEnvName        = "TUNER_WINDOW_SIZE"
	ResidualThresholdEnvName = "TUNER_RESIDUAL_THRESHOLD"

	DefaultEstimatorMode     = "ekf"
	DefaultWindowSize        = 10
	DefaultResidualThreshold = 0.5
)
