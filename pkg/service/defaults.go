package service

// Environment variable names and defaults for tuner behaviour.
const (
	WarmUpCyclesEnvName = "TUNER_WARM_UP_CYCLES"
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

// Environment variable name and default for the init-fit quality threshold.
const (
	InitFitThresholdEnvName = "TUNER_INIT_FIT_THRESHOLD"
	DefaultInitFitThreshold = 10.0
)

// Environment variable name and default for the identifiability guard. When > 0, the
// estimators reject a fit whose Jacobian condition number exceeds this value — the
// signature of a degenerate, unidentifiable solution (collapsed beta/gamma) produced when
// the observation window lacks operating-point spread (e.g. a single-replica deployment at
// steady load). Healthy fits sit well below ~100; degenerate fits exceed a few thousand.
// Set to 0 to disable the guard.
const (
	MaxConditionNumberEnvName = "TUNER_MAX_CONDITION_NUMBER"
	DefaultMaxConditionNumber = 1000.0
)

// Default field values used when the ParameterStore has a model/accelerator entry
// that is not present in the Controller's current ModelData.
const (
	DefaultAccCount     = 1
	DefaultMaxBatchSize = 256
)
