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

// relative step size used in numerical differentiation
const DefaultStepSize = float32(0.01)

// small value
const Epsilon = float32(0.0001)
