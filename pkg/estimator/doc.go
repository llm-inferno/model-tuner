// Package estimator provides pure estimation primitives for LLM inference parameter tuning.
//
// It contains two estimators and the shared types they depend on:
//
//   - [InitEstimator]: collects K initial observations then runs Nelder-Mead to fit
//     an initial (alpha, beta, gamma) before handing off to the EKF or SWNM.
//
//   - [SlidingWindowEstimator]: maintains a fixed-capacity circular buffer of recent
//     observations and re-runs Nelder-Mead on every Fit() call.
//
// [GuessInitState] provides an algebraic cold-start estimate from a single observation.
// Both estimators use it as a Nelder-Mead warm-start and fallback.
//
// This package has no dependency on HTTP routing or the optimizer-light config types.
// It depends only on pkg/core (for EnvironmentPrefillDecode) and the queue-analysis
// analyzer (for the queueing model objective function).
package estimator
