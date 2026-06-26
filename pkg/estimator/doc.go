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
// [GuessInitState] provides a cold-start estimate from a single observation. Given a seed
// [alpha,beta,gamma] (via SetSeed) it pins the unidentifiable gamma to the seed and solves
// alpha,beta from the observation — falling back to the full seed if that is degenerate;
// with no seed it uses the legacy algebraic heuristic (alpha = baseFactor * ITL). Both
// estimators use it as a Nelder-Mead warm-start and fallback.
//
// This package has no dependency on HTTP routing or the optimizer-light config types.
// It depends only on pkg/core (for EnvironmentPrefillDecode) and the queue-analysis
// analyzer (for the queueing model objective function).
package estimator
