// Package service provides TunerService, the orchestration layer for LLM inference
// parameter tuning. It sits between the estimation primitives in pkg/estimator and the
// HTTP adapter in tunerservice.
//
// TunerService groups per-replica ServerSpecs by (model, accelerator), runs an
// InitEstimator cold-start phase for each new pair, then dispatches subsequent
// observations to either a SlidingWindowEstimator (SWNM mode) or the EKF Tuner
// (pkg/core). Tuned parameters are stored in a ParameterStore for state continuity
// across tuning cycles.
//
// Consumers that want estimation without the HTTP layer can import this package directly
// and call [TunerService.Tune], [TunerService.GetParams], and [TunerService.Merge]
// without running a gin server.
package service
