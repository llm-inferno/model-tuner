// Package tunerservice provides an EKF-based model parameter tuning service designed
// for integration with the llm-inferno control-loop.
//
// # Overview
//
// LLM inference servers are modeled with three parameters (alpha, beta, gamma) that
// describe iteration time as a function of token workload:
//
//	iterationTime = alpha + beta*computedTokens + gamma*transferredTokens
//
// This package continuously refines those parameters using an Extended Kalman Filter
// (EKF) fed by per-replica performance observations. Tuned parameters are stored in a
// [ParameterStore] keyed by model/accelerator pair and returned as
// optimizer-light ModelData, ready for direct use by the Optimizer.
//
// # Control-Loop Integration
//
// The intended usage from the control-loop Controller is:
//
//  1. Call the Collector to obtain ServerCollectorInfo (includes ReplicaSpecs).
//  2. POST ReplicaSpecs to /tune — receives updated ModelData.
//  3. Set SystemData.Spec.Models to the returned ModelData.
//  4. POST SystemData to the Optimizer as usual.
//
// Alternatively, GET /getparams?model=<name>&accelerator=<acc> retrieves the most
// recently stored parameters for a specific model/accelerator pair without triggering
// a new tuning cycle.
//
// # EKF Features
//
// The EKF implementation includes three features ported from the
// llm-d-workload-variant-autoscaler:
//
//   - State continuity: previously tuned alpha/beta/gamma and their covariance matrix
//     are restored at the start of each tuning cycle, so the filter converges faster
//     over time rather than re-initializing from scratch.
//
//   - Initial state guessing: on first observation, alpha/beta/gamma are derived
//     algebraically from observed TTFT and ITL using the paper's queueing model
//     equations, providing a warm start instead of cold defaults.
//
//   - NIS validation: after each EKF update the Normalized Innovation Squared
//     (NIS = y^T S^-1 y) is checked against a chi-squared threshold (7.378 for 2 DOF
//     at 97.5%). Updates that exceed the threshold are rejected and the filter is
//     rolled back to its previous state, preventing parameter divergence on outlier
//     observations.
//
// # Grouping and Multi-Replica Tuning
//
// Incoming ReplicaSpecs are grouped by (Model, Accelerator). Within each group, one
// EKF predict+update cycle is run per replica that has active traffic (ArrivalRate > 0),
// giving the filter multiple independent observations per tuning call — one per live pod.
//
// # Configuration
//
// Filter and model parameters are loaded from JSON config files via the CONFIG_DATA_DIR
// environment variable (default: config-data). A model-specific file
// (<model>-config-data.json) is used when present; otherwise default-config-data.json
// is loaded as a fallback.
//
// # HTTP API
//
//	POST /tune
//	  Body:     []config.ServerSpec   (ReplicaSpecs from the Collector)
//	  Response: config.ModelData      (updated alpha/beta/gamma per model/accelerator)
//
//	GET /getparams?model=<name>&accelerator=<acc>
//	  Response: JSON with alpha, beta, gamma, NIS, lastUpdated
package tunerservice
