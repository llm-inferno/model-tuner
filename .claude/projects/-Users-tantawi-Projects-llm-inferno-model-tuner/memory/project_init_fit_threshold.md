---
name: TUNER_INIT_FIT_THRESHOLD feature
description: One-time EKF fallback selector for SWNM mode based on InitEstimator.Fit() quality
type: project
---

Added in PR #13 (merged 2026-04-21), closes issue #12.

When `TUNER_ESTIMATOR_MODE=sliding-window`, if `InitEstimator.Fit()` objective value exceeds `TUNER_INIT_FIT_THRESHOLD` (default 10.0), the `(model, accelerator)` pair is permanently routed to EKF instead of SWNM. Set to `0` to disable.

**Why:** At low utilisation the loss surface is flat and Nelder-Mead converges to a degenerate solution (e.g. granite_8b: alpha≈149, beta≈0, gamma≈0, funcValue=158064). SWNM's warm-start then propagates this bad result permanently. EKF handles poor starting points via sequential Bayesian updates and the NIS gate.

**How to apply:** When debugging SWNM behaviour at low utilisation, check the `funcValue` log from `InitEstimator: Fit complete`. If >> 10, the pair should have fallen back to EKF — verify `ekfFallbacks` is set in logs.
