package estimator

import "github.com/llm-inferno/model-tuner/pkg/core"

// GuessInitState derives initial alpha, beta, gamma from observed TTFT and ITL using the
// queueing model equations from the paper:
//
//	TTFT = alpha + (beta + gamma) * inputTokens           (eq 12)
//	ITL  = alpha + beta + gamma * (inputTokens + (outputTokens+1)/2)  (eq 13)
//
// gamma is the only parameter that is unidentifiable from a single operating point (the latency
// equations cannot separate beta from gamma without spread in the token sizes). When a valid
// seed = [alpha0, beta0, gamma0] is supplied (e.g. the config initState), GuessInitState pins
// gamma to the seed's gamma and solves alpha and beta from the observation — with gamma fixed
// the two equations make alpha, beta jointly identifiable, so no arbitrary constant is needed.
// This avoids the cold-start failure (issue #17) where a load/batch-induced latency excess is
// misattributed to gamma, inflating it ~20x into an infeasible regime. If that solve is
// degenerate (non-positive alpha/beta), it falls back to the full seed.
//
// With no usable seed it falls back to the legacy heuristic alpha = baseFactor * ITL. Returns
// nil if the derivation yields non-positive parameters and no seed is available.
func GuessInitState(env *core.EnvironmentPrefillDecode, seed []float64) []float64 {
	if env == nil || !env.Valid() {
		return nil
	}
	ttft := float64(env.AvgTTFT)
	itl := float64(env.AvgITL)
	inputToks := float64(env.AvgInputTokens)
	outputToks := float64(env.AvgOutputTokens)

	if ttft <= 0 || itl <= 0 || inputToks <= 0 || outputToks <= 0 {
		return nil
	}

	// Seed-anchored path: pin gamma to the seed, solve alpha and beta from the observation.
	if seedValid(seed) {
		gamma := seed[2]
		// From eq 12 - eq 13 with gamma fixed:
		//   TTFT - ITL = beta*(inputToks - 1) - gamma*(outputToks+1)/2
		if inputToks > 1 {
			beta := (ttft - itl + gamma*(outputToks+1)/2) / (inputToks - 1)
			alpha := itl - beta - gamma*(inputToks+(outputToks+1)/2)
			if alpha > 0 && beta > 0 {
				return []float64{alpha, beta, gamma}
			}
		}
		// Degenerate solve: fall back to the full seed (still feasible).
		return []float64{seed[0], seed[1], seed[2]}
	}

	// Legacy heuristic (no seed available).
	alpha := baseFactor * itl
	sumBetaGamma := (ttft - alpha) / inputToks
	if sumBetaGamma < 0 {
		return nil
	}

	denominator := inputToks + (outputToks+1)/2 - 1
	if denominator <= 0 {
		return nil
	}
	gamma := ((itl - alpha) - sumBetaGamma) / denominator
	beta := sumBetaGamma - gamma

	if alpha <= 0 || beta <= 0 || gamma <= 0 {
		return nil
	}
	return []float64{alpha, beta, gamma}
}

// seedValid reports whether seed is a usable [alpha, beta, gamma] anchor (3 positive entries).
func seedValid(seed []float64) bool {
	return len(seed) == 3 && seed[0] > 0 && seed[1] > 0 && seed[2] > 0
}
