package estimator

import (
	"math"

	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
	"gonum.org/v1/gonum/mat"
)

// residualVector returns the per-observation relative residuals (dTTFT, dITL) for params
// x=[alpha,beta,gamma], evaluated via the full queueing model. The boolean is false if any
// observation cannot be evaluated (model error or non-positive value).
func residualVector(obs []fitObservation, x []float64) ([]float64, bool) {
	if x[0] <= 0 || x[1] <= 0 || x[2] <= 0 {
		return nil, false
	}
	r := make([]float64, 0, 2*len(obs))
	for _, o := range obs {
		qConfig := &analyzer.Configuration{
			MaxBatchSize: o.MaxBatch,
			MaxQueueSize: o.MaxQueueSize,
			ServiceParms: &analyzer.ServiceParms{
				Alpha: float32(x[0]), Beta: float32(x[1]), Gamma: float32(x[2]),
			},
		}
		requestSize := &analyzer.RequestSize{
			AvgInputTokens:  o.AvgInputTokens,
			AvgOutputTokens: o.AvgOutputTokens,
		}
		qa, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
		if err != nil {
			return nil, false
		}
		metrics, err := qa.Analyze(float32(o.Lambda / 60))
		if err != nil {
			return nil, false
		}
		ttftModel := float64(metrics.AvgTTFT)
		itlModel := float64(metrics.AvgTokenTime)
		if o.AvgTTFT <= 0 || o.AvgITL <= 0 || ttftModel <= 0 || itlModel <= 0 {
			return nil, false
		}
		r = append(r, (ttftModel-o.AvgTTFT)/o.AvgTTFT, (itlModel-o.AvgITL)/o.AvgITL)
	}
	return r, true
}

// fitConditionNumber estimates the practical identifiability of a fit: the ratio of the
// largest to smallest singular value of the residual Jacobian taken with respect to the
// log of each parameter (relative perturbations, so the measure is scale-invariant across
// alpha~O(10), beta~O(0.01), gamma~O(1e-4)). A parameter direction the data cannot pin
// down — e.g. beta and gamma when every observation shares one token operating point —
// produces a vanishing singular value and hence a very large (or infinite) condition
// number. Returns +Inf when the window is underdetermined (fewer residuals than
// parameters) or cannot be evaluated.
func fitConditionNumber(obs []fitObservation, x []float64) float64 {
	const relEps = 1e-3
	m := 2 * len(obs)
	n := len(x)
	if n == 0 || m < n {
		return math.Inf(1)
	}
	jac := mat.NewDense(m, n, nil)
	for k := 0; k < n; k++ {
		up := append([]float64(nil), x...)
		dn := append([]float64(nil), x...)
		up[k] = x[k] * (1 + relEps)
		dn[k] = x[k] * (1 - relEps)
		rUp, okUp := residualVector(obs, up)
		rDn, okDn := residualVector(obs, dn)
		if !okUp || !okDn {
			return math.Inf(1)
		}
		// Central difference w.r.t. ln(x_k): d(ln x_k) = relEps, so the column is
		// (rUp - rDn) / (2*relEps).
		for i := 0; i < m; i++ {
			jac.Set(i, k, (rUp[i]-rDn[i])/(2*relEps))
		}
	}
	var svd mat.SVD
	if !svd.Factorize(jac, mat.SVDThin) {
		return math.Inf(1)
	}
	sv := svd.Values(nil) // descending order
	if len(sv) == 0 {
		return math.Inf(1)
	}
	sMax := sv[0]
	sMin := sv[len(sv)-1]
	if sMin <= 0 {
		return math.Inf(1)
	}
	return sMax / sMin
}
