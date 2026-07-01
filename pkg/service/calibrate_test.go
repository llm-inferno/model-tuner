package service

import (
	"testing"

	"github.com/llm-inferno/queue-analysis/pkg/analyzer"

	optconfig "github.com/llm-inferno/optimizer-light/pkg/config"
)

// groundTruthMetrics runs the queue-analysis model at a given operating point with known
// parameters to produce a consistent (TTFT, ITL) pair — the calibration sweep's "measurement".
func groundTruthMetrics(t *testing.T, rpm, inTok, outTok float64, maxBatch int, p [3]float64) (ttft, itl float32) {
	t.Helper()
	qa, err := analyzer.NewLLMQueueAnalyzer(
		&analyzer.Configuration{
			MaxBatchSize: maxBatch,
			MaxQueueSize: 0,
			ServiceParms: &analyzer.ServiceParms{Alpha: float32(p[0]), Beta: float32(p[1]), Gamma: float32(p[2])},
		},
		&analyzer.RequestSize{AvgInputTokens: float32(inTok), AvgOutputTokens: float32(outTok)},
	)
	if err != nil {
		t.Fatalf("build analyzer: %v", err)
	}
	m, err := qa.Analyze(float32(rpm / 60.0))
	if err != nil {
		t.Fatalf("analyze rpm=%.1f: %v", rpm, err)
	}
	return m.AvgTTFT, m.AvgTokenTime
}

// sweepSpec builds one synthetic sweep point as a ServerSpec, mirroring what the Collector's
// /sweep handler produces from a measured /simulate result.
func sweepSpec(t *testing.T, model, acc string, rpm, inTok, outTok float64, maxBatch int, p [3]float64) optconfig.ServerSpec {
	ttft, itl := groundTruthMetrics(t, rpm, inTok, outTok, maxBatch, p)
	return optconfig.ServerSpec{
		Name:         model,
		Model:        model,
		MaxBatchSize: maxBatch,
		CurrentAlloc: optconfig.AllocationData{
			Accelerator: acc,
			MaxBatch:    maxBatch,
			TTFTAverage: ttft,
			ITLAverage:  itl,
			Load: optconfig.ServerLoadSpec{
				ArrivalRate:  float32(rpm),
				AvgInTokens:  int(inTok),
				AvgOutTokens: int(outTok),
			},
		},
	}
}

// TestCalibrate_RecoversParametersFromSweep verifies the heart of benchmarking-on-the-fly: a
// deliberately-diverse batch (arrival-rate ramp + skewed token mixes) lets the joint fit recover
// performance parameters well enough to predict a held-out operating point — something a single
// operating point cannot do. The recovered params need not match the originals exactly (gamma is
// weakly excited at low occupancy), so we assert generalisation: prediction at a holdout point.
func TestCalibrate_RecoversParametersFromSweep(t *testing.T) {
	const model, acc = "qwen_2_5_14b", "H100"
	const maxBatch = 128
	truth := [3]float64{12.0, 0.04, 0.00006}

	specs := []optconfig.ServerSpec{
		sweepSpec(t, model, acc, 30, 512, 256, maxBatch, truth),
		sweepSpec(t, model, acc, 60, 512, 256, maxBatch, truth),
		sweepSpec(t, model, acc, 90, 512, 256, maxBatch, truth),
		sweepSpec(t, model, acc, 120, 512, 256, maxBatch, truth),
		sweepSpec(t, model, acc, 150, 512, 256, maxBatch, truth),
		sweepSpec(t, model, acc, 60, 1024, 128, maxBatch, truth), // input-heavy: excites beta
		sweepSpec(t, model, acc, 60, 256, 512, maxBatch, truth),  // output-heavy: excites gamma
	}

	ts := NewTunerService(3, 3, false, false, DefaultWindowSize, DefaultResidualThreshold, 0)
	ts.SetMaxConditionNumber(DefaultMaxConditionNumber)

	md, err := ts.Calibrate(specs)
	if err != nil {
		t.Fatalf("Calibrate failed: %v", err)
	}
	if len(md.PerfData) != 1 {
		t.Fatalf("expected 1 calibrated entry, got %d", len(md.PerfData))
	}

	params := ts.GetParams(model, acc)
	if params == nil {
		t.Fatal("expected calibrated params in store, got nil")
	}
	if params.Alpha <= 0 || params.Beta <= 0 || params.Gamma <= 0 {
		t.Fatalf("non-physical calibrated params: alpha=%g beta=%g gamma=%g",
			params.Alpha, params.Beta, params.Gamma)
	}
	// Graduated so the warm-up gate no longer blocks the pair.
	if params.UpdateCount < 3 {
		t.Fatalf("expected graduated UpdateCount >= warmUpCycles(3), got %d", params.UpdateCount)
	}

	// Generalisation: predict a held-out operating point with the recovered params and compare to
	// the ground truth. Within the swept regime, an identifiable fit must predict it closely.
	wantTTFT, wantITL := groundTruthMetrics(t, 75, 768, 192, maxBatch, truth)
	gotTTFT, gotITL := groundTruthMetrics(t, 75, 768, 192, maxBatch,
		[3]float64{float64(params.Alpha), float64(params.Beta), float64(params.Gamma)})

	if rel := relErr(gotTTFT, wantTTFT); rel > 0.20 {
		t.Errorf("holdout TTFT off by %.1f%% (got %.2f want %.2f)", rel*100, gotTTFT, wantTTFT)
	}
	if rel := relErr(gotITL, wantITL); rel > 0.20 {
		t.Errorf("holdout ITL off by %.1f%% (got %.2f want %.2f)", rel*100, gotITL, wantITL)
	}
}

// TestCalibrate_RejectsSingleOperatingPoint verifies the identifiability guard: a batch with no
// operating-point spread (the same point repeated) is the unidentifiable case the feature exists to
// avoid, and Calibrate must reject it rather than store a degenerate fit.
func TestCalibrate_RejectsSingleOperatingPoint(t *testing.T) {
	const model, acc = "qwen_2_5_14b", "H100"
	const maxBatch = 128
	truth := [3]float64{12.0, 0.04, 0.00006}

	specs := []optconfig.ServerSpec{
		sweepSpec(t, model, acc, 60, 512, 256, maxBatch, truth),
		sweepSpec(t, model, acc, 60, 512, 256, maxBatch, truth),
		sweepSpec(t, model, acc, 60, 512, 256, maxBatch, truth),
	}

	ts := NewTunerService(3, 3, false, false, DefaultWindowSize, DefaultResidualThreshold, 0)
	ts.SetMaxConditionNumber(DefaultMaxConditionNumber)

	if _, err := ts.Calibrate(specs); err == nil {
		t.Fatal("expected Calibrate to reject an ill-conditioned single-operating-point batch, got nil error")
	}
	if ts.GetParams(model, acc) != nil {
		t.Fatal("expected no stored params after a rejected calibration")
	}
}

// TestCalibrate_ResponseExcludesGroupsThatFailedThisCall verifies that a group which fails
// calibration in the current call does not leak previously-stored params into the response. A
// pair is first calibrated and stored; a later call re-submits that pair as an unidentifiable
// single-operating-point batch (rejected) alongside a second pair that calibrates cleanly. The
// response must contain only the freshly-calibrated pair, not the stale-params rejected one.
func TestCalibrate_ResponseExcludesGroupsThatFailedThisCall(t *testing.T) {
	const acc = "H100"
	const staleModel, freshModel = "qwen_2_5_14b", "llama_3_8b"
	const maxBatch = 128
	truth := [3]float64{12.0, 0.04, 0.00006}

	ts := NewTunerService(3, 3, false, false, DefaultWindowSize, DefaultResidualThreshold, 0)
	ts.SetMaxConditionNumber(DefaultMaxConditionNumber)

	// Pre-populate the store: calibrate the stale pair with a proper spread.
	seed := []optconfig.ServerSpec{
		sweepSpec(t, staleModel, acc, 30, 512, 256, maxBatch, truth),
		sweepSpec(t, staleModel, acc, 60, 512, 256, maxBatch, truth),
		sweepSpec(t, staleModel, acc, 90, 512, 256, maxBatch, truth),
		sweepSpec(t, staleModel, acc, 120, 512, 256, maxBatch, truth),
		sweepSpec(t, staleModel, acc, 60, 1024, 128, maxBatch, truth),
		sweepSpec(t, staleModel, acc, 60, 256, 512, maxBatch, truth),
	}
	if _, err := ts.Calibrate(seed); err != nil {
		t.Fatalf("initial Calibrate failed: %v", err)
	}
	if ts.GetParams(staleModel, acc) == nil {
		t.Fatal("expected stored params after initial calibration")
	}

	// Mixed call: the stale pair as an unidentifiable batch (rejected) plus a fresh pair that
	// calibrates cleanly.
	mixed := []optconfig.ServerSpec{
		sweepSpec(t, staleModel, acc, 60, 512, 256, maxBatch, truth),
		sweepSpec(t, staleModel, acc, 60, 512, 256, maxBatch, truth),
		sweepSpec(t, staleModel, acc, 60, 512, 256, maxBatch, truth),
		sweepSpec(t, freshModel, acc, 30, 512, 256, maxBatch, truth),
		sweepSpec(t, freshModel, acc, 60, 512, 256, maxBatch, truth),
		sweepSpec(t, freshModel, acc, 90, 512, 256, maxBatch, truth),
		sweepSpec(t, freshModel, acc, 120, 512, 256, maxBatch, truth),
		sweepSpec(t, freshModel, acc, 60, 1024, 128, maxBatch, truth),
		sweepSpec(t, freshModel, acc, 60, 256, 512, maxBatch, truth),
	}
	md, err := ts.Calibrate(mixed)
	if err != nil {
		t.Fatalf("mixed Calibrate failed: %v", err)
	}
	if len(md.PerfData) != 1 {
		t.Fatalf("expected exactly 1 entry (the fresh pair), got %d: %+v", len(md.PerfData), md.PerfData)
	}
	if md.PerfData[0].Name != freshModel {
		t.Fatalf("expected only the freshly-calibrated pair %q in response, got %q (stale params leaked)",
			freshModel, md.PerfData[0].Name)
	}
}

func relErr(got, want float32) float64 {
	if want == 0 {
		return 0
	}
	d := float64(got - want)
	if d < 0 {
		d = -d
	}
	return d / float64(want)
}
