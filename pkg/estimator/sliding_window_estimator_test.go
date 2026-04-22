package estimator

import (
	"math"
	"testing"

	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
)

func TestSlidingWindowEstimator_IsReady(t *testing.T) {
	swe := NewSlidingWindowEstimator(5, 3, 0.5)
	env := makeTestEnv(10, 50, 5, 100, 500, 64)

	if swe.IsReady() {
		t.Fatal("should not be ready with 0 observations")
	}
	swe.AddObservation(env)
	if swe.IsReady() {
		t.Fatal("should not be ready with 1 observation (minObs=3)")
	}
	swe.AddObservation(env)
	swe.AddObservation(env)
	if !swe.IsReady() {
		t.Fatal("should be ready with 3 observations (minObs=3, windowSize=5)")
	}
}

func TestSlidingWindowEstimator_AddObservation_Caps(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 1, 0.5)
	e1 := makeTestEnv(10, 50, 5, 100, 500, 64)
	e2 := makeTestEnv(20, 60, 6, 110, 510, 64)
	e3 := makeTestEnv(30, 70, 7, 120, 520, 64)
	e4 := makeTestEnv(40, 80, 8, 130, 530, 64)

	swe.AddObservation(e1)
	swe.AddObservation(e2)
	swe.AddObservation(e3)
	swe.AddObservation(e4)

	if len(swe.window) != 3 {
		t.Fatalf("expected window size 3, got %d", len(swe.window))
	}
	if swe.window[0].Lambda != float64(e2.Lambda) {
		t.Errorf("expected oldest entry lambda=20, got %v", swe.window[0].Lambda)
	}
	if swe.window[2].Lambda != float64(e4.Lambda) {
		t.Errorf("expected newest entry lambda=40, got %v", swe.window[2].Lambda)
	}
}

func TestSlidingWindowEstimator_AddObservation_IgnoresNilAndInvalid(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 1, 0.5)
	swe.AddObservation(nil)
	invalid := core.NewEnvironmentPrefillDecode(10, 0, 0, 64, 100, 500, 0, 0)
	swe.AddObservation(invalid)
	if len(swe.window) != 0 {
		t.Fatalf("expected 0 observations stored, got %d", len(swe.window))
	}
}

func TestSlidingWindowEstimator_Seed(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 1, 0.5)
	obs := []fitObservation{
		{Lambda: 10}, {Lambda: 20}, {Lambda: 30}, {Lambda: 40}, {Lambda: 50},
	}
	swe.Seed(obs)
	if len(swe.window) != 3 {
		t.Fatalf("expected window capped at 3, got %d", len(swe.window))
	}
	if swe.window[0].Lambda != 30 {
		t.Errorf("expected lambda=30 at index 0, got %v", swe.window[0].Lambda)
	}
}

func TestSlidingWindowEstimator_Fit_EmptyError(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 1, 0.5)
	_, err := swe.Fit()
	if err == nil {
		t.Fatal("expected error from Fit() on empty window")
	}
}

func TestSlidingWindowEstimator_Fit_ParameterRecovery(t *testing.T) {
	trueAlpha := float32(16.78)
	trueBeta := float32(0.073)
	trueGamma := float32(0.00228)
	maxBatch := 64

	type opPoint struct {
		lambda float32
		inTok  float32
		outTok float32
	}
	ops := []opPoint{
		{lambda: 10, inTok: 90, outTok: 670},
		{lambda: 15, inTok: 120, outTok: 900},
		{lambda: 20, inTok: 150, outTok: 1100},
		{lambda: 12, inTok: 100, outTok: 750},
	}

	swe := NewSlidingWindowEstimator(len(ops), 1, 0.5)
	for _, op := range ops {
		qConfig := &analyzer.Configuration{
			MaxBatchSize: maxBatch,
			ServiceParms: &analyzer.ServiceParms{Alpha: trueAlpha, Beta: trueBeta, Gamma: trueGamma},
		}
		requestSize := &analyzer.RequestSize{AvgInputTokens: op.inTok, AvgOutputTokens: op.outTok}
		qa, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
		if err != nil {
			t.Fatalf("failed to create analyzer: %v", err)
		}
		metrics, err := qa.Analyze(op.lambda / 60)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}
		env := core.NewEnvironmentPrefillDecode(op.lambda, 0, 0, maxBatch, op.inTok, op.outTok,
			metrics.AvgTTFT, metrics.AvgTokenTime)
		swe.AddObservation(env)
	}

	if !swe.IsReady() {
		t.Fatal("expected window to be ready")
	}
	fitted, err := swe.Fit()
	if err != nil {
		t.Fatalf("Fit() returned error: %v", err)
	}

	tolerance := 0.10
	checkParam := func(name string, got, want float64) {
		t.Helper()
		relErr := math.Abs(got-want) / want
		if relErr > tolerance {
			t.Errorf("param %s: got %.6f, want %.6f (%.1f%% > %.0f%%)",
				name, got, want, relErr*100, tolerance*100)
		}
	}
	checkParam("alpha", fitted[0], float64(trueAlpha))
	checkParam("beta", fitted[1], float64(trueBeta))
	checkParam("gamma", fitted[2], float64(trueGamma))
}

func TestSlidingWindowEstimator_Fit_ResidualRejection(t *testing.T) {
	trueAlpha := float32(16.78)
	trueBeta := float32(0.073)
	trueGamma := float32(0.00228)
	maxBatch := 64

	type opPoint struct {
		lambda float32
		inTok  float32
		outTok float32
	}
	cleanOps := []opPoint{
		{lambda: 10, inTok: 90, outTok: 670},
		{lambda: 15, inTok: 120, outTok: 900},
		{lambda: 20, inTok: 150, outTok: 1100},
		{lambda: 12, inTok: 100, outTok: 750},
	}

	swe := NewSlidingWindowEstimator(5, 1, 0.5)

	addSynthetic := func(op opPoint) {
		qConfig := &analyzer.Configuration{
			MaxBatchSize: maxBatch,
			ServiceParms: &analyzer.ServiceParms{Alpha: trueAlpha, Beta: trueBeta, Gamma: trueGamma},
		}
		requestSize := &analyzer.RequestSize{AvgInputTokens: op.inTok, AvgOutputTokens: op.outTok}
		qa, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
		if err != nil {
			t.Fatalf("failed to create analyzer: %v", err)
		}
		metrics, err := qa.Analyze(op.lambda / 60)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}
		env := core.NewEnvironmentPrefillDecode(op.lambda, 0, 0, maxBatch, op.inTok, op.outTok,
			metrics.AvgTTFT, metrics.AvgTokenTime)
		swe.AddObservation(env)
	}

	for _, op := range cleanOps {
		addSynthetic(op)
	}
	outlier := core.NewEnvironmentPrefillDecode(10, 0, 0, maxBatch, 90, 670, 500, 5)
	swe.AddObservation(outlier)

	if !swe.IsReady() {
		t.Fatal("expected window to be ready")
	}
	fitted, err := swe.Fit()
	if err != nil {
		t.Fatalf("Fit() with outlier returned error: %v", err)
	}

	tolerance := 0.15
	checkParam := func(name string, got, want float64) {
		t.Helper()
		relErr := math.Abs(got-want) / want
		if relErr > tolerance {
			t.Errorf("param %s: got %.6f, want %.6f (%.1f%% > %.0f%%) — outlier not rejected?",
				name, got, want, relErr*100, tolerance*100)
		}
	}
	checkParam("alpha", fitted[0], float64(trueAlpha))
	checkParam("beta", fitted[1], float64(trueBeta))
	checkParam("gamma", fitted[2], float64(trueGamma))
}

func TestSlidingWindowEstimator_FilterOutliers_DisabledWhenThresholdZero(t *testing.T) {
	swe := NewSlidingWindowEstimator(3, 1, 0)
	obs := []fitObservation{
		{Lambda: 10, AvgTTFT: 50, AvgITL: 5, AvgInputTokens: 100, AvgOutputTokens: 500, MaxBatch: 64},
		{Lambda: 20, AvgTTFT: 60, AvgITL: 6, AvgInputTokens: 110, AvgOutputTokens: 510, MaxBatch: 64},
	}
	x := []float64{1.0, 0.01, 0.001}
	kept := swe.filterOutliers(obs, x)
	if len(kept) != len(obs) {
		t.Errorf("expected all %d kept when threshold=0, got %d", len(obs), len(kept))
	}
}
