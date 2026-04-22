package estimator

import (
	"math"
	"testing"

	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
)

func makeTestEnv(lambda, ttft, itl float32, inTok, outTok float32, maxBatch int) *core.EnvironmentPrefillDecode {
	return core.NewEnvironmentPrefillDecode(
		lambda, 0, 0, maxBatch, inTok, outTok, ttft, itl,
	)
}

func makeTestEnvWithQueue(lambda, ttft, itl float32, inTok, outTok float32, maxBatch, maxQueue int) *core.EnvironmentPrefillDecode {
	env := core.NewEnvironmentPrefillDecode(lambda, 0, 0, maxBatch, inTok, outTok, ttft, itl)
	env.MaxQueueSize = maxQueue
	return env
}

func TestInitEstimator_HoldBack(t *testing.T) {
	ie := NewInitEstimator(3, true)
	if !ie.HoldBack() {
		t.Fatal("expected HoldBack=true")
	}
	ie2 := NewInitEstimator(3, false)
	if ie2.HoldBack() {
		t.Fatal("expected HoldBack=false")
	}
}

func TestInitEstimator_IsReady(t *testing.T) {
	ie := NewInitEstimator(3, true)
	env := makeTestEnv(10, 50, 5, 100, 500, 64)

	if ie.IsReady() {
		t.Fatal("should not be ready with 0 observations")
	}
	ie.AddObservation(env)
	if ie.IsReady() {
		t.Fatal("should not be ready with 1 observation")
	}
	ie.AddObservation(env)
	if ie.IsReady() {
		t.Fatal("should not be ready with 2 observations")
	}
	ie.AddObservation(env)
	if !ie.IsReady() {
		t.Fatal("should be ready with 3 observations (minObs=3)")
	}
}

func TestInitEstimator_AddObservation_IgnoresNilAndInvalid(t *testing.T) {
	ie := NewInitEstimator(3, true)
	ie.AddObservation(nil)
	invalid := core.NewEnvironmentPrefillDecode(10, 0, 0, 64, 100, 500, 0, 0)
	ie.AddObservation(invalid)
	if len(ie.observations) != 0 {
		t.Fatalf("expected 0 observations stored, got %d", len(ie.observations))
	}
}

func TestInitEstimator_AddObservation_StoresFields(t *testing.T) {
	ie := NewInitEstimator(1, false)
	env := makeTestEnvWithQueue(30, 45.5, 6.2, 120, 700, 64, 128)
	ie.AddObservation(env)
	if len(ie.observations) != 1 {
		t.Fatalf("expected 1 observation, got %d", len(ie.observations))
	}
	obs := ie.observations[0]
	if obs.Lambda != float64(env.Lambda) {
		t.Errorf("Lambda mismatch: got %v want %v", obs.Lambda, env.Lambda)
	}
	if obs.MaxBatch != env.MaxBatchSize {
		t.Errorf("MaxBatch mismatch: got %v want %v", obs.MaxBatch, env.MaxBatchSize)
	}
	if obs.MaxQueueSize != env.MaxQueueSize {
		t.Errorf("MaxQueueSize mismatch: got %v want %v", obs.MaxQueueSize, env.MaxQueueSize)
	}
	if obs.AvgTTFT != float64(env.AvgTTFT) {
		t.Errorf("AvgTTFT mismatch: got %v want %v", obs.AvgTTFT, env.AvgTTFT)
	}
}

func TestInitEstimator_Fit_ParameterRecovery(t *testing.T) {
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
		{lambda: 18, inTok: 165, outTok: 1250},
	}

	ie := NewInitEstimator(len(ops), false)

	for _, op := range ops {
		qConfig := &analyzer.Configuration{
			MaxBatchSize: maxBatch,
			MaxQueueSize: 0,
			ServiceParms: &analyzer.ServiceParms{
				Alpha: trueAlpha,
				Beta:  trueBeta,
				Gamma: trueGamma,
			},
		}
		requestSize := &analyzer.RequestSize{
			AvgInputTokens:  op.inTok,
			AvgOutputTokens: op.outTok,
		}
		qa, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
		if err != nil {
			t.Fatalf("failed to create analyzer: %v", err)
		}
		metrics, err := qa.Analyze(op.lambda / 60)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}

		env := core.NewEnvironmentPrefillDecode(
			op.lambda, 0, 0, maxBatch,
			op.inTok, op.outTok,
			metrics.AvgTTFT, metrics.AvgTokenTime,
		)
		ie.AddObservation(env)
	}

	if !ie.IsReady() {
		t.Fatal("estimator should be ready after adding all observations")
	}

	fitted, err := ie.Fit()
	if err != nil {
		t.Fatalf("Fit() returned error: %v", err)
	}
	if len(fitted) != 3 {
		t.Fatalf("expected 3 params, got %d", len(fitted))
	}

	tolerance := 0.10
	checkParam := func(name string, got, want float64) {
		t.Helper()
		relErr := math.Abs(got-want) / want
		if relErr > tolerance {
			t.Errorf("param %s: got %.6f, want %.6f (relative error %.1f%% > %.0f%%)",
				name, got, want, relErr*100, tolerance*100)
		}
	}
	checkParam("alpha", fitted[0], float64(trueAlpha))
	checkParam("beta", fitted[1], float64(trueBeta))
	checkParam("gamma", fitted[2], float64(trueGamma))
}

func TestInitEstimator_Fit_PoorStartingPoint(t *testing.T) {
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
		{lambda: 18, inTok: 165, outTok: 1250},
	}

	ie := NewInitEstimator(len(ops), false)

	for _, op := range ops {
		qConfig := &analyzer.Configuration{
			MaxBatchSize: maxBatch,
			MaxQueueSize: 0,
			ServiceParms: &analyzer.ServiceParms{
				Alpha: trueAlpha,
				Beta:  trueBeta,
				Gamma: trueGamma,
			},
		}
		requestSize := &analyzer.RequestSize{
			AvgInputTokens:  op.inTok,
			AvgOutputTokens: op.outTok,
		}
		qa, err := analyzer.NewLLMQueueAnalyzer(qConfig, requestSize)
		if err != nil {
			t.Fatalf("failed to create analyzer: %v", err)
		}
		metrics, err := qa.Analyze(op.lambda / 60)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}
		env := core.NewEnvironmentPrefillDecode(
			op.lambda, 0, 0, maxBatch,
			op.inTok, op.outTok,
			metrics.AvgTTFT, metrics.AvgTokenTime,
		)
		ie.AddObservation(env)
	}

	poorX0 := []float64{1.0, 0.01, 0.0001}
	fitted, err := ie.fitWithX0(poorX0)
	if err != nil {
		t.Fatalf("fitWithX0() returned error: %v", err)
	}
	if len(fitted) != 3 {
		t.Fatalf("expected 3 params, got %d", len(fitted))
	}

	tolerance := 0.10
	checkParam := func(name string, got, want float64) {
		t.Helper()
		relErr := math.Abs(got-want) / want
		if relErr > tolerance {
			t.Errorf("param %s: got %.6f, want %.6f (relative error %.1f%% > %.0f%%)",
				name, got, want, relErr*100, tolerance*100)
		}
	}
	checkParam("alpha", fitted[0], float64(trueAlpha))
	checkParam("beta", fitted[1], float64(trueBeta))
	checkParam("gamma", fitted[2], float64(trueGamma))
}

func TestInitEstimator_LastFitFuncValue_ZeroBeforeFit(t *testing.T) {
	ie := NewInitEstimator(1, false)
	if ie.LastFitFuncValue() != 0 {
		t.Errorf("expected 0 before Fit(), got %f", ie.LastFitFuncValue())
	}
}

func TestInitEstimator_LastFitFuncValue_NonNegativeAfterFit(t *testing.T) {
	ie := NewInitEstimator(1, false)
	ie.AddObservation(makeTestEnv(15, 55, 6, 120, 700, 64))
	if _, err := ie.Fit(); err != nil {
		t.Fatalf("Fit() returned error: %v", err)
	}
	fv := ie.LastFitFuncValue()
	if fv < 0 {
		t.Errorf("expected non-negative funcValue after Fit(), got %f", fv)
	}
}

func TestInitEstimator_LastFitFuncValue_MaxFloatOnFallback(t *testing.T) {
	ie := NewInitEstimator(1, false)
	ie.AddObservation(makeTestEnv(15, 55, 6, 120, 700, 64))
	ie.fitWithX0([]float64{0, 0, 0})
	if ie.LastFitFuncValue() != math.MaxFloat64 {
		t.Errorf("expected math.MaxFloat64 after fallback, got %f", ie.LastFitFuncValue())
	}
}
