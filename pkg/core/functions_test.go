package core

import (
	"testing"

	"github.com/llm-inferno/model-tuner/pkg/config"
	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
	"gonum.org/v1/gonum/mat"
)

// TestSystemFunc_UsesEnvMaxQueueSize verifies that the EKF system function uses
// the MaxQueueSize stored on the environment rather than the hardcoded 10×maxBatchSize.
//
// Strategy: generate reference observations with MaxQueueSize=128 using the queue
// analyzer directly, then confirm the system function (fed an environment with
// MaxQueueSize=128) predicts the same values. A system function running with the
// old hardcoded 10×128=1280 queue would produce different predictions at load
// levels where queue occupancy matters.
func TestSystemFunc_UsesEnvMaxQueueSize(t *testing.T) {
	const (
		alpha        = float32(8.0)
		beta         = float32(0.016)
		gamma        = float32(0.0005)
		maxBatchSize = 64
		maxQueueSize = 128 // explicit — NOT 10×64=640
		lambda       = float32(200) // high load: queue depth is meaningful
		inTok        = float32(2048)
		outTok       = float32(1024)
	)

	// Reference metrics produced by the queue-analysis model with MaxQueueSize=128.
	refConfig := &analyzer.Configuration{
		MaxBatchSize: maxBatchSize,
		MaxQueueSize: maxQueueSize,
		ServiceParms: &analyzer.ServiceParms{Alpha: alpha, Beta: beta, Gamma: gamma},
	}
	qa, err := analyzer.NewLLMQueueAnalyzer(refConfig, &analyzer.RequestSize{
		AvgInputTokens: inTok, AvgOutputTokens: outTok,
	})
	if err != nil {
		t.Fatalf("create reference analyzer: %v", err)
	}
	ref, err := qa.Analyze(float32(lambda / 60))
	if err != nil {
		t.Fatalf("reference analyze: %v", err)
	}

	// Confirm that MaxQueueSize=640 (old heuristic) gives different predictions,
	// so the test is actually sensitive to the queue size choice.
	wrongConfig := &analyzer.Configuration{
		MaxBatchSize: maxBatchSize,
		MaxQueueSize: 10 * maxBatchSize, // old hardcoded value
		ServiceParms: &analyzer.ServiceParms{Alpha: alpha, Beta: beta, Gamma: gamma},
	}
	qaWrong, err := analyzer.NewLLMQueueAnalyzer(wrongConfig, &analyzer.RequestSize{
		AvgInputTokens: inTok, AvgOutputTokens: outTok,
	})
	if err != nil {
		t.Fatalf("create wrong analyzer: %v", err)
	}
	wrong, err := qaWrong.Analyze(float32(lambda / 60))
	if err != nil {
		t.Fatalf("wrong analyze: %v", err)
	}
	if ref.AvgTTFT == wrong.AvgTTFT && ref.AvgTokenTime == wrong.AvgTokenTime {
		t.Skip("MaxQueueSize=128 and MaxQueueSize=640 produce identical metrics at this load; choose higher lambda")
	}

	// Build a tuner with the environment carrying MaxQueueSize=128.
	cfg := &config.ConfigData{
		FilterData: config.FilterData{GammaFactor: 1.0, ErrorLevel: 0.5, TPercentile: 1.96},
		ModelData: config.ModelData{
			InitState:            []float64{float64(alpha), float64(beta), float64(gamma)},
			PercentChange:        []float64{10.0, 10.0, 10.0},
			BoundedState:         false,
			ExpectedObservations: []float64{float64(ref.AvgTTFT), float64(ref.AvgTokenTime)},
		},
	}
	env := NewEnvironmentPrefillDecode(lambda, 0, 0, maxBatchSize, inTok, outTok, ref.AvgTTFT, ref.AvgTokenTime)
	env.MaxQueueSize = maxQueueSize

	tuner, err := NewTuner(cfg, env)
	if err != nil {
		t.Fatalf("create tuner: %v", err)
	}

	// The system function evaluated at the true state should predict the reference metrics.
	sysFunc := NewQueueModelSystemFuncCreatorPrefillDecode(tuner).Create()
	state := mat.NewVecDense(3, []float64{float64(alpha), float64(beta), float64(gamma)})
	predicted := sysFunc(state)

	const tol = 0.01 // 1% relative tolerance
	gotTTFT := float32(predicted.AtVec(0))
	gotITL := float32(predicted.AtVec(1))

	if relErr(gotTTFT, ref.AvgTTFT) > tol {
		t.Errorf("TTFT: system func predicted %.4f, reference (MaxQueueSize=128) %.4f, wrong (MaxQueueSize=640) %.4f",
			gotTTFT, ref.AvgTTFT, wrong.AvgTTFT)
	}
	if relErr(gotITL, ref.AvgTokenTime) > tol {
		t.Errorf("ITL: system func predicted %.4f, reference (MaxQueueSize=128) %.4f, wrong (MaxQueueSize=640) %.4f",
			gotITL, ref.AvgTokenTime, wrong.AvgTokenTime)
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
