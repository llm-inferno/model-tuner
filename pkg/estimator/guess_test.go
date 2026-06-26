package estimator

import (
	"math"
	"testing"

	"github.com/llm-inferno/model-tuner/pkg/core"
)

// env builds a prefill-decode environment from the fields GuessInitState reads.
func env(ttft, itl, inTok, outTok float32) *core.EnvironmentPrefillDecode {
	return core.NewEnvironmentPrefillDecode(100, 0, 0, 128, inTok, outTok, ttft, itl)
}

// With a seed, GuessInitState pins gamma to the seed and solves alpha/beta from the observation,
// keeping gamma feasible — unlike the legacy heuristic, which inflates gamma ~20x on this run17
// operating point (issue #17).
func TestGuessInitState_SeedAnchored_PinsGammaSolvesAlphaBeta(t *testing.T) {
	seed := []float64{5.0, 0.05, 0.00005} // default config initState
	e := env(80.48, 15.911, 1123, 487)    // run17 armB-high cycle 1

	got := GuessInitState(e, seed)
	if got == nil {
		t.Fatal("expected params, got nil")
	}
	if got[2] != seed[2] {
		t.Fatalf("gamma must be pinned to seed: want %g, got %g", seed[2], got[2])
	}
	if got[0] <= 0 || got[1] <= 0 {
		t.Fatalf("expected positive alpha/beta, got alpha=%g beta=%g", got[0], got[1])
	}
	// alpha/beta solved from the observation (verified by hand: alpha~15.79, beta~0.0576).
	if math.Abs(got[0]-15.79) > 0.1 || math.Abs(got[1]-0.0576) > 0.001 {
		t.Errorf("alpha/beta off: got alpha=%g beta=%g (want ~15.79, ~0.0576)", got[0], got[1])
	}

	// Contrast: the legacy (nil-seed) path inflates gamma far above the feasible seed.
	legacy := GuessInitState(e, nil)
	if legacy == nil {
		t.Fatal("legacy guess returned nil")
	}
	if legacy[2] <= 10*seed[2] {
		t.Fatalf("expected legacy gamma to be inflated vs seed; legacy=%g seed=%g", legacy[2], seed[2])
	}
	if got[2] >= legacy[2] {
		t.Errorf("seed-anchored gamma (%g) should be far below legacy gamma (%g)", got[2], legacy[2])
	}
}

// When the seed-anchored solve is degenerate (here TTFT < ITL drives beta negative), GuessInitState
// returns the full seed rather than emitting non-positive params (WI-3 last resort).
func TestGuessInitState_SeedAnchored_DegenerateFallsBackToSeed(t *testing.T) {
	seed := []float64{5.0, 0.05, 0.00005}
	e := env(5, 20, 1000, 400) // TTFT < ITL → beta solve goes negative

	got := GuessInitState(e, seed)
	if got == nil {
		t.Fatal("expected seed fallback, got nil")
	}
	for i := range seed {
		if got[i] != seed[i] {
			t.Fatalf("expected full seed on degenerate solve: want %v, got %v", seed, got)
		}
	}
}

// A nil (or invalid) seed reproduces the legacy heuristic alpha = baseFactor * ITL.
func TestGuessInitState_NilSeed_Legacy(t *testing.T) {
	e := env(80.48, 15.911, 1123, 487)

	got := GuessInitState(e, nil)
	if got == nil {
		t.Fatal("expected legacy params, got nil")
	}
	if math.Abs(got[0]-baseFactor*15.911) > 1e-6 {
		t.Errorf("legacy alpha should be baseFactor*ITL=%g, got %g", baseFactor*15.911, got[0])
	}

	// An under-length seed is treated as no seed.
	if alt := GuessInitState(e, []float64{1.0, 2.0}); alt == nil || math.Abs(alt[0]-got[0]) > 1e-9 {
		t.Errorf("invalid seed should fall through to legacy path")
	}
}
