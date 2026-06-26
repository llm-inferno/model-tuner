package estimator

import (
	"math"
	"reflect"
	"testing"
)

// When the current window is unidentifiable (all observations at one operating point),
// Fit must not adopt the resulting degenerate solution; it holds the last good fit.
func TestSlidingWindowFit_HoldsLastGoodWhenIllConditioned(t *testing.T) {
	swe := NewSlidingWindowEstimator(10, 2, 0)
	swe.SetMaxConditionNumber(1000)

	lastGood := []float64{8.0, 0.016, 0.0005}
	swe.SeedLastFit(lastGood)

	// Collinear window: every observation at the same operating point.
	obsParams := []float64{8.0, 0.016, 0.0005}
	swe.Seed([]fitObservation{
		mkObs(t, obsParams, 15, 2000, 1000, 128, 2048),
		mkObs(t, obsParams, 15, 2000, 1000, 128, 2048),
		mkObs(t, obsParams, 15, 2000, 1000, 128, 2048),
	})

	got, err := swe.Fit()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(got, lastGood) {
		t.Fatalf("expected Fit to hold last-good params %v on ill-conditioned window, got %v", lastGood, got)
	}
}

// With no prior fit to fall back on, an ill-conditioned window must yield the analytical
// GuessInitState (positive, plausible) rather than a degenerate Nelder-Mead solution.
func TestSlidingWindowFit_FallsBackToGuessWhenIllConditionedNoPrior(t *testing.T) {
	swe := NewSlidingWindowEstimator(10, 2, 0)
	swe.SetMaxConditionNumber(1000)

	obsParams := []float64{8.0, 0.016, 0.0005}
	swe.Seed([]fitObservation{
		mkObs(t, obsParams, 15, 2000, 1000, 128, 2048),
		mkObs(t, obsParams, 15, 2000, 1000, 128, 2048),
	})

	got, err := swe.Fit()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := GuessInitState(swe.window[len(swe.window)-1].toEnv())
	if want == nil {
		t.Fatal("test setup: GuessInitState returned nil")
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("expected GuessInitState fallback %v, got %v", want, got)
	}
}

// The InitEstimator must not graduate warm-up on a degenerate fit: an ill-conditioned
// observation set falls back to the analytical GuessInitState.
func TestInitEstimatorFit_FallsBackToGuessWhenIllConditioned(t *testing.T) {
	ie := NewInitEstimator(2, true)
	ie.SetMaxConditionNumber(1000)

	p := []float64{8.0, 0.016, 0.0005}
	ie.observations = append(ie.observations,
		mkObs(t, p, 15, 2000, 1000, 128, 2048),
		mkObs(t, p, 15, 2000, 1000, 128, 2048),
	)

	got, err := ie.Fit()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := GuessInitState(ie.observations[0].toEnv())
	if want == nil {
		t.Fatal("test setup: GuessInitState returned nil")
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("expected GuessInitState fallback %v, got %v", want, got)
	}
}

// An ill-conditioned init fallback to GuessInitState is a deliberate, usable result — it
// must NOT report a MaxFloat64 funcValue, which the service interprets as a poor fit and
// uses to escalate the pair to the unguarded EKF path (defeating the guard).
func TestInitEstimatorFit_IllConditionedFallbackNotFlaggedPoor(t *testing.T) {
	ie := NewInitEstimator(2, true)
	ie.SetMaxConditionNumber(1000)

	p := []float64{8.0, 0.016, 0.0005}
	ie.observations = append(ie.observations,
		mkObs(t, p, 15, 2000, 1000, 128, 2048),
		mkObs(t, p, 15, 2000, 1000, 128, 2048),
	)

	if _, err := ie.Fit(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ie.LastFitFuncValue() == math.MaxFloat64 {
		t.Fatal("ill-conditioned GuessInitState fallback must not report MaxFloat64 funcValue " +
			"(would trigger EKF fallback in the service)")
	}
}

// HeldLastGoodFit reports true only when Fit held a genuine prior on an ill-conditioned
// window — the trigger for the transient EKF excursion (issue #19). It must be false after a
// well-excited fit and after the cold-start GuessInitState fallback (no prior to hold).
func TestSlidingWindowFit_HeldLastGoodFitSignal(t *testing.T) {
	collinear := []fitObservation{
		mkObs(t, []float64{8.0, 0.016, 0.0005}, 15, 2000, 1000, 128, 2048),
		mkObs(t, []float64{8.0, 0.016, 0.0005}, 15, 2000, 1000, 128, 2048),
	}

	// Ill-conditioned with a prior fit: signal true.
	held := NewSlidingWindowEstimator(10, 2, 0)
	held.SetMaxConditionNumber(1000)
	held.SeedLastFit([]float64{8.0, 0.016, 0.0005})
	held.Seed(collinear)
	if _, err := held.Fit(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !held.HeldLastGoodFit() {
		t.Fatal("expected HeldLastGoodFit true when holding a prior on ill-conditioned window")
	}

	// Ill-conditioned with no prior (cold start): signal false.
	cold := NewSlidingWindowEstimator(10, 2, 0)
	cold.SetMaxConditionNumber(1000)
	cold.Seed(collinear)
	if _, err := cold.Fit(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cold.HeldLastGoodFit() {
		t.Fatal("expected HeldLastGoodFit false on cold-start GuessInitState fallback (no prior)")
	}

	// Well-excited fit: signal false.
	good := NewSlidingWindowEstimator(10, 2, 0)
	good.SetMaxConditionNumber(1000)
	good.Seed([]fitObservation{
		mkObs(t, []float64{8.0, 0.016, 0.0005}, 12, 500, 400, 128, 2048),
		mkObs(t, []float64{8.0, 0.016, 0.0005}, 15, 1500, 1000, 128, 2048),
		mkObs(t, []float64{8.0, 0.016, 0.0005}, 18, 2500, 1600, 128, 2048),
	})
	if _, err := good.Fit(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if good.HeldLastGoodFit() {
		t.Fatal("expected HeldLastGoodFit false after a well-excited (accepted) fit")
	}
}

// A well-excited window must pass the guard and return the fitted params (not a fallback).
func TestSlidingWindowFit_AcceptsWellExcitedFit(t *testing.T) {
	swe := NewSlidingWindowEstimator(10, 2, 0)
	swe.SetMaxConditionNumber(1000)

	p := []float64{8.0, 0.016, 0.0005}
	swe.Seed([]fitObservation{
		mkObs(t, p, 12, 500, 400, 128, 2048),
		mkObs(t, p, 15, 1500, 1000, 128, 2048),
		mkObs(t, p, 18, 2500, 1600, 128, 2048),
	})

	got, err := swe.Fit()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Fit should recover params close to the generating values, not a fallback/degenerate set.
	if got[0] < 6 || got[0] > 10 || got[1] < 0.010 || got[2] < 1e-4 {
		t.Fatalf("expected well-excited fit near %v, got %v", p, got)
	}
}
