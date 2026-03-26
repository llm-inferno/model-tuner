package core

import (
	"fmt"

	kalman "github.com/llm-inferno/kalman-filter/pkg/core"
	"gonum.org/v1/gonum/mat"
)

// Stasher saves and restores the EKF state (X) and covariance (P) for rollback on validation failure.
type Stasher struct {
	filter *kalman.ExtendedKalmanFilter
	X      *mat.VecDense
	P      *mat.Dense
}

// NewStasher creates a Stasher bound to the given filter.
func NewStasher(filter *kalman.ExtendedKalmanFilter) (*Stasher, error) {
	if filter == nil {
		return nil, fmt.Errorf("filter cannot be nil")
	}
	return &Stasher{filter: filter}, nil
}

// Stash saves a copy of the current filter state and covariance.
func (s *Stasher) Stash() error {
	if s.filter.X == nil || s.filter.P == nil {
		return fmt.Errorf("filter state or covariance is nil")
	}
	s.X = mat.VecDenseCopyOf(s.filter.X)
	s.P = mat.DenseCopyOf(s.filter.P)
	return nil
}

// UnStash restores the previously stashed state and covariance into the filter.
func (s *Stasher) UnStash() error {
	if s.X == nil || s.P == nil {
		return fmt.Errorf("no stashed state available")
	}
	s.filter.X.CloneFromVec(s.X)
	s.filter.P.Copy(s.P)
	return nil
}
