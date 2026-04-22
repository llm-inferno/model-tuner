package service

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

// LearnedParameters holds the tuned parameters for one model/accelerator pair.
type LearnedParameters struct {
	Alpha       float32
	Beta        float32
	Gamma       float32
	NIS         float64
	UpdateCount int
	Covariance  [][]float64
	LastUpdated time.Time
}

// CovarianceMatrix converts the stored slice representation back to a mat.Dense.
func (lp *LearnedParameters) CovarianceMatrix() *mat.Dense {
	n := len(lp.Covariance)
	if n == 0 {
		return nil
	}
	data := make([]float64, n*n)
	for i, row := range lp.Covariance {
		copy(data[i*n:], row)
	}
	return mat.NewDense(n, n, data)
}

// ParameterStore is a thread-safe in-memory store of LearnedParameters keyed by "modelName/accelerator".
type ParameterStore struct {
	mu     sync.RWMutex
	params map[string]*LearnedParameters
}

// NewParameterStore creates an empty ParameterStore.
func NewParameterStore() *ParameterStore {
	return &ParameterStore{params: make(map[string]*LearnedParameters)}
}

func makeKey(model, accelerator string) string {
	return fmt.Sprintf("%s/%s", model, accelerator)
}

// splitKey splits a "model/accelerator" key back into its components.
// If the model name itself contains slashes, only the last slash is used.
func splitKey(key string) (model, accelerator string) {
	idx := strings.LastIndex(key, "/")
	if idx < 0 {
		return key, ""
	}
	return key[:idx], key[idx+1:]
}

// Get returns the stored parameters for a model/accelerator pair, or nil if not found.
func (ps *ParameterStore) Get(model, accelerator string) *LearnedParameters {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	return ps.params[makeKey(model, accelerator)]
}

// Set stores parameters for a model/accelerator pair.
func (ps *ParameterStore) Set(model, accelerator string, params *LearnedParameters) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.params[makeKey(model, accelerator)] = params
}

// GetAll returns a snapshot of all stored parameters.
func (ps *ParameterStore) GetAll() map[string]*LearnedParameters {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	out := make(map[string]*LearnedParameters, len(ps.params))
	for k, v := range ps.params {
		out[k] = v
	}
	return out
}

func covToSlice(p *mat.Dense) [][]float64 {
	if p == nil {
		return nil
	}
	n, _ := p.Dims()
	out := make([][]float64, n)
	for i := range n {
		out[i] = make([]float64, n)
		for j := range n {
			out[i][j] = p.At(i, j)
		}
	}
	return out
}
