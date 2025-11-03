package observer

import "github.com/llm-inferno/model-tuner/pkg/core"

// Observer interface for getting the environment
type Observer interface {
	GetEnvironment() core.Environment
}

// abstract class
type BaseObserver struct {
}
