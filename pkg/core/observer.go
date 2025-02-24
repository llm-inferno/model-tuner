package core

type Observer interface {
	GetEnvironment() *Environment
}

// abstract class
type BaseObserver struct {
}
