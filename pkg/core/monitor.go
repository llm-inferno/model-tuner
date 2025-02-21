package core

type Monitor interface {
	GetEnvironment() *Environment
}

// abstract class
type BaseMonitor struct {
}
