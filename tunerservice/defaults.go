package tunerservice

const (
	CollectorHostEnvName = "COLLECTOR_HOST"
	CollectorPortEnvName = "COLLECTOR_PORT"
)

const CollectEnvVerb = "getenv"

const DefaultTunerPeriodSeconds int = 60 // periodicity of tuner

var CollectorURL string
