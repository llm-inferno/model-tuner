package main

import (
	"log"
	"log/slog"
	"os"
	"strconv"

	pkgconfig "github.com/llm-inferno/model-tuner/pkg/config"
	"github.com/llm-inferno/model-tuner/tunerservice"
)

func main() {
	host := os.Getenv(tunerservice.TunerHostEnvName)
	if host == "" {
		host = tunerservice.DefaultTunerHost
	}
	port := os.Getenv(tunerservice.TunerPortEnvName)
	if port == "" {
		port = tunerservice.DefaultTunerPort
	}

	warmUpCycles := pkgconfig.DefaultWarmUpCycles
	if v := os.Getenv(tunerservice.WarmUpCyclesEnvName); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			warmUpCycles = n
		}
	}

	initObs := tunerservice.DefaultInitObs
	if v := os.Getenv(tunerservice.InitObsEnvName); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			initObs = n
		}
	}

	holdBack := tunerservice.DefaultInitHoldBack
	if v := os.Getenv(tunerservice.InitHoldBackEnvName); v != "" {
		holdBack = v == "true" || v == "1"
	}

	service := tunerservice.NewTunerService(warmUpCycles, initObs, holdBack)
	server := tunerservice.NewTunerServer(service)

	slog.Info("Starting TunerService",
		"host", host, "port", port,
		"warmUpCycles", warmUpCycles,
		"initObs", initObs,
		"holdBack", holdBack)
	if err := server.Run(host, port); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
