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

	useSliding := os.Getenv(tunerservice.EstimatorModeEnvName) == "sliding-window"

	windowSize := tunerservice.DefaultWindowSize
	if v := os.Getenv(tunerservice.WindowSizeEnvName); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			windowSize = n
		}
	}

	residualThreshold := tunerservice.DefaultResidualThreshold
	if v := os.Getenv(tunerservice.ResidualThresholdEnvName); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
			residualThreshold = f
		}
	}

	initFitThreshold := tunerservice.DefaultInitFitThreshold
	if v := os.Getenv(tunerservice.InitFitThresholdEnvName); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
			initFitThreshold = f
		}
	}

	service := tunerservice.NewTunerService(warmUpCycles, initObs, holdBack, useSliding, windowSize, residualThreshold, initFitThreshold)
	server := tunerservice.NewTunerServer(service)

	estimatorMode := tunerservice.DefaultEstimatorMode
	if useSliding {
		estimatorMode = "sliding-window"
	}
	slog.Info("Starting TunerService",
		"host", host, "port", port,
		"warmUpCycles", warmUpCycles,
		"initObs", initObs,
		"holdBack", holdBack,
		"estimatorMode", estimatorMode,
		"windowSize", windowSize,
		"residualThreshold", residualThreshold,
		"initFitThreshold", initFitThreshold)
	if err := server.Run(host, port); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
