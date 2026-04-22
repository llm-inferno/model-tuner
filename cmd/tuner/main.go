package main

import (
	"log"
	"log/slog"
	"os"
	"strconv"

	pkgconfig "github.com/llm-inferno/model-tuner/pkg/config"
	pkgsvc "github.com/llm-inferno/model-tuner/pkg/service"
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
	if v := os.Getenv(pkgsvc.WarmUpCyclesEnvName); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			warmUpCycles = n
		}
	}

	initObs := pkgsvc.DefaultInitObs
	if v := os.Getenv(pkgsvc.InitObsEnvName); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			initObs = n
		}
	}

	holdBack := pkgsvc.DefaultInitHoldBack
	if v := os.Getenv(pkgsvc.InitHoldBackEnvName); v != "" {
		holdBack = v == "true" || v == "1"
	}

	useSliding := os.Getenv(pkgsvc.EstimatorModeEnvName) == "sliding-window"

	windowSize := pkgsvc.DefaultWindowSize
	if v := os.Getenv(pkgsvc.WindowSizeEnvName); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			windowSize = n
		}
	}

	residualThreshold := pkgsvc.DefaultResidualThreshold
	if v := os.Getenv(pkgsvc.ResidualThresholdEnvName); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
			residualThreshold = f
		}
	}

	initFitThreshold := pkgsvc.DefaultInitFitThreshold
	if v := os.Getenv(pkgsvc.InitFitThresholdEnvName); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
			initFitThreshold = f
		}
	}

	service := pkgsvc.NewTunerService(warmUpCycles, initObs, holdBack, useSliding, windowSize, residualThreshold, initFitThreshold)
	server := tunerservice.NewTunerServer(service)

	estimatorMode := pkgsvc.DefaultEstimatorMode
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
