package main

import (
	"log"
	"log/slog"
	"os"

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

	service := tunerservice.NewTunerService()
	server := tunerservice.NewTunerServer(service)

	slog.Info("Starting TunerService", "host", host, "port", port)
	if err := server.Run(host, port); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
