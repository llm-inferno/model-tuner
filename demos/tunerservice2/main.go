package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"os"
	"time"

	optconfig "github.com/llm-inferno/optimizer-light/pkg/config"
	"github.com/llm-inferno/model-tuner/pkg/tunerservice2"
)

func main() {
	host := os.Getenv("TUNER_HOST")
	if host == "" {
		host = "localhost"
	}
	port := os.Getenv("TUNER_PORT")
	if port == "" {
		port = "8081"
	}

	service := tunerservice2.NewTunerService()
	server := tunerservice2.NewTunerServer(service)

	// Run server in background and send a test request after a short delay.
	go func() {
		time.Sleep(500 * time.Millisecond)
		baseURL := fmt.Sprintf("http://%s:%s", host, port)
		testTune(baseURL)
		testGetParams(baseURL, "llama3-8b", "A100")
	}()

	slog.Info("Starting TunerService2", "host", host, "port", port)
	if err := server.Run(host, port); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

// testTune posts synthetic ReplicaSpecs and prints the returned ModelData.
func testTune(baseURL string) {
	replicas := []optconfig.ServerSpec{
		{
			Name:  "llama3-8b/pod-0",
			Model: "llama3-8b",
			CurrentAlloc: optconfig.AllocationData{
				Accelerator: "A100",
				MaxBatch:    256,
				TTFTAverage: 120.0,
				ITLAverage:  15.0,
				Load: optconfig.ServerLoadSpec{
					ArrivalRate:  30.0,
					AvgInTokens:  512,
					AvgOutTokens: 128,
				},
			},
			MaxBatchSize: 256,
		},
		{
			Name:  "llama3-8b/pod-1",
			Model: "llama3-8b",
			CurrentAlloc: optconfig.AllocationData{
				Accelerator: "A100",
				MaxBatch:    256,
				TTFTAverage: 125.0,
				ITLAverage:  16.0,
				Load: optconfig.ServerLoadSpec{
					ArrivalRate:  28.0,
					AvgInTokens:  520,
					AvgOutTokens: 130,
				},
			},
			MaxBatchSize: 256,
		},
	}

	body, _ := json.Marshal(replicas)
	resp, err := http.Post(baseURL+"/tune", "application/json", bytes.NewReader(body))
	if err != nil {
		slog.Error("POST /tune failed", "err", err)
		return
	}
	defer func() { _ = resp.Body.Close() }()
	out, _ := io.ReadAll(resp.Body)
	slog.Info("POST /tune response", "status", resp.StatusCode, "body", string(out))
}

// testGetParams queries /getparams for a specific model/accelerator pair.
func testGetParams(baseURL, model, accelerator string) {
	url := fmt.Sprintf("%s/getparams?model=%s&accelerator=%s", baseURL, model, accelerator)
	resp, err := http.Get(url)
	if err != nil {
		slog.Error("GET /getparams failed", "err", err)
		return
	}
	defer func() { _ = resp.Body.Close() }()
	out, _ := io.ReadAll(resp.Body)
	slog.Info("GET /getparams response", "status", resp.StatusCode, "body", string(out))
}
