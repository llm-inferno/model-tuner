package core

import (
	"fmt"
	"log"
	"os"

	"github.ibm.com/modeling-analysis/model-tuner/pkg/config"
	"github.ibm.com/modeling-analysis/model-tuner/pkg/metrics"
)

// observer is prom client that knows where to access the prometheus, and which queries to use to get the measurements from the environment
type RealObserver struct {
	BaseObserver
	promClient *metrics.PrometheusClient
}

func NewRealObserver() (*RealObserver, error) {
	secretToken := os.Getenv("TOKEN")
	if secretToken == "" {
		return nil, fmt.Errorf("TOKEN is not set")
	}
	client, err := metrics.NewPrometheusClient(config.PrometheusAddress, secretToken)
	if err != nil {
		return nil, err
	}
	return &RealObserver{
		BaseObserver: BaseObserver{},
		promClient:   client,
	}, nil
}

func (obs *RealObserver) GetEnvironment() *Environment {
	var namespace, modelName, duration string
	namespace, modelName, duration = "platform-opt", "opt-125", "1m"
	metrics := map[string]string{
		"rpm":          fmt.Sprintf(`sum(rate(vllm:request_success_total{namespace="%s", model_name=~".*%s.*"}[%s]))`, namespace, modelName, duration),
		"avgNumTokens": fmt.Sprintf(`sum(rate(vllm:generation_tokens_total{namespace="%s", model_name=~".*%s.*"}[%s])) / sum(rate(vllm:request_success_total{namespace="%s", model_name=~".*%s.*"}[%s]))`, namespace, modelName, duration, namespace, modelName, duration),
		"batchSize":    fmt.Sprintf(`avg_over_time(vllm:num_requests_running{namespace="%s", model_name=~".*%s.*"}[%s])`, namespace, modelName, duration),
		"avgWaitTime":  fmt.Sprintf(`sum(rate(vllm:request_queue_time_seconds_sum{namespace="%s", model_name=~".*%s.*"}[%s])) / sum(rate(vllm:request_queue_time_seconds_count{namespace="%s", model_name=~".*%s.*"}[%s]))`, namespace, modelName, duration, namespace, modelName, duration),
		"avgTokenTime": fmt.Sprintf(`sum(rate(vllm:time_per_output_token_seconds_sum{namespace="%s", model_name=~".*%s.*"}[%s])) / sum(rate(vllm:time_per_output_token_seconds_count{namespace="%s", model_name=~".*%s.*"}[%s]))`, namespace, modelName, duration, namespace, modelName, duration),
		"throughput":   fmt.Sprintf(`sum(rate(vllm:request_success_total{namespace="%s", model_name=~".*%s.*"}[%s])) / sum(rate(vllm:time_per_output_token_seconds_count{namespace="%s", model_name=~".*%s.*"}[%s]))`, namespace, modelName, duration, namespace, modelName, duration),
	}

	results := make(map[string]float64)
	for key, query := range metrics {
		value, err := obs.fetchMetric(query)
		if err != nil {
			log.Printf("Error fetching %s: %v", key, err)
			return nil
		}
		results[key] = value
	}

	return &Environment{
		Lambda:              float32(results["rpm"]) * 60,
		MaxBatchSize:        256, //int(maxBatchSize),
		BatchSize:           int(results["batchSize"]),
		AvgTokensPerRequest: float32(results["avgNumTokens"]),
		AvgQueueTime:        float32(results["avgWaitTime"]) * 1000,
		AvgTokenTime:        float32(results["avgTokenTime"]) * 1000,
		Throughput:          float32(results["throughput"]) * 60,
	}
}

func (obs *RealObserver) fetchMetric(query string) (float64, error) {
	value, err := obs.promClient.Query(query)
	if err != nil {
		return 0, err
	}
	return value, nil
}
