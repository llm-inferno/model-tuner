/*
Online Observer acts as Prometheus client and the environement struct is returned using Prometheus queries
To be able to use Online Observer, the user must set the secret bearer token and prometheus address as environment variable
*/

package observer

import (
	"fmt"
	"log"
	"os"

	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/model-tuner/pkg/metrics"
)

// observer is prom client that knows where to access the prometheus, and which queries to use to get the measurements from the environment
type OnlineObserver struct {
	BaseObserver
	promClient *metrics.PrometheusClient
}

func NewOnlineObserver() (*OnlineObserver, error) {
	secretToken := os.Getenv("TOKEN")
	if secretToken == "" {
		return nil, fmt.Errorf("TOKEN is not set")
	}
	promAddress := os.Getenv("PROMETHEUS_ADDRESS")
	client, err := metrics.NewPrometheusClient(promAddress, secretToken)
	if err != nil {
		return nil, err
	}
	return &OnlineObserver{
		BaseObserver: BaseObserver{},
		promClient:   client,
	}, nil
}

func (obs *OnlineObserver) GetEnvironment() core.Environment {
	var namespace, modelName, duration string
	namespace, modelName, duration = "platform-opt", "opt-125", "1m"
	metrics := map[string]string{
		"rpm":             fmt.Sprintf(`sum(rate(vllm:request_success_total{namespace="%s", model_name=~".*%s.*"}[%s]))`, namespace, modelName, duration),
		"AvgOutputTokens": fmt.Sprintf(`sum(rate(vllm:generation_tokens_total{namespace="%s", model_name=~".*%s.*"}[%s])) / sum(rate(vllm:request_success_total{namespace="%s", model_name=~".*%s.*"}[%s]))`, namespace, modelName, duration, namespace, modelName, duration),
		"batchSize":       fmt.Sprintf(`avg(avg_over_time(vllm:num_requests_running{namespace="%s", model_name=~".*%s.*"}[%s]))`, namespace, modelName, duration),
		"avgWaitTime":     fmt.Sprintf(`sum(rate(vllm:request_queue_time_seconds_sum{namespace="%s", model_name=~".*%s.*"}[%s])) / sum(rate(vllm:request_queue_time_seconds_count{namespace="%s", model_name=~".*%s.*"}[%s]))`, namespace, modelName, duration, namespace, modelName, duration),
		"avgTokenTime":    fmt.Sprintf(`sum(rate(vllm:time_per_output_token_seconds_sum{namespace="%s", model_name=~".*%s.*"}[%s])) / sum(rate(vllm:time_per_output_token_seconds_count{namespace="%s", model_name=~".*%s.*"}[%s]))`, namespace, modelName, duration, namespace, modelName, duration),
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

	// create environment with input parameters
	env := core.NewEnvironmentDecode(float32(results["rpm"])*60, float32(results["batchSize"]), float32(results["avgWaitTime"])*1000, 8,
		float32(results["AvgOutputTokens"]), float32(results["avgTokenTime"])*1000)
	return env
}

func (obs *OnlineObserver) fetchMetric(query string) (float64, error) {
	value, err := obs.promClient.Query(query)
	if err != nil {
		return 0, err
	}
	return value, nil
}
