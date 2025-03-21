package metrics

import (
	"context"
	"fmt"
	"time"

	"github.com/prometheus/client_golang/api"
	v1 "github.com/prometheus/client_golang/api/prometheus/v1"
	cfg "github.com/prometheus/common/config"
	"github.com/prometheus/common/model"
)

type PrometheusClient struct {
	client api.Client
}

func NewPrometheusClient(url, secretToken string) (*PrometheusClient, error) {
	client, err := api.NewClient(api.Config{
		Address: url,
		RoundTripper: cfg.NewAuthorizationCredentialsRoundTripper(
			"Bearer",
			cfg.NewInlineSecret(secretToken),
			api.DefaultRoundTripper,
		),
	})
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err)
	}
	return &PrometheusClient{
		client: client,
	}, nil
}

func (pc *PrometheusClient) Query(query string) (float64, error) {
	v1api := v1.NewAPI(pc.client)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	result, warnings, err := v1api.Query(ctx, query, time.Now(), v1.WithTimeout(5*time.Second))
	if err != nil {
		return 0, fmt.Errorf("error querying Prometheus: %v", err)
	}
	if len(warnings) > 0 {
		fmt.Printf("Warnings: %v\n", warnings)
	}
	if vector, ok := result.(model.Vector); ok && len(vector) > 0 {
		return float64(vector[0].Value), nil
	}

	return 0, fmt.Errorf("no data for query: %s", query)
}
