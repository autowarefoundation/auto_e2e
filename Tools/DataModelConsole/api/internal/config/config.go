// Package config loads the API server configuration from environment
// variables with sane in-cluster defaults (see deploy/k8s/api-deployment.yaml).
package config

import (
	"log/slog"
	"os"
	"time"
)

// Config holds all runtime configuration for the console API server.
type Config struct {
	Port            string
	AWSRegion       string
	DatasetsBucket  string
	ArtifactsBucket string
	MLflowURL       string
	FlyteURL        string
	PresignExpiry   time.Duration
	// CORSOrigin is the value for Access-Control-Allow-Origin.
	// "*" for development; set to the console origin in production.
	CORSOrigin string
	// FlyteProject / FlyteDomain scope the Flyte proxy queries.
	FlyteProject string
	FlyteDomain  string
}

// Load reads configuration from the environment, applying defaults.
func Load() *Config {
	cfg := &Config{
		Port:            getenv("PORT", "8080"),
		AWSRegion:       getenv("AWS_REGION", "us-west-2"),
		DatasetsBucket:  getenv("DATASETS_BUCKET", "auto-e2e-platform-datasets-381491877296"),
		ArtifactsBucket: getenv("ARTIFACTS_BUCKET", "auto-e2e-platform-artifacts-381491877296"),
		MLflowURL:       getenv("MLFLOW_URL", "http://mlflow.mlflow.svc.cluster.local:5000"),
		FlyteURL:        getenv("FLYTE_URL", "http://flyteadmin.flyte.svc.cluster.local:80"),
		CORSOrigin:      getenv("CORS_ORIGIN", "*"),
		FlyteProject:    getenv("FLYTE_PROJECT", "auto-e2e"),
		FlyteDomain:     getenv("FLYTE_DOMAIN", "development"),
	}

	expiry := getenv("PRESIGN_EXPIRY", "15m")
	d, err := time.ParseDuration(expiry)
	if err != nil {
		slog.Warn("invalid PRESIGN_EXPIRY, falling back to 15m", "value", expiry, "error", err)
		d = 15 * time.Minute
	}
	cfg.PresignExpiry = d
	return cfg
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
