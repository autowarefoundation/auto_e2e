package service

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// upstreamResult carries a proxied upstream response body + status.
type upstreamResult struct {
	Status int
	Body   []byte
}

// httpGetJSON performs a GET against an upstream JSON API with query params.
func httpGetJSON(ctx context.Context, client *http.Client, base, p string, q url.Values) (*upstreamResult, error) {
	u, err := url.Parse(base)
	if err != nil {
		return nil, fmt.Errorf("parse upstream url %q: %w", base, err)
	}
	u.Path, err = url.JoinPath(u.Path, p)
	if err != nil {
		return nil, fmt.Errorf("join upstream path %q: %w", p, err)
	}
	u = u.JoinPath() // normalize
	full := u.String()
	if len(q) > 0 {
		full += "?" + q.Encode()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, full, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20)) // 32MiB guard
	if err != nil {
		return nil, err
	}
	return &upstreamResult{Status: resp.StatusCode, Body: body}, nil
}

// MLflowService proxies read-only queries to the in-cluster MLflow REST API.
type MLflowService struct {
	baseURL string
	client  *http.Client
}

// NewMLflowService creates the proxy for the given MLflow base URL.
func NewMLflowService(baseURL string) *MLflowService {
	return &MLflowService{
		baseURL: baseURL,
		client:  &http.Client{Timeout: 30 * time.Second},
	}
}

// SearchExperiments proxies GET /api/2.0/mlflow/experiments/search.
func (m *MLflowService) SearchExperiments(ctx context.Context, maxResults, pageToken string) (*upstreamResult, error) {
	q := url.Values{}
	if maxResults != "" {
		q.Set("max_results", maxResults)
	}
	if pageToken != "" {
		q.Set("page_token", pageToken)
	}
	return httpGetJSON(ctx, m.client, m.baseURL, "/api/2.0/mlflow/experiments/search", q)
}

// SearchRuns proxies GET /api/2.0/mlflow/runs/search for one experiment.
func (m *MLflowService) SearchRuns(ctx context.Context, experimentID, maxResults, pageToken string) (*upstreamResult, error) {
	q := url.Values{}
	q.Set("experiment_ids", experimentID)
	if maxResults != "" {
		q.Set("max_results", maxResults)
	}
	if pageToken != "" {
		q.Set("page_token", pageToken)
	}
	return httpGetJSON(ctx, m.client, m.baseURL, "/api/2.0/mlflow/runs/search", q)
}

// GetRun proxies GET /api/2.0/mlflow/runs/get.
func (m *MLflowService) GetRun(ctx context.Context, runID string) (*upstreamResult, error) {
	q := url.Values{}
	q.Set("run_id", runID)
	return httpGetJSON(ctx, m.client, m.baseURL, "/api/2.0/mlflow/runs/get", q)
}

// SearchRegisteredModels proxies GET /api/2.0/mlflow/registered-models/search.
func (m *MLflowService) SearchRegisteredModels(ctx context.Context, maxResults, pageToken string) (*upstreamResult, error) {
	q := url.Values{}
	if maxResults != "" {
		q.Set("max_results", maxResults)
	}
	if pageToken != "" {
		q.Set("page_token", pageToken)
	}
	return httpGetJSON(ctx, m.client, m.baseURL, "/api/2.0/mlflow/registered-models/search", q)
}

// Ping checks MLflow reachability (used by /readyz extended checks).
func (m *MLflowService) Ping(ctx context.Context) error {
	res, err := m.SearchExperiments(ctx, "1", "")
	if err != nil {
		return err
	}
	if res.Status >= 500 {
		return fmt.Errorf("mlflow returned %d", res.Status)
	}
	return nil
}
