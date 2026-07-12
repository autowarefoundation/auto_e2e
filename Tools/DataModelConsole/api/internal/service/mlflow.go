package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

// UpstreamResult carries a proxied upstream response body + status.
type UpstreamResult struct {
	Status int
	Body   []byte
}

// httpGetJSON performs a GET against an upstream JSON API with query params.
func httpGetJSON(ctx context.Context, client *http.Client, base, p string, q url.Values) (*UpstreamResult, error) {
	u, err := url.Parse(base)
	if err != nil {
		return nil, fmt.Errorf("parse upstream url %q: %w", base, err)
	}
	u.Path, err = url.JoinPath(u.Path, p)
	if err != nil {
		return nil, fmt.Errorf("join upstream path %q: %w", p, err)
	}
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
	return &UpstreamResult{Status: resp.StatusCode, Body: body}, nil
}

// httpPostJSON performs a POST with a JSON body against an upstream API.
func httpPostJSON(ctx context.Context, client *http.Client, base, p string, payload any) (*UpstreamResult, error) {
	u, err := url.Parse(base)
	if err != nil {
		return nil, fmt.Errorf("parse upstream url %q: %w", base, err)
	}
	u.Path, err = url.JoinPath(u.Path, p)
	if err != nil {
		return nil, fmt.Errorf("join upstream path %q: %w", p, err)
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if err != nil {
		return nil, err
	}
	return &UpstreamResult{Status: resp.StatusCode, Body: body}, nil
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
func (m *MLflowService) SearchExperiments(ctx context.Context, maxResults, pageToken string) (*UpstreamResult, error) {
	q := url.Values{}
	if maxResults != "" {
		q.Set("max_results", maxResults)
	}
	if pageToken != "" {
		q.Set("page_token", pageToken)
	}
	return httpGetJSON(ctx, m.client, m.baseURL, "/api/2.0/mlflow/experiments/search", q)
}

// SearchRuns proxies POST /api/2.0/mlflow/runs/search for one experiment
// (the MLflow REST API only accepts POST for runs/search; the console
// endpoint stays GET and translates here).
func (m *MLflowService) SearchRuns(ctx context.Context, experimentID, maxResults, pageToken string) (*UpstreamResult, error) {
	body := map[string]any{
		"experiment_ids": []string{experimentID},
		"order_by":       []string{"attributes.start_time DESC"},
	}
	if maxResults != "" {
		var n int
		if _, err := fmt.Sscanf(maxResults, "%d", &n); err == nil && n > 0 {
			body["max_results"] = n
		}
	}
	if pageToken != "" {
		body["page_token"] = pageToken
	}
	return httpPostJSON(ctx, m.client, m.baseURL, "/api/2.0/mlflow/runs/search", body)
}

// GetRun proxies GET /api/2.0/mlflow/runs/get.
func (m *MLflowService) GetRun(ctx context.Context, runID string) (*UpstreamResult, error) {
	q := url.Values{}
	q.Set("run_id", runID)
	return httpGetJSON(ctx, m.client, m.baseURL, "/api/2.0/mlflow/runs/get", q)
}

// SearchRegisteredModels proxies GET /api/2.0/mlflow/registered-models/search.
func (m *MLflowService) SearchRegisteredModels(ctx context.Context, maxResults, pageToken string) (*UpstreamResult, error) {
	q := url.Values{}
	if maxResults != "" {
		q.Set("max_results", maxResults)
	}
	if pageToken != "" {
		q.Set("page_token", pageToken)
	}
	return httpGetJSON(ctx, m.client, m.baseURL, "/api/2.0/mlflow/registered-models/search", q)
}

// RunStats aggregates dashboard KPIs from MLflow: total run count across all
// experiments and the eval/ade metric of the most recent run that reports it.
// Runs are paged newest-first with a hard cap so a large tracking server
// cannot stall the dashboard.
func (m *MLflowService) RunStats(ctx context.Context) (totalRuns int, latestADE *float64, err error) {
	// 1. Collect experiment IDs.
	expRes, err := m.SearchExperiments(ctx, "1000", "")
	if err != nil {
		return 0, nil, err
	}
	if expRes.Status != http.StatusOK {
		return 0, nil, fmt.Errorf("mlflow experiments/search returned %d", expRes.Status)
	}
	var expBody struct {
		Experiments []struct {
			ExperimentID string `json:"experiment_id"`
		} `json:"experiments"`
	}
	if err := json.Unmarshal(expRes.Body, &expBody); err != nil {
		return 0, nil, fmt.Errorf("decode experiments: %w", err)
	}
	ids := make([]string, 0, len(expBody.Experiments))
	for _, e := range expBody.Experiments {
		ids = append(ids, e.ExperimentID)
	}
	if len(ids) == 0 {
		return 0, nil, nil
	}

	// 2. Page runs newest-first across all experiments (cap: 10 pages x 1000).
	// Reuse model.NormalizeMLflowRuns to fold the runs (its metric decode is
	// lenient — MLflow serializes NaN/Infinity, and some metric values, as JSON
	// STRINGS; a strict float64 decode here would fail the whole page and
	// wrongly report MLflow as empty/unavailable). Only the page token is read
	// separately.
	token := ""
	for page := 0; page < 10; page++ {
		res, err := httpPostJSON(ctx, m.client, m.baseURL, "/api/2.0/mlflow/runs/search", map[string]any{
			"experiment_ids": ids,
			"max_results":    1000,
			"order_by":       []string{"attributes.start_time DESC"},
			"page_token":     token,
		})
		if err != nil {
			return 0, nil, err
		}
		if res.Status != http.StatusOK {
			return 0, nil, fmt.Errorf("mlflow runs/search returned %d", res.Status)
		}
		runs, err := model.NormalizeMLflowRuns(res.Body)
		if err != nil {
			return 0, nil, fmt.Errorf("decode runs: %w", err)
		}
		var pageToken struct {
			NextPageToken string `json:"next_page_token"`
		}
		_ = json.Unmarshal(res.Body, &pageToken) // token absent => last page
		totalRuns += len(runs)
		if latestADE == nil {
			for _, run := range runs {
				if v, ok := run.Metrics["eval/ade"]; ok {
					vv := v
					latestADE = &vv
					break
				}
			}
		}
		if pageToken.NextPageToken == "" || len(runs) == 0 {
			break
		}
		token = pageToken.NextPageToken
	}
	return totalRuns, latestADE, nil
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
