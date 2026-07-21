package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

const (
	// MaxProxyResponseBytes bounds each buffered MLflow or Flyte response. Four
	// concurrent reads therefore reserve at most 32 MiB of response payload in
	// the 512 MiB API pod, leaving room for JSON decoding and other endpoints.
	MaxProxyResponseBytes int64 = 8 << 20 // 8 MiB

	maxConcurrentProxyRequests = 4
	runStatsCacheTTL           = 10 * time.Second
)

// ErrProxyResponseTooLarge reports that an upstream response exceeded the
// bounded proxy buffer.
var ErrProxyResponseTooLarge = errors.New("upstream response exceeds proxy limit")

// proxyRequestSlots is shared by every MLflowService and FlyteService in the
// process. The services use the same HTTP helpers below, so separate service
// instances cannot multiply the response-buffer budget.
var proxyRequestSlots = make(chan struct{}, maxConcurrentProxyRequests)

// UpstreamResult carries a proxied upstream response body + status.
type UpstreamResult struct {
	Status int
	Body   []byte
}

func doUpstreamJSON(ctx context.Context, client *http.Client, req *http.Request) (*UpstreamResult, error) {
	select {
	case proxyRequestSlots <- struct{}{}:
		defer func() { <-proxyRequestSlots }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, MaxProxyResponseBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > MaxProxyResponseBytes {
		return nil, fmt.Errorf("%w: maximum is %d bytes", ErrProxyResponseTooLarge, MaxProxyResponseBytes)
	}
	return &UpstreamResult{Status: resp.StatusCode, Body: body}, nil
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

	return doUpstreamJSON(ctx, client, req)
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

	return doUpstreamJSON(ctx, client, req)
}

// MLflowService proxies read-only queries to the in-cluster MLflow REST API.
type MLflowService struct {
	baseURL string
	client  *http.Client

	statsMu     sync.Mutex
	statsCache  cachedRunStats
	statsFlight *runStatsFlight
}

type runStatsValue struct {
	totalRuns int
	latestADE float64
	hasADE    bool
}

func (v runStatsValue) result() (int, *float64, error) {
	if !v.hasADE {
		return v.totalRuns, nil, nil
	}
	latestADE := v.latestADE
	return v.totalRuns, &latestADE, nil
}

type cachedRunStats struct {
	value     runStatsValue
	expiresAt time.Time
	valid     bool
}

type runStatsFlight struct {
	done  chan struct{}
	value runStatsValue
	err   error
}

// NewMLflowService creates the proxy for the given MLflow base URL.
func NewMLflowService(baseURL string) *MLflowService {
	return &MLflowService{
		baseURL: baseURL,
		client:  &http.Client{Timeout: 30 * time.Second},
	}
}

// SearchExperiments proxies GET /api/2.0/mlflow/experiments/search.
func (m *MLflowService) SearchExperiments(ctx context.Context, maxResults int, pageToken string) (*UpstreamResult, error) {
	q := url.Values{}
	if maxResults > 0 {
		q.Set("max_results", strconv.Itoa(maxResults))
	}
	if pageToken != "" {
		q.Set("page_token", pageToken)
	}
	return httpGetJSON(ctx, m.client, m.baseURL, "/api/2.0/mlflow/experiments/search", q)
}

// SearchRuns proxies POST /api/2.0/mlflow/runs/search for one experiment
// (the MLflow REST API only accepts POST for runs/search; the console
// endpoint stays GET and translates here).
func (m *MLflowService) SearchRuns(ctx context.Context, experimentID string, maxResults int, pageToken string) (*UpstreamResult, error) {
	body := map[string]any{
		"experiment_ids": []string{experimentID},
		"order_by":       []string{"attributes.start_time DESC"},
	}
	if maxResults > 0 {
		body["max_results"] = maxResults
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
func (m *MLflowService) SearchRegisteredModels(ctx context.Context, maxResults int, pageToken string) (*UpstreamResult, error) {
	q := url.Values{}
	if maxResults > 0 {
		q.Set("max_results", strconv.Itoa(maxResults))
	}
	if pageToken != "" {
		q.Set("page_token", pageToken)
	}
	return httpGetJSON(ctx, m.client, m.baseURL, "/api/2.0/mlflow/registered-models/search", q)
}

// RunStats aggregates dashboard KPIs from MLflow: total run count across all
// experiments and the eval/ade metric of the most recent run that reports it.
// Successful results are cached briefly, and concurrent cache misses share one
// fetch. Callers waiting for that fetch still honor their own context.
func (m *MLflowService) RunStats(ctx context.Context) (totalRuns int, latestADE *float64, err error) {
	if err := ctx.Err(); err != nil {
		return 0, nil, err
	}

	m.statsMu.Lock()
	if m.statsCache.valid && time.Now().Before(m.statsCache.expiresAt) {
		value := m.statsCache.value
		m.statsMu.Unlock()
		return value.result()
	}
	if flight := m.statsFlight; flight != nil {
		m.statsMu.Unlock()
		select {
		case <-flight.done:
			if err := ctx.Err(); err != nil {
				return 0, nil, err
			}
			if flight.err != nil {
				return 0, nil, flight.err
			}
			return flight.value.result()
		case <-ctx.Done():
			return 0, nil, ctx.Err()
		}
	}

	flight := &runStatsFlight{done: make(chan struct{})}
	m.statsFlight = flight
	m.statsMu.Unlock()

	value, fetchErr := m.fetchRunStats(ctx)

	m.statsMu.Lock()
	flight.value = value
	flight.err = fetchErr
	if fetchErr == nil {
		m.statsCache = cachedRunStats{
			value:     value,
			expiresAt: time.Now().Add(runStatsCacheTTL),
			valid:     true,
		}
	}
	m.statsFlight = nil
	close(flight.done)
	m.statsMu.Unlock()

	if fetchErr != nil {
		return 0, nil, fetchErr
	}
	return value.result()
}

func (m *MLflowService) fetchRunStats(ctx context.Context) (runStatsValue, error) {
	experimentIDs, err := m.allExperimentIDs(ctx)
	if err != nil {
		return runStatsValue{}, err
	}
	if len(experimentIDs) == 0 {
		return runStatsValue{}, nil
	}

	value := runStatsValue{}
	seenTokens := make(map[string]struct{})
	seenRunIDs := make(map[string]struct{})
	token := ""

	for {
		if err := ctx.Err(); err != nil {
			return runStatsValue{}, err
		}
		if _, seen := seenTokens[token]; seen {
			return runStatsValue{}, errors.New("mlflow runs pagination token cycle")
		}
		seenTokens[token] = struct{}{}

		payload := map[string]any{
			"experiment_ids": experimentIDs,
			"max_results":    1000,
			"order_by":       []string{"attributes.start_time DESC"},
		}
		if token != "" {
			payload["page_token"] = token
		}
		res, err := httpPostJSON(
			ctx,
			m.client,
			m.baseURL,
			"/api/2.0/mlflow/runs/search",
			payload,
		)
		if err != nil {
			return runStatsValue{}, err
		}
		if res.Status != http.StatusOK {
			return runStatsValue{}, fmt.Errorf("mlflow runs/search returned %d", res.Status)
		}
		page, err := model.NormalizeMLflowRunsPage(res.Body)
		if err != nil {
			return runStatsValue{}, fmt.Errorf("decode runs: %w", err)
		}
		for _, run := range page.Items {
			if run.RunID == "" {
				return runStatsValue{}, errors.New("mlflow runs/search returned a run without an id")
			}
			if _, seen := seenRunIDs[run.RunID]; seen {
				continue
			}
			seenRunIDs[run.RunID] = struct{}{}
			value.totalRuns++
			if value.hasADE {
				continue
			}
			if ade, ok := run.Metrics["eval/ade"]; ok &&
				!math.IsNaN(ade) && !math.IsInf(ade, 0) {
				value.latestADE = ade
				value.hasADE = true
			}
		}
		if page.NextPageToken == "" {
			return value, nil
		}
		token = page.NextPageToken
	}
}

func (m *MLflowService) allExperimentIDs(ctx context.Context) ([]string, error) {
	var experimentIDs []string
	seenExperimentIDs := make(map[string]struct{})
	seenTokens := make(map[string]struct{})
	token := ""

	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if _, seen := seenTokens[token]; seen {
			return nil, errors.New("mlflow experiments pagination token cycle")
		}
		seenTokens[token] = struct{}{}

		res, err := m.SearchExperiments(ctx, 1000, token)
		if err != nil {
			return nil, err
		}
		if res.Status != http.StatusOK {
			return nil, fmt.Errorf("mlflow experiments/search returned %d", res.Status)
		}
		page, err := model.NormalizeMLflowExperimentsPage(res.Body)
		if err != nil {
			return nil, fmt.Errorf("decode experiments: %w", err)
		}
		for _, experiment := range page.Items {
			if experiment.ExperimentID == "" {
				return nil, errors.New("mlflow experiments/search returned an experiment without an id")
			}
			if _, seen := seenExperimentIDs[experiment.ExperimentID]; seen {
				continue
			}
			seenExperimentIDs[experiment.ExperimentID] = struct{}{}
			experimentIDs = append(experimentIDs, experiment.ExperimentID)
		}
		if page.NextPageToken == "" {
			return experimentIDs, nil
		}
		token = page.NextPageToken
	}
}

// Ping checks MLflow reachability (used by /readyz extended checks).
func (m *MLflowService) Ping(ctx context.Context) error {
	res, err := m.SearchExperiments(ctx, 1, "")
	if err != nil {
		return err
	}
	if res.Status >= 500 {
		return fmt.Errorf("mlflow returned %d", res.Status)
	}
	return nil
}
