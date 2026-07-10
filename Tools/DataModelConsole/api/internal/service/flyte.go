package service

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"time"
)

// FlyteService proxies read-only queries to the in-cluster Flyte Admin
// HTTP gateway (flyteadmin exposes its gRPC API over HTTP+JSON).
type FlyteService struct {
	baseURL string
	project string
	domain  string
	client  *http.Client
}

// NewFlyteService creates the proxy scoped to project/domain.
func NewFlyteService(baseURL, project, domain string) *FlyteService {
	return &FlyteService{
		baseURL: baseURL,
		project: project,
		domain:  domain,
		client:  &http.Client{Timeout: 30 * time.Second},
	}
}

// ListExecutions proxies GET /api/v1/executions/{project}/{domain}.
func (f *FlyteService) ListExecutions(ctx context.Context, limit, token string) (*upstreamResult, error) {
	q := url.Values{}
	if limit != "" {
		q.Set("limit", limit)
	}
	if token != "" {
		q.Set("token", token)
	}
	// Newest-first is what the console dashboard shows.
	q.Set("sort_by.key", "created_at")
	q.Set("sort_by.direction", "DESCENDING")
	p := fmt.Sprintf("/api/v1/executions/%s/%s", f.project, f.domain)
	return httpGetJSON(ctx, f.client, f.baseURL, p, q)
}

// GetExecution proxies GET /api/v1/executions/{project}/{domain}/{name}.
func (f *FlyteService) GetExecution(ctx context.Context, name string) (*upstreamResult, error) {
	p := fmt.Sprintf("/api/v1/executions/%s/%s/%s", f.project, f.domain, name)
	return httpGetJSON(ctx, f.client, f.baseURL, p, nil)
}
