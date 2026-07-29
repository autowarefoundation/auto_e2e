package service

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
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

type FlyteLaunchPlanID struct {
	ResourceType int    `json:"resourceType"`
	Project      string `json:"project"`
	Domain       string `json:"domain"`
	Name         string `json:"name"`
	Version      string `json:"version"`
}

type FlyteExecutionCreateResult struct {
	ExecutionID string
	Created     bool
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
func (f *FlyteService) ListExecutions(ctx context.Context, limit, token string) (*UpstreamResult, error) {
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
func (f *FlyteService) GetExecution(ctx context.Context, name string) (*UpstreamResult, error) {
	p := fmt.Sprintf("/api/v1/executions/%s/%s/%s", f.project, f.domain, name)
	return httpGetJSON(ctx, f.client, f.baseURL, p, nil)
}

// LatestLaunchPlan resolves the newest immutable version of a named LaunchPlan.
func (f *FlyteService) LatestLaunchPlan(
	ctx context.Context,
	name string,
) (FlyteLaunchPlanID, error) {
	var identifier FlyteLaunchPlanID
	if name == "" {
		return identifier, fmt.Errorf("Flyte LaunchPlan name is empty")
	}
	query := url.Values{
		"limit":             {"1"},
		"sort_by.key":       {"created_at"},
		"sort_by.direction": {"DESCENDING"},
	}
	path := fmt.Sprintf(
		"/api/v1/launch_plans/%s/%s/%s",
		f.project, f.domain, name,
	)
	result, err := httpGetJSON(ctx, f.client, f.baseURL, path, query)
	if err != nil {
		return identifier, fmt.Errorf("list Flyte LaunchPlans: %w", err)
	}
	if result.Status != http.StatusOK {
		return identifier, fmt.Errorf(
			"list Flyte LaunchPlans returned status %d", result.Status,
		)
	}
	var response struct {
		LaunchPlans []struct {
			ID FlyteLaunchPlanID `json:"id"`
		} `json:"launchPlans"`
	}
	if err := json.Unmarshal(result.Body, &response); err != nil {
		return identifier, fmt.Errorf("decode Flyte LaunchPlans: %w", err)
	}
	if len(response.LaunchPlans) != 1 {
		return identifier, fmt.Errorf("Flyte LaunchPlan %q is not registered", name)
	}
	identifier = response.LaunchPlans[0].ID
	if identifier.ResourceType != 3 ||
		identifier.Project != f.project ||
		identifier.Domain != f.domain ||
		identifier.Name != name ||
		identifier.Version == "" {
		return FlyteLaunchPlanID{}, fmt.Errorf(
			"Flyte LaunchPlan identity differs from requested scope",
		)
	}
	return identifier, nil
}

func flyteLiteral(value any) (map[string]any, error) {
	primitive := make(map[string]any, 1)
	switch typed := value.(type) {
	case string:
		primitive["stringValue"] = typed
	case int:
		primitive["integer"] = strconv.Itoa(typed)
	case int64:
		primitive["integer"] = strconv.FormatInt(typed, 10)
	case float64:
		primitive["floatValue"] = typed
	case bool:
		primitive["boolean"] = typed
	default:
		return nil, fmt.Errorf("unsupported Flyte input type %T", value)
	}
	return map[string]any{
		"scalar": map[string]any{"primitive": primitive},
	}, nil
}

func flyteLiteralMap(inputs map[string]any) (map[string]any, error) {
	literals := make(map[string]any, len(inputs))
	for name, value := range inputs {
		if name == "" {
			return nil, fmt.Errorf("Flyte input name is empty")
		}
		literal, err := flyteLiteral(value)
		if err != nil {
			return nil, fmt.Errorf("encode Flyte input %s: %w", name, err)
		}
		literals[name] = literal
	}
	return map[string]any{"literals": literals}, nil
}

// CreateLaunchPlanExecution starts one execution with explicit typed inputs.
func (f *FlyteService) CreateLaunchPlanExecution(
	ctx context.Context,
	launchPlan FlyteLaunchPlanID,
	executionName string,
	inputs map[string]any,
) (FlyteExecutionCreateResult, error) {
	literalMap, err := flyteLiteralMap(inputs)
	if err != nil {
		return FlyteExecutionCreateResult{}, err
	}
	payload := map[string]any{
		"project": f.project,
		"domain":  f.domain,
		"name":    executionName,
		"spec": map[string]any{
			"launchPlan": launchPlan,
			"metadata": map[string]any{
				"mode":      "MANUAL",
				"principal": "data-model-console",
				"nesting":   0,
			},
			"disableAll": true,
		},
		"inputs": literalMap,
	}
	return f.createExecution(ctx, "/api/v1/executions", payload, executionName)
}

// RelaunchExecution retries an existing execution with the exact same inputs.
func (f *FlyteService) RelaunchExecution(
	ctx context.Context,
	executionID string,
	relaunchName string,
) (FlyteExecutionCreateResult, error) {
	payload := map[string]any{
		"id": map[string]any{
			"project": f.project,
			"domain":  f.domain,
			"name":    executionID,
		},
		"name":           relaunchName,
		"overwriteCache": false,
	}
	return f.createExecution(
		ctx, "/api/v1/executions/relaunch", payload, relaunchName,
	)
}

func (f *FlyteService) createExecution(
	ctx context.Context,
	path string,
	payload map[string]any,
	requestedName string,
) (FlyteExecutionCreateResult, error) {
	result, err := httpPostJSON(ctx, f.client, f.baseURL, path, payload)
	if err != nil {
		return FlyteExecutionCreateResult{}, fmt.Errorf(
			"create Flyte execution: %w", err,
		)
	}
	if result.Status == http.StatusConflict {
		return FlyteExecutionCreateResult{
			ExecutionID: requestedName,
			Created:     false,
		}, nil
	}
	if result.Status != http.StatusOK && result.Status != http.StatusCreated {
		return FlyteExecutionCreateResult{}, fmt.Errorf(
			"create Flyte execution returned status %d", result.Status,
		)
	}
	var response struct {
		ID struct {
			Project string `json:"project"`
			Domain  string `json:"domain"`
			Name    string `json:"name"`
		} `json:"id"`
	}
	if err := json.Unmarshal(result.Body, &response); err != nil {
		return FlyteExecutionCreateResult{}, fmt.Errorf(
			"decode Flyte execution create response: %w", err,
		)
	}
	if response.ID.Project != f.project ||
		response.ID.Domain != f.domain ||
		response.ID.Name != requestedName {
		return FlyteExecutionCreateResult{}, fmt.Errorf(
			"created Flyte execution identity differs from request",
		)
	}
	return FlyteExecutionCreateResult{
		ExecutionID: response.ID.Name,
		Created:     true,
	}, nil
}
