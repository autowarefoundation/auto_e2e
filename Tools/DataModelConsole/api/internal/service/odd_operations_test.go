package service

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func validODDOperationsConfig(t *testing.T) ODDOperationsConfig {
	t.Helper()
	body, err := json.Marshal(oddLabelerInputTemplate{
		OpenAIModel:                   "nvidia/Cosmos3-Nano",
		OpenAIModelRevision:           strings.Repeat("a", 40),
		BedrockMapModelID:             "us.anthropic.claude-sonnet-4-6",
		BedrockMapModelRevision:       "claude-sonnet-4-6",
		LabelerImageDigest:            "sha256:" + strings.Repeat("b", 64),
		LabelerSourceRevision:         strings.Repeat("c", 40),
		CameraAnchorIntervalS:         1,
		MaximumCameraAnchors:          128,
		TriggerContextS:               1,
		RefinementConfidenceThreshold: 0.65,
		SmokeMaximumScenes:            2,
		SceneConcurrency:              40,
		OpenAIConcurrency:             10,
		BedrockConcurrency:            20,
	})
	if err != nil {
		t.Fatal(err)
	}
	return ODDOperationsConfig{
		Enabled:        true,
		LaunchPlanName: "odd-dataset-labeler",
		InputsJSON:     string(body),
	}
}

func TestODDOperationsLaunchPinsDatasetPublication(t *testing.T) {
	s3Service, _ := newPublicationTestService(t)
	var executionPayload map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			switch {
			case r.Method == http.MethodGet &&
				r.URL.Path ==
					"/api/v1/launch_plans/auto-e2e/development/odd-dataset-labeler":
				_, _ = w.Write([]byte(`{"launchPlans":[{"id":{
					"resourceType":3,
					"project":"auto-e2e",
					"domain":"development",
					"name":"odd-dataset-labeler",
					"version":"immutable-v5"
				}}]}`))
			case r.Method == http.MethodGet &&
				r.URL.Path == "/api/v1/executions/auto-e2e/development":
				_, _ = w.Write([]byte(`{"executions":[]}`))
			case r.Method == http.MethodPost &&
				r.URL.Path == "/api/v1/executions":
				if err := json.NewDecoder(r.Body).Decode(&executionPayload); err != nil {
					t.Fatal(err)
				}
				name := executionPayload["name"].(string)
				w.WriteHeader(http.StatusCreated)
				_ = json.NewEncoder(w).Encode(map[string]any{
					"id": map[string]any{
						"project": "auto-e2e",
						"domain":  "development",
						"name":    name,
					},
				})
			default:
				http.NotFound(w, r)
			}
		},
	))
	defer upstream.Close()

	operations, err := NewODDOperationsService(
		s3Service,
		NewFlyteService(upstream.URL, "auto-e2e", "development"),
		validODDOperationsConfig(t),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := operations.Launch(
		t.Context(), "kitscenes", "v2.1", "smoke",
	)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Created ||
		!strings.HasPrefix(result.ExecutionID, "odd-") ||
		result.LaunchPlanVersion != "immutable-v5" {
		t.Fatalf("launch result = %+v", result)
	}
	literals := executionPayload["inputs"].(map[string]any)["literals"].(map[string]any)
	primitive := func(name string) map[string]any {
		return literals[name].(map[string]any)["scalar"].(map[string]any)["primitive"].(map[string]any)
	}
	if primitive("dataset_name")["stringValue"] != "kitscenes" ||
		primitive("dataset_version")["stringValue"] != "v2.1" ||
		primitive("dataset_manifest_uri")["stringValue"] !=
			"s3://datasets/kitscenes/v2.1/shards/manifest.json" ||
		primitive("dataset_manifest_sha256")["stringValue"] == "" ||
		primitive("maximum_scenes")["integer"] != "2" ||
		primitive("publication_scope")["stringValue"] != "smoke" {
		t.Fatalf("launch inputs = %#v", literals)
	}
}

func TestODDOperationsRequireExplicitFullAndImmutableConfig(t *testing.T) {
	disabledConfig := validODDOperationsConfig(t)
	operations, err := NewODDOperationsService(nil, nil, ODDOperationsConfig{})
	if err != nil || operations.Enabled() {
		t.Fatalf("disabled operations = %+v, %v", operations, err)
	}

	mutable := disabledConfig
	var template map[string]any
	if err := json.Unmarshal([]byte(mutable.InputsJSON), &template); err != nil {
		t.Fatal(err)
	}
	template["openai_model_revision"] = "deployed"
	body, err := json.Marshal(template)
	if err != nil {
		t.Fatal(err)
	}
	mutable.InputsJSON = string(body)
	if _, err := NewODDOperationsService(
		&S3Service{}, &FlyteService{}, mutable,
	); !errors.Is(err, ErrODDInvalidOperation) {
		t.Fatalf("mutable revision error = %v", err)
	}

	s3Service, _ := newPublicationTestService(t)
	operations, err = NewODDOperationsService(
		s3Service,
		NewFlyteService("http://unused", "auto-e2e", "development"),
		disabledConfig,
	)
	if err != nil {
		t.Fatal(err)
	}
	_, err = operations.Launch(t.Context(), "kitscenes", "v2.1", "full")
	if !errors.Is(err, ErrODDFullRunsDisabled) {
		t.Fatalf("full launch error = %v", err)
	}
}

func TestODDOperationsRetryOnlyFailedDatasetLabeler(t *testing.T) {
	var retriedExecution string
	upstream := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			switch {
			case r.Method == http.MethodGet &&
				r.URL.Path ==
					"/api/v1/executions/auto-e2e/development/odd-failed":
				_, _ = w.Write([]byte(`{
					"id":{"name":"odd-failed"},
					"spec":{"launchPlan":{"name":"odd-dataset-labeler"}},
					"closure":{"phase":"FAILED"}
				}`))
			case r.Method == http.MethodGet &&
				r.URL.Path == "/api/v1/executions/auto-e2e/development":
				_, _ = w.Write([]byte(`{"executions":[]}`))
			case r.Method == http.MethodPost &&
				r.URL.Path == "/api/v1/executions/relaunch":
				var payload struct {
					ID struct {
						Name string `json:"name"`
					} `json:"id"`
					Name string `json:"name"`
				}
				if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
					t.Fatal(err)
				}
				retriedExecution = payload.ID.Name
				_ = json.NewEncoder(w).Encode(map[string]any{
					"id": map[string]any{
						"project": "auto-e2e",
						"domain":  "development",
						"name":    payload.Name,
					},
				})
			default:
				http.NotFound(w, r)
			}
		},
	))
	defer upstream.Close()

	config := validODDOperationsConfig(t)
	operations, err := NewODDOperationsService(
		&S3Service{},
		NewFlyteService(upstream.URL, "auto-e2e", "development"),
		config,
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := operations.Retry(t.Context(), "odd-failed")
	if err != nil {
		t.Fatal(err)
	}
	if retriedExecution != "odd-failed" ||
		!result.Created ||
		!strings.HasPrefix(result.ExecutionID, "odr-") {
		t.Fatalf("retry result = %+v, source = %q", result, retriedExecution)
	}
}

func TestODDOperationsRejectNonFailedOrNonODDRetry(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/v1/executions/auto-e2e/development/odd-succeeded":
				_, _ = w.Write([]byte(`{
					"id":{"name":"odd-succeeded"},
					"spec":{"launchPlan":{"name":"odd-dataset-labeler"}},
					"closure":{"phase":"SUCCEEDED"}
				}`))
			case "/api/v1/executions/auto-e2e/development/training-failed":
				_, _ = w.Write([]byte(`{
					"id":{"name":"training-failed"},
					"spec":{"launchPlan":{"name":"wf_train_il"}},
					"closure":{"phase":"FAILED"}
				}`))
			default:
				http.NotFound(w, r)
			}
		},
	))
	defer upstream.Close()

	operations, err := NewODDOperationsService(
		&S3Service{},
		NewFlyteService(upstream.URL, "auto-e2e", "development"),
		validODDOperationsConfig(t),
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, executionID := range []string{"odd-succeeded", "training-failed"} {
		if _, err := operations.Retry(
			t.Context(), executionID,
		); !errors.Is(err, ErrODDInvalidOperation) {
			t.Fatalf("retry %s error = %v", executionID, err)
		}
	}
}
