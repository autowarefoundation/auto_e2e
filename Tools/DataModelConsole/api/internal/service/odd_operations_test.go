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
	configSHA256 := strings.Repeat("2", 64)
	body, err := json.Marshal(oddLabelerInputTemplate{
		OntologyVersion:                 "odd_ontology_v1.0.1",
		OntologySHA256:                  strings.Repeat("1", 64),
		LabelerBundleVersion:            "odd_dataset_labeler_v6",
		LabelerConfigURI:                "s3://datasets/odd/configs/" + configSHA256 + ".json",
		LabelerConfigSHA256:             configSHA256,
		EnabledSources:                  []string{"map_route", "gnss_ins", "vlm", "image_qc", "fusion"},
		RoadVLMProvider:                 "openai_compatible",
		RoadVLMModel:                    "nvidia/Cosmos3-Nano",
		RoadVLMModelRevision:            strings.Repeat("a", 40),
		RoadVLMPromptBundleSHA256:       strings.Repeat("3", 64),
		RoadVLMDecodingConfigSHA256:     strings.Repeat("4", 64),
		MapResolverProvider:             "amazon_bedrock",
		MapResolverModelID:              "us.anthropic.claude-sonnet-4-6",
		MapResolverModelRevision:        "claude-sonnet-4-6",
		MapResolverPromptBundleSHA256:   strings.Repeat("5", 64),
		MapResolverDecodingConfigSHA256: strings.Repeat("6", 64),
		FusionConfigSHA256:              strings.Repeat("7", 64),
		CalibrationBundleSHA256:         strings.Repeat("8", 64),
		LabelerImageDigest:              "sha256:" + strings.Repeat("9", 64),
		LabelerSourceRevision:           strings.Repeat("b", 40),
		CameraAnchorIntervalS:           1,
		MaximumCameraAnchors:            128,
		TriggerContextS:                 1,
		RefinementConfidenceThreshold:   0.65,
		SmokeMaximumScenes:              2,
		SceneConcurrency:                40,
		OpenAIConcurrency:               10,
		BedrockConcurrency:              20,
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
	if len(literals) != 35 ||
		primitive("dataset_name")["stringValue"] != "kitscenes" ||
		primitive("dataset_version")["stringValue"] != "v2.1" ||
		primitive("dataset_manifest_uri")["stringValue"] !=
			"s3://datasets/kitscenes/v2.1/shards/manifest.json" ||
		primitive("dataset_manifest_sha256")["stringValue"] == "" ||
		primitive("ontology_version")["stringValue"] != "odd_ontology_v1.0.1" ||
		primitive("ontology_sha256")["stringValue"] != strings.Repeat("1", 64) ||
		primitive("labeler_bundle_version")["stringValue"] !=
			"odd_dataset_labeler_v6" ||
		primitive("labeler_config_sha256")["stringValue"] !=
			strings.Repeat("2", 64) ||
		primitive("road_vlm_provider")["stringValue"] !=
			"openai_compatible" ||
		primitive("map_resolver_provider")["stringValue"] !=
			"amazon_bedrock" ||
		primitive("fusion_config_sha256")["stringValue"] !=
			strings.Repeat("7", 64) ||
		primitive("calibration_bundle_sha256")["stringValue"] !=
			strings.Repeat("8", 64) ||
		primitive("publication_prefix")["stringValue"] !=
			"kitscenes/v2.1/odd" ||
		primitive("maximum_scenes")["integer"] != "2" ||
		primitive("publication_scope")["stringValue"] != "smoke" {
		t.Fatalf("launch inputs = %#v", literals)
	}
	sources := literals["enabled_sources"].(map[string]any)["collection"].(map[string]any)["literals"].([]any)
	if len(sources) != 5 {
		t.Fatalf("enabled source literals = %#v", sources)
	}
	if _, exists := literals["openai_model"]; exists {
		t.Fatalf("legacy OpenAI model input remains: %#v", literals)
	}
	if _, exists := literals["bedrock_map_model_id"]; exists {
		t.Fatalf("legacy Bedrock model input remains: %#v", literals)
	}
}

func TestODDOperationsRequireExplicitFullAndImmutableConfig(t *testing.T) {
	disabledConfig := validODDOperationsConfig(t)
	operations, err := NewODDOperationsService(nil, nil, ODDOperationsConfig{})
	if err != nil || operations.Enabled() {
		t.Fatalf("disabled operations = %+v, %v", operations, err)
	}

	mutations := map[string]func(map[string]any){
		"mutable revision": func(template map[string]any) {
			template["road_vlm_model_revision"] = "deployed"
		},
		"invalid digest": func(template map[string]any) {
			template["ontology_sha256"] = "not-a-digest"
		},
		"mutable config URI": func(template map[string]any) {
			template["labeler_config_uri"] = "s3://datasets/odd/configs/latest.json"
		},
		"missing fusion": func(template map[string]any) {
			template["enabled_sources"] = []string{"vlm"}
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			invalid := disabledConfig
			var template map[string]any
			if err := json.Unmarshal(
				[]byte(invalid.InputsJSON), &template,
			); err != nil {
				t.Fatal(err)
			}
			mutate(template)
			body, err := json.Marshal(template)
			if err != nil {
				t.Fatal(err)
			}
			invalid.InputsJSON = string(body)
			if _, err := NewODDOperationsService(
				&S3Service{}, &FlyteService{}, invalid,
			); !errors.Is(err, ErrODDInvalidOperation) {
				t.Fatalf("invalid config error = %v", err)
			}
		})
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
