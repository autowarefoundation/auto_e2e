package service

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFlyteLaunchPlanExecutionUsesTypedInputs(t *testing.T) {
	var createPayload map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			switch {
			case r.Method == http.MethodGet &&
				r.URL.Path ==
					"/api/v1/launch_plans/auto-e2e/development/odd-dataset-labeler":
				if r.URL.Query().Get("limit") != "1" ||
					r.URL.Query().Get("sort_by.direction") != "DESCENDING" {
					t.Fatalf("launch plan query = %s", r.URL.RawQuery)
				}
				_, _ = w.Write([]byte(`{"launchPlans":[{"id":{
					"resourceType":"LAUNCH_PLAN",
					"project":"auto-e2e",
					"domain":"development",
					"name":"odd-dataset-labeler",
					"version":"immutable-v5"
				}}]}`))
			case r.Method == http.MethodPost &&
				r.URL.Path == "/api/v1/executions":
				if err := json.NewDecoder(r.Body).Decode(&createPayload); err != nil {
					t.Fatal(err)
				}
				w.WriteHeader(http.StatusCreated)
				_, _ = w.Write([]byte(`{"id":{
					"project":"auto-e2e",
					"domain":"development",
					"name":"odd-0123456789abcdef"
				}}`))
			default:
				http.NotFound(w, r)
			}
		},
	))
	defer upstream.Close()

	flyte := NewFlyteService(upstream.URL, "auto-e2e", "development")
	launchPlan, err := flyte.LatestLaunchPlan(
		t.Context(), "odd-dataset-labeler",
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := flyte.CreateLaunchPlanExecution(
		t.Context(),
		launchPlan,
		"odd-0123456789abcdef",
		map[string]any{
			"dataset_name":                    "kitscenes",
			"maximum_camera_anchors":          128,
			"camera_anchor_interval_s":        1.0,
			"refinement_confidence_threshold": 0.65,
			"enabled":                         true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.ExecutionID != "odd-0123456789abcdef" || !result.Created {
		t.Fatalf("create result = %+v", result)
	}

	spec := createPayload["spec"].(map[string]any)
	identifier := spec["launchPlan"].(map[string]any)
	if identifier["resourceType"] != "LAUNCH_PLAN" ||
		identifier["name"] != "odd-dataset-labeler" ||
		identifier["version"] != "immutable-v5" ||
		spec["disableAll"] != true {
		t.Fatalf("execution spec = %#v", spec)
	}
	literals := createPayload["inputs"].(map[string]any)["literals"].(map[string]any)
	primitive := func(name string) map[string]any {
		return literals[name].(map[string]any)["scalar"].(map[string]any)["primitive"].(map[string]any)
	}
	if primitive("dataset_name")["stringValue"] != "kitscenes" ||
		primitive("maximum_camera_anchors")["integer"] != "128" ||
		primitive("camera_anchor_interval_s")["floatValue"] != 1.0 ||
		primitive("refinement_confidence_threshold")["floatValue"] != 0.65 ||
		primitive("enabled")["boolean"] != true {
		t.Fatalf("typed literals = %#v", literals)
	}
}

func TestFlyteExecutionCreationIsIdempotentOnConflict(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/api/v1/executions/relaunch" {
				http.NotFound(w, r)
				return
			}
			var payload struct {
				ID struct {
					Project string `json:"project"`
					Domain  string `json:"domain"`
					Name    string `json:"name"`
				} `json:"id"`
				Name           string `json:"name"`
				OverwriteCache bool   `json:"overwriteCache"`
			}
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Fatal(err)
			}
			if payload.ID.Project != "auto-e2e" ||
				payload.ID.Domain != "development" ||
				payload.ID.Name != "odd-failed" ||
				payload.Name != "odr-0123456789abcdef" ||
				payload.OverwriteCache {
				t.Fatalf("relaunch payload = %+v", payload)
			}
			w.WriteHeader(http.StatusConflict)
		},
	))
	defer upstream.Close()

	result, err := NewFlyteService(
		upstream.URL, "auto-e2e", "development",
	).RelaunchExecution(
		t.Context(), "odd-failed", "odr-0123456789abcdef",
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.ExecutionID != "odr-0123456789abcdef" || result.Created {
		t.Fatalf("idempotent relaunch result = %+v", result)
	}
}
