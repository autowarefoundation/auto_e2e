package service

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestExperimentServiceJoinsExplicitLineage(t *testing.T) {
	mlflow := httptest.NewServer(http.HandlerFunc(func(
		w http.ResponseWriter,
		r *http.Request,
	) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/2.0/mlflow/experiments/search":
			_, _ = w.Write([]byte(`{"experiments":[
				{"experiment_id":"8","name":"imitation-learning"},
				{"experiment_id":"10","name":"conn-check"}
			]}`))
		case "/api/2.0/mlflow/runs/search":
			var request struct {
				ExperimentIDs []string `json:"experiment_ids"`
			}
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Fatalf("decode run search: %v", err)
			}
			if len(request.ExperimentIDs) != 2 {
				t.Errorf("experiment ids = %v", request.ExperimentIDs)
			}
			_, _ = w.Write([]byte(`{"runs":[
				{"info":{"run_id":"new","run_name":"route-on","experiment_id":"8","status":"FINISHED","start_time":3000,"end_time":4000},
				 "data":{
					"params":[
						{"key":"data/dataset","value":"KITScenes"},
						{"key":"data/dataset_version","value":"v3.0"},
						{"key":"model/enable_route_conditioning","value":"True"},
						{"key":"train/validation_scope","value":"full"}
					],
					"tags":[
						{"key":"pipeline","value":"imitation-learning"},
						{"key":"ctx/eval_execution_id","value":"exec-new"}
					],
					"metrics":[
						{"key":"val/ade","value":3.8},
						{"key":"val/fde","value":11.0},
						{"key":"eval/ade","value":3.5},
						{"key":"eval/fde","value":10.2},
						{"key":"eval/gate_pass","value":0}
					]}},
				{"info":{"run_id":"old","run_name":"legacy","experiment_id":"8","status":"FINISHED","start_time":2000,"end_time":2500},
				 "data":{"params":[
					{"key":"data/dataset","value":"KITScenes"},
					{"key":"ctx/train_execution_id","value":"exec-old"}
				 ]}},
				{"info":{"run_id":"missing","run_name":"unlinked","experiment_id":"8","status":"FINISHED","start_time":1000,"end_time":1500},
				 "data":{"params":[{"key":"data/dataset","value":"L2D"}]}},
				{"info":{"run_id":"check","run_name":"connection-check","experiment_id":"10","status":"FINISHED","start_time":5000,"end_time":5001},
				 "data":{"metrics":[{"key":"eval/ade","value":1.23}]}}
			]}`))
		case "/api/2.0/mlflow/model-versions/search":
			_, _ = w.Write([]byte(`{"model_versions":[
				{"name":"auto-e2e-driving-policy","version":"42","run_id":"new","status":"READY",
				 "tags":[
					{"key":"checkpoint_role","value":"best"},
					{"key":"train_execution_id","value":"exec-new"}
				 ]}
			]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer mlflow.Close()

	flyte := httptest.NewServer(http.HandlerFunc(func(
		w http.ResponseWriter,
		r *http.Request,
	) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"executions":[
			{"id":{"name":"exec-new"},"closure":{
				"phase":"SUCCEEDED","startedAt":"2026-07-25T00:00:00Z",
				"duration":"60s","workflowId":{
					"name":".flytegen.Platform.pipelines.workflows.wf_full_pipeline"}}},
			{"id":{"name":"exec-old"},"closure":{
				"phase":"FAILED","startedAt":"2026-07-24T00:00:00Z",
				"duration":"30s","workflowId":{"name":"wf_train_il"}}}
		]}`))
	}))
	defer flyte.Close()

	service := NewExperimentService(
		NewMLflowService(mlflow.URL),
		NewFlyteService(flyte.URL, "auto-e2e", "development"),
		"https://mlflow.example",
		"https://flyte.example/console",
	)
	response, err := service.List(t.Context())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if response.Summary.Total != 3 || response.Summary.Evaluated != 1 ||
		response.Summary.Registered != 1 || response.Summary.Unlinked != 1 {
		t.Fatalf("summary = %+v", response.Summary)
	}
	if len(response.Experiments) != 3 {
		t.Fatalf("records = %d, want 3", len(response.Experiments))
	}

	newRun := response.Experiments[0]
	if newRun.RunID != "new" || newRun.LineageStatus != "complete" ||
		newRun.PrimaryExecutionID != "exec-new" {
		t.Errorf("new run lineage = %+v", newRun)
	}
	if newRun.TrainExecution == nil ||
		newRun.EvalExecution == nil ||
		newRun.TrainExecution.ExecutionID != "exec-new" ||
		newRun.EvalExecution.WorkflowName != "wf_full_pipeline" {
		t.Errorf("execution refs = train:%+v eval:%+v",
			newRun.TrainExecution, newRun.EvalExecution)
	}
	if newRun.Evaluation == nil || newRun.Validation == nil ||
		*newRun.Evaluation.ADE != 3.5 || *newRun.Validation.ADE != 3.8 {
		t.Errorf("metric sources = eval:%+v val:%+v",
			newRun.Evaluation, newRun.Validation)
	}
	if newRun.RouteConditioning == nil || !*newRun.RouteConditioning {
		t.Errorf("route conditioning = %v", newRun.RouteConditioning)
	}
	if len(newRun.ModelVersions) != 1 ||
		newRun.ModelVersions[0].Version != "42" ||
		newRun.ModelVersions[0].Role != "best" {
		t.Errorf("model versions = %+v", newRun.ModelVersions)
	}
	if newRun.MLflowURL !=
		"https://mlflow.example/#/experiments/8/runs/new" {
		t.Errorf("mlflow url = %q", newRun.MLflowURL)
	}
	if newRun.PrimaryExecutionURL !=
		"https://flyte.example/console/projects/auto-e2e/domains/development/executions/exec-new" {
		t.Errorf("flyte url = %q", newRun.PrimaryExecutionURL)
	}

	if response.Experiments[1].LineageStatus != "complete" ||
		response.Experiments[2].LineageStatus != "missing" {
		t.Errorf("legacy/missing lineage = %q/%q",
			response.Experiments[1].LineageStatus,
			response.Experiments[2].LineageStatus)
	}
}

func TestExperimentPaginationRejectsTokenCycle(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(
		w http.ResponseWriter,
		r *http.Request,
	) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/api/2.0/mlflow/experiments/search" {
			_, _ = w.Write([]byte(
				`{"experiments":[],"next_page_token":"cycle"}`,
			))
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	service := NewExperimentService(
		NewMLflowService(server.URL),
		NewFlyteService(server.URL, "auto-e2e", "development"),
		"", "",
	)
	if _, err := service.List(t.Context()); err == nil {
		t.Fatal("List succeeded, want pagination cycle error")
	}
}
