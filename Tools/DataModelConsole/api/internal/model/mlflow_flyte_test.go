package model

import (
	"encoding/json"
	"testing"
)

// Representative payloads mirror the real MLflow REST / Flyte Admin JSON shapes
// (the in-cluster services are unreachable from a dev laptop, so these encode
// the documented contracts the normalizers must handle).

func TestNormalizeMLflowExperiments(t *testing.T) {
	body := []byte(`{"experiments":[
		{"experiment_id":"1","name":"il-train","artifact_location":"s3://a/1","lifecycle_stage":"active","last_update_time":1699000000000},
		{"experiment_id":"2","name":"sweep","artifact_location":"s3://a/2","lifecycle_stage":"active","last_update_time":"1699000001000"}
	]}`)
	out, err := NormalizeMLflowExperiments(body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("got %d experiments, want 2", len(out))
	}
	if out[0].ExperimentID != "1" || out[0].Name != "il-train" || out[0].LastUpdateTime != 1699000000000 {
		t.Errorf("exp0 = %+v", out[0])
	}
	// last_update_time as a STRING must still coerce to int64.
	if out[1].LastUpdateTime != 1699000001000 {
		t.Errorf("exp1 last_update_time = %d, want 1699000001000", out[1].LastUpdateTime)
	}
}

func TestNormalizeMLflowRuns(t *testing.T) {
	body := []byte(`{"runs":[
		{"info":{"run_id":"r1","run_name":"","experiment_id":"1","status":"FINISHED","start_time":1699000000000,"end_time":1699000100000},
		 "data":{"params":[{"key":"lr","value":"3e-4"},{"key":"mlflow.runName","value":"tagged-name"}],
		         "metrics":[{"key":"eval/ade","value":2.5},{"key":"eval/ade","value":9.9},{"key":"loss","value":"0.31"}]}}
	]}`)
	out, err := NormalizeMLflowRuns(body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("got %d runs, want 1", len(out))
	}
	r := out[0]
	if r.RunID != "r1" || r.Status != "FINISHED" || r.StartTime != 1699000000000 {
		t.Errorf("run info = %+v", r)
	}
	// run_name empty in info => fall back to the mlflow.runName param.
	if r.RunName != "tagged-name" {
		t.Errorf("run_name = %q, want tagged-name", r.RunName)
	}
	if r.Params["lr"] != "3e-4" {
		t.Errorf("param lr = %q", r.Params["lr"])
	}
	// First metric value for a key wins (upstream is newest-first).
	if r.Metrics["eval/ade"] != 2.5 {
		t.Errorf("metric eval/ade = %v, want 2.5", r.Metrics["eval/ade"])
	}
	// Metric value as a STRING must coerce.
	if r.Metrics["loss"] != 0.31 {
		t.Errorf("metric loss = %v, want 0.31", r.Metrics["loss"])
	}
}

func TestNormalizeMLflowRuns_NonFiniteMetricIsEncodable(t *testing.T) {
	// MLflow serializes NaN/Infinity as JSON strings. The normalized run must be
	// JSON-encodable (encoding/json cannot marshal NaN/Inf), so non-finite
	// metric values coerce to 0.
	body := []byte(`{"runs":[{"info":{"run_id":"r1"},
		"data":{"metrics":[{"key":"loss","value":"NaN"},{"key":"grad","value":"Infinity"},{"key":"ade","value":2.0}]}}]}`)
	runs, err := NormalizeMLflowRuns(body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	m := runs[0].Metrics
	if m["loss"] != 0 || m["grad"] != 0 {
		t.Errorf("non-finite metrics not zeroed: %+v", m)
	}
	if m["ade"] != 2.0 {
		t.Errorf("finite metric changed: %v", m["ade"])
	}
	if _, err := json.Marshal(runs); err != nil {
		t.Errorf("normalized runs must be JSON-encodable, got: %v", err)
	}
}

func TestNormalizeMLflowModels(t *testing.T) {
	body := []byte(`{"registered_models":[
		{"name":"planner","latest_versions":[{"version":"3","current_stage":"Production","run_id":"r9","status":"READY"}]}
	]}`)
	out, err := NormalizeMLflowModels(body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out) != 1 || out[0].Name != "planner" {
		t.Fatalf("models = %+v", out)
	}
	v := out[0].LatestVersions[0]
	if v.Version != "3" || v.Stage != "Production" || v.RunID != "r9" || v.Status != "READY" {
		t.Errorf("version = %+v (current_stage must map to Stage)", v)
	}
}

func TestNormalizeFlyteExecutions(t *testing.T) {
	body := []byte(`{"executions":[
		{"id":{"name":"abc123"},
		 "spec":{"launchPlan":{"name":"lp_train_il"}},
		 "closure":{"phase":"SUCCEEDED","createdAt":"2026-07-10T00:00:00Z","startedAt":"2026-07-10T00:00:05Z","duration":"642.5s","workflowId":{"name":"wf_train_il"}}}
	]}`)
	out, err := NormalizeFlyteExecutions(body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("got %d executions, want 1", len(out))
	}
	e := out[0]
	if e.ExecutionID != "abc123" || e.WorkflowName != "wf_train_il" || e.Phase != "SUCCEEDED" {
		t.Errorf("exec = %+v", e)
	}
	if e.StartedAt != "2026-07-10T00:00:05Z" {
		t.Errorf("started_at = %q (should prefer startedAt over createdAt)", e.StartedAt)
	}
	if e.DurationS != 642 {
		t.Errorf("duration_s = %d, want 642", e.DurationS)
	}
}

func TestNormalizeFlyteExecution_Single(t *testing.T) {
	// The get-by-id endpoint returns the UNWRAPPED execution (no envelope).
	body := []byte(`{"id":{"name":"exec9"},
		"spec":{"launchPlan":{"name":"lp_train_il"}},
		"closure":{"phase":"FAILED","createdAt":"2026-07-10T00:00:00Z","startedAt":"2026-07-10T00:00:03Z","duration":"12.0s","workflowId":{"name":"wf_train_il"}}}`)
	e, err := NormalizeFlyteExecution(body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if e.ExecutionID != "exec9" || e.WorkflowName != "wf_train_il" || e.Phase != "FAILED" {
		t.Errorf("exec = %+v", e)
	}
	if e.StartedAt != "2026-07-10T00:00:03Z" || e.DurationS != 12 {
		t.Errorf("started=%q duration=%d", e.StartedAt, e.DurationS)
	}
}

func TestNormalizeFlyteExecutions_Fallbacks(t *testing.T) {
	// No startedAt / no workflowId.name => fall back to createdAt / launchPlan.
	body := []byte(`{"executions":[
		{"id":{"name":"e2"},"spec":{"launchPlan":{"name":"lp_eval"}},
		 "closure":{"phase":"RUNNING","createdAt":"2026-07-11T00:00:00Z","workflowId":{}}}
	]}`)
	out, err := NormalizeFlyteExecutions(body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	e := out[0]
	if e.WorkflowName != "lp_eval" {
		t.Errorf("workflow_name = %q, want lp_eval (launchPlan fallback)", e.WorkflowName)
	}
	if e.StartedAt != "2026-07-11T00:00:00Z" {
		t.Errorf("started_at = %q, want createdAt fallback", e.StartedAt)
	}
	if e.DurationS != 0 {
		t.Errorf("duration_s = %d, want 0 (absent)", e.DurationS)
	}
}
