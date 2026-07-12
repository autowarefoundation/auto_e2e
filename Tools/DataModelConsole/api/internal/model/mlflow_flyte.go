package model

import "encoding/json"

// This file normalizes the verbose, nested MLflow REST and Flyte Admin JSON
// shapes into the FLAT shapes the console frontend consumes. The proxy used to
// relay the raw upstream bodies, but the frontend types (MLflowExperiment /
// MLflowRun / MLflowRegisteredModel / FlyteExecution) are flat, so a raw
// pass-through crashed the UI on real data. Normalizing server-side keeps the
// contract in one place and lets us unit-test it against representative
// payloads (the in-cluster services are not reachable from a dev laptop).

// ---------------------------------------------------------------------------
// MLflow
// ---------------------------------------------------------------------------

// MLflowExperiment is the flat experiment shape the frontend expects.
type MLflowExperiment struct {
	ExperimentID   string `json:"experiment_id"`
	Name           string `json:"name"`
	ArtifactLocn   string `json:"artifact_location"`
	LifecycleStage string `json:"lifecycle_stage"`
	RunCount       int    `json:"run_count"`
	LastUpdateTime int64  `json:"last_update_time"`
}

// MLflowRun is the flat run shape the frontend expects: info.* lifted to the
// top level, and data.params / data.metrics folded into flat maps (latest
// value per metric key).
type MLflowRun struct {
	RunID        string             `json:"run_id"`
	RunName      string             `json:"run_name"`
	ExperimentID string             `json:"experiment_id"`
	Status       string             `json:"status"`
	StartTime    int64              `json:"start_time"`
	EndTime      int64              `json:"end_time"`
	Params       map[string]string  `json:"params"`
	Metrics      map[string]float64 `json:"metrics"`
}

// MLflowModelVersion is one registered-model version (flat).
type MLflowModelVersion struct {
	Version string `json:"version"`
	Stage   string `json:"stage"`
	RunID   string `json:"run_id"`
	Status  string `json:"status"`
}

// MLflowRegisteredModel is the flat registered-model shape.
type MLflowRegisteredModel struct {
	Name           string               `json:"name"`
	LatestVersions []MLflowModelVersion `json:"latest_versions"`
}

// NormalizeMLflowExperiments decodes the upstream experiments/search envelope
// ({"experiments":[{experiment_id,name,artifact_location,lifecycle_stage,
// last_update_time,...}]}) into the flat list. RunCount is not part of that
// response (MLflow does not return it), so it stays 0.
func NormalizeMLflowExperiments(body []byte) ([]MLflowExperiment, error) {
	var env struct {
		Experiments []struct {
			ExperimentID   string          `json:"experiment_id"`
			Name           string          `json:"name"`
			ArtifactLocn   string          `json:"artifact_location"`
			LifecycleStage string          `json:"lifecycle_stage"`
			LastUpdateTime json.RawMessage `json:"last_update_time"`
		} `json:"experiments"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return nil, err
	}
	out := make([]MLflowExperiment, 0, len(env.Experiments))
	for _, e := range env.Experiments {
		out = append(out, MLflowExperiment{
			ExperimentID:   e.ExperimentID,
			Name:           e.Name,
			ArtifactLocn:   e.ArtifactLocn,
			LifecycleStage: e.LifecycleStage,
			LastUpdateTime: asInt64(e.LastUpdateTime),
		})
	}
	return out, nil
}

// NormalizeMLflowRuns decodes the upstream runs/search envelope
// ({"runs":[{info:{...}, data:{params:[{key,value}], metrics:[{key,value}]}}]})
// into the flat run list, folding params/metrics into maps (latest metric value
// per key wins — the upstream lists newest first).
func NormalizeMLflowRuns(body []byte) ([]MLflowRun, error) {
	var env struct {
		Runs []rawRun `json:"runs"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return nil, err
	}
	out := make([]MLflowRun, 0, len(env.Runs))
	for _, r := range env.Runs {
		out = append(out, r.flatten())
	}
	return out, nil
}

// NormalizeMLflowRun decodes the single-run get envelope ({"run":{...}}).
func NormalizeMLflowRun(body []byte) (MLflowRun, error) {
	var env struct {
		Run rawRun `json:"run"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return MLflowRun{}, err
	}
	return env.Run.flatten(), nil
}

// rawRun mirrors the upstream nested run shape for flattening.
type rawRun struct {
	Info struct {
		RunID        string          `json:"run_id"`
		RunName      string          `json:"run_name"`
		ExperimentID string          `json:"experiment_id"`
		Status       string          `json:"status"`
		StartTime    json.RawMessage `json:"start_time"`
		EndTime      json.RawMessage `json:"end_time"`
	} `json:"info"`
	Data struct {
		Params []struct {
			Key   string `json:"key"`
			Value string `json:"value"`
		} `json:"params"`
		Metrics []struct {
			Key   string          `json:"key"`
			Value json.RawMessage `json:"value"`
		} `json:"metrics"`
	} `json:"data"`
}

func (r rawRun) flatten() MLflowRun {
	params := map[string]string{}
	for _, p := range r.Data.Params {
		params[p.Key] = p.Value
	}
	metrics := map[string]float64{}
	for _, m := range r.Data.Metrics {
		if _, seen := metrics[m.Key]; !seen {
			metrics[m.Key] = asFloat64(m.Value)
		}
	}
	// run_name lives under info in modern MLflow; fall back to the params tag.
	name := r.Info.RunName
	if name == "" {
		name = params["mlflow.runName"]
	}
	return MLflowRun{
		RunID:        r.Info.RunID,
		RunName:      name,
		ExperimentID: r.Info.ExperimentID,
		Status:       r.Info.Status,
		StartTime:    asInt64(r.Info.StartTime),
		EndTime:      asInt64(r.Info.EndTime),
		Params:       params,
		Metrics:      metrics,
	}
}

// NormalizeMLflowModels decodes the registered-models/search envelope
// ({"registered_models":[{name, latest_versions:[{version,current_stage,
// run_id,status}]}]}) into the flat model list.
func NormalizeMLflowModels(body []byte) ([]MLflowRegisteredModel, error) {
	var env struct {
		RegisteredModels []struct {
			Name           string `json:"name"`
			LatestVersions []struct {
				Version      string `json:"version"`
				CurrentStage string `json:"current_stage"`
				RunID        string `json:"run_id"`
				Status       string `json:"status"`
			} `json:"latest_versions"`
		} `json:"registered_models"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return nil, err
	}
	out := make([]MLflowRegisteredModel, 0, len(env.RegisteredModels))
	for _, m := range env.RegisteredModels {
		versions := make([]MLflowModelVersion, 0, len(m.LatestVersions))
		for _, v := range m.LatestVersions {
			versions = append(versions, MLflowModelVersion{
				Version: v.Version,
				Stage:   v.CurrentStage,
				RunID:   v.RunID,
				Status:  v.Status,
			})
		}
		out = append(out, MLflowRegisteredModel{Name: m.Name, LatestVersions: versions})
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Flyte
// ---------------------------------------------------------------------------

// FlyteExecution is the flat execution shape the frontend expects.
type FlyteExecution struct {
	ExecutionID  string `json:"execution_id"`
	WorkflowName string `json:"workflow_name"`
	Phase        string `json:"phase"`
	StartedAt    string `json:"started_at"`
	DurationS    int64  `json:"duration_s"`
}

// NormalizeFlyteExecutions decodes the Flyte Admin executions list
// ({"executions":[{id:{name}, closure:{phase, workflowId:{name}|created_at|
// duration}, spec:{launchPlan:{name}}}]}) into the flat list.
func NormalizeFlyteExecutions(body []byte) ([]FlyteExecution, error) {
	var env struct {
		Executions []rawFlyteExecution `json:"executions"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return nil, err
	}
	out := make([]FlyteExecution, 0, len(env.Executions))
	for _, e := range env.Executions {
		out = append(out, e.flatten())
	}
	return out, nil
}

// NormalizeFlyteExecution decodes a SINGLE Flyte Admin execution (the get-by-id
// endpoint returns the unwrapped Execution message {id,closure,spec}, NOT the
// {"executions":[...]} envelope) into the flat FlyteExecution shape.
func NormalizeFlyteExecution(body []byte) (FlyteExecution, error) {
	var raw rawFlyteExecution
	if err := json.Unmarshal(body, &raw); err != nil {
		return FlyteExecution{}, err
	}
	return raw.flatten(), nil
}

type rawFlyteExecution struct {
	ID struct {
		Name string `json:"name"`
	} `json:"id"`
	Spec struct {
		LaunchPlan struct {
			Name string `json:"name"`
		} `json:"launchPlan"`
	} `json:"spec"`
	Closure struct {
		Phase      string `json:"phase"`
		CreatedAt  string `json:"createdAt"`
		StartedAt  string `json:"startedAt"`
		Duration   string `json:"duration"` // e.g. "123.4s"
		WorkflowID struct {
			Name string `json:"name"`
		} `json:"workflowId"`
	} `json:"closure"`
}

func (e rawFlyteExecution) flatten() FlyteExecution {
	started := e.Closure.StartedAt
	if started == "" {
		started = e.Closure.CreatedAt
	}
	name := e.Closure.WorkflowID.Name
	if name == "" {
		name = e.Spec.LaunchPlan.Name
	}
	return FlyteExecution{
		ExecutionID:  e.ID.Name,
		WorkflowName: name,
		Phase:        e.Closure.Phase,
		StartedAt:    started,
		DurationS:    parseDurationSeconds(e.Closure.Duration),
	}
}
