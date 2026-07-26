package model

import "time"

// DisplacementMetrics keeps validation and held-out evaluation scores
// distinct. Callers must never silently substitute one source for the other.
type DisplacementMetrics struct {
	ADE      *float64 `json:"ade,omitempty"`
	FDE      *float64 `json:"fde,omitempty"`
	GatePass *bool    `json:"gate_pass,omitempty"`
}

// ExperimentExecution is a Flyte execution linked to an MLflow run.
type ExperimentExecution struct {
	ExecutionID  string `json:"execution_id"`
	WorkflowName string `json:"workflow_name"`
	Phase        string `json:"phase"`
	StartedAt    string `json:"started_at"`
	DurationS    int64  `json:"duration_s"`
	URL          string `json:"url,omitempty"`
}

// ExperimentModelVersion is one immutable Registry version produced by a run.
type ExperimentModelVersion struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Role    string `json:"role,omitempty"`
	Status  string `json:"status"`
	URL     string `json:"url,omitempty"`
}

// ExperimentRecord is the cross-system view of one MLflow run and its Flyte
// executions and Registry outputs.
type ExperimentRecord struct {
	RunID          string `json:"run_id"`
	RunName        string `json:"run_name"`
	ExperimentID   string `json:"experiment_id"`
	ExperimentName string `json:"experiment_name"`
	MLflowStatus   string `json:"mlflow_status"`
	StartTime      int64  `json:"start_time"`
	EndTime        int64  `json:"end_time"`

	Dataset             string `json:"dataset"`
	DatasetVersion      string `json:"dataset_version"`
	DataFingerprint     string `json:"data_fingerprint,omitempty"`
	ValidationScope     string `json:"validation_scope,omitempty"`
	ValidationSplitID   string `json:"validation_split_id,omitempty"`
	Backbone            string `json:"backbone,omitempty"`
	FusionMode          string `json:"fusion_mode,omitempty"`
	RouteConditioning   *bool  `json:"route_conditioning,omitempty"`
	Seed                string `json:"seed,omitempty"`
	Epochs              string `json:"epochs,omitempty"`
	EpochsCompleted     string `json:"epochs_completed,omitempty"`
	LineageStatus       string `json:"lineage_status"`
	PrimaryExecutionID  string `json:"primary_execution_id,omitempty"`
	PrimaryExecutionURL string `json:"primary_execution_url,omitempty"`
	MLflowURL           string `json:"mlflow_url,omitempty"`

	Evaluation     *DisplacementMetrics     `json:"evaluation,omitempty"`
	Validation     *DisplacementMetrics     `json:"validation,omitempty"`
	TrainExecution *ExperimentExecution     `json:"train_execution,omitempty"`
	EvalExecution  *ExperimentExecution     `json:"eval_execution,omitempty"`
	ModelVersions  []ExperimentModelVersion `json:"model_versions"`
	Params         map[string]string        `json:"params"`
	Tags           map[string]string        `json:"tags"`
	Metrics        map[string]float64       `json:"metrics"`
}

// ExperimentSummary reports counts for the current unfiltered data set.
type ExperimentSummary struct {
	Total      int `json:"total"`
	Running    int `json:"running"`
	Failed     int `json:"failed"`
	Evaluated  int `json:"evaluated"`
	Registered int `json:"registered"`
	Unlinked   int `json:"unlinked"`
}

// ExperimentsResponse is GET /api/v1/experiments.
type ExperimentsResponse struct {
	GeneratedAt time.Time          `json:"generated_at"`
	Summary     ExperimentSummary  `json:"summary"`
	Experiments []ExperimentRecord `json:"experiments"`
}
