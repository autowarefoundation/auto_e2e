package service

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

const experimentPageSize = 1000

// ExperimentService joins read-only Flyte, MLflow, and Model Registry data.
type ExperimentService struct {
	mlflow          *MLflowService
	flyte           *FlyteService
	mlflowPublicURL string
	flyteConsoleURL string
}

// NewExperimentService creates the cross-system experiment reader.
func NewExperimentService(
	mlflow *MLflowService,
	flyte *FlyteService,
	mlflowPublicURL string,
	flyteConsoleURL string,
) *ExperimentService {
	return &ExperimentService{
		mlflow:          mlflow,
		flyte:           flyte,
		mlflowPublicURL: strings.TrimRight(mlflowPublicURL, "/"),
		flyteConsoleURL: strings.TrimRight(flyteConsoleURL, "/"),
	}
}

// List returns model-bearing MLflow runs enriched with explicit Flyte and
// Registry lineage. Health-check runs without a dataset coordinate are omitted.
func (s *ExperimentService) List(
	ctx context.Context,
) (model.ExperimentsResponse, error) {
	catalog, err := s.listMLflowExperiments(ctx)
	if err != nil {
		return model.ExperimentsResponse{}, err
	}
	if len(catalog) == 0 {
		return model.ExperimentsResponse{
			GeneratedAt: time.Now().UTC(),
			Experiments: []model.ExperimentRecord{},
		}, nil
	}

	experimentIDs := make([]string, 0, len(catalog))
	experimentNames := make(map[string]string, len(catalog))
	for _, experiment := range catalog {
		experimentIDs = append(experimentIDs, experiment.ExperimentID)
		experimentNames[experiment.ExperimentID] = experiment.Name
	}

	var (
		runs       []model.MLflowRun
		versions   []model.MLflowModelVersion
		executions []model.FlyteExecution
		runErr     error
		versionErr error
		flyteErr   error
		wg         sync.WaitGroup
	)
	wg.Add(3)
	go func() {
		defer wg.Done()
		runs, runErr = s.listMLflowRuns(ctx, experimentIDs)
	}()
	go func() {
		defer wg.Done()
		versions, versionErr = s.listModelVersions(ctx)
	}()
	go func() {
		defer wg.Done()
		executions, flyteErr = s.listFlyteExecutions(ctx)
	}()
	wg.Wait()
	if runErr != nil {
		return model.ExperimentsResponse{}, runErr
	}
	if versionErr != nil {
		return model.ExperimentsResponse{}, versionErr
	}
	if flyteErr != nil {
		return model.ExperimentsResponse{}, flyteErr
	}

	executionByID := make(map[string]model.FlyteExecution, len(executions))
	for _, execution := range executions {
		executionByID[execution.ExecutionID] = execution
	}
	versionsByRun := make(map[string][]model.MLflowModelVersion)
	for _, version := range versions {
		versionsByRun[version.RunID] = append(
			versionsByRun[version.RunID], version,
		)
	}

	records := make([]model.ExperimentRecord, 0, len(runs))
	for _, run := range runs {
		if !isModelExperimentRun(run) {
			continue
		}
		records = append(records, s.buildRecord(
			run,
			experimentNames[run.ExperimentID],
			versionsByRun[run.RunID],
			executionByID,
		))
	}
	sort.SliceStable(records, func(i, j int) bool {
		return records[i].StartTime > records[j].StartTime
	})

	response := model.ExperimentsResponse{
		GeneratedAt: time.Now().UTC(),
		Experiments: records,
	}
	for _, record := range records {
		response.Summary.Total++
		status := record.MLflowStatus
		if execution, ok := executionByID[record.PrimaryExecutionID]; ok {
			status = execution.Phase
		}
		switch status {
		case "RUNNING", "SUCCEEDING", "QUEUED", "SCHEDULED":
			response.Summary.Running++
		case "FAILED", "FAILING", "ABORTED", "ABORTING", "TIMED_OUT", "KILLED":
			response.Summary.Failed++
		}
		if record.Evaluation != nil &&
			(record.Evaluation.ADE != nil || record.Evaluation.FDE != nil) {
			response.Summary.Evaluated++
		}
		response.Summary.Registered += len(record.ModelVersions)
		if record.LineageStatus == "missing" {
			response.Summary.Unlinked++
		}
	}
	return response, nil
}

func (s *ExperimentService) buildRecord(
	run model.MLflowRun,
	experimentName string,
	versions []model.MLflowModelVersion,
	executionByID map[string]model.FlyteExecution,
) model.ExperimentRecord {
	trainID := firstNonEmpty(
		run.Params["ctx/train_execution_id"],
		run.Tags["ctx/train_execution_id"],
	)
	evalID := firstNonEmpty(
		run.Tags["ctx/eval_execution_id"],
		run.Params["ctx/eval_execution_id"],
	)
	if trainID == "" {
		for _, version := range versions {
			if id := version.Tags["train_execution_id"]; id != "" {
				trainID = id
				break
			}
		}
	}

	primaryID := firstNonEmpty(trainID, evalID)
	lineageStatus := "missing"
	if primaryID != "" {
		lineageStatus = "partial"
		if _, ok := executionByID[primaryID]; ok {
			lineageStatus = "complete"
		}
	}

	var routeConditioning *bool
	if raw, ok := run.Params["model/enable_route_conditioning"]; ok {
		if parsed, err := strconv.ParseBool(raw); err == nil {
			routeConditioning = &parsed
		}
	}

	modelVersions := make([]model.ExperimentModelVersion, 0, len(versions))
	sort.SliceStable(versions, func(i, j int) bool {
		left, _ := strconv.Atoi(versions[i].Version)
		right, _ := strconv.Atoi(versions[j].Version)
		return left > right
	})
	for _, version := range versions {
		modelVersions = append(modelVersions, model.ExperimentModelVersion{
			Name:    version.Name,
			Version: version.Version,
			Role:    version.Tags["checkpoint_role"],
			Status:  version.Status,
			URL:     s.modelVersionURL(version.Name, version.Version),
		})
	}

	record := model.ExperimentRecord{
		RunID:              run.RunID,
		RunName:            run.RunName,
		ExperimentID:       run.ExperimentID,
		ExperimentName:     experimentName,
		MLflowStatus:       run.Status,
		StartTime:          run.StartTime,
		EndTime:            run.EndTime,
		Dataset:            run.Params["data/dataset"],
		DatasetVersion:     run.Params["data/dataset_version"],
		DataFingerprint:    run.Params["data/fingerprint"],
		ValidationScope:    run.Params["train/validation_scope"],
		ValidationSplitID:  run.Params["train/validation_split_id"],
		Backbone:           run.Params["model/backbone"],
		FusionMode:         run.Params["model/fusion_mode"],
		RouteConditioning:  routeConditioning,
		Seed:               run.Params["train/seed"],
		Epochs:             run.Params["train/epochs"],
		EpochsCompleted:    run.Tags["train/epochs_completed"],
		LineageStatus:      lineageStatus,
		PrimaryExecutionID: primaryID,
		MLflowURL: s.mlflowRunURL(
			run.ExperimentID, run.RunID,
		),
		Evaluation:     displacementMetrics(run.Metrics, "eval/"),
		Validation:     displacementMetrics(run.Metrics, "val/"),
		TrainExecution: s.executionRef(trainID, executionByID),
		EvalExecution:  s.executionRef(evalID, executionByID),
		ModelVersions:  modelVersions,
		Params:         run.Params,
		Tags:           run.Tags,
		Metrics:        run.Metrics,
	}
	record.PrimaryExecutionURL = s.flyteExecutionURL(primaryID)
	return record
}

func displacementMetrics(
	metrics map[string]float64,
	prefix string,
) *model.DisplacementMetrics {
	ade, hasADE := metrics[prefix+"ade"]
	fde, hasFDE := metrics[prefix+"fde"]
	gate, hasGate := metrics[prefix+"gate_pass"]
	if !hasADE && !hasFDE && !hasGate {
		return nil
	}
	out := &model.DisplacementMetrics{}
	if hasADE {
		out.ADE = &ade
	}
	if hasFDE {
		out.FDE = &fde
	}
	if hasGate {
		passed := gate == 1
		out.GatePass = &passed
	}
	return out
}

func (s *ExperimentService) executionRef(
	executionID string,
	executionByID map[string]model.FlyteExecution,
) *model.ExperimentExecution {
	if executionID == "" || executionID == "?" || executionID == "local" {
		return nil
	}
	execution := executionByID[executionID]
	return &model.ExperimentExecution{
		ExecutionID:  executionID,
		WorkflowName: execution.WorkflowName,
		Phase:        execution.Phase,
		StartedAt:    execution.StartedAt,
		DurationS:    execution.DurationS,
		URL:          s.flyteExecutionURL(executionID),
	}
}

func (s *ExperimentService) listMLflowExperiments(
	ctx context.Context,
) ([]model.MLflowExperiment, error) {
	var out []model.MLflowExperiment
	err := paginateTokens("mlflow experiments", func(token string) (string, error) {
		result, err := s.mlflow.SearchExperiments(
			ctx, experimentPageSize, token,
		)
		if err != nil {
			return "", err
		}
		if result.Status != http.StatusOK {
			return "", fmt.Errorf(
				"mlflow experiments/search returned %d", result.Status,
			)
		}
		page, err := model.NormalizeMLflowExperimentsPage(result.Body)
		if err != nil {
			return "", fmt.Errorf("decode mlflow experiments: %w", err)
		}
		out = append(out, page.Items...)
		return page.NextPageToken, nil
	})
	return out, err
}

func (s *ExperimentService) listMLflowRuns(
	ctx context.Context,
	experimentIDs []string,
) ([]model.MLflowRun, error) {
	var out []model.MLflowRun
	err := paginateTokens("mlflow runs", func(token string) (string, error) {
		result, err := s.mlflow.SearchRunsForExperiments(
			ctx, experimentIDs, experimentPageSize, token,
		)
		if err != nil {
			return "", err
		}
		if result.Status != http.StatusOK {
			return "", fmt.Errorf(
				"mlflow runs/search returned %d", result.Status,
			)
		}
		page, err := model.NormalizeMLflowRunsPage(result.Body)
		if err != nil {
			return "", fmt.Errorf("decode mlflow runs: %w", err)
		}
		out = append(out, page.Items...)
		return page.NextPageToken, nil
	})
	return out, err
}

func (s *ExperimentService) listModelVersions(
	ctx context.Context,
) ([]model.MLflowModelVersion, error) {
	var out []model.MLflowModelVersion
	err := paginateTokens("mlflow model versions", func(token string) (string, error) {
		result, err := s.mlflow.SearchModelVersions(
			ctx, experimentPageSize, token,
		)
		if err != nil {
			return "", err
		}
		if result.Status != http.StatusOK {
			return "", fmt.Errorf(
				"mlflow model-versions/search returned %d", result.Status,
			)
		}
		page, err := model.NormalizeMLflowModelVersionsPage(result.Body)
		if err != nil {
			return "", fmt.Errorf("decode mlflow model versions: %w", err)
		}
		out = append(out, page.Items...)
		return page.NextPageToken, nil
	})
	return out, err
}

func (s *ExperimentService) listFlyteExecutions(
	ctx context.Context,
) ([]model.FlyteExecution, error) {
	var out []model.FlyteExecution
	err := paginateTokens("flyte executions", func(token string) (string, error) {
		result, err := s.flyte.ListExecutions(
			ctx, strconv.Itoa(experimentPageSize), token,
		)
		if err != nil {
			return "", err
		}
		if result.Status != http.StatusOK {
			return "", fmt.Errorf(
				"flyte executions list returned %d", result.Status,
			)
		}
		page, err := model.NormalizeFlyteExecutionsPage(result.Body)
		if err != nil {
			return "", fmt.Errorf("decode flyte executions: %w", err)
		}
		out = append(out, page.Items...)
		return page.NextPageToken, nil
	})
	return out, err
}

func paginateTokens(
	name string,
	fetch func(token string) (next string, err error),
) error {
	token := ""
	seen := map[string]struct{}{}
	for {
		if _, ok := seen[token]; ok {
			return fmt.Errorf("%s pagination token cycle", name)
		}
		seen[token] = struct{}{}
		next, err := fetch(token)
		if err != nil {
			return err
		}
		if next == "" {
			return nil
		}
		token = next
	}
}

func isModelExperimentRun(run model.MLflowRun) bool {
	if strings.TrimSpace(run.Params["data/dataset"]) != "" {
		return true
	}
	switch run.Tags["pipeline"] {
	case "imitation-learning", "offline-rl":
		return true
	default:
		return false
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" && value != "?" && value != "local" {
			return value
		}
	}
	return ""
}

func (s *ExperimentService) mlflowRunURL(
	experimentID string,
	runID string,
) string {
	if s.mlflowPublicURL == "" {
		return ""
	}
	return s.mlflowPublicURL + "/#/experiments/" +
		url.PathEscape(experimentID) + "/runs/" + url.PathEscape(runID)
}

func (s *ExperimentService) modelVersionURL(
	name string,
	version string,
) string {
	if s.mlflowPublicURL == "" || name == "" || version == "" {
		return ""
	}
	return s.mlflowPublicURL + "/#/models/" + url.PathEscape(name) +
		"/versions/" + url.PathEscape(version)
}

func (s *ExperimentService) flyteExecutionURL(executionID string) string {
	if s.flyteConsoleURL == "" || executionID == "" {
		return ""
	}
	return s.flyteConsoleURL + "/projects/" +
		url.PathEscape(s.flyte.project) + "/domains/" +
		url.PathEscape(s.flyte.domain) + "/executions/" +
		url.PathEscape(executionID)
}
