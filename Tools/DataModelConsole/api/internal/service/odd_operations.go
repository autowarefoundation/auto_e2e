package service

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

var (
	ErrODDOperationsDisabled = errors.New("ODD operations are disabled")
	ErrODDFullRunsDisabled   = errors.New("ODD full runs are disabled")
	ErrODDOperationConflict  = errors.New("another ODD operation is running")
	ErrODDInvalidOperation   = errors.New("invalid ODD operation")
)

type ODDOperationsConfig struct {
	Enabled        bool
	AllowFull      bool
	LaunchPlanName string
	InputsJSON     string
}

type oddLabelerInputTemplate struct {
	OpenAIModel                   string  `json:"openai_model"`
	OpenAIModelRevision           string  `json:"openai_model_revision"`
	BedrockMapModelID             string  `json:"bedrock_map_model_id"`
	BedrockMapModelRevision       string  `json:"bedrock_map_model_revision"`
	LabelerImageDigest            string  `json:"labeler_image_digest"`
	LabelerSourceRevision         string  `json:"labeler_source_revision"`
	CameraAnchorIntervalS         float64 `json:"camera_anchor_interval_s"`
	MaximumCameraAnchors          int     `json:"maximum_camera_anchors"`
	TriggerContextS               float64 `json:"trigger_context_s"`
	RefinementConfidenceThreshold float64 `json:"refinement_confidence_threshold"`
	SmokeMaximumScenes            int     `json:"smoke_maximum_scenes"`
	SceneConcurrency              int     `json:"scene_concurrency"`
	OpenAIConcurrency             int     `json:"openai_concurrency"`
	BedrockConcurrency            int     `json:"bedrock_concurrency"`
}

type ODDOperationResult struct {
	Action            string `json:"action"`
	ExecutionID       string `json:"execution_id"`
	Created           bool   `json:"created"`
	LaunchPlanVersion string `json:"launch_plan_version,omitempty"`
}

type ODDOperationsService struct {
	s3             *S3Service
	flyte          *FlyteService
	enabled        bool
	allowFull      bool
	launchPlanName string
	template       oddLabelerInputTemplate
}

func NewODDOperationsService(
	s3 *S3Service,
	flyte *FlyteService,
	config ODDOperationsConfig,
) (*ODDOperationsService, error) {
	service := &ODDOperationsService{
		s3:             s3,
		flyte:          flyte,
		enabled:        config.Enabled,
		allowFull:      config.Enabled && config.AllowFull,
		launchPlanName: config.LaunchPlanName,
	}
	if !config.Enabled {
		return service, nil
	}
	if s3 == nil || flyte == nil || config.LaunchPlanName != "odd-dataset-labeler" {
		return nil, fmt.Errorf(
			"%w: ODD operation dependencies or LaunchPlan are invalid",
			ErrODDInvalidOperation,
		)
	}
	template, err := decodeODDLabelerInputTemplate(config.InputsJSON)
	if err != nil {
		return nil, err
	}
	service.template = template
	return service, nil
}

func decodeODDLabelerInputTemplate(
	body string,
) (oddLabelerInputTemplate, error) {
	var template oddLabelerInputTemplate
	decoder := json.NewDecoder(bytes.NewBufferString(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&template); err != nil {
		return template, fmt.Errorf(
			"%w: decode ODD labeler inputs: %v",
			ErrODDInvalidOperation, err,
		)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return template, fmt.Errorf(
			"%w: ODD labeler inputs must contain one object",
			ErrODDInvalidOperation,
		)
	}
	if err := template.validate(); err != nil {
		return oddLabelerInputTemplate{}, err
	}
	return template, nil
}

func (t oddLabelerInputTemplate) validate() error {
	required := map[string]string{
		"openai_model":               t.OpenAIModel,
		"openai_model_revision":      t.OpenAIModelRevision,
		"bedrock_map_model_id":       t.BedrockMapModelID,
		"bedrock_map_model_revision": t.BedrockMapModelRevision,
		"labeler_image_digest":       t.LabelerImageDigest,
		"labeler_source_revision":    t.LabelerSourceRevision,
	}
	for name, value := range required {
		if value == "" || len(value) > 256 {
			return fmt.Errorf(
				"%w: %s is missing or too long",
				ErrODDInvalidOperation, name,
			)
		}
	}
	for _, revision := range []string{
		t.OpenAIModelRevision,
		t.BedrockMapModelRevision,
	} {
		switch strings.ToLower(revision) {
		case "latest", "deployed", "main", "unknown":
			return fmt.Errorf(
				"%w: mutable model revision %q",
				ErrODDInvalidOperation, revision,
			)
		}
	}
	if !strings.HasPrefix(t.LabelerImageDigest, "sha256:") ||
		!validOddDigest(strings.TrimPrefix(t.LabelerImageDigest, "sha256:")) ||
		!validSourceRevision(t.LabelerSourceRevision) {
		return fmt.Errorf(
			"%w: labeler image/source identity is invalid",
			ErrODDInvalidOperation,
		)
	}
	if t.CameraAnchorIntervalS <= 0 || t.CameraAnchorIntervalS > 60 ||
		t.MaximumCameraAnchors <= 0 || t.MaximumCameraAnchors > 1024 ||
		t.TriggerContextS < 0 || t.TriggerContextS > 10 ||
		t.RefinementConfidenceThreshold < 0 ||
		t.RefinementConfidenceThreshold > 1 ||
		t.SmokeMaximumScenes <= 0 || t.SmokeMaximumScenes > 100 ||
		t.SceneConcurrency <= 0 || t.SceneConcurrency > 100 ||
		t.OpenAIConcurrency <= 0 || t.OpenAIConcurrency > 100 ||
		t.BedrockConcurrency <= 0 || t.BedrockConcurrency > 100 {
		return fmt.Errorf(
			"%w: ODD sampling or concurrency input is outside bounds",
			ErrODDInvalidOperation,
		)
	}
	return nil
}

func validSourceRevision(value string) bool {
	return (len(value) == 40 || len(value) == 64) &&
		validLowerHex(value)
}

func validLowerHex(value string) bool {
	for _, char := range value {
		if (char < '0' || char > '9') && (char < 'a' || char > 'f') {
			return false
		}
	}
	return value != ""
}

func (s *ODDOperationsService) Enabled() bool {
	return s != nil && s.enabled
}

func (s *ODDOperationsService) AllowFull() bool {
	return s != nil && s.allowFull
}

func (s *ODDOperationsService) Launch(
	ctx context.Context,
	dataset string,
	version string,
	publicationScope string,
) (ODDOperationResult, error) {
	if !s.Enabled() {
		return ODDOperationResult{}, ErrODDOperationsDisabled
	}
	if publicationScope != "smoke" && publicationScope != "full" {
		return ODDOperationResult{}, fmt.Errorf(
			"%w: unsupported publication scope",
			ErrODDInvalidOperation,
		)
	}
	if publicationScope == "full" && !s.allowFull {
		return ODDOperationResult{}, ErrODDFullRunsDisabled
	}
	manifest, err := s.s3.loadPublicationManifest(ctx, dataset, version)
	if err != nil {
		return ODDOperationResult{}, fmt.Errorf(
			"%w: load dataset publication: %v",
			ErrODDInvalidOperation, err,
		)
	}
	launchPlan, err := s.flyte.LatestLaunchPlan(ctx, s.launchPlanName)
	if err != nil {
		return ODDOperationResult{}, err
	}
	maximumScenes := s.template.SmokeMaximumScenes
	if publicationScope == "full" {
		maximumScenes = 0
	}
	inputs := map[string]any{
		"dataset_name":                    dataset,
		"dataset_version":                 version,
		"dataset_manifest_uri":            fmt.Sprintf("s3://%s/%s/%s/shards/manifest.json", s.s3.bucket, dataset, version),
		"dataset_manifest_sha256":         manifest.SHA256,
		"datasets_bucket":                 s.s3.bucket,
		"openai_model":                    s.template.OpenAIModel,
		"openai_model_revision":           s.template.OpenAIModelRevision,
		"bedrock_map_model_id":            s.template.BedrockMapModelID,
		"bedrock_map_model_revision":      s.template.BedrockMapModelRevision,
		"labeler_image_digest":            s.template.LabelerImageDigest,
		"labeler_source_revision":         s.template.LabelerSourceRevision,
		"camera_anchor_interval_s":        s.template.CameraAnchorIntervalS,
		"maximum_camera_anchors":          s.template.MaximumCameraAnchors,
		"trigger_context_s":               s.template.TriggerContextS,
		"refinement_confidence_threshold": s.template.RefinementConfidenceThreshold,
		"maximum_scenes":                  maximumScenes,
		"scene_concurrency":               s.template.SceneConcurrency,
		"openai_concurrency":              s.template.OpenAIConcurrency,
		"bedrock_concurrency":             s.template.BedrockConcurrency,
		"publication_scope":               publicationScope,
	}
	executionName, err := oddExecutionName("odd", map[string]any{
		"launch_plan": launchPlan,
		"inputs":      inputs,
	})
	if err != nil {
		return ODDOperationResult{}, err
	}
	if existing, conflict, err := s.activeODDExecution(ctx, executionName); err != nil {
		return ODDOperationResult{}, err
	} else if conflict {
		return ODDOperationResult{}, ErrODDOperationConflict
	} else if existing {
		return ODDOperationResult{
			Action:            "launch",
			ExecutionID:       executionName,
			Created:           false,
			LaunchPlanVersion: launchPlan.Version,
		}, nil
	}
	created, err := s.flyte.CreateLaunchPlanExecution(
		ctx, launchPlan, executionName, inputs,
	)
	if err != nil {
		return ODDOperationResult{}, err
	}
	return ODDOperationResult{
		Action:            "launch",
		ExecutionID:       created.ExecutionID,
		Created:           created.Created,
		LaunchPlanVersion: launchPlan.Version,
	}, nil
}

func (s *ODDOperationsService) Retry(
	ctx context.Context,
	executionID string,
) (ODDOperationResult, error) {
	if !s.Enabled() {
		return ODDOperationResult{}, ErrODDOperationsDisabled
	}
	result, err := s.flyte.GetExecution(ctx, executionID)
	if err != nil {
		return ODDOperationResult{}, err
	}
	if result.Status != 200 {
		return ODDOperationResult{}, fmt.Errorf(
			"%w: Flyte execution lookup returned status %d",
			ErrODDInvalidOperation, result.Status,
		)
	}
	execution, err := model.NormalizeFlyteExecution(result.Body)
	if err != nil {
		return ODDOperationResult{}, fmt.Errorf(
			"%w: decode Flyte execution: %v",
			ErrODDInvalidOperation, err,
		)
	}
	if execution.ExecutionID != executionID ||
		!isODDExecutionName(execution.WorkflowName) ||
		!isRetryableFlytePhase(execution.Phase) {
		return ODDOperationResult{}, fmt.Errorf(
			"%w: execution is not a failed ODD Dataset Labeler run",
			ErrODDInvalidOperation,
		)
	}
	relaunchName, err := oddExecutionName(
		"odr", map[string]any{"execution_id": executionID},
	)
	if err != nil {
		return ODDOperationResult{}, err
	}
	if existing, conflict, err := s.activeODDExecution(ctx, relaunchName); err != nil {
		return ODDOperationResult{}, err
	} else if conflict {
		return ODDOperationResult{}, ErrODDOperationConflict
	} else if existing {
		return ODDOperationResult{
			Action:      "retry",
			ExecutionID: relaunchName,
			Created:     false,
		}, nil
	}
	created, err := s.flyte.RelaunchExecution(
		ctx, executionID, relaunchName,
	)
	if err != nil {
		return ODDOperationResult{}, err
	}
	return ODDOperationResult{
		Action:      "retry",
		ExecutionID: created.ExecutionID,
		Created:     created.Created,
	}, nil
}

func oddExecutionName(prefix string, identity any) (string, error) {
	body, err := json.Marshal(identity)
	if err != nil {
		return "", fmt.Errorf("encode ODD execution identity: %w", err)
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(body))
	return prefix + "-" + digest[:16], nil
}

func isODDExecutionName(name string) bool {
	normalized := strings.ToLower(name)
	return normalized == "odd-dataset-labeler" ||
		strings.HasSuffix(normalized, "wf_generate_odd_labelset") ||
		strings.HasSuffix(normalized, "wf-generate-odd-labelset")
}

func isActiveFlytePhase(phase string) bool {
	switch phase {
	case "QUEUED", "RUNNING", "SUCCEEDING":
		return true
	default:
		return false
	}
}

func isRetryableFlytePhase(phase string) bool {
	switch phase {
	case "FAILED", "ABORTED", "TIMED_OUT":
		return true
	default:
		return false
	}
}

func (s *ODDOperationsService) activeODDExecution(
	ctx context.Context,
	requestedName string,
) (existing bool, conflict bool, err error) {
	token := ""
	seen := make(map[string]struct{})
	for pageNumber := 0; pageNumber < 10; pageNumber++ {
		if _, found := seen[token]; found {
			return false, false, fmt.Errorf("Flyte execution pagination cycle")
		}
		seen[token] = struct{}{}
		result, err := s.flyte.ListExecutions(ctx, "1000", token)
		if err != nil {
			return false, false, err
		}
		if result.Status != 200 {
			return false, false, fmt.Errorf(
				"list Flyte executions returned status %d", result.Status,
			)
		}
		page, err := model.NormalizeFlyteExecutionsPage(result.Body)
		if err != nil {
			return false, false, err
		}
		for _, execution := range page.Items {
			if !isODDExecutionName(execution.WorkflowName) ||
				!isActiveFlytePhase(execution.Phase) {
				continue
			}
			if execution.ExecutionID == requestedName {
				return true, false, nil
			}
			return false, true, nil
		}
		token = page.NextPageToken
		if token == "" {
			return false, false, nil
		}
	}
	return false, false, fmt.Errorf("Flyte execution pagination exceeds limit")
}
