package service

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

const maxOddJSONBytes = 64 << 20

// ErrODDUnavailable means the selected dataset has no complete ready LabelSet.
var ErrODDUnavailable = errors.New("ODD LabelSet unavailable")

// ErrODDInvalidQuery means a structured search request is not executable.
var ErrODDInvalidQuery = errors.New("invalid ODD search query")

type ODDArtifact struct {
	Key           string `json:"key"`
	SHA256        string `json:"sha256"`
	ByteSize      int64  `json:"byte_size"`
	ContentType   string `json:"content_type,omitempty"`
	Format        string `json:"format,omitempty"`
	RowCount      int64  `json:"row_count,omitempty"`
	SchemaVersion string `json:"schema_version,omitempty"`
	Authoritative bool   `json:"authoritative,omitempty"`
}

type ODDQualityState struct {
	SchemaVersion       string `json:"schema_version"`
	StructuralStatus    string `json:"structural_status"`
	AuditStatus         string `json:"audit_status"`
	CertificationStatus string `json:"certification_status"`
}

type oddCoverageKey struct {
	Key             string `json:"key"`
	QualityTier     string `json:"quality_tier"`
	SupportState    string `json:"support_state"`
	AttemptedCount  int64  `json:"attempted_count"`
	SuccessfulCount int64  `json:"successful_count"`
}

type oddCoverageDocument struct {
	LabelSetID string           `json:"labelset_id"`
	Keys       []oddCoverageKey `json:"keys"`
}

type oddStatisticKey struct {
	Key                     string  `json:"key"`
	ValidSceneCount         int64   `json:"valid_scene_count"`
	EligibleSceneCount      int64   `json:"eligible_scene_count"`
	ObservableSceneCoverage float64 `json:"observable_scene_coverage"`
}

type oddStatisticsDocument struct {
	LabelSetID string            `json:"labelset_id"`
	Keys       []oddStatisticKey `json:"keys"`
}

type ODDManifest struct {
	SchemaVersion         string                 `json:"schema_version"`
	Status                string                 `json:"status"`
	LabelSetID            string                 `json:"labelset_id"`
	DatasetName           string                 `json:"dataset_name"`
	DatasetVersion        string                 `json:"dataset_version"`
	DatasetManifestURI    string                 `json:"dataset_manifest_uri"`
	DatasetManifestSHA256 string                 `json:"dataset_manifest_sha256"`
	OntologyVersion       string                 `json:"ontology_version"`
	OntologySHA256        string                 `json:"ontology_sha256"`
	LabelerVersion        string                 `json:"labeler_version"`
	LabelerImageDigest    string                 `json:"labeler_image_digest"`
	LabelerSourceRevision string                 `json:"labeler_source_revision"`
	PublicationScope      string                 `json:"publication_scope"`
	ExpectedSceneCount    int                    `json:"expected_scene_count"`
	SceneCount            int                    `json:"scene_count"`
	OpenAICompatible      map[string]string      `json:"openai_compatible"`
	Quality               ODDQualityState        `json:"quality"`
	Artifacts             map[string]ODDArtifact `json:"artifacts"`
}

type oddPointer struct {
	SchemaVersion  string `json:"schema_version"`
	Status         string `json:"status"`
	DatasetName    string `json:"dataset_name"`
	DatasetVersion string `json:"dataset_version"`
	LabelSetID     string `json:"labelset_id"`
	ManifestKey    string `json:"manifest_key"`
	ManifestSHA256 string `json:"manifest_sha256"`
}

type ODDSceneObservationSummary struct {
	Key              string   `json:"key"`
	Status           string   `json:"status"`
	Values           []string `json:"values"`
	Source           string   `json:"source"`
	Confidence       float64  `json:"confidence"`
	DurationNS       int64    `json:"duration_ns"`
	FirstTimestampNS int64    `json:"first_timestamp_ns"`
	IntervalCount    int      `json:"interval_count,omitempty"`
	CameraID         string   `json:"camera_id,omitempty"`
	ActorTrackUID    string   `json:"actor_track_uid,omitempty"`
	EventUID         string   `json:"event_uid,omitempty"`
}

type ODDSceneEventSummary struct {
	EventUID         string   `json:"event_uid"`
	PrimaryEventKey  string   `json:"primary_event_key"`
	PrimaryValues    []string `json:"primary_values"`
	StartTimestampNS int64    `json:"start_timestamp_ns"`
	EndTimestampNS   int64    `json:"end_timestamp_ns"`
	Status           string   `json:"status"`
	Confidence       float64  `json:"confidence"`
	ActorTrackUIDs   []string `json:"actor_track_uids"`
	Outcome          string   `json:"outcome"`
}

type ODDSceneSummary struct {
	SceneUID         string                       `json:"scene_uid"`
	ShardName        string                       `json:"shard_name"`
	RecordKey        string                       `json:"record_key"`
	RecordSHA256     string                       `json:"record_sha256"`
	RecordByteSize   int64                        `json:"record_byte_size"`
	StartTimestampNS int64                        `json:"start_timestamp_ns"`
	EndTimestampNS   int64                        `json:"end_timestamp_ns"`
	DistanceM        float64                      `json:"distance_m"`
	Observations     []ODDSceneObservationSummary `json:"observations"`
	Events           []ODDSceneEventSummary       `json:"events,omitempty"`
	Matched          []ODDSceneObservationSummary `json:"matched,omitempty"`
	MatchedDuration  int64                        `json:"matched_duration_ns,omitempty"`
	MatchConfidence  float64                      `json:"match_confidence,omitempty"`
	FirstMatchedNS   int64                        `json:"first_matched_timestamp_ns,omitempty"`
}

type oddSceneIndex struct {
	SchemaVersion string            `json:"schema_version"`
	LabelSetID    string            `json:"labelset_id"`
	Scenes        []ODDSceneSummary `json:"scenes"`
}

type ODDSearchResponse struct {
	Dataset        string            `json:"dataset"`
	Version        string            `json:"version"`
	LabelSetID     string            `json:"labelset_id"`
	Scenes         []ODDSceneSummary `json:"scenes"`
	Total          int               `json:"total"`
	Limit          int               `json:"limit"`
	Offset         int               `json:"offset"`
	More           bool              `json:"more"`
	ManifestSHA256 string            `json:"manifest_sha256"`
}

type ODDEvidenceResponse struct {
	Dataset             string            `json:"dataset"`
	Version             string            `json:"version"`
	LabelSetID          string            `json:"labelset_id"`
	SceneUID            string            `json:"scene_uid"`
	Observation         json.RawMessage   `json:"observation"`
	SupportingEvidence  []json.RawMessage `json:"supporting_evidence"`
	ConflictingEvidence []json.RawMessage `json:"conflicting_evidence"`
	RelatedEvents       []json.RawMessage `json:"related_events"`
	SceneProvenance     json.RawMessage   `json:"scene_provenance"`
	ManifestSHA256      string            `json:"manifest_sha256"`
}

type ODDSearchPredicate struct {
	Key               string   `json:"key"`
	Operator          string   `json:"operator"`
	Values            []string `json:"values"`
	Statuses          []string `json:"statuses"`
	Sources           []string `json:"sources"`
	MinimumConfidence float64  `json:"minimum_confidence"`
	MinimumDurationNS int64    `json:"minimum_duration_ns"`
	CameraID          string   `json:"camera_id"`
	ActorTrackUID     string   `json:"actor_track_uid"`
}

type ODDSearchGroup struct {
	Logic      string               `json:"logic"`
	Predicates []ODDSearchPredicate `json:"predicates"`
	Groups     []ODDSearchGroup     `json:"groups"`
}

type ODDStructuredSearchRequest struct {
	Query      ODDSearchGroup `json:"query"`
	Sort       string         `json:"sort"`
	Descending bool           `json:"descending"`
	Limit      int            `json:"limit"`
	Offset     int            `json:"offset"`
}

func validOddDigest(value string) bool {
	if len(value) != 64 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil && strings.ToLower(value) == value
}

func (s *S3Service) oddObject(
	ctx context.Context,
	key string,
	expectedSHA256 string,
	expectedByteSize int64,
) ([]byte, error) {
	output, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return nil, fmt.Errorf("%w: read %s: %v", ErrODDUnavailable, key, err)
	}
	defer output.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(output.Body, maxOddJSONBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read ODD object %s: %w", key, err)
	}
	if len(payload) > maxOddJSONBytes {
		return nil, fmt.Errorf("ODD object exceeds size cap: %s", key)
	}
	if expectedByteSize > 0 && int64(len(payload)) != expectedByteSize {
		return nil, fmt.Errorf(
			"ODD object size differs for %s: expected=%d actual=%d",
			key, expectedByteSize, len(payload),
		)
	}
	if expectedSHA256 != "" {
		actual := fmt.Sprintf("%x", sha256.Sum256(payload))
		if actual != expectedSHA256 {
			return nil, fmt.Errorf(
				"ODD object digest differs for %s: expected=%s actual=%s",
				key, expectedSHA256, actual,
			)
		}
	}
	return payload, nil
}

func (s *S3Service) loadODDManifest(
	ctx context.Context,
	dataset string,
	version string,
) (ODDManifest, string, error) {
	var manifest ODDManifest
	pointerKey := fmt.Sprintf("%s/%s/odd/latest.json", dataset, version)
	body, err := s.oddObject(ctx, pointerKey, "", 0)
	if err != nil {
		return manifest, "", err
	}
	var pointer oddPointer
	if err := json.Unmarshal(body, &pointer); err != nil {
		return manifest, "", fmt.Errorf("decode ODD pointer: %w", err)
	}
	expectedRoot := fmt.Sprintf("%s/%s/odd/labelsets/", dataset, version)
	if pointer.SchemaVersion != "odd_labelset_pointer_v1" ||
		pointer.Status != "ready" ||
		pointer.DatasetName != dataset ||
		pointer.DatasetVersion != version ||
		pointer.LabelSetID == "" ||
		!strings.HasPrefix(pointer.ManifestKey, expectedRoot) ||
		!strings.HasSuffix(pointer.ManifestKey, "/manifest.json") ||
		!validOddDigest(pointer.ManifestSHA256) {
		return manifest, "", fmt.Errorf("invalid ODD ready pointer")
	}
	body, err = s.oddObject(
		ctx, pointer.ManifestKey, pointer.ManifestSHA256, 0,
	)
	if err != nil {
		return manifest, "", err
	}
	if err := json.Unmarshal(body, &manifest); err != nil {
		return manifest, "", fmt.Errorf("decode ODD manifest: %w", err)
	}
	if manifest.SchemaVersion != "odd_labelset_manifest_v1" ||
		manifest.Status != "ready" ||
		manifest.LabelSetID != pointer.LabelSetID ||
		manifest.DatasetName != dataset ||
		manifest.DatasetVersion != version ||
		manifest.SceneCount <= 0 ||
		!validOddDigest(manifest.DatasetManifestSHA256) ||
		!validOddDigest(manifest.OntologySHA256) {
		return ODDManifest{}, "", fmt.Errorf("invalid ODD manifest")
	}
	for _, name := range []string{"ontology", "statistics", "scene_index"} {
		artifact, found := manifest.Artifacts[name]
		if !found ||
			!strings.HasPrefix(artifact.Key, expectedRoot) ||
			!validOddDigest(artifact.SHA256) ||
			artifact.ByteSize <= 0 ||
			artifact.ByteSize > maxOddJSONBytes {
			return ODDManifest{}, "", fmt.Errorf(
				"invalid ODD %s artifact", name,
			)
		}
	}
	return manifest, pointer.ManifestSHA256, nil
}

func (s *S3Service) oddArtifact(
	ctx context.Context,
	dataset string,
	version string,
	name string,
) ([]byte, ODDManifest, string, error) {
	manifest, manifestSHA, err := s.loadODDManifest(ctx, dataset, version)
	if err != nil {
		return nil, manifest, "", err
	}
	artifact := manifest.Artifacts[name]
	body, err := s.oddObject(
		ctx, artifact.Key, artifact.SHA256, artifact.ByteSize,
	)
	return body, manifest, manifestSHA, err
}

// ODDOntology returns the complete registry, including zero-count candidates.
func (s *S3Service) ODDOntology(
	ctx context.Context,
	dataset string,
	version string,
) (json.RawMessage, ODDManifest, string, error) {
	body, manifest, digest, err := s.oddArtifact(
		ctx, dataset, version, "ontology",
	)
	if err != nil {
		return nil, manifest, digest, err
	}
	coverageBody, coverageManifest, _, err := s.oddArtifact(
		ctx, dataset, version, "quality_coverage",
	)
	if err != nil {
		return nil, manifest, digest, err
	}
	statisticsBody, statisticsManifest, _, err := s.oddArtifact(
		ctx, dataset, version, "statistics",
	)
	if err != nil {
		return nil, manifest, digest, err
	}
	if coverageManifest.LabelSetID != manifest.LabelSetID ||
		statisticsManifest.LabelSetID != manifest.LabelSetID {
		return nil, manifest, digest, fmt.Errorf(
			"ODD ontology support artifacts differ from LabelSet",
		)
	}
	enriched, err := enrichODDOntology(
		body,
		coverageBody,
		statisticsBody,
		dataset,
		version,
		manifest.LabelSetID,
	)
	if err != nil {
		return nil, manifest, digest, err
	}
	return json.RawMessage(enriched), manifest, digest, nil
}

func canonicalODDSupportState(value string, qualityTier string) (string, bool) {
	switch value {
	case "supported_certified", "supported_experimental",
		"unsupported_missing_source", "disabled_pending_audit":
		return value, true
	case "supported_observed":
		if qualityTier == "certified" {
			return "supported_certified", true
		}
		return "supported_experimental", true
	case "attempted_no_valid", "unsupported_reported":
		return "unsupported_missing_source", true
	default:
		return "", false
	}
}

func enrichODDOntology(
	ontologyBody []byte,
	coverageBody []byte,
	statisticsBody []byte,
	dataset string,
	version string,
	labelSetID string,
) ([]byte, error) {
	var ontology map[string]any
	if err := json.Unmarshal(ontologyBody, &ontology); err != nil {
		return nil, fmt.Errorf("decode ODD ontology: %w", err)
	}
	labels, ok := ontology["labels"].([]any)
	if !ok {
		return nil, fmt.Errorf("ODD ontology labels are invalid")
	}
	var coverage oddCoverageDocument
	if err := json.Unmarshal(coverageBody, &coverage); err != nil {
		return nil, fmt.Errorf("decode ODD coverage: %w", err)
	}
	var statistics oddStatisticsDocument
	if err := json.Unmarshal(statisticsBody, &statistics); err != nil {
		return nil, fmt.Errorf("decode ODD statistics: %w", err)
	}
	if coverage.LabelSetID != labelSetID || statistics.LabelSetID != labelSetID {
		return nil, fmt.Errorf("ODD ontology support identity differs")
	}
	coverageByKey := make(map[string]oddCoverageKey, len(coverage.Keys))
	for _, row := range coverage.Keys {
		if row.Key == "" {
			return nil, fmt.Errorf("ODD coverage has an empty key")
		}
		if _, found := coverageByKey[row.Key]; found {
			return nil, fmt.Errorf("ODD coverage has duplicate key %s", row.Key)
		}
		coverageByKey[row.Key] = row
	}
	statisticsByKey := make(map[string]oddStatisticKey, len(statistics.Keys))
	for _, row := range statistics.Keys {
		if row.Key == "" {
			return nil, fmt.Errorf("ODD statistics has an empty key")
		}
		if _, found := statisticsByKey[row.Key]; found {
			return nil, fmt.Errorf("ODD statistics has duplicate key %s", row.Key)
		}
		statisticsByKey[row.Key] = row
	}
	seen := make(map[string]struct{}, len(labels))
	for _, rawLabel := range labels {
		label, ok := rawLabel.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("ODD ontology label is invalid")
		}
		key, ok := label["key"].(string)
		if !ok || key == "" {
			return nil, fmt.Errorf("ODD ontology label key is invalid")
		}
		if _, found := seen[key]; found {
			return nil, fmt.Errorf("ODD ontology has duplicate key %s", key)
		}
		seen[key] = struct{}{}
		coverageRow, found := coverageByKey[key]
		if !found {
			return nil, fmt.Errorf("ODD coverage is missing key %s", key)
		}
		statisticsRow, found := statisticsByKey[key]
		if !found {
			return nil, fmt.Errorf("ODD statistics is missing key %s", key)
		}
		supportState, valid := canonicalODDSupportState(
			coverageRow.SupportState,
			coverageRow.QualityTier,
		)
		if !valid {
			return nil, fmt.Errorf(
				"ODD coverage has invalid support state %s",
				coverageRow.SupportState,
			)
		}
		label["dataset_support"] = map[string]any{
			"support_state":             supportState,
			"quality_tier":              coverageRow.QualityTier,
			"valid_scene_count":         statisticsRow.ValidSceneCount,
			"eligible_scene_count":      statisticsRow.EligibleSceneCount,
			"observable_scene_coverage": statisticsRow.ObservableSceneCoverage,
			"attempted_count":           coverageRow.AttemptedCount,
			"successful_count":          coverageRow.SuccessfulCount,
		}
	}
	if len(seen) != len(coverageByKey) || len(seen) != len(statisticsByKey) {
		return nil, fmt.Errorf(
			"ODD ontology, coverage, and statistics keys differ",
		)
	}
	ontology["dataset_name"] = dataset
	ontology["dataset_version"] = version
	ontology["labelset_id"] = labelSetID
	enriched, err := json.Marshal(ontology)
	if err != nil {
		return nil, fmt.Errorf("encode enriched ODD ontology: %w", err)
	}
	return enriched, nil
}

// ODDStatistics returns the precomputed scene-native dataset composition.
func (s *S3Service) ODDStatistics(
	ctx context.Context,
	dataset string,
	version string,
) (json.RawMessage, ODDManifest, string, error) {
	body, manifest, digest, err := s.oddArtifact(
		ctx, dataset, version, "statistics",
	)
	return json.RawMessage(body), manifest, digest, err
}

func (s *S3Service) oddScenes(
	ctx context.Context,
	dataset string,
	version string,
) (oddSceneIndex, ODDManifest, string, error) {
	var index oddSceneIndex
	body, manifest, digest, err := s.oddArtifact(
		ctx, dataset, version, "scene_index",
	)
	if err != nil {
		return index, manifest, digest, err
	}
	if err := json.Unmarshal(body, &index); err != nil {
		return index, manifest, digest, fmt.Errorf(
			"decode ODD scene index: %w", err,
		)
	}
	if (index.SchemaVersion != "odd_scene_index_v1" &&
		index.SchemaVersion != "odd_scene_index_v2") ||
		index.LabelSetID != manifest.LabelSetID ||
		len(index.Scenes) != manifest.SceneCount {
		return oddSceneIndex{}, manifest, digest, fmt.Errorf(
			"ODD scene index differs from manifest",
		)
	}
	root := fmt.Sprintf(
		"%s/%s/odd/labelsets/%s/scenes/",
		dataset, version, manifest.LabelSetID,
	)
	seen := make(map[string]struct{}, len(index.Scenes))
	for _, scene := range index.Scenes {
		if scene.SceneUID == "" ||
			scene.ShardName == "" ||
			!strings.HasPrefix(scene.RecordKey, root) ||
			!validOddDigest(scene.RecordSHA256) ||
			scene.RecordByteSize <= 0 ||
			scene.RecordByteSize > maxOddJSONBytes ||
			scene.EndTimestampNS <= scene.StartTimestampNS {
			return oddSceneIndex{}, manifest, digest, fmt.Errorf(
				"invalid ODD scene index entry",
			)
		}
		if _, found := seen[scene.SceneUID]; found {
			return oddSceneIndex{}, manifest, digest, fmt.Errorf(
				"duplicate ODD scene uid",
			)
		}
		seen[scene.SceneUID] = struct{}{}
	}
	return index, manifest, digest, nil
}

func containsODDValue(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func intersectsODDValues(left []string, right []string) bool {
	for _, value := range left {
		if containsODDValue(right, value) {
			return true
		}
	}
	return false
}

func matchODDPredicate(
	observation ODDSceneObservationSummary,
	predicate ODDSearchPredicate,
) bool {
	if predicate.Key != "" && observation.Key != predicate.Key {
		return false
	}
	statuses := predicate.Statuses
	if len(statuses) == 0 && len(predicate.Values) > 0 {
		statuses = []string{"valid"}
	}
	if len(statuses) > 0 && !containsODDValue(statuses, observation.Status) {
		return false
	}
	if len(predicate.Sources) > 0 &&
		!containsODDValue(predicate.Sources, observation.Source) {
		return false
	}
	if observation.Confidence < predicate.MinimumConfidence ||
		observation.DurationNS < predicate.MinimumDurationNS {
		return false
	}
	if predicate.CameraID != "" && observation.CameraID != predicate.CameraID {
		return false
	}
	if predicate.ActorTrackUID != "" &&
		observation.ActorTrackUID != predicate.ActorTrackUID {
		return false
	}
	operator := predicate.Operator
	if operator == "" {
		operator = "contains"
	}
	switch operator {
	case "exists":
		return true
	case "contains", "equals", "in":
		return len(predicate.Values) == 0 ||
			intersectsODDValues(observation.Values, predicate.Values)
	case "not_equals":
		return observation.Status == "valid" &&
			len(observation.Values) > 0 &&
			!intersectsODDValues(observation.Values, predicate.Values)
	default:
		return false
	}
}

func validateODDGroup(group ODDSearchGroup, depth int) error {
	if depth > 3 {
		return fmt.Errorf("%w: nesting exceeds three levels", ErrODDInvalidQuery)
	}
	logic := group.Logic
	if logic == "" {
		logic = "and"
	}
	if logic != "and" && logic != "or" {
		return fmt.Errorf("%w: logic must be and or or", ErrODDInvalidQuery)
	}
	if len(group.Predicates)+len(group.Groups) == 0 {
		return fmt.Errorf("%w: group is empty", ErrODDInvalidQuery)
	}
	if len(group.Predicates) > 16 || len(group.Groups) > 8 {
		return fmt.Errorf("%w: group is too large", ErrODDInvalidQuery)
	}
	for _, predicate := range group.Predicates {
		switch predicate.Operator {
		case "", "exists", "contains", "equals", "in", "not_equals":
		default:
			return fmt.Errorf("%w: unsupported operator", ErrODDInvalidQuery)
		}
		if predicate.MinimumConfidence < 0 ||
			predicate.MinimumConfidence > 1 ||
			predicate.MinimumDurationNS < 0 {
			return fmt.Errorf("%w: invalid numeric filter", ErrODDInvalidQuery)
		}
	}
	for _, child := range group.Groups {
		if err := validateODDGroup(child, depth+1); err != nil {
			return err
		}
	}
	return nil
}

func matchODDGroup(
	scene ODDSceneSummary,
	group ODDSearchGroup,
) (bool, []ODDSceneObservationSummary) {
	logic := group.Logic
	if logic == "" {
		logic = "and"
	}
	results := make([]bool, 0, len(group.Predicates)+len(group.Groups))
	matches := make([]ODDSceneObservationSummary, 0)
	for _, predicate := range group.Predicates {
		predicateMatches := make([]ODDSceneObservationSummary, 0)
		for _, observation := range scene.Observations {
			if matchODDPredicate(observation, predicate) {
				predicateMatches = append(predicateMatches, observation)
			}
		}
		results = append(results, len(predicateMatches) > 0)
		matches = append(matches, predicateMatches...)
	}
	for _, child := range group.Groups {
		matched, childMatches := matchODDGroup(scene, child)
		results = append(results, matched)
		if matched {
			matches = append(matches, childMatches...)
		}
	}
	matched := logic == "and"
	for _, result := range results {
		if logic == "and" {
			matched = matched && result
		} else {
			matched = matched || result
		}
	}
	if !matched {
		return false, nil
	}
	seen := make(map[string]struct{}, len(matches))
	unique := make([]ODDSceneObservationSummary, 0, len(matches))
	for _, observation := range matches {
		identity := fmt.Sprintf(
			"%s\x00%s\x00%s\x00%d\x00%s\x00%s",
			observation.Key,
			observation.Status,
			strings.Join(observation.Values, "\x00"),
			observation.FirstTimestampNS,
			observation.CameraID,
			observation.ActorTrackUID,
		)
		if _, found := seen[identity]; found {
			continue
		}
		seen[identity] = struct{}{}
		unique = append(unique, observation)
	}
	return true, unique
}

func decorateODDMatch(
	scene ODDSceneSummary,
	matches []ODDSceneObservationSummary,
) ODDSceneSummary {
	scene.Matched = matches
	scene.FirstMatchedNS = scene.EndTimestampNS
	for _, match := range matches {
		scene.MatchedDuration += match.DurationNS
		scene.MatchConfidence = max(scene.MatchConfidence, match.Confidence)
		scene.FirstMatchedNS = min(
			scene.FirstMatchedNS,
			match.FirstTimestampNS,
		)
	}
	if len(matches) == 0 {
		scene.FirstMatchedNS = 0
	}
	return scene
}

// SearchODDScenesStructured searches scene summaries, never training samples.
func (s *S3Service) SearchODDScenesStructured(
	ctx context.Context,
	dataset string,
	version string,
	request ODDStructuredSearchRequest,
) (ODDSearchResponse, error) {
	if err := validateODDGroup(request.Query, 1); err != nil {
		return ODDSearchResponse{}, err
	}
	if request.Limit <= 0 {
		request.Limit = 50
	}
	if request.Limit > 200 || request.Offset < 0 {
		return ODDSearchResponse{}, fmt.Errorf(
			"%w: invalid pagination", ErrODDInvalidQuery,
		)
	}
	switch request.Sort {
	case "", "scene_uid", "confidence", "matched_duration",
		"scene_duration", "recording_time":
	default:
		return ODDSearchResponse{}, fmt.Errorf(
			"%w: invalid sort", ErrODDInvalidQuery,
		)
	}
	index, manifest, manifestSHA, err := s.oddScenes(ctx, dataset, version)
	if err != nil {
		return ODDSearchResponse{}, err
	}
	matches := make([]ODDSceneSummary, 0)
	for _, scene := range index.Scenes {
		matched, observations := matchODDGroup(scene, request.Query)
		if matched {
			matches = append(matches, decorateODDMatch(scene, observations))
		}
	}
	sort.SliceStable(matches, func(i, j int) bool {
		var less bool
		switch request.Sort {
		case "confidence":
			less = matches[i].MatchConfidence < matches[j].MatchConfidence
		case "matched_duration":
			less = matches[i].MatchedDuration < matches[j].MatchedDuration
		case "scene_duration":
			left := matches[i].EndTimestampNS - matches[i].StartTimestampNS
			right := matches[j].EndTimestampNS - matches[j].StartTimestampNS
			less = left < right
		case "recording_time":
			less = matches[i].StartTimestampNS < matches[j].StartTimestampNS
		default:
			return matches[i].SceneUID < matches[j].SceneUID
		}
		if request.Sort != "" && request.Sort != "scene_uid" {
			var equal bool
			switch request.Sort {
			case "confidence":
				equal = matches[i].MatchConfidence == matches[j].MatchConfidence
			case "matched_duration":
				equal = matches[i].MatchedDuration == matches[j].MatchedDuration
			case "scene_duration":
				equal = matches[i].EndTimestampNS-matches[i].StartTimestampNS ==
					matches[j].EndTimestampNS-matches[j].StartTimestampNS
			case "recording_time":
				equal = matches[i].StartTimestampNS == matches[j].StartTimestampNS
			}
			if equal {
				return matches[i].SceneUID < matches[j].SceneUID
			}
		}
		if request.Descending {
			return !less
		}
		return less
	})
	total := len(matches)
	offset := min(request.Offset, total)
	end := min(total, offset+request.Limit)
	return ODDSearchResponse{
		Dataset:        dataset,
		Version:        version,
		LabelSetID:     manifest.LabelSetID,
		Scenes:         matches[offset:end],
		Total:          total,
		Limit:          request.Limit,
		Offset:         offset,
		More:           end < total,
		ManifestSHA256: manifestSHA,
	}, nil
}

// SearchODDScenes preserves the original single-predicate GET API.
func (s *S3Service) SearchODDScenes(
	ctx context.Context,
	dataset string,
	version string,
	key string,
	value string,
	status string,
	source string,
	limit int,
	offset int,
) (ODDSearchResponse, error) {
	predicate := ODDSearchPredicate{Key: key, Operator: "contains"}
	if value != "" {
		predicate.Values = []string{value}
	}
	if status != "" {
		predicate.Statuses = []string{status}
	}
	if source != "" {
		predicate.Sources = []string{source}
	}
	return s.SearchODDScenesStructured(
		ctx,
		dataset,
		version,
		ODDStructuredSearchRequest{
			Query: ODDSearchGroup{
				Logic:      "and",
				Predicates: []ODDSearchPredicate{predicate},
			},
			Sort:   "scene_uid",
			Limit:  limit,
			Offset: offset,
		},
	)
}

// ODDScene returns one coalesced scene record from the pinned LabelSet.
func (s *S3Service) ODDScene(
	ctx context.Context,
	dataset string,
	version string,
	sceneUID string,
) (json.RawMessage, ODDManifest, string, error) {
	index, manifest, digest, err := s.oddScenes(ctx, dataset, version)
	if err != nil {
		return nil, manifest, digest, err
	}
	for _, scene := range index.Scenes {
		if scene.SceneUID != sceneUID {
			continue
		}
		body, err := s.oddObject(
			ctx,
			scene.RecordKey,
			scene.RecordSHA256,
			scene.RecordByteSize,
		)
		if err != nil {
			return nil, manifest, digest, err
		}
		var identity struct {
			SceneUID       string `json:"scene_uid"`
			DatasetName    string `json:"dataset_name"`
			DatasetVersion string `json:"dataset_version"`
		}
		if err := json.Unmarshal(body, &identity); err != nil ||
			identity.SceneUID != sceneUID ||
			identity.DatasetName != dataset ||
			identity.DatasetVersion != version {
			return nil, manifest, digest, fmt.Errorf(
				"ODD scene record differs from index coordinate",
			)
		}
		return json.RawMessage(body), manifest, digest, err
	}
	return nil, manifest, digest, ErrNotFound
}

// ODDEvidence returns one observation with only its referenced evidence and events.
func (s *S3Service) ODDEvidence(
	ctx context.Context,
	dataset string,
	version string,
	sceneUID string,
	observationUID string,
) (ODDEvidenceResponse, ODDManifest, string, error) {
	body, manifest, digest, err := s.ODDScene(
		ctx, dataset, version, sceneUID,
	)
	if err != nil {
		return ODDEvidenceResponse{}, manifest, digest, err
	}
	var record struct {
		SceneUID     string            `json:"scene_uid"`
		Observations []json.RawMessage `json:"observations"`
		Evidence     []json.RawMessage `json:"evidence"`
		Events       []json.RawMessage `json:"events"`
		Provenance   json.RawMessage   `json:"provenance"`
	}
	if err := json.Unmarshal(body, &record); err != nil {
		return ODDEvidenceResponse{}, manifest, digest, fmt.Errorf(
			"decode ODD scene evidence: %w", err,
		)
	}
	var selected json.RawMessage
	var observation struct {
		ObservationUID          string   `json:"observation_uid"`
		EventUID                string   `json:"event_uid"`
		EvidenceUIDs            []string `json:"evidence_uids"`
		ConflictingEvidenceUIDs []string `json:"conflicting_evidence_uids"`
	}
	for _, raw := range record.Observations {
		var identity struct {
			ObservationUID string `json:"observation_uid"`
		}
		if err := json.Unmarshal(raw, &identity); err != nil ||
			identity.ObservationUID == "" {
			return ODDEvidenceResponse{}, manifest, digest, fmt.Errorf(
				"invalid observation in ODD scene record",
			)
		}
		if identity.ObservationUID != observationUID {
			continue
		}
		if selected != nil {
			return ODDEvidenceResponse{}, manifest, digest, fmt.Errorf(
				"duplicate ODD observation uid",
			)
		}
		selected = raw
		if err := json.Unmarshal(raw, &observation); err != nil {
			return ODDEvidenceResponse{}, manifest, digest, fmt.Errorf(
				"decode ODD observation: %w", err,
			)
		}
	}
	if selected == nil {
		return ODDEvidenceResponse{}, manifest, digest, ErrNotFound
	}

	evidenceByUID := make(map[string]json.RawMessage, len(record.Evidence))
	for _, raw := range record.Evidence {
		var identity struct {
			EvidenceUID string `json:"evidence_uid"`
		}
		if err := json.Unmarshal(raw, &identity); err != nil ||
			identity.EvidenceUID == "" {
			return ODDEvidenceResponse{}, manifest, digest, fmt.Errorf(
				"invalid evidence in ODD scene record",
			)
		}
		if _, found := evidenceByUID[identity.EvidenceUID]; found {
			return ODDEvidenceResponse{}, manifest, digest, fmt.Errorf(
				"duplicate ODD evidence uid",
			)
		}
		evidenceByUID[identity.EvidenceUID] = raw
	}
	resolveEvidence := func(uids []string) ([]json.RawMessage, error) {
		resolved := make([]json.RawMessage, 0, len(uids))
		for _, uid := range uids {
			raw, found := evidenceByUID[uid]
			if !found {
				return nil, fmt.Errorf(
					"ODD observation references missing evidence",
				)
			}
			resolved = append(resolved, raw)
		}
		return resolved, nil
	}
	supporting, err := resolveEvidence(observation.EvidenceUIDs)
	if err != nil {
		return ODDEvidenceResponse{}, manifest, digest, err
	}
	conflicting, err := resolveEvidence(observation.ConflictingEvidenceUIDs)
	if err != nil {
		return ODDEvidenceResponse{}, manifest, digest, err
	}

	events := make([]json.RawMessage, 0)
	for _, raw := range record.Events {
		var event struct {
			EventUID        string   `json:"event_uid"`
			ObservationUIDs []string `json:"observation_uids"`
		}
		if err := json.Unmarshal(raw, &event); err != nil ||
			event.EventUID == "" {
			return ODDEvidenceResponse{}, manifest, digest, fmt.Errorf(
				"invalid event in ODD scene record",
			)
		}
		if event.EventUID == observation.EventUID ||
			containsODDValue(event.ObservationUIDs, observationUID) {
			events = append(events, raw)
		}
	}
	provenance := record.Provenance
	if len(provenance) == 0 {
		provenance = json.RawMessage(`{}`)
	}
	response := ODDEvidenceResponse{
		Dataset:             dataset,
		Version:             version,
		LabelSetID:          manifest.LabelSetID,
		SceneUID:            record.SceneUID,
		Observation:         selected,
		SupportingEvidence:  supporting,
		ConflictingEvidence: conflicting,
		RelatedEvents:       events,
		SceneProvenance:     provenance,
		ManifestSHA256:      digest,
	}
	return response, manifest, digest, nil
}
