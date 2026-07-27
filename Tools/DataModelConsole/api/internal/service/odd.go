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

type ODDArtifact struct {
	Key      string `json:"key"`
	SHA256   string `json:"sha256"`
	ByteSize int64  `json:"byte_size"`
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
	SceneCount            int                    `json:"scene_count"`
	OpenAICompatible      map[string]string      `json:"openai_compatible"`
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
}

type ODDSceneSummary struct {
	SceneUID         string                       `json:"scene_uid"`
	ShardName        string                       `json:"shard_name"`
	RecordKey        string                       `json:"record_key"`
	StartTimestampNS int64                        `json:"start_timestamp_ns"`
	EndTimestampNS   int64                        `json:"end_timestamp_ns"`
	DistanceM        float64                      `json:"distance_m"`
	Observations     []ODDSceneObservationSummary `json:"observations"`
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
	body, err := s.oddObject(ctx, pointerKey, "")
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
	body, err = s.oddObject(ctx, pointer.ManifestKey, pointer.ManifestSHA256)
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
	body, err := s.oddObject(ctx, artifact.Key, artifact.SHA256)
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
	return json.RawMessage(body), manifest, digest, err
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
	if index.SchemaVersion != "odd_scene_index_v1" ||
		index.LabelSetID != manifest.LabelSetID ||
		len(index.Scenes) != manifest.SceneCount {
		return oddSceneIndex{}, manifest, digest, fmt.Errorf(
			"ODD scene index differs from manifest",
		)
	}
	return index, manifest, digest, nil
}

// SearchODDScenes searches scene summaries, never overlapping training samples.
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
	index, manifest, manifestSHA, err := s.oddScenes(ctx, dataset, version)
	if err != nil {
		return ODDSearchResponse{}, err
	}
	matches := make([]ODDSceneSummary, 0)
	for _, scene := range index.Scenes {
		matched := false
		for _, observation := range scene.Observations {
			if key != "" && observation.Key != key {
				continue
			}
			if status != "" && observation.Status != status {
				continue
			}
			if source != "" && observation.Source != source {
				continue
			}
			if value != "" {
				found := false
				for _, candidate := range observation.Values {
					if candidate == value {
						found = true
						break
					}
				}
				if !found {
					continue
				}
			}
			matched = true
			break
		}
		if matched {
			matches = append(matches, scene)
		}
	}
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].SceneUID < matches[j].SceneUID
	})
	total := len(matches)
	if offset > total {
		offset = total
	}
	end := min(total, offset+limit)
	return ODDSearchResponse{
		Dataset:        dataset,
		Version:        version,
		LabelSetID:     manifest.LabelSetID,
		Scenes:         matches[offset:end],
		Total:          total,
		Limit:          limit,
		Offset:         offset,
		More:           end < total,
		ManifestSHA256: manifestSHA,
	}, nil
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
		root := fmt.Sprintf(
			"%s/%s/odd/labelsets/%s/scenes/",
			dataset, version, manifest.LabelSetID,
		)
		if !strings.HasPrefix(scene.RecordKey, root) {
			return nil, manifest, digest, fmt.Errorf(
				"ODD scene record key escapes LabelSet",
			)
		}
		body, err := s.oddObject(ctx, scene.RecordKey, "")
		return json.RawMessage(body), manifest, digest, err
	}
	return nil, manifest, digest, ErrNotFound
}
