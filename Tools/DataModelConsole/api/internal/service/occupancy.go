package service

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/url"
	"path"
	"sort"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

const (
	occupancySetSchema           = "semantic_occupancy_set_v2"
	occupancySetPrefix           = "semantic-occupancy-sets/schema=v2"
	maxOccupancySetManifestBytes = 8 << 20
	maxOccupancyModels           = 128
)

var occupancyClassNames = map[string]struct{}{
	"drivable_area":        {},
	"lane_area":            {},
	"lane_boundary":        {},
	"intersection":         {},
	"crosswalk":            {},
	"stop_line":            {},
	"vehicle":              {},
	"vulnerable_road_user": {},
	"other_obstacle":       {},
}

type occupancySetModelSource struct {
	CodeLicenseSPDX         string `json:"code_license_spdx"`
	Config                  string `json:"config"`
	LicenseSPDX             string `json:"license_spdx"`
	Repository              string `json:"repository"`
	RepositoryRevision      string `json:"repository_revision"`
	TrainingDataLicenseSPDX string `json:"training_data_license_spdx"`
	WeightSHA256            string `json:"weight_sha256"`
	WeightSourceURL         string `json:"weight_source_url"`
}

type occupancySetShard struct {
	ByteSize       int64  `json:"byte_size"`
	S3Key          string `json:"s3_key"`
	SampleCount    int    `json:"sample_count"`
	SHA256         string `json:"sha256"`
	Shard          string `json:"shard"`
	TeacherPresent bool   `json:"teacher_present"`
}

type occupancySetManifest struct {
	SchemaVersion         string                  `json:"schema_version"`
	ArtifactKind          string                  `json:"artifact_kind"`
	ArtifactSchema        string                  `json:"artifact_schema"`
	CreatedAt             string                  `json:"created_at"`
	Dataset               string                  `json:"dataset"`
	DatasetVersion        string                  `json:"dataset_version"`
	DatasetManifestSHA256 string                  `json:"dataset_manifest_sha256"`
	DisplayName           string                  `json:"display_name"`
	GeometryID            string                  `json:"geometry_id"`
	HeadVersion           string                  `json:"head_version"`
	InputContract         string                  `json:"input_contract"`
	Limitations           []string                `json:"limitations"`
	ModelArtifactID       string                  `json:"model_artifact_id"`
	ModelFamily           string                  `json:"model_family"`
	ModelSource           occupancySetModelSource `json:"model_source"`
	ProducerConfig        map[string]any          `json:"producer_config"`
	SampleCount           int                     `json:"sample_count"`
	ShardCount            int                     `json:"shard_count"`
	Shards                []occupancySetShard     `json:"shards"`
	SupportedClasses      []string                `json:"supported_classes"`
	TaxonomyVersion       string                  `json:"taxonomy_version"`
	TeacherAvailable      bool                    `json:"teacher_available"`

	SHA256      string                       `json:"-"`
	ShardByName map[string]occupancySetShard `json:"-"`
}

func occupancySetDiscoveryPrefix(dataset, version string) string {
	return fmt.Sprintf(
		"%s/dataset=%s/version=%s/",
		occupancySetPrefix,
		dataset,
		version,
	)
}

func occupancySetManifestKey(
	dataset, version, modelArtifactID, datasetManifestSHA256 string,
) string {
	return fmt.Sprintf(
		"%smodel=%s/manifest=%s/manifest.json",
		occupancySetDiscoveryPrefix(dataset, version),
		modelArtifactID,
		datasetManifestSHA256,
	)
}

func validOccupancySegment(value string) bool {
	if value == "" {
		return false
	}
	for _, char := range value {
		if (char < 'a' || char > 'z') &&
			(char < 'A' || char > 'Z') &&
			(char < '0' || char > '9') &&
			char != '_' && char != '-' && char != '.' {
			return false
		}
	}
	return true
}

func validateTrimmedStrings(
	values []string,
	label string,
	allowed map[string]struct{},
) error {
	if len(values) == 0 {
		return fmt.Errorf("%s must not be empty", label)
	}
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if value == "" || strings.TrimSpace(value) != value {
			return fmt.Errorf("%s contains an empty or untrimmed value", label)
		}
		if _, duplicate := seen[value]; duplicate {
			return fmt.Errorf("%s contains duplicate %q", label, value)
		}
		if allowed != nil {
			if _, ok := allowed[value]; !ok {
				return fmt.Errorf("%s contains unsupported value %q", label, value)
			}
		}
		seen[value] = struct{}{}
	}
	return nil
}

func decodeOccupancySetManifest(
	body []byte,
	key, dataset, version, datasetManifestSHA256 string,
) (*occupancySetManifest, error) {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	var manifest occupancySetManifest
	if err := decoder.Decode(&manifest); err != nil {
		return nil, fmt.Errorf("decode occupancy set manifest: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return nil, fmt.Errorf("decode occupancy set manifest: %w", err)
	}
	if manifest.SchemaVersion != occupancySetSchema {
		return nil, fmt.Errorf(
			"unsupported occupancy set schema %q",
			manifest.SchemaVersion,
		)
	}
	if manifest.ArtifactKind != "native-semantic-occupancy" &&
		manifest.ArtifactKind != "detection-derived-occupancy" {
		return nil, fmt.Errorf(
			"unsupported occupancy artifact kind %q",
			manifest.ArtifactKind,
		)
	}
	if manifest.Dataset != dataset ||
		manifest.DatasetVersion != version ||
		manifest.DatasetManifestSHA256 != datasetManifestSHA256 {
		return nil, fmt.Errorf("occupancy set dataset publication differs")
	}
	if !isLowerHexDigest(manifest.ModelArtifactID) ||
		!isLowerHexDigest(manifest.ModelSource.WeightSHA256) ||
		key != occupancySetManifestKey(
			dataset,
			version,
			manifest.ModelArtifactID,
			datasetManifestSHA256,
		) {
		return nil, fmt.Errorf("occupancy set model identity is invalid")
	}
	if manifest.ArtifactSchema != "v1" ||
		!validOccupancySegment(manifest.GeometryID) ||
		!validOccupancySegment(manifest.HeadVersion) ||
		!validOccupancySegment(manifest.TaxonomyVersion) ||
		!validOccupancySegment(manifest.InputContract) {
		return nil, fmt.Errorf("occupancy set rendering contract is invalid")
	}
	if _, err := time.Parse(time.RFC3339, manifest.CreatedAt); err != nil {
		return nil, fmt.Errorf("occupancy set created_at is invalid: %w", err)
	}
	for label, value := range map[string]string{
		"display_name":                            manifest.DisplayName,
		"model_family":                            manifest.ModelFamily,
		"model_source.code_license_spdx":          manifest.ModelSource.CodeLicenseSPDX,
		"model_source.config":                     manifest.ModelSource.Config,
		"model_source.license_spdx":               manifest.ModelSource.LicenseSPDX,
		"model_source.repository":                 manifest.ModelSource.Repository,
		"model_source.revision":                   manifest.ModelSource.RepositoryRevision,
		"model_source.training_data_license_spdx": manifest.ModelSource.TrainingDataLicenseSPDX,
		"model_source.weight_source_url":          manifest.ModelSource.WeightSourceURL,
	} {
		if value == "" || strings.TrimSpace(value) != value {
			return nil, fmt.Errorf("%s must be a non-empty trimmed string", label)
		}
	}
	repository, err := url.Parse(manifest.ModelSource.Repository)
	if err != nil || repository.Scheme != "https" || repository.Host == "" {
		return nil, fmt.Errorf("occupancy set repository must be an HTTPS URL")
	}
	weightSource, err := url.Parse(manifest.ModelSource.WeightSourceURL)
	if err != nil ||
		(weightSource.Scheme != "https" || weightSource.Host == "") &&
			(weightSource.Scheme != "urn" ||
				weightSource.Opaque !=
					"sha256:"+manifest.ModelSource.WeightSHA256) {
		return nil, fmt.Errorf(
			"occupancy set weight source must be HTTPS or its SHA-256 URN",
		)
	}
	if len(manifest.ProducerConfig) == 0 {
		return nil, fmt.Errorf("occupancy set producer_config must not be empty")
	}
	for key := range manifest.ProducerConfig {
		if key == "" || strings.TrimSpace(key) != key {
			return nil, fmt.Errorf(
				"occupancy set producer_config has an invalid key",
			)
		}
	}
	if err := validateTrimmedStrings(
		manifest.SupportedClasses,
		"supported_classes",
		occupancyClassNames,
	); err != nil {
		return nil, err
	}
	if err := validateTrimmedStrings(
		manifest.Limitations,
		"limitations",
		nil,
	); err != nil {
		return nil, err
	}
	if manifest.SampleCount <= 0 ||
		manifest.ShardCount <= 0 ||
		manifest.ShardCount != len(manifest.Shards) {
		return nil, fmt.Errorf("occupancy set counts are invalid")
	}

	manifest.ShardByName = make(
		map[string]occupancySetShard,
		len(manifest.Shards),
	)
	totalSamples := 0
	previousShard := ""
	for _, entry := range manifest.Shards {
		if !validPublishedShardName(entry.Shard) ||
			entry.ByteSize <= 0 ||
			entry.ByteSize > MaxSemanticOccupancyBytes ||
			entry.SampleCount <= 0 ||
			!isLowerHexDigest(entry.SHA256) ||
			entry.TeacherPresent != manifest.TeacherAvailable {
			return nil, fmt.Errorf(
				"occupancy set shard %q is invalid",
				entry.Shard,
			)
		}
		if previousShard != "" && entry.Shard <= previousShard {
			return nil, fmt.Errorf(
				"occupancy set shards are duplicate or unsorted at %q",
				entry.Shard,
			)
		}
		expectedKey := fmt.Sprintf(
			"semantic-occupancy/schema=%s/model=%s/manifest=%s/"+
				"geometry=%s/taxonomy=%s/head=%s/dataset=%s/shard=%s/"+
				"occupancy.bin.gz",
			manifest.ArtifactSchema,
			manifest.ModelArtifactID,
			manifest.DatasetManifestSHA256,
			manifest.GeometryID,
			manifest.TaxonomyVersion,
			manifest.HeadVersion,
			manifest.Dataset,
			entry.Shard,
		)
		if entry.S3Key != expectedKey ||
			path.Base(entry.S3Key) != "occupancy.bin.gz" {
			return nil, fmt.Errorf(
				"occupancy set shard %q has a non-canonical body key",
				entry.Shard,
			)
		}
		previousShard = entry.Shard
		totalSamples += entry.SampleCount
		manifest.ShardByName[entry.Shard] = entry
	}
	if totalSamples != manifest.SampleCount {
		return nil, fmt.Errorf(
			"occupancy set sample count differs: manifest=%d shards=%d",
			manifest.SampleCount,
			totalSamples,
		)
	}
	digest := sha256.Sum256(body)
	manifest.SHA256 = hex.EncodeToString(digest[:])
	return &manifest, nil
}

func (s *S3Service) occupancyArtifactsBucket() string {
	if s.artifactsBucket != "" {
		return s.artifactsBucket
	}
	return s.bucket
}

func (s *S3Service) loadOccupancySetManifest(
	ctx context.Context,
	key, dataset, version, datasetManifestSHA256 string,
) (*occupancySetManifest, error) {
	body, err := s.getObjectBytesFromBucket(
		ctx,
		s.occupancyArtifactsBucket(),
		key,
		maxOccupancySetManifestBytes,
	)
	if err != nil {
		return nil, err
	}
	manifest, err := decodeOccupancySetManifest(
		body,
		key,
		dataset,
		version,
		datasetManifestSHA256,
	)
	if err != nil {
		return nil, err
	}
	head, err := s.client.HeadObject(ctx, &s3.HeadObjectInput{
		Bucket: aws.String(s.occupancyArtifactsBucket()),
		Key:    aws.String(key),
	})
	if err != nil {
		if isS3NotFound(err) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("head occupancy set manifest: %w", err)
	}
	if aws.ToInt64(head.ContentLength) != int64(len(body)) ||
		metadataValue(head.Metadata, "manifest-sha256") != manifest.SHA256 ||
		metadataValue(head.Metadata, "dataset-manifest-sha256") !=
			datasetManifestSHA256 ||
		metadataValue(head.Metadata, "schema") != occupancySetSchema {
		return nil, fmt.Errorf("occupancy set manifest object identity mismatch")
	}
	return manifest, nil
}

func occupancyModelForShard(
	manifest *occupancySetManifest,
	shard string,
) (model.SemanticOccupancyModel, bool) {
	entry, ok := manifest.ShardByName[shard]
	if !ok {
		return model.SemanticOccupancyModel{}, false
	}
	producerConfig := make(map[string]any, len(manifest.ProducerConfig))
	for key, value := range manifest.ProducerConfig {
		producerConfig[key] = value
	}
	return model.SemanticOccupancyModel{
		ModelArtifactID:       manifest.ModelArtifactID,
		DisplayName:           manifest.DisplayName,
		ModelFamily:           manifest.ModelFamily,
		ArtifactKind:          manifest.ArtifactKind,
		ArtifactSchema:        manifest.ArtifactSchema,
		CreatedAt:             manifest.CreatedAt,
		DatasetManifestSHA256: manifest.DatasetManifestSHA256,
		GeometryID:            manifest.GeometryID,
		TaxonomyVersion:       manifest.TaxonomyVersion,
		HeadVersion:           manifest.HeadVersion,
		InputContract:         manifest.InputContract,
		SupportedClasses:      append([]string(nil), manifest.SupportedClasses...),
		TeacherAvailable:      manifest.TeacherAvailable,
		Limitations:           append([]string(nil), manifest.Limitations...),
		ModelSource: model.SemanticOccupancyModelSource{
			CodeLicenseSPDX:         manifest.ModelSource.CodeLicenseSPDX,
			Config:                  manifest.ModelSource.Config,
			LicenseSPDX:             manifest.ModelSource.LicenseSPDX,
			Repository:              manifest.ModelSource.Repository,
			RepositoryRevision:      manifest.ModelSource.RepositoryRevision,
			TrainingDataLicenseSPDX: manifest.ModelSource.TrainingDataLicenseSPDX,
			WeightSHA256:            manifest.ModelSource.WeightSHA256,
			WeightSourceURL:         manifest.ModelSource.WeightSourceURL,
		},
		ProducerConfig:   producerConfig,
		SampleCount:      manifest.SampleCount,
		ShardCount:       manifest.ShardCount,
		ShardSampleCount: entry.SampleCount,
	}, true
}

// ListSemanticOccupancyModels discovers model-set manifests independently
// from trajectory overlay records and returns only models containing shard.
func (s *S3Service) ListSemanticOccupancyModels(
	ctx context.Context,
	dataset, version, shard string,
) ([]model.SemanticOccupancyModel, string, error) {
	var err error
	version, err = s.publishedVersion(ctx, dataset, version)
	if err != nil {
		return nil, "", err
	}
	if _, err := s.publishedShard(ctx, dataset, version, shard); err != nil {
		return nil, version, err
	}
	publication, err := s.loadPublicationManifest(ctx, dataset, version)
	if err != nil {
		return nil, version, err
	}

	prefix := occupancySetDiscoveryPrefix(dataset, version)
	paginator := s3.NewListObjectsV2Paginator(
		s.client,
		&s3.ListObjectsV2Input{
			Bucket: aws.String(s.occupancyArtifactsBucket()),
			Prefix: aws.String(prefix),
		},
	)
	keys := make([]string, 0)
	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, version, fmt.Errorf(
				"list semantic occupancy models: %w",
				err,
			)
		}
		for _, object := range page.Contents {
			key := aws.ToString(object.Key)
			if !strings.HasSuffix(
				key,
				"/manifest="+publication.SHA256+"/manifest.json",
			) {
				continue
			}
			keys = append(keys, key)
			if len(keys) > maxOccupancyModels {
				return nil, version, fmt.Errorf(
					"semantic occupancy model count exceeds %d",
					maxOccupancyModels,
				)
			}
		}
	}
	sort.Strings(keys)

	models := make([]model.SemanticOccupancyModel, 0, len(keys))
	for _, key := range keys {
		manifest, err := s.loadOccupancySetManifest(
			ctx,
			key,
			dataset,
			version,
			publication.SHA256,
		)
		if err != nil {
			return nil, version, fmt.Errorf(
				"validate semantic occupancy model %q: %w",
				key,
				err,
			)
		}
		if entry, ok := occupancyModelForShard(manifest, shard); ok {
			models = append(models, entry)
		}
	}
	sort.Slice(models, func(i, j int) bool {
		if models[i].CreatedAt != models[j].CreatedAt {
			return models[i].CreatedAt > models[j].CreatedAt
		}
		return models[i].ModelArtifactID < models[j].ModelArtifactID
	})
	return models, version, nil
}

// GetPublishedSemanticOccupancyBody verifies one independently published
// model-set manifest and its selected shard body.
func (s *S3Service) GetPublishedSemanticOccupancyBody(
	ctx context.Context,
	dataset, version, shard, modelArtifactID string,
) (*SemanticOccupancyBody, string, error) {
	var err error
	version, err = s.publishedVersion(ctx, dataset, version)
	if err != nil {
		return nil, "", err
	}
	if _, err := s.publishedShard(ctx, dataset, version, shard); err != nil {
		return nil, version, err
	}
	publication, err := s.loadPublicationManifest(ctx, dataset, version)
	if err != nil {
		return nil, version, err
	}
	manifest, err := s.loadOccupancySetManifest(
		ctx,
		occupancySetManifestKey(
			dataset,
			version,
			modelArtifactID,
			publication.SHA256,
		),
		dataset,
		version,
		publication.SHA256,
	)
	if err != nil {
		return nil, version, err
	}
	entry, ok := manifest.ShardByName[shard]
	if !ok {
		return nil, version, ErrNotFound
	}
	payload, err := s.getObjectBytesFromBucket(
		ctx,
		s.occupancyArtifactsBucket(),
		entry.S3Key,
		MaxSemanticOccupancyBytes,
	)
	if err != nil {
		return nil, version, err
	}
	if int64(len(payload)) != entry.ByteSize {
		return nil, version, fmt.Errorf(
			"semantic occupancy size mismatch: manifest=%d body=%d",
			entry.ByteSize,
			len(payload),
		)
	}
	digest := sha256.Sum256(payload)
	if hex.EncodeToString(digest[:]) != entry.SHA256 {
		return nil, version, fmt.Errorf("semantic occupancy SHA-256 mismatch")
	}
	if len(payload) < 2 || payload[0] != 0x1f || payload[1] != 0x8b {
		return nil, version, fmt.Errorf("semantic occupancy body is not gzip")
	}
	return &SemanticOccupancyBody{
		Descriptor: model.SemanticOccupancyDescriptor{
			ModelArtifactID: manifest.ModelArtifactID,
			Schema:          manifest.ArtifactSchema,
			GeometryID:      manifest.GeometryID,
			TaxonomyVersion: manifest.TaxonomyVersion,
			HeadVersion:     manifest.HeadVersion,
			SHA256:          entry.SHA256,
			ByteSize:        entry.ByteSize,
		},
		Payload: payload,
	}, version, nil
}
