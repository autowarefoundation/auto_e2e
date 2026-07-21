package service

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

const (
	publicationSchema           = "v2"
	maxPublicationManifestBytes = 4 << 20
	publicationHeadConcurrency  = 32
)

type publicationShardEntry struct {
	Name            string              `json:"name"`
	Key             string              `json:"key"`
	ByteSize        int64               `json:"byte_size"`
	ETag            string              `json:"etag"`
	ContentIdentity string              `json:"content_identity"`
	Rig             publicationArtifact `json:"rig"`
	LastModified    time.Time           `json:"-"`
}

type publicationArtifact struct {
	Key    string `json:"key"`
	SHA256 string `json:"sha256"`
}

type publicationGeoArtifacts struct {
	SummaryKey    string  `json:"summary_key"`
	HeatmapKey    string  `json:"heatmap_key"`
	SamplePoseKey *string `json:"sample_pose_key"`
	HeatmapSHA256 string  `json:"heatmap_sha256"`
}

// publicationManifest is the immutable v2.1+ readiness gate written last by
// finalize_dataset_publication. shard_entries is the authoritative allowlist;
// S3 prefix contents are never treated as the publication inventory.
type publicationManifest struct {
	SchemaVersion string `json:"schema_version"`
	Status        string `json:"status"`
	Dataset       string `json:"dataset"`
	Version       string `json:"version"`

	TotalSamples int `json:"total_samples"`
	// ReasoningLabelCount includes successful and explicit-abstention records.
	ReasoningLabelCount int `json:"reasoning_label_count"`
	Shards              int `json:"shards"`
	ShardCount          int `json:"shard_count"`
	RigCount            int `json:"rig_count"`
	Episodes            int `json:"episodes"`
	NumViews            int `json:"num_views"`

	HasMap        bool `json:"has_map"`
	HasWorldModel bool `json:"has_world_model"`
	HasReasoning  bool `json:"has_reasoning_labels"`
	HasGPS        bool `json:"has_gps"`

	ShardEntries []publicationShardEntry  `json:"shard_entries"`
	GeoArtifacts *publicationGeoArtifacts `json:"geo_artifacts"`

	SHA256      string                           `json:"-"`
	ShardByName map[string]publicationShardEntry `json:"-"`
}

func decodePublicationManifest(
	body []byte,
	dataset, version string,
) (*publicationManifest, error) {
	decoder := json.NewDecoder(bytes.NewReader(body))
	var manifest publicationManifest
	if err := decoder.Decode(&manifest); err != nil {
		return nil, fmt.Errorf("decode publication manifest: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return nil, err
	}
	if manifest.SchemaVersion != publicationSchema {
		return nil, fmt.Errorf(
			"unsupported publication schema %q", manifest.SchemaVersion,
		)
	}
	if manifest.Status != "ready" {
		return nil, fmt.Errorf(
			"publication status is %q, want ready", manifest.Status,
		)
	}
	if manifest.Dataset != dataset || manifest.Version != version {
		return nil, fmt.Errorf(
			"publication coordinate is %s/%s, want %s/%s",
			manifest.Dataset, manifest.Version, dataset, version,
		)
	}
	if manifest.TotalSamples <= 0 {
		return nil, fmt.Errorf("publication has no samples")
	}
	if manifest.ReasoningLabelCount < 0 ||
		manifest.ReasoningLabelCount > manifest.TotalSamples ||
		manifest.HasReasoning != (manifest.ReasoningLabelCount > 0) {
		return nil, fmt.Errorf(
			"publication reasoning counts disagree: has_reasoning=%v count=%d samples=%d",
			manifest.HasReasoning,
			manifest.ReasoningLabelCount,
			manifest.TotalSamples,
		)
	}
	if manifest.Shards <= 0 ||
		manifest.Shards != manifest.ShardCount ||
		manifest.ShardCount != len(manifest.ShardEntries) {
		return nil, fmt.Errorf(
			"publication shard counts disagree: shards=%d shard_count=%d entries=%d",
			manifest.Shards, manifest.ShardCount, len(manifest.ShardEntries),
		)
	}

	manifest.ShardByName = make(
		map[string]publicationShardEntry, len(manifest.ShardEntries),
	)
	rigs := make(map[string]struct{})
	previousName := ""
	for i := range manifest.ShardEntries {
		entry := &manifest.ShardEntries[i]
		if !validPublishedShardName(entry.Name) {
			return nil, fmt.Errorf("invalid published shard name %q", entry.Name)
		}
		expectedKey := shardsPrefix(dataset, version) + entry.Name
		if entry.Key != expectedKey {
			return nil, fmt.Errorf(
				"published shard %q has non-canonical key %q",
				entry.Name, entry.Key,
			)
		}
		if entry.ByteSize <= 0 {
			return nil, fmt.Errorf(
				"published shard %q has invalid size %d",
				entry.Name, entry.ByteSize,
			)
		}
		etag, ok := canonicalS3ETag(entry.ETag)
		if !ok || !isLowerHexDigest(entry.ContentIdentity) {
			return nil, fmt.Errorf(
				"published shard %q has invalid content identity",
				entry.Name,
			)
		}
		entry.ETag = etag
		expectedRigKey := fmt.Sprintf(
			"%s/%s/rig/%s.json",
			dataset,
			version,
			entry.Rig.SHA256,
		)
		if !isLowerHexDigest(entry.Rig.SHA256) ||
			entry.Rig.Key != expectedRigKey {
			return nil, fmt.Errorf(
				"published shard %q has invalid rig artifact",
				entry.Name,
			)
		}
		rigs[entry.Rig.Key] = struct{}{}
		if previousName != "" && entry.Name <= previousName {
			return nil, fmt.Errorf(
				"published shard entries are duplicate or unsorted at %q",
				entry.Name,
			)
		}
		previousName = entry.Name
		manifest.ShardByName[entry.Name] = *entry
	}
	if manifest.RigCount <= 0 || manifest.RigCount != len(rigs) {
		return nil, fmt.Errorf(
			"publication rig count disagrees: rig_count=%d entries=%d",
			manifest.RigCount,
			len(rigs),
		)
	}
	if manifest.HasGPS {
		if err := validateGeoArtifacts(
			manifest.GeoArtifacts, dataset, version,
		); err != nil {
			return nil, err
		}
	} else if manifest.GeoArtifacts != nil {
		return nil, fmt.Errorf("publication without GPS has geo artifacts")
	}

	digest := sha256.Sum256(body)
	manifest.SHA256 = hex.EncodeToString(digest[:])
	return &manifest, nil
}

func ensureJSONEOF(decoder *json.Decoder) error {
	var extra json.RawMessage
	err := decoder.Decode(&extra)
	if err == io.EOF {
		return nil
	}
	if err != nil {
		return fmt.Errorf("decode publication manifest trailing data: %w", err)
	}
	return fmt.Errorf("publication manifest contains multiple JSON values")
}

func validateGeoArtifacts(
	artifacts *publicationGeoArtifacts,
	dataset, version string,
) error {
	if artifacts == nil {
		return fmt.Errorf("GPS publication has no geo artifacts")
	}
	prefix := fmt.Sprintf("%s/%s/geo/", dataset, version)
	if artifacts.SummaryKey != prefix+"summary.json" ||
		artifacts.HeatmapKey != prefix+"heatmap.geojson.gz" ||
		!isLowerHexDigest(artifacts.HeatmapSHA256) {
		return fmt.Errorf("publication geo artifacts are invalid")
	}
	if artifacts.SamplePoseKey != nil &&
		*artifacts.SamplePoseKey != prefix+"sample_pose.parquet" {
		return fmt.Errorf("publication sample pose artifact is invalid")
	}
	return nil
}

func validPublishedShardName(name string) bool {
	return len(name) > len(".tar") &&
		name[len(name)-len(".tar"):] == ".tar" &&
		name != ".tar" && name != "..tar" &&
		!bytes.ContainsAny([]byte(name), `/\`)
}

func isLowerHexDigest(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	for _, char := range value {
		if (char < '0' || char > '9') && (char < 'a' || char > 'f') {
			return false
		}
	}
	return true
}

func (s *S3Service) loadPublicationManifest(
	ctx context.Context,
	dataset, version string,
) (*publicationManifest, error) {
	cacheKey := dataset + "/" + version
	s.publicationMu.Lock()
	defer s.publicationMu.Unlock()
	if manifest := s.publicationCache[cacheKey]; manifest != nil {
		return manifest, nil
	}

	manifestKey := shardsPrefix(dataset, version) + "manifest.json"
	body, err := s.getObjectBytesFromBucket(
		ctx, s.bucket, manifestKey, maxPublicationManifestBytes,
	)
	if err != nil {
		return nil, err
	}
	manifest, err := decodePublicationManifest(body, dataset, version)
	if err != nil {
		return nil, err
	}
	head, err := s.client.HeadObject(ctx, &s3.HeadObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(manifestKey),
	})
	if err != nil {
		if isS3NotFound(err) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("head publication manifest: %w", err)
	}
	if aws.ToInt64(head.ContentLength) != int64(len(body)) ||
		metadataValue(head.Metadata, "sha256") != manifest.SHA256 ||
		metadataValue(head.Metadata, "publication-schema") != publicationSchema {
		return nil, fmt.Errorf("publication manifest object identity mismatch")
	}
	if err := s.validatePublicationShards(ctx, manifest); err != nil {
		return nil, err
	}
	if s.publicationCache == nil {
		s.publicationCache = make(map[string]*publicationManifest)
	}
	s.publicationCache[cacheKey] = manifest
	return manifest, nil
}

func (s *S3Service) validatePublicationShards(
	ctx context.Context,
	manifest *publicationManifest,
) error {
	objects := make(map[string]struct {
		size         int64
		lastModified time.Time
	})
	paginator := s3.NewListObjectsV2Paginator(
		s.client,
		&s3.ListObjectsV2Input{
			Bucket: aws.String(s.bucket),
			Prefix: aws.String(shardsPrefix(manifest.Dataset, manifest.Version)),
		},
	)
	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return fmt.Errorf("list published shard inventory: %w", err)
		}
		for _, object := range page.Contents {
			key := aws.ToString(object.Key)
			if !strings.HasSuffix(key, ".tar") {
				continue
			}
			if _, exists := objects[key]; exists {
				return fmt.Errorf("duplicate S3 shard object %q", key)
			}
			objects[key] = struct {
				size         int64
				lastModified time.Time
			}{
				size:         aws.ToInt64(object.Size),
				lastModified: aws.ToTime(object.LastModified),
			}
		}
	}
	if len(objects) != len(manifest.ShardEntries) {
		return fmt.Errorf(
			"published shard inventory count mismatch: manifest=%d s3=%d",
			len(manifest.ShardEntries), len(objects),
		)
	}
	for i := range manifest.ShardEntries {
		entry := &manifest.ShardEntries[i]
		object, ok := objects[entry.Key]
		if !ok {
			return fmt.Errorf("published shard %q is missing", entry.Name)
		}
		if object.size != entry.ByteSize {
			return fmt.Errorf(
				"published shard %q size mismatch: manifest=%d s3=%d",
				entry.Name, entry.ByteSize, object.size,
			)
		}
		entry.LastModified = object.lastModified
	}

	validationCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	results := make([]error, len(manifest.ShardEntries))
	sem := make(chan struct{}, publicationHeadConcurrency)
	var wg sync.WaitGroup
	for i := range manifest.ShardEntries {
		select {
		case sem <- struct{}{}:
		case <-validationCtx.Done():
			results[i] = validationCtx.Err()
			continue
		}
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			defer func() { <-sem }()
			entry := manifest.ShardEntries[i]
			head, err := s.client.HeadObject(
				validationCtx,
				&s3.HeadObjectInput{
					Bucket: aws.String(s.bucket),
					Key:    aws.String(entry.Key),
				},
			)
			if err != nil {
				results[i] = fmt.Errorf(
					"head published shard %q: %w", entry.Name, err,
				)
				cancel()
				return
			}
			if aws.ToInt64(head.ContentLength) != entry.ByteSize ||
				!sameS3ETag(aws.ToString(head.ETag), entry.ETag) ||
				metadataValue(
					head.Metadata, "source-identity",
				) != entry.ContentIdentity {
				results[i] = fmt.Errorf(
					"published shard %q object identity mismatch", entry.Name,
				)
				cancel()
			}
		}(i)
	}
	wg.Wait()
	for _, err := range results {
		if err != nil {
			return err
		}
	}
	for _, entry := range manifest.ShardEntries {
		manifest.ShardByName[entry.Name] = entry
	}
	rigs := make(map[string]struct{}, manifest.RigCount)
	for _, entry := range manifest.ShardEntries {
		if _, ok := rigs[entry.Rig.Key]; ok {
			continue
		}
		head, err := s.client.HeadObject(
			ctx,
			&s3.HeadObjectInput{
				Bucket: aws.String(s.bucket),
				Key:    aws.String(entry.Rig.Key),
			},
		)
		if err != nil {
			return fmt.Errorf(
				"head published rig %q: %w", entry.Rig.Key, err,
			)
		}
		if aws.ToInt64(head.ContentLength) <= 0 ||
			metadataValue(head.Metadata, "sha256") != entry.Rig.SHA256 ||
			metadataValue(
				head.Metadata, "publication-schema",
			) != publicationSchema {
			return fmt.Errorf(
				"published rig %q object identity mismatch",
				entry.Rig.Key,
			)
		}
		rigs[entry.Rig.Key] = struct{}{}
	}
	return nil
}

func (s *S3Service) publishedVersion(
	ctx context.Context,
	dataset, requested string,
) (string, error) {
	if isVersionDir(requested) {
		if requiresPublicationManifest(requested) {
			if _, err := s.loadPublicationManifest(
				ctx, dataset, requested,
			); err != nil {
				return "", err
			}
		} else if !s.versionHasShards(ctx, dataset, requested) {
			return "", ErrNotFound
		}
		return requested, nil
	}
	version := s.resolveVersion(ctx, dataset)
	if requiresPublicationManifest(version) {
		if _, err := s.loadPublicationManifest(
			ctx, dataset, version,
		); err != nil {
			return "", err
		}
	}
	return version, nil
}

func (s *S3Service) publishedShard(
	ctx context.Context,
	dataset, version, shard string,
) (publicationShardEntry, error) {
	manifest, err := s.loadPublicationManifest(ctx, dataset, version)
	if err != nil {
		return publicationShardEntry{}, err
	}
	entry, ok := manifest.ShardByName[shard]
	if !ok {
		return publicationShardEntry{}, ErrNotFound
	}
	return entry, nil
}

func (s *S3Service) publishedShardKey(
	ctx context.Context,
	dataset, requestedVersion, shard string,
) (string, string, int64, error) {
	version, key, size, _, err := s.publishedShardObject(
		ctx, dataset, requestedVersion, shard,
	)
	return version, key, size, err
}

func (s *S3Service) publishedShardObject(
	ctx context.Context,
	dataset, requestedVersion, shard string,
) (string, string, int64, string, error) {
	version, err := s.publishedVersion(ctx, dataset, requestedVersion)
	if err != nil {
		return "", "", 0, "", err
	}
	if requiresPublicationManifest(version) {
		entry, err := s.publishedShard(ctx, dataset, version, shard)
		if err != nil {
			return "", "", 0, "", err
		}
		return version, entry.Key, entry.ByteSize, entry.ETag, nil
	}
	return version, shardsPrefix(dataset, version) + shard, 0, "", nil
}

func metadataValue(metadata map[string]string, key string) string {
	for candidate, value := range metadata {
		if strings.EqualFold(candidate, key) {
			return value
		}
	}
	return ""
}

func canonicalS3ETag(value string) (string, bool) {
	value = strings.TrimSpace(value)
	if len(value) >= 2 && value[0] == '"' && value[len(value)-1] == '"' {
		value = value[1 : len(value)-1]
	}
	if value == "" {
		return "", false
	}
	for _, char := range value {
		if char == '"' || char < 0x20 || char == 0x7f {
			return "", false
		}
	}
	return `"` + value + `"`, true
}

func sameS3ETag(left, right string) bool {
	leftCanonical, leftOK := canonicalS3ETag(left)
	rightCanonical, rightOK := canonicalS3ETag(right)
	return leftOK && rightOK && leftCanonical == rightCanonical
}
