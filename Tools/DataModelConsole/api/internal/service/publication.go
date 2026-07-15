package service

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
)

const (
	publicationSchema           = "v1"
	maxPublicationManifestBytes = 4 << 20
)

type publicationShardEntry struct {
	Name            string `json:"name"`
	Key             string `json:"key"`
	ByteSize        int64  `json:"byte_size"`
	ETag            string `json:"etag"`
	ContentIdentity string `json:"content_identity"`
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
	Shards       int `json:"shards"`
	ShardCount   int `json:"shard_count"`
	Episodes     int `json:"episodes"`
	NumViews     int `json:"num_views"`

	HasMap        bool `json:"has_map"`
	HasWorldModel bool `json:"has_world_model"`
	HasGPS        bool `json:"has_gps"`

	ShardEntries []publicationShardEntry  `json:"shard_entries"`
	Rig          publicationArtifact      `json:"rig"`
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
	previousName := ""
	for _, entry := range manifest.ShardEntries {
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
		if entry.ETag == "" || !isLowerHexDigest(entry.ContentIdentity) {
			return nil, fmt.Errorf(
				"published shard %q has invalid content identity",
				entry.Name,
			)
		}
		if previousName != "" && entry.Name <= previousName {
			return nil, fmt.Errorf(
				"published shard entries are duplicate or unsorted at %q",
				entry.Name,
			)
		}
		previousName = entry.Name
		manifest.ShardByName[entry.Name] = entry
	}

	expectedRigKey := fmt.Sprintf("%s/%s/rig/projection.json", dataset, version)
	if manifest.Rig.Key != expectedRigKey ||
		!isLowerHexDigest(manifest.Rig.SHA256) {
		return nil, fmt.Errorf("publication rig artifact is invalid")
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
		name != ".tar" &&
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
