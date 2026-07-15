package service

import (
	"encoding/json"
	"strings"
	"testing"
)

func validPublicationValue() map[string]any {
	digest := strings.Repeat("a", 64)
	return map[string]any{
		"schema_version":  "v1",
		"status":          "ready",
		"dataset":         "kitscenes",
		"version":         "v2.1",
		"total_samples":   20,
		"shards":          2,
		"shard_count":     2,
		"episodes":        2,
		"num_views":       7,
		"has_map":         true,
		"has_world_model": false,
		"has_gps":         true,
		"shard_entries": []any{
			map[string]any{
				"name":             "scene-a-train-000000.tar",
				"key":              "kitscenes/v2.1/shards/scene-a-train-000000.tar",
				"byte_size":        123,
				"etag":             "etag-a",
				"content_identity": digest,
			},
			map[string]any{
				"name":             "scene-b-train-000000.tar",
				"key":              "kitscenes/v2.1/shards/scene-b-train-000000.tar",
				"byte_size":        456,
				"etag":             "etag-b",
				"content_identity": strings.Repeat("b", 64),
			},
		},
		"rig": map[string]any{
			"key":    "kitscenes/v2.1/rig/projection.json",
			"sha256": digest,
		},
		"geo_artifacts": map[string]any{
			"summary_key":     "kitscenes/v2.1/geo/summary.json",
			"heatmap_key":     "kitscenes/v2.1/geo/heatmap.geojson.gz",
			"sample_pose_key": "kitscenes/v2.1/geo/sample_pose.parquet",
			"heatmap_sha256":  strings.Repeat("c", 64),
		},
	}
}

func encodePublication(t *testing.T, value map[string]any) []byte {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func TestDecodePublicationManifestAcceptsCanonicalInventory(t *testing.T) {
	body := encodePublication(t, validPublicationValue())
	manifest, err := decodePublicationManifest(
		body, "kitscenes", "v2.1",
	)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.TotalSamples != 20 || len(manifest.ShardByName) != 2 {
		t.Fatalf("manifest = %+v", manifest)
	}
	if manifest.ShardByName["scene-b-train-000000.tar"].ByteSize != 456 {
		t.Fatal("shard allowlist was not indexed")
	}
	if !isLowerHexDigest(manifest.SHA256) {
		t.Fatalf("manifest digest = %q", manifest.SHA256)
	}
}

func TestDecodePublicationManifestRejectsInvalidGate(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(map[string]any)
	}{
		{
			name: "unsupported schema",
			mutate: func(value map[string]any) {
				value["schema_version"] = "v2"
			},
		},
		{
			name: "not ready",
			mutate: func(value map[string]any) {
				value["status"] = "building"
			},
		},
		{
			name: "wrong dataset",
			mutate: func(value map[string]any) {
				value["dataset"] = "l2d"
			},
		},
		{
			name: "wrong version",
			mutate: func(value map[string]any) {
				value["version"] = "v2.2"
			},
		},
		{
			name: "empty publication",
			mutate: func(value map[string]any) {
				value["total_samples"] = 0
			},
		},
		{
			name: "shard count mismatch",
			mutate: func(value map[string]any) {
				value["shard_count"] = 3
			},
		},
		{
			name: "non-canonical shard key",
			mutate: func(value map[string]any) {
				entries := value["shard_entries"].([]any)
				entries[0].(map[string]any)["key"] = "other/v2.1/shards/a.tar"
			},
		},
		{
			name: "duplicate shard",
			mutate: func(value map[string]any) {
				entries := value["shard_entries"].([]any)
				entries[1] = entries[0]
			},
		},
		{
			name: "unsorted shards",
			mutate: func(value map[string]any) {
				entries := value["shard_entries"].([]any)
				entries[0], entries[1] = entries[1], entries[0]
			},
		},
		{
			name: "zero shard size",
			mutate: func(value map[string]any) {
				entries := value["shard_entries"].([]any)
				entries[0].(map[string]any)["byte_size"] = 0
			},
		},
		{
			name: "invalid shard identity",
			mutate: func(value map[string]any) {
				entries := value["shard_entries"].([]any)
				entries[0].(map[string]any)["content_identity"] = "not-a-digest"
			},
		},
		{
			name: "invalid rig",
			mutate: func(value map[string]any) {
				value["rig"].(map[string]any)["key"] = "kitscenes/v2.1/rig/other.json"
			},
		},
		{
			name: "missing GPS artifacts",
			mutate: func(value map[string]any) {
				delete(value, "geo_artifacts")
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			value := validPublicationValue()
			test.mutate(value)
			if _, err := decodePublicationManifest(
				encodePublication(t, value), "kitscenes", "v2.1",
			); err == nil {
				t.Fatal("invalid publication manifest was accepted")
			}
		})
	}
}

func TestDecodePublicationManifestRejectsMalformedOrTrailingJSON(t *testing.T) {
	for _, body := range [][]byte{
		[]byte(`{"schema_version":`),
		append(encodePublication(t, validPublicationValue()), []byte(` {}`)...),
	} {
		if _, err := decodePublicationManifest(
			body, "kitscenes", "v2.1",
		); err == nil {
			t.Fatal("malformed publication manifest was accepted")
		}
	}
}

func TestValidPublishedShardNameMatchesWriterContract(t *testing.T) {
	for _, name := range []string{
		"train-000000.tar",
		"scene-a-train-000000.tar",
	} {
		if !validPublishedShardName(name) {
			t.Errorf("valid shard %q was rejected", name)
		}
	}
	for _, name := range []string{
		"",
		".tar",
		"..tar",
		"train.tar.gz",
		"nested/train.tar",
		`nested\train.tar`,
	} {
		if validPublishedShardName(name) {
			t.Errorf("invalid shard %q was accepted", name)
		}
	}
}
