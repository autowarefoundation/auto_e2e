package service

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	s3types "github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/aws/smithy-go"
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

type fakePublicationObject struct {
	body         []byte
	metadata     map[string]string
	lastModified time.Time
}

type fakePublicationS3 struct {
	mu        sync.Mutex
	objects   map[string]fakePublicationObject
	getCalls  map[string]int
	headCalls map[string]int
}

func (f *fakePublicationS3) GetObject(
	_ context.Context,
	input *s3.GetObjectInput,
	_ ...func(*s3.Options),
) (*s3.GetObjectOutput, error) {
	key := aws.ToString(input.Key)
	f.mu.Lock()
	f.getCalls[key]++
	object, ok := f.objects[key]
	f.mu.Unlock()
	if !ok {
		return nil, &smithy.GenericAPIError{Code: "NoSuchKey"}
	}
	return &s3.GetObjectOutput{
		Body:          io.NopCloser(bytes.NewReader(object.body)),
		ContentLength: aws.Int64(int64(len(object.body))),
		Metadata:      object.metadata,
	}, nil
}

func (f *fakePublicationS3) HeadBucket(
	context.Context,
	*s3.HeadBucketInput,
	...func(*s3.Options),
) (*s3.HeadBucketOutput, error) {
	return &s3.HeadBucketOutput{}, nil
}

func (f *fakePublicationS3) HeadObject(
	_ context.Context,
	input *s3.HeadObjectInput,
	_ ...func(*s3.Options),
) (*s3.HeadObjectOutput, error) {
	key := aws.ToString(input.Key)
	f.mu.Lock()
	f.headCalls[key]++
	object, ok := f.objects[key]
	f.mu.Unlock()
	if !ok {
		return nil, &smithy.GenericAPIError{Code: "NotFound"}
	}
	return &s3.HeadObjectOutput{
		ContentLength: aws.Int64(int64(len(object.body))),
		LastModified:  aws.Time(object.lastModified),
		Metadata:      object.metadata,
	}, nil
}

func (f *fakePublicationS3) ListObjectsV2(
	_ context.Context,
	input *s3.ListObjectsV2Input,
	_ ...func(*s3.Options),
) (*s3.ListObjectsV2Output, error) {
	prefix := aws.ToString(input.Prefix)
	f.mu.Lock()
	defer f.mu.Unlock()
	objects := make([]s3types.Object, 0)
	for key, object := range f.objects {
		if !strings.HasPrefix(key, prefix) {
			continue
		}
		objects = append(objects, s3types.Object{
			Key:          aws.String(key),
			Size:         aws.Int64(int64(len(object.body))),
			LastModified: aws.Time(object.lastModified),
		})
	}
	return &s3.ListObjectsV2Output{
		Contents:    objects,
		IsTruncated: aws.Bool(false),
		KeyCount:    aws.Int32(int32(len(objects))),
	}, nil
}

func newPublicationTestService(
	t *testing.T,
) (*S3Service, *fakePublicationS3) {
	t.Helper()
	body := encodePublication(t, validPublicationValue())
	manifest, err := decodePublicationManifest(
		body, "kitscenes", "v2.1",
	)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)
	objects := map[string]fakePublicationObject{
		"kitscenes/v2.1/shards/manifest.json": {
			body: body,
			metadata: map[string]string{
				"sha256":             manifest.SHA256,
				"publication-schema": publicationSchema,
			},
			lastModified: now,
		},
	}
	for _, entry := range manifest.ShardEntries {
		objects[entry.Key] = fakePublicationObject{
			body: bytes.Repeat([]byte{1}, int(entry.ByteSize)),
			metadata: map[string]string{
				"source-identity": entry.ContentIdentity,
			},
			lastModified: now,
		}
	}
	client := &fakePublicationS3{
		objects:   objects,
		getCalls:  map[string]int{},
		headCalls: map[string]int{},
	}
	service := &S3Service{
		client:           client,
		bucket:           "datasets",
		versionCache:     map[string]cachedVersion{},
		publicationCache: map[string]*publicationManifest{},
		indexSF:          map[string]*sync.WaitGroup{},
	}
	return service, client
}

func TestLoadPublicationManifestValidatesAndCachesInventory(t *testing.T) {
	service, client := newPublicationTestService(t)
	first, err := service.loadPublicationManifest(
		context.Background(), "kitscenes", "v2.1",
	)
	if err != nil {
		t.Fatal(err)
	}
	second, err := service.loadPublicationManifest(
		context.Background(), "kitscenes", "v2.1",
	)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatal("validated immutable manifest was not cached")
	}
	manifestKey := "kitscenes/v2.1/shards/manifest.json"
	if client.getCalls[manifestKey] != 1 {
		t.Fatalf("manifest GET calls = %d, want 1", client.getCalls[manifestKey])
	}
	for _, entry := range first.ShardEntries {
		if client.headCalls[entry.Key] != 1 {
			t.Fatalf(
				"shard %s HEAD calls = %d, want 1",
				entry.Name, client.headCalls[entry.Key],
			)
		}
		if entry.LastModified.IsZero() {
			t.Fatalf("shard %s lost LastModified", entry.Name)
		}
	}
}

func TestLoadPublicationManifestRejectsS3InventoryMismatch(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*fakePublicationS3)
	}{
		{
			name: "orphan shard",
			mutate: func(client *fakePublicationS3) {
				client.objects["kitscenes/v2.1/shards/orphan.tar"] =
					fakePublicationObject{
						body:     []byte("orphan"),
						metadata: map[string]string{"source-identity": strings.Repeat("d", 64)},
					}
			},
		},
		{
			name: "missing shard",
			mutate: func(client *fakePublicationS3) {
				delete(
					client.objects,
					"kitscenes/v2.1/shards/scene-a-train-000000.tar",
				)
			},
		},
		{
			name: "wrong shard size",
			mutate: func(client *fakePublicationS3) {
				key := "kitscenes/v2.1/shards/scene-a-train-000000.tar"
				object := client.objects[key]
				object.body = append(object.body, 0)
				client.objects[key] = object
			},
		},
		{
			name: "wrong source identity",
			mutate: func(client *fakePublicationS3) {
				key := "kitscenes/v2.1/shards/scene-a-train-000000.tar"
				object := client.objects[key]
				object.metadata["source-identity"] = strings.Repeat("f", 64)
				client.objects[key] = object
			},
		},
		{
			name: "wrong manifest digest",
			mutate: func(client *fakePublicationS3) {
				key := "kitscenes/v2.1/shards/manifest.json"
				object := client.objects[key]
				object.metadata["sha256"] = strings.Repeat("0", 64)
				client.objects[key] = object
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			service, client := newPublicationTestService(t)
			test.mutate(client)
			if _, err := service.loadPublicationManifest(
				context.Background(), "kitscenes", "v2.1",
			); err == nil {
				t.Fatal("invalid S3 publication inventory was accepted")
			}
		})
	}
}

func TestPublishedVersionRejectsMissingExplicitManifest(t *testing.T) {
	service, client := newPublicationTestService(t)
	delete(client.objects, "kitscenes/v2.1/shards/manifest.json")
	if _, err := service.publishedVersion(
		context.Background(), "kitscenes", "v2.1",
	); err == nil {
		t.Fatal("explicit unpublished version was accepted")
	}
}
