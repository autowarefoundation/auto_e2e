package service

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func semanticOccupancyGzipFixture(t *testing.T, marker string) []byte {
	t.Helper()
	var output bytes.Buffer
	writer := gzip.NewWriter(&output)
	if _, err := writer.Write([]byte("ASOC-" + marker)); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return output.Bytes()
}

func validOccupancySetFixture(
	t *testing.T,
	datasetManifestSHA256, modelArtifactID, shard string,
	payload []byte,
) (occupancySetManifest, string, []byte) {
	t.Helper()
	payloadDigest := sha256.Sum256(payload)
	weightSHA256 := strings.Repeat("f", 64)
	bodyKey := "semantic-occupancy/schema=v1/model=" + modelArtifactID +
		"/manifest=" + datasetManifestSHA256 +
		"/geometry=autoe2e-bev-450x300-0p4m-v1" +
		"/taxonomy=autoe2e-bev-semantic-v1" +
		"/head=bev-segmentation-head-v1" +
		"/dataset=kitscenes/shard=" + shard +
		"/occupancy.bin.gz"
	manifest := occupancySetManifest{
		SchemaVersion:         occupancySetSchema,
		ArtifactKind:          "native-semantic-occupancy",
		ArtifactSchema:        "v1",
		CreatedAt:             "2026-08-19T14:00:00Z",
		Dataset:               "kitscenes",
		DatasetVersion:        "v2.1",
		DatasetManifestSHA256: datasetManifestSHA256,
		DisplayName:           "AutoE2E Reactive BEV segmentation",
		GeometryID:            "autoe2e-bev-450x300-0p4m-v1",
		HeadVersion:           "bev-segmentation-head-v1",
		InputContract:         "autoe2e-packed-calibrated-camera-v1",
		Limitations: []string{
			"No viewer-side geometry correction is applied.",
		},
		ModelArtifactID: modelArtifactID,
		ModelFamily:     "AutoE2E Reactive",
		ModelSource: occupancySetModelSource{
			CodeLicenseSPDX:         "Apache-2.0",
			Config:                  "embedded-checkpoint-config",
			LicenseSPDX:             "NOASSERTION",
			Repository:              "https://github.com/autowarefoundation/auto_e2e",
			RepositoryRevision:      strings.Repeat("c", 40),
			TrainingDataLicenseSPDX: "NOASSERTION",
			WeightSHA256:            weightSHA256,
			WeightSourceURL:         "urn:sha256:" + weightSHA256,
		},
		ProducerConfig: map[string]any{
			"deterministic_algorithms": true,
			"probability_encoding":     "uint8-rint-gzip-level-6-v1",
		},
		SampleCount: 3,
		ShardCount:  1,
		Shards: []occupancySetShard{{
			ByteSize:       int64(len(payload)),
			S3Key:          bodyKey,
			SampleCount:    3,
			SHA256:         hex.EncodeToString(payloadDigest[:]),
			Shard:          shard,
			TeacherPresent: false,
		}},
		SupportedClasses: []string{
			"drivable_area",
			"lane_area",
			"intersection",
			"crosswalk",
			"stop_line",
			"vehicle",
			"vulnerable_road_user",
			"other_obstacle",
		},
		TaxonomyVersion:  "autoe2e-bev-semantic-v1",
		TeacherAvailable: false,
	}
	manifestKey := occupancySetManifestKey(
		manifest.Dataset,
		manifest.DatasetVersion,
		modelArtifactID,
		datasetManifestSHA256,
	)
	body, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	return manifest, manifestKey, body
}

func addOccupancySetFixture(
	t *testing.T,
	service *S3Service,
	client *fakePublicationS3,
	modelArtifactID, shard string,
) (string, []byte) {
	t.Helper()
	publication, err := service.loadPublicationManifest(
		context.Background(),
		"kitscenes",
		"v2.1",
	)
	if err != nil {
		t.Fatal(err)
	}
	payload := semanticOccupancyGzipFixture(t, modelArtifactID[:8])
	manifest, manifestKey, manifestBody := validOccupancySetFixture(
		t,
		publication.SHA256,
		modelArtifactID,
		shard,
		payload,
	)
	manifestDigest := sha256.Sum256(manifestBody)
	now := time.Date(2026, 8, 19, 14, 0, 0, 0, time.UTC)
	client.objects[manifestKey] = fakePublicationObject{
		body: manifestBody,
		metadata: map[string]string{
			"dataset-manifest-sha256": publication.SHA256,
			"manifest-sha256":         hex.EncodeToString(manifestDigest[:]),
			"schema":                  occupancySetSchema,
		},
		lastModified: now,
	}
	client.objects[manifest.Shards[0].S3Key] = fakePublicationObject{
		body:         payload,
		lastModified: now,
	}
	return manifest.Shards[0].S3Key, payload
}

func TestDecodeOccupancySetManifestBindsPublicationAndBodyKeys(t *testing.T) {
	datasetDigest := strings.Repeat("a", 64)
	modelDigest := strings.Repeat("d", 64)
	payload := semanticOccupancyGzipFixture(t, "valid")
	manifest, key, body := validOccupancySetFixture(
		t,
		datasetDigest,
		modelDigest,
		"scene-a-train-000000.tar",
		payload,
	)

	decoded, err := decodeOccupancySetManifest(
		body,
		key,
		"kitscenes",
		"v2.1",
		datasetDigest,
	)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.ModelArtifactID != modelDigest ||
		decoded.ShardByName["scene-a-train-000000.tar"].S3Key !=
			manifest.Shards[0].S3Key ||
		!isLowerHexDigest(decoded.SHA256) {
		t.Fatalf("decoded occupancy set = %+v", decoded)
	}
}

func TestDecodeOccupancySetManifestAcceptsLaneBoundaryTaxonomy(t *testing.T) {
	datasetDigest := strings.Repeat("a", 64)
	modelDigest := strings.Repeat("d", 64)
	payload := semanticOccupancyGzipFixture(t, "lane-boundary")
	manifest, key, _ := validOccupancySetFixture(
		t,
		datasetDigest,
		modelDigest,
		"scene-a-train-000000.tar",
		payload,
	)
	manifest.SupportedClasses = []string{
		"drivable_area",
		"lane_boundary",
		"intersection",
		"crosswalk",
		"stop_line",
		"vehicle",
		"vulnerable_road_user",
		"other_obstacle",
	}
	body, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := decodeOccupancySetManifest(
		body,
		key,
		"kitscenes",
		"v2.1",
		datasetDigest,
	); err != nil {
		t.Fatal(err)
	}
}

func TestDecodeOccupancySetManifestRejectsInvalidContracts(t *testing.T) {
	datasetDigest := strings.Repeat("a", 64)
	modelDigest := strings.Repeat("d", 64)
	payload := semanticOccupancyGzipFixture(t, "invalid")
	valid, key, _ := validOccupancySetFixture(
		t,
		datasetDigest,
		modelDigest,
		"scene-a-train-000000.tar",
		payload,
	)
	encode := func(value any) []byte {
		t.Helper()
		body, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		return body
	}

	tests := []struct {
		name   string
		body   func() []byte
		digest string
	}{
		{
			name: "unknown field",
			body: func() []byte {
				var value map[string]any
				if err := json.Unmarshal(encode(valid), &value); err != nil {
					t.Fatal(err)
				}
				value["unexpected"] = true
				return encode(value)
			},
			digest: datasetDigest,
		},
		{
			name: "dataset publication mismatch",
			body: func() []byte {
				return encode(valid)
			},
			digest: strings.Repeat("b", 64),
		},
		{
			name: "non canonical body key",
			body: func() []byte {
				value := valid
				value.Shards = append([]occupancySetShard(nil), valid.Shards...)
				value.Shards[0].S3Key = "semantic-occupancy/escaped.bin.gz"
				return encode(value)
			},
			digest: datasetDigest,
		},
		{
			name: "teacher disagreement",
			body: func() []byte {
				value := valid
				value.Shards = append([]occupancySetShard(nil), valid.Shards...)
				value.Shards[0].TeacherPresent = true
				return encode(value)
			},
			digest: datasetDigest,
		},
		{
			name: "unsupported class",
			body: func() []byte {
				value := valid
				value.SupportedClasses = []string{"invented"}
				return encode(value)
			},
			digest: datasetDigest,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := decodeOccupancySetManifest(
				test.body(),
				key,
				"kitscenes",
				"v2.1",
				test.digest,
			); err == nil {
				t.Fatal("invalid occupancy set manifest was accepted")
			}
		})
	}
}

func TestSemanticOccupancyCatalogIsIndependentFromTrajectoryStore(t *testing.T) {
	service, client := newPublicationTestService(t)
	firstModel := strings.Repeat("d", 64)
	secondModel := strings.Repeat("e", 64)
	firstShard := "scene-a-train-000000.tar"
	secondShard := "scene-b-train-000000.tar"
	firstBodyKey, firstPayload := addOccupancySetFixture(
		t,
		service,
		client,
		firstModel,
		firstShard,
	)
	addOccupancySetFixture(
		t,
		service,
		client,
		secondModel,
		secondShard,
	)
	if service.store != nil {
		t.Fatal("fixture unexpectedly configured a trajectory store")
	}

	models, version, err := service.ListSemanticOccupancyModels(
		context.Background(),
		"kitscenes",
		"v2.1",
		firstShard,
	)
	if err != nil {
		t.Fatal(err)
	}
	if version != "v2.1" || len(models) != 1 {
		t.Fatalf("catalog = %q %+v", version, models)
	}
	if models[0].ModelArtifactID != firstModel ||
		models[0].ArtifactKind != "native-semantic-occupancy" ||
		models[0].ModelSource.WeightSHA256 != strings.Repeat("f", 64) ||
		models[0].ModelSource.CodeLicenseSPDX != "Apache-2.0" ||
		models[0].ModelSource.LicenseSPDX != "NOASSERTION" ||
		models[0].ProducerConfig["deterministic_algorithms"] != true ||
		models[0].ShardSampleCount != 3 ||
		models[0].TeacherAvailable {
		t.Fatalf("catalog model = %+v", models[0])
	}

	body, resolvedVersion, err := service.GetPublishedSemanticOccupancyBody(
		context.Background(),
		"kitscenes",
		"v2.1",
		firstShard,
		firstModel,
	)
	if err != nil {
		t.Fatal(err)
	}
	if resolvedVersion != "v2.1" ||
		!bytes.Equal(body.Payload, firstPayload) ||
		body.Descriptor.ModelArtifactID != firstModel ||
		body.Descriptor.HeadVersion != "bev-segmentation-head-v1" {
		t.Fatalf("occupancy body = %q %+v", resolvedVersion, body)
	}

	object := client.objects[firstBodyKey]
	object.body = append([]byte(nil), object.body...)
	object.body[len(object.body)-1] ^= 0xff
	client.objects[firstBodyKey] = object
	if _, _, err := service.GetPublishedSemanticOccupancyBody(
		context.Background(),
		"kitscenes",
		"v2.1",
		firstShard,
		firstModel,
	); err == nil || !strings.Contains(err.Error(), "SHA-256 mismatch") {
		t.Fatalf("tampered body error = %v", err)
	}
}

func TestOccupancySetManifestObjectMetadataIsVerified(t *testing.T) {
	service, client := newPublicationTestService(t)
	modelID := strings.Repeat("d", 64)
	addOccupancySetFixture(
		t,
		service,
		client,
		modelID,
		"scene-a-train-000000.tar",
	)
	publication, err := service.loadPublicationManifest(
		context.Background(),
		"kitscenes",
		"v2.1",
	)
	if err != nil {
		t.Fatal(err)
	}
	key := occupancySetManifestKey(
		"kitscenes",
		"v2.1",
		modelID,
		publication.SHA256,
	)
	object := client.objects[key]
	object.metadata["manifest-sha256"] = strings.Repeat("0", 64)
	client.objects[key] = object

	if _, _, err := service.ListSemanticOccupancyModels(
		context.Background(),
		"kitscenes",
		"v2.1",
		"scene-a-train-000000.tar",
	); err == nil || !strings.Contains(err.Error(), "object identity") {
		t.Fatalf("invalid manifest metadata error = %v", err)
	}
}
