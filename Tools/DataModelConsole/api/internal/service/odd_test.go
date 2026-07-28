package service

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func oddJSON(t *testing.T, value any) []byte {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func oddDigest(body []byte) string {
	return fmt.Sprintf("%x", sha256.Sum256(body))
}

func oddTestService(
	t *testing.T,
	mutate func(map[string]fakePublicationObject),
) *S3Service {
	t.Helper()
	const (
		dataset   = "kitscenes"
		version   = "v3.0"
		labelSet  = "oddls-test"
		root      = dataset + "/" + version + "/odd/labelsets/" + labelSet
		recordKey = root + "/scenes/scene-1.json"
	)
	recordBody := oddJSON(t, map[string]any{
		"scene_uid":       "scene-1",
		"dataset_name":    dataset,
		"dataset_version": version,
		"observations":    []any{},
	})
	ontologyBody := oddJSON(t, map[string]any{
		"schema_version": "odd_ontology_registry_v1",
		"labels":         []any{},
	})
	statisticsBody := oddJSON(t, map[string]any{
		"schema_version": "odd_statistics_v1",
		"labelset_id":    labelSet,
		"scene_count":    1,
	})
	sceneIndexBody := oddJSON(t, map[string]any{
		"schema_version": "odd_scene_index_v1",
		"labelset_id":    labelSet,
		"scenes": []any{map[string]any{
			"scene_uid":          "scene-1",
			"shard_name":         "scene-1.tar",
			"record_key":         recordKey,
			"record_sha256":      oddDigest(recordBody),
			"record_byte_size":   len(recordBody),
			"start_timestamp_ns": 1,
			"end_timestamp_ns":   2,
			"distance_m":         1,
			"observations":       []any{},
		}},
	})
	artifacts := map[string]ODDArtifact{
		"ontology": {
			Key: root + "/ontology.json", SHA256: oddDigest(ontologyBody),
			ByteSize: int64(len(ontologyBody)),
		},
		"statistics": {
			Key: root + "/statistics.json", SHA256: oddDigest(statisticsBody),
			ByteSize: int64(len(statisticsBody)),
		},
		"scene_index": {
			Key: root + "/scene_index.json", SHA256: oddDigest(sceneIndexBody),
			ByteSize: int64(len(sceneIndexBody)),
		},
	}
	manifestBody := oddJSON(t, ODDManifest{
		SchemaVersion:         "odd_labelset_manifest_v1",
		Status:                "ready",
		LabelSetID:            labelSet,
		DatasetName:           dataset,
		DatasetVersion:        version,
		DatasetManifestSHA256: strings.Repeat("a", 64),
		OntologyVersion:       "odd_ontology_v1",
		OntologySHA256:        strings.Repeat("b", 64),
		LabelerVersion:        "odd_dataset_labeler_v1",
		SceneCount:            1,
		Artifacts:             artifacts,
	})
	manifestKey := root + "/manifest.json"
	pointerBody := oddJSON(t, oddPointer{
		SchemaVersion:  "odd_labelset_pointer_v1",
		Status:         "ready",
		DatasetName:    dataset,
		DatasetVersion: version,
		LabelSetID:     labelSet,
		ManifestKey:    manifestKey,
		ManifestSHA256: oddDigest(manifestBody),
	})
	objects := map[string]fakePublicationObject{
		dataset + "/" + version + "/odd/latest.json": {body: pointerBody},
		manifestKey:                  {body: manifestBody},
		artifacts["ontology"].Key:    {body: ontologyBody},
		artifacts["statistics"].Key:  {body: statisticsBody},
		artifacts["scene_index"].Key: {body: sceneIndexBody},
		recordKey:                    {body: recordBody},
	}
	if mutate != nil {
		mutate(objects)
	}
	return &S3Service{
		client: &fakePublicationS3{
			objects:   objects,
			getCalls:  make(map[string]int),
			headCalls: make(map[string]int),
		},
		bucket: "datasets",
	}
}

func TestODDSceneVerifiesPinnedRecord(t *testing.T) {
	service := oddTestService(t, nil)

	body, manifest, _, err := service.ODDScene(
		context.Background(), "kitscenes", "v3.0", "scene-1",
	)

	if err != nil {
		t.Fatal(err)
	}
	if manifest.LabelSetID != "oddls-test" ||
		!strings.Contains(string(body), `"scene_uid":"scene-1"`) {
		t.Fatalf("unexpected ODD scene response: %s", body)
	}
}

func TestODDSceneRejectsTamperedRecord(t *testing.T) {
	service := oddTestService(t, func(objects map[string]fakePublicationObject) {
		key := "kitscenes/v3.0/odd/labelsets/oddls-test/scenes/scene-1.json"
		objects[key] = fakePublicationObject{body: []byte(`{"scene_uid":"other"}`)}
	})

	_, _, _, err := service.ODDScene(
		context.Background(), "kitscenes", "v3.0", "scene-1",
	)

	if err == nil || !strings.Contains(err.Error(), "size differs") {
		t.Fatalf("tampered record error = %v", err)
	}
}

func TestODDStatisticsRejectsArtifactSizeMismatch(t *testing.T) {
	service := oddTestService(t, func(objects map[string]fakePublicationObject) {
		key := "kitscenes/v3.0/odd/labelsets/oddls-test/statistics.json"
		object := objects[key]
		object.body = append(object.body, '\n')
		objects[key] = object
	})

	_, _, _, err := service.ODDStatistics(
		context.Background(), "kitscenes", "v3.0",
	)

	if err == nil || !strings.Contains(err.Error(), "size differs") {
		t.Fatalf("artifact size error = %v", err)
	}
}
