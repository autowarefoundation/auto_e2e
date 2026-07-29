package service

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/store"
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
		"labels": []any{map[string]any{
			"key":          "odd.road.context",
			"quality_tier": "experimental",
			"values":       []any{map[string]any{"value": "urban"}},
		}},
	})
	statisticsBody := oddJSON(t, map[string]any{
		"schema_version": "odd_statistics_v1",
		"labelset_id":    labelSet,
		"scene_count":    1,
		"keys": []any{map[string]any{
			"key":                       "odd.road.context",
			"valid_scene_count":         1,
			"eligible_scene_count":      1,
			"observable_scene_coverage": 1.0,
		}},
	})
	coverageBody := oddJSON(t, map[string]any{
		"schema_version": "odd_quality_v1",
		"labelset_id":    labelSet,
		"keys": []any{map[string]any{
			"key":              "odd.road.context",
			"quality_tier":     "experimental",
			"support_state":    "supported_experimental",
			"attempted_count":  1,
			"successful_count": 1,
		}},
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
		"quality_coverage": {
			Key: root + "/quality/coverage.json", SHA256: oddDigest(coverageBody),
			ByteSize: int64(len(coverageBody)),
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
		OpenAICompatible: map[string]any{
			"model": "road-observer",
			"sampling": map[string]any{
				"camera_anchor_interval_s": 1.0,
				"maximum_camera_anchors":   128,
			},
		},
		Artifacts: artifacts,
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
		manifestKey:                       {body: manifestBody},
		artifacts["ontology"].Key:         {body: ontologyBody},
		artifacts["statistics"].Key:       {body: statisticsBody},
		artifacts["scene_index"].Key:      {body: sceneIndexBody},
		artifacts["quality_coverage"].Key: {body: coverageBody},
		recordKey:                         {body: recordBody},
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

type fakeODDMetricProjectionStore struct {
	consoleStore
	pointers []store.ODDMetricProjectionPointer
}

func (s *fakeODDMetricProjectionStore) ListReadyODDMetricProjections(
	context.Context,
	string,
	string,
	string,
) ([]store.ODDMetricProjectionPointer, error) {
	return append([]store.ODDMetricProjectionPointer(nil), s.pointers...), nil
}

func oddMetricValues(value float64) map[string]float64 {
	return map[string]float64{
		"ade_1s_m":         value,
		"ade_2s_m":         value,
		"ade_3s_m":         value,
		"ade_horizon_m":    value,
		"fde_horizon_m":    value,
		"acceleration_mae": value,
		"curvature_mae":    value,
	}
}

func oddMetricProjectionTestService(
	t *testing.T,
	mutateReport func(map[string]any),
) (*S3Service, *fakePublicationS3, *fakeODDMetricProjectionStore) {
	t.Helper()
	service := oddTestService(t, nil)
	client := service.client.(*fakePublicationS3)
	service.artifactsBucket = "artifacts"

	var labelSetPointer oddPointer
	if err := json.Unmarshal(
		client.objects["kitscenes/v3.0/odd/latest.json"].body,
		&labelSetPointer,
	); err != nil {
		t.Fatal(err)
	}
	modelArtifactID := strings.Repeat("1", 64)
	projectionID := strings.Repeat("2", 64)
	evaluationManifestSHA256 := strings.Repeat("3", 64)
	validationSampleUIDDigest := strings.Repeat("4", 64)
	root := fmt.Sprintf(
		"odd_metric_projections/schema=v1/dataset=kitscenes/version=v3.0/"+
			"labelset=oddls-test/model=%s/projection=%s",
		modelArtifactID,
		projectionID,
	)
	reportKey := root + "/report.json"
	manifestKey := root + "/manifest.json"
	report := map[string]any{
		"schema_version":            "odd_model_metric_projection_v1",
		"projection_policy_version": "odd_interval_projection_v1",
		"metric_policy_version":     "control_displacement_seed_mean_v1",
		"frequency_hz":              10,
		"horizon_steps":             64,
		"horizon_seconds":           6.4,
		"observation_join":          "start <= anchor < end",
		"event_join": "event_start < anchor + model_horizon " +
			"and event_end > anchor",
		"seed_aggregation":          "arithmetic_mean",
		"sample_uid_digest":         validationSampleUIDDigest,
		"sample_count":              3820,
		"scene_count":               40,
		"samples_with_observations": 3800,
		"samples_with_events":       100,
		"overall": map[string]any{
			"sample_count": 3820,
			"scene_count":  40,
			"metrics":      oddMetricValues(1.0),
		},
		"slices": []any{map[string]any{
			"kind":         "observation",
			"key":          "odd.road.context",
			"value":        "urban",
			"status":       "valid",
			"sample_count": 100,
			"scene_count":  20,
			"metrics":      oddMetricValues(1.1),
		}},
		"status":        "ready",
		"projection_id": projectionID,
		"model": map[string]any{
			"artifact_sha256":       modelArtifactID,
			"registered_model_name": "auto-e2e-driving-policy",
			"model_version":         42,
			"run_id":                "run-42",
		},
		"evaluation_dataset": map[string]any{
			"dataset":                 "kitscenes",
			"version":                 "v2.2",
			"manifest_uri":            "s3://datasets/kitscenes/v2.2/manifest.json",
			"manifest_sha256":         evaluationManifestSHA256,
			"overlay_manifest_key":    "overlays/model/manifest.json",
			"overlay_manifest_sha256": strings.Repeat("5", 64),
			"overlay_cache_identity":  strings.Repeat("6", 64),
		},
		"labelset": map[string]any{
			"dataset":                 "kitscenes",
			"version":                 "v3.0",
			"labelset_id":             "oddls-test",
			"manifest_key":            labelSetPointer.ManifestKey,
			"manifest_sha256":         labelSetPointer.ManifestSHA256,
			"dataset_manifest_sha256": strings.Repeat("a", 64),
		},
		"validation": map[string]any{
			"strategy":          "exact_group_fraction",
			"split_id":          "split-v1",
			"group_count":       40,
			"sample_count":      3820,
			"sample_uid_digest": validationSampleUIDDigest,
		},
	}
	if mutateReport != nil {
		mutateReport(report)
	}
	reportBody := oddJSON(t, report)
	reportSHA256 := oddDigest(reportBody)
	projectionManifest := map[string]any{
		"schema_version":                     "odd_model_metric_projection_manifest_v1",
		"status":                             "ready",
		"projection_id":                      projectionID,
		"projection_schema_version":          "odd_model_metric_projection_v1",
		"projection_policy_version":          "odd_interval_projection_v1",
		"metric_policy_version":              "control_displacement_seed_mean_v1",
		"model_artifact_sha256":              modelArtifactID,
		"labelset_id":                        "oddls-test",
		"labelset_manifest_sha256":           labelSetPointer.ManifestSHA256,
		"evaluation_dataset_manifest_sha256": evaluationManifestSHA256,
		"validation_sample_uid_digest":       validationSampleUIDDigest,
		"sample_count":                       3820,
		"artifacts": map[string]any{
			"report": map[string]any{
				"key":          reportKey,
				"sha256":       reportSHA256,
				"byte_size":    len(reportBody),
				"content_type": "application/json",
			},
			"samples": map[string]any{
				"key":          root + "/samples.jsonl.gz",
				"sha256":       strings.Repeat("7", 64),
				"byte_size":    4096,
				"content_type": "application/x-ndjson",
			},
		},
	}
	manifestBody := oddJSON(t, projectionManifest)
	client.objects[reportKey] = fakePublicationObject{body: reportBody}
	client.objects[manifestKey] = fakePublicationObject{body: manifestBody}
	fakeStore := &fakeODDMetricProjectionStore{
		pointers: []store.ODDMetricProjectionPointer{{
			ProjectionID:                    projectionID,
			ProjectionPolicyVersion:         "odd_interval_projection_v1",
			MetricPolicyVersion:             "control_displacement_seed_mean_v1",
			LabelSetID:                      "oddls-test",
			LabelSetManifestSHA256:          labelSetPointer.ManifestSHA256,
			ModelArtifactID:                 modelArtifactID,
			RegisteredModelName:             "auto-e2e-driving-policy",
			ModelVersion:                    42,
			RunID:                           "run-42",
			EvaluationDataset:               "kitscenes",
			EvaluationVersion:               "v2.2",
			EvaluationDatasetManifestSHA256: evaluationManifestSHA256,
			ValidationSampleUIDDigest:       validationSampleUIDDigest,
			SampleCount:                     3820,
			SceneCount:                      40,
			ManifestKey:                     manifestKey,
			ManifestSHA256:                  oddDigest(manifestBody),
			ManifestByteSize:                int64(len(manifestBody)),
			ReportKey:                       reportKey,
			ReportSHA256:                    reportSHA256,
			ReportByteSize:                  int64(len(reportBody)),
			ArtifactsBucket:                 "artifacts",
		}},
	}
	service.store = fakeStore
	return service, client, fakeStore
}

func oddStructuredSearchService(t *testing.T) *S3Service {
	t.Helper()
	service := oddTestService(t, nil)
	client := service.client.(*fakePublicationS3)
	const root = "kitscenes/v3.0/odd/labelsets/oddls-test"
	scenes := []ODDSceneSummary{
		{
			SceneUID:         "scene-a",
			ShardName:        "scene-a.tar",
			RecordKey:        root + "/scenes/scene-a.json",
			RecordSHA256:     strings.Repeat("c", 64),
			RecordByteSize:   100,
			StartTimestampNS: 100,
			EndTimestampNS:   1_100,
			DistanceM:        10,
			Observations: []ODDSceneObservationSummary{
				{
					Key: "odd.road.context", Status: "valid",
					Values: []string{"urban"}, Source: "map_route",
					Confidence: 0.9, DurationNS: 800,
					FirstTimestampNS: 100, IntervalCount: 1,
				},
				{
					Key: "odd.environment.sky", Status: "unavailable",
					Source: "vlm", Confidence: 0,
					DurationNS: 1_000, FirstTimestampNS: 100,
					IntervalCount: 1,
				},
			},
		},
		{
			SceneUID:         "scene-b",
			ShardName:        "scene-b.tar",
			RecordKey:        root + "/scenes/scene-b.json",
			RecordSHA256:     strings.Repeat("d", 64),
			RecordByteSize:   100,
			StartTimestampNS: 200,
			EndTimestampNS:   2_200,
			DistanceM:        20,
			Observations: []ODDSceneObservationSummary{
				{
					Key: "odd.road.context", Status: "valid",
					Values: []string{"suburban"}, Source: "map_route",
					Confidence: 0.7, DurationNS: 1_500,
					FirstTimestampNS: 200, IntervalCount: 2,
				},
				{
					Key: "perception.object.visibility", Status: "valid",
					Values: []string{"partially_visible"}, Source: "vlm",
					Confidence: 0.85, DurationNS: 600,
					FirstTimestampNS: 400, IntervalCount: 2,
					CameraID: "front", ActorTrackUID: "vehicle-1",
				},
			},
		},
		{
			SceneUID:         "scene-c",
			ShardName:        "scene-c.tar",
			RecordKey:        root + "/scenes/scene-c.json",
			RecordSHA256:     strings.Repeat("e", 64),
			RecordByteSize:   100,
			StartTimestampNS: 300,
			EndTimestampNS:   3_300,
			DistanceM:        30,
			Observations: []ODDSceneObservationSummary{
				{
					Key: "odd.road.context", Status: "valid",
					Values: []string{"rural"}, Source: "fusion",
					Confidence: 0.8, DurationNS: 2_500,
					FirstTimestampNS: 300, IntervalCount: 1,
				},
				{
					Key: "event.vehicle.interaction", Status: "valid",
					Values: []string{"cut_in"}, Source: "fusion",
					Confidence: 0.95, DurationNS: 500,
					FirstTimestampNS: 500, IntervalCount: 1,
					ActorTrackUID: "vehicle-2", EventUID: "event-1",
				},
			},
			Events: []ODDSceneEventSummary{
				{
					EventUID: "event-1", PrimaryEventKey: "event.vehicle.interaction",
					PrimaryValues:    []string{"cut_in"},
					StartTimestampNS: 500, EndTimestampNS: 1_000,
					Status: "valid", Confidence: 0.95,
					ActorTrackUIDs: []string{"vehicle-2"},
					Outcome:        "unresolved",
				},
			},
		},
	}
	indexBody := oddJSON(t, oddSceneIndex{
		SchemaVersion: "odd_scene_index_v2",
		LabelSetID:    "oddls-test",
		Scenes:        scenes,
	})
	indexKey := root + "/scene_index.json"
	client.objects[indexKey] = fakePublicationObject{body: indexBody}

	manifestKey := root + "/manifest.json"
	var manifest ODDManifest
	if err := json.Unmarshal(client.objects[manifestKey].body, &manifest); err != nil {
		t.Fatal(err)
	}
	manifest.SceneCount = len(scenes)
	artifact := manifest.Artifacts["scene_index"]
	artifact.SHA256 = oddDigest(indexBody)
	artifact.ByteSize = int64(len(indexBody))
	manifest.Artifacts["scene_index"] = artifact
	manifestBody := oddJSON(t, manifest)
	client.objects[manifestKey] = fakePublicationObject{body: manifestBody}

	pointerKey := "kitscenes/v3.0/odd/latest.json"
	var pointer oddPointer
	if err := json.Unmarshal(client.objects[pointerKey].body, &pointer); err != nil {
		t.Fatal(err)
	}
	pointer.ManifestSHA256 = oddDigest(manifestBody)
	client.objects[pointerKey] = fakePublicationObject{body: oddJSON(t, pointer)}
	return service
}

func oddEvidenceTestService(
	t *testing.T,
	mutateRecord func(map[string]any),
) *S3Service {
	t.Helper()
	service := oddTestService(t, nil)
	client := service.client.(*fakePublicationS3)
	const root = "kitscenes/v3.0/odd/labelsets/oddls-test"
	record := map[string]any{
		"scene_uid":       "scene-1",
		"dataset_name":    "kitscenes",
		"dataset_version": "v3.0",
		"provenance": map[string]any{
			"labeler_version": "odd_dataset_labeler_v1",
		},
		"observations": []any{
			map[string]any{
				"observation_uid":           "observation-1",
				"scene_uid":                 "scene-1",
				"key":                       "event.ego.maneuver",
				"status":                    "ambiguous",
				"values":                    []string{},
				"confidence":                0.6,
				"source":                    "fusion",
				"start_timestamp_ns":        10,
				"end_timestamp_ns":          90,
				"evidence_uids":             []string{"evidence-support"},
				"conflicting_evidence_uids": []string{"evidence-conflict"},
				"event_uid":                 "event-1",
				"provenance": map[string]any{
					"fusion_version": "odd_fusion_v1",
				},
			},
		},
		"evidence": []any{
			map[string]any{
				"evidence_uid": "evidence-support",
				"label_key":    "event.ego.maneuver",
				"status":       "valid",
				"values":       []string{"turn_left"},
				"source":       "gnss_ins",
				"confidence":   0.91,
				"scope": map[string]any{
					"scene_uid":          "scene-1",
					"start_timestamp_ns": 10,
					"end_timestamp_ns":   90,
				},
				"provenance": map[string]any{
					"labeler_name": "trajectory_resolver",
				},
			},
			map[string]any{
				"evidence_uid": "evidence-conflict",
				"label_key":    "event.ego.maneuver",
				"status":       "valid",
				"values":       []string{"turn_right"},
				"source":       "vlm",
				"confidence":   0.72,
				"scope": map[string]any{
					"scene_uid":          "scene-1",
					"start_timestamp_ns": 10,
					"end_timestamp_ns":   90,
				},
				"provenance": map[string]any{
					"labeler_name": "road_vlm",
					"model_name":   "road-observer",
				},
			},
		},
		"events": []any{
			map[string]any{
				"event_uid":          "event-1",
				"scene_uid":          "scene-1",
				"primary_event_key":  "event.ego.maneuver",
				"observation_uids":   []string{"observation-1"},
				"start_timestamp_ns": 10,
				"end_timestamp_ns":   90,
				"phases": []any{
					map[string]any{
						"phase":              "active",
						"start_timestamp_ns": 10,
						"end_timestamp_ns":   90,
					},
				},
			},
		},
	}
	if mutateRecord != nil {
		mutateRecord(record)
	}
	recordBody := oddJSON(t, record)
	recordKey := root + "/scenes/scene-1.json"
	client.objects[recordKey] = fakePublicationObject{body: recordBody}

	indexKey := root + "/scene_index.json"
	var index oddSceneIndex
	if err := json.Unmarshal(client.objects[indexKey].body, &index); err != nil {
		t.Fatal(err)
	}
	index.SchemaVersion = "odd_scene_index_v2"
	index.Scenes[0].RecordSHA256 = oddDigest(recordBody)
	index.Scenes[0].RecordByteSize = int64(len(recordBody))
	indexBody := oddJSON(t, index)
	client.objects[indexKey] = fakePublicationObject{body: indexBody}

	manifestKey := root + "/manifest.json"
	var manifest ODDManifest
	if err := json.Unmarshal(client.objects[manifestKey].body, &manifest); err != nil {
		t.Fatal(err)
	}
	artifact := manifest.Artifacts["scene_index"]
	artifact.SHA256 = oddDigest(indexBody)
	artifact.ByteSize = int64(len(indexBody))
	manifest.Artifacts["scene_index"] = artifact
	manifestBody := oddJSON(t, manifest)
	client.objects[manifestKey] = fakePublicationObject{body: manifestBody}

	pointerKey := "kitscenes/v3.0/odd/latest.json"
	var pointer oddPointer
	if err := json.Unmarshal(client.objects[pointerKey].body, &pointer); err != nil {
		t.Fatal(err)
	}
	pointer.ManifestSHA256 = oddDigest(manifestBody)
	client.objects[pointerKey] = fakePublicationObject{body: oddJSON(t, pointer)}
	return service
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

func TestODDManifestPreservesNestedSamplingProvenance(t *testing.T) {
	service := oddTestService(t, nil)

	manifest, _, err := service.loadODDManifest(
		context.Background(), "kitscenes", "v3.0",
	)

	if err != nil {
		t.Fatal(err)
	}
	sampling, ok := manifest.OpenAICompatible["sampling"].(map[string]any)
	if !ok ||
		sampling["camera_anchor_interval_s"] != 1.0 ||
		sampling["maximum_camera_anchors"] != float64(128) {
		t.Fatalf(
			"nested OpenAI-compatible sampling provenance = %#v",
			manifest.OpenAICompatible,
		)
	}
}

func TestODDManifestReportsUnstartedWithoutReadyPointer(t *testing.T) {
	service := oddTestService(t, func(objects map[string]fakePublicationObject) {
		delete(objects, "kitscenes/v3.0/odd/latest.json")
	})

	_, _, err := service.loadODDManifest(
		context.Background(), "kitscenes", "v3.0",
	)

	if !errors.Is(err, ErrODDNotStarted) {
		t.Fatalf("missing ready pointer error = %v", err)
	}
}

func TestODDMetricProjectionsVerifyAndReturnReport(t *testing.T) {
	service, _, _ := oddMetricProjectionTestService(t, nil)

	response, manifest, digest, err := service.ODDMetricProjections(
		context.Background(),
		"kitscenes",
		"v3.0",
	)

	if err != nil {
		t.Fatal(err)
	}
	if response.Dataset != "kitscenes" ||
		response.Version != "v3.0" ||
		response.LabelSetID != "oddls-test" ||
		response.LabelSetManifestSHA256 != digest ||
		manifest.LabelSetID != response.LabelSetID ||
		len(response.Projections) != 1 {
		t.Fatalf("ODD metric projection response = %+v", response)
	}
	var report oddMetricProjectionReport
	if err := json.Unmarshal(response.Projections[0], &report); err != nil {
		t.Fatal(err)
	}
	if report.Model.ModelVersion != 42 ||
		report.SampleCount != 3820 ||
		report.SceneCount != 40 ||
		report.Overall.Metrics["ade_horizon_m"] != 1.0 ||
		len(report.Slices) != 1 {
		t.Fatalf("ODD metric projection report = %+v", report)
	}
}

func TestODDMetricProjectionsRejectTamperedObjects(t *testing.T) {
	for _, test := range []struct {
		name       string
		mutateBody func([]byte) []byte
		want       string
	}{
		{
			name: "size",
			mutateBody: func(body []byte) []byte {
				return append(body, '\n')
			},
			want: "size differs",
		},
		{
			name: "digest",
			mutateBody: func(body []byte) []byte {
				tampered := append([]byte(nil), body...)
				tampered[len(tampered)-2] ^= 1
				return tampered
			},
			want: "digest differs",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			service, client, fakeStore := oddMetricProjectionTestService(
				t,
				nil,
			)
			reportKey := fakeStore.pointers[0].ReportKey
			object := client.objects[reportKey]
			object.body = test.mutateBody(object.body)
			client.objects[reportKey] = object

			_, _, _, err := service.ODDMetricProjections(
				context.Background(),
				"kitscenes",
				"v3.0",
			)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("tampered report error = %v", err)
			}
		})
	}
}

func TestODDMetricProjectionsRejectSemanticIdentityDrift(t *testing.T) {
	service, _, _ := oddMetricProjectionTestService(
		t,
		func(report map[string]any) {
			labelSet := report["labelset"].(map[string]any)
			labelSet["manifest_sha256"] = strings.Repeat("f", 64)
		},
	)

	_, _, _, err := service.ODDMetricProjections(
		context.Background(),
		"kitscenes",
		"v3.0",
	)

	if err == nil || !strings.Contains(err.Error(), "LabelSet identity differs") {
		t.Fatalf("semantic identity drift error = %v", err)
	}
}

func TestODDOntologyIncludesCanonicalDatasetSupport(t *testing.T) {
	service := oddTestService(t, nil)

	body, manifest, _, err := service.ODDOntology(
		context.Background(), "kitscenes", "v3.0",
	)

	if err != nil {
		t.Fatal(err)
	}
	if manifest.LabelSetID != "oddls-test" {
		t.Fatalf("labelset = %q", manifest.LabelSetID)
	}
	var document struct {
		DatasetName    string `json:"dataset_name"`
		DatasetVersion string `json:"dataset_version"`
		LabelSetID     string `json:"labelset_id"`
		Labels         []struct {
			Key            string `json:"key"`
			DatasetSupport struct {
				SupportState            string  `json:"support_state"`
				ValidSceneCount         int64   `json:"valid_scene_count"`
				EligibleSceneCount      int64   `json:"eligible_scene_count"`
				ObservableSceneCoverage float64 `json:"observable_scene_coverage"`
			} `json:"dataset_support"`
		} `json:"labels"`
	}
	if err := json.Unmarshal(body, &document); err != nil {
		t.Fatal(err)
	}
	if document.DatasetName != "kitscenes" ||
		document.DatasetVersion != "v3.0" ||
		document.LabelSetID != "oddls-test" ||
		len(document.Labels) != 1 ||
		document.Labels[0].Key != "odd.road.context" ||
		document.Labels[0].DatasetSupport.SupportState !=
			"supported_experimental" ||
		document.Labels[0].DatasetSupport.ValidSceneCount != 1 ||
		document.Labels[0].DatasetSupport.EligibleSceneCount != 1 ||
		document.Labels[0].DatasetSupport.ObservableSceneCoverage != 1 {
		t.Fatalf("unexpected enriched ontology: %+v", document)
	}
}

func TestODDOntologyNormalizesLegacySupportState(t *testing.T) {
	state, ok := canonicalODDSupportState(
		"supported_observed",
		"experimental",
	)
	if !ok || state != "supported_experimental" {
		t.Fatalf("legacy support state = %q, %v", state, ok)
	}
	state, ok = canonicalODDSupportState("attempted_no_valid", "experimental")
	if !ok || state != "unsupported_missing_source" {
		t.Fatalf("legacy unavailable state = %q, %v", state, ok)
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

func TestODDStructuredSearchSupportsNestedLogicAndScope(t *testing.T) {
	service := oddStructuredSearchService(t)

	response, err := service.SearchODDScenesStructured(
		context.Background(),
		"kitscenes",
		"v3.0",
		ODDStructuredSearchRequest{
			Query: ODDSearchGroup{
				Logic: "and",
				Predicates: []ODDSearchPredicate{
					{
						Key: "odd.road.context", Operator: "in",
						Values:            []string{"suburban", "rural"},
						MinimumConfidence: 0.75,
					},
				},
				Groups: []ODDSearchGroup{
					{
						Logic: "or",
						Predicates: []ODDSearchPredicate{
							{
								Key:      "perception.object.visibility",
								Operator: "equals",
								Values:   []string{"partially_visible"},
								CameraID: "front", ActorTrackUID: "vehicle-1",
								MinimumDurationNS: 500,
							},
							{
								Key:      "event.vehicle.interaction",
								Operator: "contains", Values: []string{"cut_in"},
								ActorTrackUID: "vehicle-2",
							},
						},
					},
				},
			},
			Sort:  "scene_uid",
			Limit: 50,
		},
	)

	if err != nil {
		t.Fatal(err)
	}
	if response.Total != 1 || response.Scenes[0].SceneUID != "scene-c" {
		t.Fatalf("nested search response = %+v", response)
	}
	if len(response.Scenes[0].Matched) != 2 ||
		response.Scenes[0].MatchedDuration != 3_000 ||
		response.Scenes[0].MatchConfidence != 0.95 {
		t.Fatalf("match decoration = %+v", response.Scenes[0])
	}
}

func TestODDNotEqualsExcludesUnavailableObservations(t *testing.T) {
	service := oddStructuredSearchService(t)

	response, err := service.SearchODDScenesStructured(
		context.Background(),
		"kitscenes",
		"v3.0",
		ODDStructuredSearchRequest{
			Query: ODDSearchGroup{
				Predicates: []ODDSearchPredicate{
					{
						Key:      "odd.environment.sky",
						Operator: "not_equals",
						Values:   []string{"clear"},
					},
				},
			},
			Limit: 50,
		},
	)

	if err != nil {
		t.Fatal(err)
	}
	if response.Total != 0 {
		t.Fatalf("unavailable observation matched not_equals: %+v", response)
	}
}

func TestODDStructuredSearchSortsAndPaginatesV2Events(t *testing.T) {
	service := oddStructuredSearchService(t)

	response, err := service.SearchODDScenesStructured(
		context.Background(),
		"kitscenes",
		"v3.0",
		ODDStructuredSearchRequest{
			Query: ODDSearchGroup{
				Predicates: []ODDSearchPredicate{
					{Key: "odd.road.context", Operator: "exists"},
				},
			},
			Sort:       "confidence",
			Descending: true,
			Limit:      1,
			Offset:     1,
		},
	)

	if err != nil {
		t.Fatal(err)
	}
	if response.Total != 3 || len(response.Scenes) != 1 ||
		response.Scenes[0].SceneUID != "scene-c" || !response.More {
		t.Fatalf("paginated response = %+v", response)
	}
	if len(response.Scenes[0].Events) != 1 ||
		response.Scenes[0].Events[0].Outcome != "unresolved" ||
		response.Scenes[0].Events[0].ActorTrackUIDs[0] != "vehicle-2" {
		t.Fatalf("v2 event summary = %+v", response.Scenes[0].Events)
	}
}

func TestODDStructuredSearchRejectsInvalidRequests(t *testing.T) {
	service := oddStructuredSearchService(t)

	_, err := service.SearchODDScenesStructured(
		context.Background(),
		"kitscenes",
		"v3.0",
		ODDStructuredSearchRequest{
			Query: ODDSearchGroup{
				Logic: "not",
				Predicates: []ODDSearchPredicate{
					{Key: "odd.road.context", Operator: "exists"},
				},
			},
			Limit: 50,
		},
	)

	if err == nil || !strings.Contains(err.Error(), ErrODDInvalidQuery.Error()) {
		t.Fatalf("invalid query error = %v", err)
	}
}

func TestODDEvidenceReturnsReferencedSourcesAndEvent(t *testing.T) {
	service := oddEvidenceTestService(t, nil)

	response, manifest, digest, err := service.ODDEvidence(
		context.Background(),
		"kitscenes",
		"v3.0",
		"scene-1",
		"observation-1",
	)

	if err != nil {
		t.Fatal(err)
	}
	if response.LabelSetID != manifest.LabelSetID ||
		response.ManifestSHA256 != digest ||
		len(response.SupportingEvidence) != 1 ||
		len(response.ConflictingEvidence) != 1 ||
		len(response.RelatedEvents) != 1 {
		t.Fatalf("evidence response = %+v", response)
	}
	if !strings.Contains(
		string(response.SupportingEvidence[0]),
		`"labeler_name":"trajectory_resolver"`,
	) || !strings.Contains(
		string(response.ConflictingEvidence[0]),
		`"model_name":"road-observer"`,
	) || !strings.Contains(
		string(response.RelatedEvents[0]),
		`"phase":"active"`,
	) || !strings.Contains(
		string(response.SceneProvenance),
		`"labeler_version":"odd_dataset_labeler_v1"`,
	) {
		t.Fatalf("focused evidence content = %+v", response)
	}
}

func TestODDEvidenceRejectsMissingReference(t *testing.T) {
	service := oddEvidenceTestService(t, func(record map[string]any) {
		evidence := record["evidence"].([]any)
		record["evidence"] = evidence[:1]
	})

	_, _, _, err := service.ODDEvidence(
		context.Background(),
		"kitscenes",
		"v3.0",
		"scene-1",
		"observation-1",
	)

	if err == nil || !strings.Contains(err.Error(), "missing evidence") {
		t.Fatalf("missing evidence error = %v", err)
	}
}

func TestODDEvidenceReturnsNotFoundForUnknownObservation(t *testing.T) {
	service := oddEvidenceTestService(t, nil)

	_, _, _, err := service.ODDEvidence(
		context.Background(),
		"kitscenes",
		"v3.0",
		"scene-1",
		"unknown",
	)

	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("unknown observation error = %v", err)
	}
}
