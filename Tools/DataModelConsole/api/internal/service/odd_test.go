package service

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

const testLabelSetID = "oddls-0123456789abcdef0123456789abcdef"

func sha256Hex(body []byte) string {
	return fmt.Sprintf("%x", sha256.Sum256(body))
}

func mustJSON(t *testing.T, value any) []byte {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// newODDTestService returns a service with no objects; callers add
// publications and LabelSets.
func newODDTestService() (*S3Service, *fakePublicationS3) {
	client := &fakePublicationS3{
		objects:   map[string]fakePublicationObject{},
		getCalls:  map[string]int{},
		headCalls: map[string]int{},
	}
	return &S3Service{
		client:           client,
		bucket:           "datasets",
		versionCache:     map[string]cachedVersion{},
		publicationCache: map[string]*publicationManifest{},
		indexSF:          map[string]*shardIndexBuild{},
	}, client
}

// addTestPublication writes a valid two-scene kitscenes publication at
// version. sampleCounts gives the samples of partitions scene-a and scene-b;
// partitionIDs, when given, renames those partitions without renaming shards.
func addTestPublication(
	t *testing.T,
	client *fakePublicationS3,
	version string,
	sampleCounts [2]int,
	partitionIDs ...string,
) string {
	t.Helper()
	if len(partitionIDs) == 0 {
		partitionIDs = []string{"scene-a", "scene-b"}
	}
	rigBody := []byte(`{"dataset":"kitscenes","geometry_type":"pinhole"}`)
	rig := map[string]any{
		"key":    fmt.Sprintf("kitscenes/%s/rig/%s.json", version, sha256Hex(rigBody)),
		"sha256": sha256Hex(rigBody),
	}
	value := map[string]any{
		"schema_version":  publicationSchema,
		"status":          "ready",
		"dataset":         "kitscenes",
		"version":         version,
		"total_samples":   sampleCounts[0] + sampleCounts[1],
		"shards":          2,
		"shard_count":     2,
		"rig_count":       1,
		"episodes":        2,
		"num_views":       7,
		"has_gps":         false,
		"has_map":         true,
		"has_world_model": false,
		"partitions": []any{
			map[string]any{"partition_id": partitionIDs[0], "sample_count": sampleCounts[0]},
			map[string]any{"partition_id": partitionIDs[1], "sample_count": sampleCounts[1]},
		},
		"shard_entries": []any{},
	}
	entries := make([]any, 0, 2)
	for _, name := range []string{"scene-a", "scene-b"} {
		entries = append(entries, map[string]any{
			"name":             name + "-train-000000.tar",
			"key":              fmt.Sprintf("kitscenes/%s/shards/%s-train-000000.tar", version, name),
			"byte_size":        100,
			"etag":             "etag-" + name + "-" + version,
			"content_identity": strings.Repeat("d", 64),
			"rig":              rig,
		})
	}
	value["shard_entries"] = entries
	body := mustJSON(t, value)
	now := time.Date(2026, 10, 21, 0, 0, 0, 0, time.UTC)
	client.objects[fmt.Sprintf("kitscenes/%s/shards/manifest.json", version)] = fakePublicationObject{
		body: body,
		metadata: map[string]string{
			"sha256":             sha256Hex(body),
			"publication-schema": publicationSchema,
		},
		lastModified: now,
	}
	for _, raw := range entries {
		entry := raw.(map[string]any)
		client.objects[entry["key"].(string)] = fakePublicationObject{
			body:         bytes.Repeat([]byte{1}, 100),
			etag:         entry["etag"].(string),
			metadata:     map[string]string{"source-identity": strings.Repeat("d", 64)},
			lastModified: now,
		}
	}
	client.objects[rig["key"].(string)] = fakePublicationObject{
		body: rigBody,
		metadata: map[string]string{
			"sha256":             rig["sha256"].(string),
			"publication-schema": publicationSchema,
		},
		lastModified: now,
	}
	return sha256Hex(body)
}

func testODDScenes() []any {
	return []any{
		map[string]any{
			"scene_uid":          "uid-a",
			"shard_name":         "scene-a-train-000000.tar",
			"start_timestamp_ns": 1_000_000_000,
			"end_timestamp_ns":   21_000_000_000,
			"distance_m":         120.5,
			"observations": []any{
				map[string]any{
					"key": "odd.road.context", "status": "valid",
					"values": []string{"urban"}, "first_timestamp_ns": 5_000_000_000,
				},
				map[string]any{
					"key": "odd.road.context", "status": "valid",
					"values": []string{"urban"}, "first_timestamp_ns": 3_000_000_000,
				},
				map[string]any{
					"key": "odd.environment.sky", "status": "valid",
					"values": []string{"clear"}, "first_timestamp_ns": 1_000_000_000,
				},
			},
		},
		map[string]any{
			"scene_uid":          "uid-b",
			"shard_name":         "scene-b-train-000000.tar",
			"start_timestamp_ns": 0,
			"end_timestamp_ns":   11_000_000_000,
			"distance_m":         40,
			"observations": []any{
				// Only valid observations are searchable.
				map[string]any{
					"key": "odd.road.context", "status": "ambiguous",
					"values": []string{"urban"}, "first_timestamp_ns": 0,
				},
				map[string]any{
					"key": "odd.road.context", "status": "valid",
					"values": []string{"rural"}, "first_timestamp_ns": 0,
				},
				map[string]any{
					"key": "odd.environment.sky", "status": "valid",
					"values": []string{"clear"}, "first_timestamp_ns": 0,
				},
			},
		},
	}
}

type testLabelSet struct {
	version          string
	publicationSHA   string
	scenes           []any
	tamperIndex      bool
	omitSceneIndex   bool
	pointerLabelSet  string
	pointerVersion   string
	manifestSceneCnt int
}

// addTestLabelSet writes pointer, manifest and scene index for a LabelSet of
// the kitscenes publication at spec.version.
func addTestLabelSet(t *testing.T, client *fakePublicationS3, spec testLabelSet) {
	t.Helper()
	root := fmt.Sprintf("kitscenes/%s/odd/labelsets/%s/", spec.version, testLabelSetID)
	index := mustJSON(t, map[string]any{
		"schema_version": oddSceneIndexSchema,
		"labelset_id":    testLabelSetID,
		"scenes":         spec.scenes,
	})
	sceneCount := len(spec.scenes)
	if spec.manifestSceneCnt != 0 {
		sceneCount = spec.manifestSceneCnt
	}
	manifest := mustJSON(t, map[string]any{
		"schema_version":          oddManifestSchema,
		"status":                  "ready",
		"labelset_id":             testLabelSetID,
		"dataset_name":            "kitscenes",
		"dataset_version":         spec.version,
		"dataset_manifest_sha256": spec.publicationSHA,
		"scene_count":             sceneCount,
		"quality": map[string]any{
			"structural_status":    "passed",
			"audit_status":         "pending_human_audit",
			"certification_status": "experimental",
		},
		"artifacts": map[string]any{
			"scene_index": map[string]any{
				"key":       root + "scene_index.json",
				"sha256":    sha256Hex(index),
				"byte_size": len(index),
			},
		},
	})
	pointerLabelSet := testLabelSetID
	if spec.pointerLabelSet != "" {
		pointerLabelSet = spec.pointerLabelSet
	}
	pointerVersion := spec.version
	if spec.pointerVersion != "" {
		pointerVersion = spec.pointerVersion
	}
	pointer := mustJSON(t, map[string]any{
		"schema_version":  oddPointerSchema,
		"status":          "ready",
		"dataset_name":    "kitscenes",
		"dataset_version": pointerVersion,
		"labelset_id":     pointerLabelSet,
		"manifest_key":    fmt.Sprintf("kitscenes/%s/odd/labelsets/%s/manifest.json", pointerVersion, pointerLabelSet),
		"manifest_sha256": sha256Hex(manifest),
	})
	client.objects[fmt.Sprintf("kitscenes/%s/odd/latest.json", spec.version)] = fakePublicationObject{body: pointer}
	client.objects[root+"manifest.json"] = fakePublicationObject{body: manifest}
	if spec.omitSceneIndex {
		return
	}
	if spec.tamperIndex {
		index = append([]byte(nil), index...)
		index[len(index)-2] = ' '
	}
	client.objects[root+"scene_index.json"] = fakePublicationObject{body: index}
}

func TestSearchODDScenesCarriesLabelsToSameScenePack(t *testing.T) {
	service, client := newODDTestService()
	labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
	addTestPublication(t, client, "v3.7", [2]int{10, 5})
	addTestLabelSet(t, client, testLabelSet{
		version: "v3.0", publicationSHA: labelSHA, scenes: testODDScenes(),
	})

	response, err := service.SearchODDScenes(
		context.Background(), "kitscenes", "v3.7",
		[]model.ODDCondition{{Key: "odd.road.context", Value: "urban"}},
		50, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if response.Version != "v3.7" || response.LabelVersion != "v3.0" ||
		response.LabelSetID != testLabelSetID || response.LabeledScenes != 2 {
		t.Fatalf("unexpected coordinates: %+v", response)
	}
	if response.Quality.CertificationStatus != "experimental" {
		t.Fatalf("quality = %+v", response.Quality)
	}
	if len(response.Scenes) != 1 || response.Page.Total != 1 {
		t.Fatalf("scenes = %+v, ambiguous observation must not match", response.Scenes)
	}
	scene := response.Scenes[0]
	if scene.SceneUID != "uid-a" || scene.Shard != "scene-a-train-000000.tar" ||
		scene.DurationNS != 20_000_000_000 || scene.DistanceM != 120.5 {
		t.Fatalf("scene = %+v", scene)
	}
	if len(scene.Matches) != 1 || scene.Matches[0].FirstOffsetNS != 2_000_000_000 {
		t.Fatalf("matches = %+v, want earliest offset 2s", scene.Matches)
	}
	wantFacets := []model.ODDFacet{
		{Key: "odd.environment.sky", Values: []model.ODDFacetValue{{Value: "clear", Scenes: 1}}},
		{Key: "odd.road.context", Values: []model.ODDFacetValue{{Value: "urban", Scenes: 1}}},
	}
	if fmt.Sprint(response.Facets) != fmt.Sprint(wantFacets) {
		t.Fatalf("facets = %+v, want %+v", response.Facets, wantFacets)
	}
}

func TestSearchODDScenesWithoutConditionsCountsAllScenes(t *testing.T) {
	service, client := newODDTestService()
	labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
	addTestLabelSet(t, client, testLabelSet{
		version: "v3.0", publicationSHA: labelSHA, scenes: testODDScenes(),
	})

	response, err := service.SearchODDScenes(
		context.Background(), "kitscenes", "v3.0", nil, 1, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	if response.Page.Total != 2 || len(response.Scenes) != 1 ||
		response.Scenes[0].SceneUID != "uid-b" || response.Page.More {
		t.Fatalf("page = %+v scenes = %+v", response.Page, response.Scenes)
	}
	wantContext := []model.ODDFacetValue{
		{Value: "rural", Scenes: 1}, {Value: "urban", Scenes: 1},
	}
	sky := response.Facets[0]
	if sky.Key != "odd.environment.sky" || sky.Values[0].Scenes != 2 ||
		fmt.Sprint(response.Facets[1].Values) != fmt.Sprint(wantContext) {
		t.Fatalf("facets = %+v", response.Facets)
	}
}

func TestSearchODDScenesRequiresEveryCondition(t *testing.T) {
	service, client := newODDTestService()
	labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
	addTestLabelSet(t, client, testLabelSet{
		version: "v3.0", publicationSHA: labelSHA, scenes: testODDScenes(),
	})

	response, err := service.SearchODDScenes(
		context.Background(), "kitscenes", "v3.0",
		[]model.ODDCondition{
			{Key: "odd.environment.sky", Value: "clear"},
			{Key: "odd.road.context", Value: "rural"},
		},
		50, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Scenes) != 1 || response.Scenes[0].SceneUID != "uid-b" ||
		len(response.Scenes[0].Matches) != 2 {
		t.Fatalf("scenes = %+v", response.Scenes)
	}
}

func TestSearchODDScenesReportsNoLabels(t *testing.T) {
	cases := map[string]func(*testing.T, *fakePublicationS3){
		"no pointer": func(t *testing.T, client *fakePublicationS3) {
			addTestPublication(t, client, "v3.7", [2]int{10, 5})
		},
		"different scene inventory": func(t *testing.T, client *fakePublicationS3) {
			labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
			addTestPublication(t, client, "v3.7", [2]int{10, 6})
			addTestLabelSet(t, client, testLabelSet{
				version: "v3.0", publicationSHA: labelSHA, scenes: testODDScenes(),
			})
		},
		"same total, different samples per partition": func(t *testing.T, client *fakePublicationS3) {
			labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
			addTestPublication(t, client, "v3.7", [2]int{5, 10})
			addTestLabelSet(t, client, testLabelSet{
				version: "v3.0", publicationSHA: labelSHA, scenes: testODDScenes(),
			})
		},
		"same shards and counts, different partition ids": func(t *testing.T, client *fakePublicationS3) {
			labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
			addTestPublication(t, client, "v3.7", [2]int{10, 5}, "scene-a", "scene-c")
			addTestLabelSet(t, client, testLabelSet{
				version: "v3.0", publicationSHA: labelSHA, scenes: testODDScenes(),
			})
		},
		"labels only on a newer version": func(t *testing.T, client *fakePublicationS3) {
			addTestPublication(t, client, "v3.7", [2]int{10, 5})
			labelSHA := addTestPublication(t, client, "v3.8", [2]int{10, 5})
			addTestLabelSet(t, client, testLabelSet{
				version: "v3.8", publicationSHA: labelSHA, scenes: testODDScenes(),
			})
		},
	}
	for name, setup := range cases {
		t.Run(name, func(t *testing.T) {
			service, client := newODDTestService()
			setup(t, client)
			_, err := service.SearchODDScenes(
				context.Background(), "kitscenes", "v3.7", nil, 50, 0,
			)
			if !errors.Is(err, ErrNotFound) {
				t.Fatalf("err = %v, want ErrNotFound", err)
			}
		})
	}
}

func TestSearchODDScenesRejectsUnverifiedLabelSet(t *testing.T) {
	cases := map[string]func(*testing.T, *fakePublicationS3){
		"tampered scene index": func(t *testing.T, client *fakePublicationS3) {
			labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
			addTestLabelSet(t, client, testLabelSet{
				version: "v3.0", publicationSHA: labelSHA,
				scenes: testODDScenes(), tamperIndex: true,
			})
		},
		"missing scene index": func(t *testing.T, client *fakePublicationS3) {
			labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
			addTestLabelSet(t, client, testLabelSet{
				version: "v3.0", publicationSHA: labelSHA,
				scenes: testODDScenes(), omitSceneIndex: true,
			})
		},
		"labels for another publication": func(t *testing.T, client *fakePublicationS3) {
			addTestPublication(t, client, "v3.0", [2]int{10, 5})
			addTestLabelSet(t, client, testLabelSet{
				version: "v3.0", publicationSHA: strings.Repeat("e", 64),
				scenes: testODDScenes(),
			})
		},
		"scene count differs from publication": func(t *testing.T, client *fakePublicationS3) {
			labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
			addTestLabelSet(t, client, testLabelSet{
				version: "v3.0", publicationSHA: labelSHA,
				scenes: testODDScenes()[:1], manifestSceneCnt: 1,
			})
		},
		"scene outside publication": func(t *testing.T, client *fakePublicationS3) {
			labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
			scenes := testODDScenes()
			scenes[1].(map[string]any)["shard_name"] = "scene-z-train-000000.tar"
			addTestLabelSet(t, client, testLabelSet{
				version: "v3.0", publicationSHA: labelSHA, scenes: scenes,
			})
		},
		"value outside the search contract": func(t *testing.T, client *fakePublicationS3) {
			labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
			scenes := testODDScenes()
			observation := scenes[0].(map[string]any)["observations"].([]any)[0].(map[string]any)
			observation["values"] = []string{"urban", "Urban"}
			addTestLabelSet(t, client, testLabelSet{
				version: "v3.0", publicationSHA: labelSHA, scenes: scenes,
			})
		},
		"label before its scene starts": func(t *testing.T, client *fakePublicationS3) {
			labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
			scenes := testODDScenes()
			observation := scenes[0].(map[string]any)["observations"].([]any)[2].(map[string]any)
			delete(observation, "first_timestamp_ns")
			addTestLabelSet(t, client, testLabelSet{
				version: "v3.0", publicationSHA: labelSHA, scenes: scenes,
			})
		},
		"pointer for another version": func(t *testing.T, client *fakePublicationS3) {
			labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
			addTestLabelSet(t, client, testLabelSet{
				version: "v3.0", publicationSHA: labelSHA,
				scenes: testODDScenes(), pointerVersion: "v2.9",
			})
		},
		"malformed labelset id": func(t *testing.T, client *fakePublicationS3) {
			labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
			addTestLabelSet(t, client, testLabelSet{
				version: "v3.0", publicationSHA: labelSHA,
				scenes: testODDScenes(), pointerLabelSet: "oddls-../../x",
			})
		},
	}
	for name, setup := range cases {
		t.Run(name, func(t *testing.T) {
			service, client := newODDTestService()
			setup(t, client)
			_, err := service.SearchODDScenes(
				context.Background(), "kitscenes", "v3.0", nil, 50, 0,
			)
			if err == nil || errors.Is(err, ErrNotFound) {
				t.Fatalf("err = %v, want a validation error", err)
			}
		})
	}
}

func TestSameSceneInventoryComparesShardNames(t *testing.T) {
	manifest := func(names ...string) *publicationManifest {
		out := &publicationManifest{TotalSamples: 10}
		for _, name := range names {
			out.ShardEntries = append(out.ShardEntries, publicationShardEntry{Name: name})
			out.Partitions = append(out.Partitions, publicationPartition{
				PartitionID: strings.TrimSuffix(name, ".tar"), SampleCount: 5,
			})
		}
		return out
	}
	if !sameSceneInventory(manifest("a.tar", "b.tar"), manifest("a.tar", "b.tar")) {
		t.Fatal("identical inventories differ")
	}
	left, right := manifest("a.tar", "b.tar"), manifest("a.tar", "b.tar")
	right.ShardEntries[1].Name = "c.tar"
	if sameSceneInventory(left, right) {
		t.Fatal("different shard names compared equal")
	}
	if sameSceneInventory(&publicationManifest{TotalSamples: 10}, &publicationManifest{TotalSamples: 10}) {
		t.Fatal("publications without partitions compared equal")
	}
	for name, mutate := range map[string]func(*publicationManifest){
		"empty partition id": func(m *publicationManifest) { m.Partitions[0].PartitionID = "" },
		"duplicate partition id": func(m *publicationManifest) {
			m.Partitions[1].PartitionID = m.Partitions[0].PartitionID
		},
		"partition total differs from samples": func(m *publicationManifest) {
			m.Partitions[0].SampleCount, m.Partitions[1].SampleCount = 4, 4
		},
	} {
		left, right := manifest("a.tar", "b.tar"), manifest("a.tar", "b.tar")
		mutate(left)
		mutate(right)
		if sameSceneInventory(left, right) {
			t.Errorf("%s compared equal", name)
		}
	}
}

func TestSearchODDScenesCachesLabelSet(t *testing.T) {
	service, client := newODDTestService()
	labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
	addTestLabelSet(t, client, testLabelSet{
		version: "v3.0", publicationSHA: labelSHA, scenes: testODDScenes(),
	})
	indexKey := fmt.Sprintf(
		"kitscenes/v3.0/odd/labelsets/%s/scene_index.json", testLabelSetID,
	)
	now := time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC)
	restore := nowFunc
	nowFunc = func() time.Time { return now }
	t.Cleanup(func() { nowFunc = restore })
	addTestPublication(t, client, "v3.7", [2]int{10, 5})
	pointerKey := "kitscenes/v3.0/odd/latest.json"
	for _, version := range []string{"v3.0", "v3.0", "v3.7"} {
		if _, err := service.SearchODDScenes(
			context.Background(), "kitscenes", version, nil, 50, 0,
		); err != nil {
			t.Fatal(err)
		}
	}
	// v3.7 resolves its own pointer but reuses the parsed v3.0 LabelSet.
	if client.getCalls[indexKey] != 1 || client.getCalls[pointerKey] != 2 {
		t.Fatalf(
			"scene index GETs = %d, pointer GETs = %d, want 1 and 2",
			client.getCalls[indexKey], client.getCalls[pointerKey],
		)
	}
	now = now.Add(versionTTL)
	if _, err := service.SearchODDScenes(
		context.Background(), "kitscenes", "v3.0", nil, 50, 0,
	); err != nil {
		t.Fatal(err)
	}
	if client.getCalls[pointerKey] != 3 || client.getCalls[indexKey] != 1 {
		t.Fatalf(
			"after TTL pointer GETs = %d, index GETs = %d, want 3 and 1",
			client.getCalls[pointerKey], client.getCalls[indexKey],
		)
	}
}

func TestSearchODDScenesCachesNoLabelsUntilTTL(t *testing.T) {
	service, client := newODDTestService()
	now := time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC)
	restore := nowFunc
	nowFunc = func() time.Time { return now }
	t.Cleanup(func() { nowFunc = restore })
	labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
	search := func() error {
		_, err := service.SearchODDScenes(
			context.Background(), "kitscenes", "v3.0", nil, 50, 0,
		)
		return err
	}
	if err := search(); !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v, want ErrNotFound", err)
	}
	addTestLabelSet(t, client, testLabelSet{
		version: "v3.0", publicationSHA: labelSHA, scenes: testODDScenes(),
	})
	if err := search(); !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v, want the cached ErrNotFound within TTL", err)
	}
	now = now.Add(versionTTL)
	if err := search(); err != nil {
		t.Fatalf("err = %v, want labels after TTL", err)
	}
}

func TestSearchODDScenesDoesNotCacheErrors(t *testing.T) {
	service, client := newODDTestService()
	labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
	addTestLabelSet(t, client, testLabelSet{
		version: "v3.0", publicationSHA: labelSHA,
		scenes: testODDScenes(), omitSceneIndex: true,
	})
	search := func() error {
		_, err := service.SearchODDScenes(
			context.Background(), "kitscenes", "v3.0", nil, 50, 0,
		)
		return err
	}
	if err := search(); err == nil || errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v, want a read error", err)
	}
	addTestLabelSet(t, client, testLabelSet{
		version: "v3.0", publicationSHA: labelSHA, scenes: testODDScenes(),
	})
	if err := search(); err != nil {
		t.Fatalf("err = %v, want success once the index exists", err)
	}
}

func TestSearchODDScenesDropsReplacedLabelSet(t *testing.T) {
	service, client := newODDTestService()
	now := time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC)
	restore := nowFunc
	nowFunc = func() time.Time { return now }
	t.Cleanup(func() { nowFunc = restore })
	labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
	search := func() *model.ODDSearchResponse {
		t.Helper()
		response, err := service.SearchODDScenes(
			context.Background(), "kitscenes", "v3.0", nil, 50, 0,
		)
		if err != nil {
			t.Fatal(err)
		}
		return response
	}
	addTestLabelSet(t, client, testLabelSet{
		version: "v3.0", publicationSHA: labelSHA, scenes: testODDScenes(),
	})
	first := search().ManifestSHA256
	scenes := testODDScenes()
	scenes[0].(map[string]any)["distance_m"] = 99.0
	addTestLabelSet(t, client, testLabelSet{
		version: "v3.0", publicationSHA: labelSHA, scenes: scenes,
	})
	now = now.Add(versionTTL)
	second := search().ManifestSHA256
	if first == second {
		t.Fatal("replacement LabelSet has the same manifest digest")
	}
	if len(service.oddLabelSets) != 1 || service.oddLabelSets[second] == nil {
		t.Fatalf("parsed LabelSets = %d, want only the replacement", len(service.oddLabelSets))
	}
}
