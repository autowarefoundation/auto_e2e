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

	"github.com/parquet-go/parquet-go"

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

const sec = int64(1_000_000_000)

func row(key, status string, start, end int64, confidence float32, source string, values ...string) map[string]any {
	return map[string]any{
		"key": key, "status": status, "values": values,
		"start": start, "end": end, "confidence": confidence, "source": source,
	}
}

// testODDScenes is two scenes. Scene a runs 1-21 s, scene b 0-11 s.
func testODDScenes() []any {
	return []any{
		map[string]any{
			"scene_uid":          "uid-a",
			"shard_name":         "scene-a-train-000000.tar",
			"start_timestamp_ns": 1 * sec,
			"end_timestamp_ns":   21 * sec,
			"distance_m":         120.5,
			"observations": []any{
				row("odd.road.context", "valid", 5*sec, 9*sec, 0.9, "map_route", "urban"),
				row("odd.road.context", "valid", 3*sec, 4*sec, 0.6, "vlm", "urban"),
				row("odd.environment.sky", "valid", 1*sec, 21*sec, 0.95, "vlm", "clear"),
			},
		},
		map[string]any{
			"scene_uid":          "uid-b",
			"shard_name":         "scene-b-train-000000.tar",
			"start_timestamp_ns": 0,
			"end_timestamp_ns":   11 * sec,
			"distance_m":         40,
			"observations": []any{
				// Only valid observations are searchable.
				row("odd.road.context", "ambiguous", 0, 2*sec, 0.5, "vlm", "urban"),
				row("odd.road.context", "valid", 0, 11*sec, 0.8, "map_route", "rural"),
				row("odd.environment.sky", "valid", 0, 11*sec, 0.9, "vlm", "clear"),
				row("odd.environment.day_phase", "valid", 0, 11*sec, 0.7, "vlm", "dusk"),
			},
		},
	}
}

func testObservationRows(scenes []any) []oddObservationRow {
	rows := make([]oddObservationRow, 0)
	for _, raw := range scenes {
		scene := raw.(map[string]any)
		for _, rawRow := range scene["observations"].([]any) {
			r := rawRow.(map[string]any)
			uid := scene["scene_uid"].(string)
			if override, ok := r["scene_uid"].(string); ok {
				uid = override
			}
			var camera *string
			if c, ok := r["camera_id"].(string); ok {
				camera = &c
			}
			rows = append(rows, oddObservationRow{
				SceneUID:   uid,
				Key:        r["key"].(string),
				Status:     r["status"].(string),
				Values:     r["values"].([]string),
				Confidence: r["confidence"].(float32),
				Source:     r["source"].(string),
				Start:      r["start"].(int64),
				End:        r["end"].(int64),
				CameraID:   camera,
			})
		}
	}
	return rows
}

func encodeObservations(t *testing.T, rows []oddObservationRow) []byte {
	t.Helper()
	var buffer bytes.Buffer
	writer := parquet.NewGenericWriter[oddObservationRow](&buffer)
	if _, err := writer.Write(rows); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return buffer.Bytes()
}

type testLabelSet struct {
	version            string
	publicationSHA     string
	scenes             []any
	tamperIndex        bool
	omitSceneIndex     bool
	tamperObservations bool
	rowCountDelta      int
	pointerLabelSet    string
	pointerVersion     string
	manifestSceneCnt   int
}

// addTestLabelSet writes pointer, manifest, scene index and observations for
// a LabelSet of the kitscenes publication at spec.version.
func addTestLabelSet(t *testing.T, client *fakePublicationS3, spec testLabelSet) {
	t.Helper()
	root := fmt.Sprintf("kitscenes/%s/odd/labelsets/%s/", spec.version, testLabelSetID)
	index := mustJSON(t, map[string]any{
		"schema_version": oddSceneIndexSchema,
		"labelset_id":    testLabelSetID,
		"scenes":         spec.scenes,
	})
	rows := testObservationRows(spec.scenes)
	observations := encodeObservations(t, rows)
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
			"observations_parquet": map[string]any{
				"key":       root + "observations/part-00000.parquet",
				"sha256":    sha256Hex(observations),
				"byte_size": len(observations),
				"row_count": len(rows) + spec.rowCountDelta,
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
	if spec.tamperObservations {
		observations = append([]byte(nil), observations...)
		observations[len(observations)/2] ^= 0xff
	}
	client.objects[root+"observations/part-00000.parquet"] = fakePublicationObject{body: observations}
	if spec.omitSceneIndex {
		return
	}
	if spec.tamperIndex {
		index = append([]byte(nil), index...)
		index[len(index)-2] = ' '
	}
	client.objects[root+"scene_index.json"] = fakePublicationObject{body: index}
}

func cond(key string, values ...string) model.ODDQueryCondition {
	return model.ODDQueryCondition{Key: key, Values: values}
}

func query(op string, groups ...model.ODDQueryGroup) model.ODDQuery {
	return model.ODDQuery{Op: op, Groups: append([]model.ODDQueryGroup{}, groups...)}
}

func group(op string, conditions ...model.ODDQueryCondition) model.ODDQueryGroup {
	return model.ODDQueryGroup{Op: op, Conditions: conditions}
}

func newODDFixture(t *testing.T) *S3Service {
	t.Helper()
	service, client := newODDTestService()
	labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
	addTestLabelSet(t, client, testLabelSet{
		version: "v3.0", publicationSHA: labelSHA, scenes: testODDScenes(),
	})
	return service
}

func searchUIDs(t *testing.T, service *S3Service, q model.ODDQuery) []string {
	t.Helper()
	response, err := service.SearchODDScenes(context.Background(), "kitscenes", "v3.0", q, 50, 0)
	if err != nil {
		t.Fatal(err)
	}
	uids := make([]string, len(response.Scenes))
	for i, scene := range response.Scenes {
		uids[i] = scene.SceneUID
	}
	return uids
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
		query("and", group("and", cond("odd.road.context", "urban"))), 50, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if response.Version != "v3.7" || response.LabelVersion != "v3.0" ||
		response.LabelSetID != testLabelSetID || response.LabeledScenes != 2 ||
		response.Quality.CertificationStatus != "experimental" {
		t.Fatalf("unexpected coordinates: %+v", response.ODDLabelSetInfo)
	}
	if len(response.Scenes) != 1 || response.Page.Total != 1 {
		t.Fatalf("scenes = %+v, ambiguous observation must not match", response.Scenes)
	}
	scene := response.Scenes[0]
	if scene.SceneUID != "uid-a" || scene.Shard != "scene-a-train-000000.tar" ||
		scene.DurationNS != 20*sec || scene.DistanceM != 120.5 {
		t.Fatalf("scene = %+v", scene)
	}
	want := []model.ODDLane{{Key: "odd.road.context", Intervals: []model.ODDInterval{
		{StartNS: 2 * sec, EndNS: 3 * sec, Values: []string{"urban"}, Source: "vlm", Confidence: float64(float32(0.6))},
		{StartNS: 4 * sec, EndNS: 8 * sec, Values: []string{"urban"}, Source: "map_route", Confidence: float64(float32(0.9))},
	}}}
	if fmt.Sprint(scene.Lanes) != fmt.Sprint(want) {
		t.Fatalf("lanes = %+v, want %+v (offsets from scene start, in start order)", scene.Lanes, want)
	}
	if len(response.Facets) != 2 || len(response.Vocabulary) != 3 {
		t.Fatalf("facets = %d keys, vocabulary = %d keys; want matched 2, all 3",
			len(response.Facets), len(response.Vocabulary))
	}
}

func TestSearchODDScenesWithoutConditionsCountsAllScenes(t *testing.T) {
	service := newODDFixture(t)
	response, err := service.SearchODDScenes(
		context.Background(), "kitscenes", "v3.0", query("and"), 1, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	if response.Page.Total != 2 || len(response.Scenes) != 1 ||
		response.Scenes[0].SceneUID != "uid-b" || response.Page.More ||
		len(response.Scenes[0].Lanes) != 0 {
		t.Fatalf("page = %+v scenes = %+v", response.Page, response.Scenes)
	}
	var roadContext model.ODDFacet
	for _, facet := range response.Facets {
		if facet.Key == "odd.road.context" {
			roadContext = facet
		}
	}
	wantValues := []model.ODDFacetValue{{Value: "rural", Scenes: 1}, {Value: "urban", Scenes: 1}}
	if fmt.Sprint(roadContext.Values) != fmt.Sprint(wantValues) ||
		fmt.Sprint(roadContext.Sources) != "[map_route vlm]" {
		t.Fatalf("context facet = %+v", roadContext)
	}
}

func TestMatchODDQueryOperators(t *testing.T) {
	service := newODDFixture(t)
	withDuration := cond("odd.road.context", "urban")
	withDuration.MinDurationS = 2
	withConfidence := cond("odd.road.context", "urban")
	withConfidence.MinConfidence = 0.95
	fromVLM := cond("odd.road.context", "rural")
	fromVLM.Sources = []string{"vlm"}
	notDusk := cond("odd.environment.day_phase", "dusk")
	notDusk.Not = true
	cases := map[string]struct {
		q    model.ODDQuery
		want string
	}{
		"AND within a group":        {query("and", group("and", cond("odd.environment.sky", "clear"), cond("odd.road.context", "rural"))), "[uid-b]"},
		"OR within a group":         {query("and", group("or", cond("odd.road.context", "urban"), cond("odd.environment.day_phase", "dusk"))), "[uid-a uid-b]"},
		"IN over values":            {query("and", group("and", cond("odd.road.context", "rural", "motorway"))), "[uid-b]"},
		"NOT":                       {query("and", group("and", notDusk)), "[uid-a]"},
		"min duration":              {query("and", group("and", withDuration)), "[uid-a]"},
		"min confidence":            {query("and", group("and", withConfidence)), "[]"},
		"source":                    {query("and", group("and", fromVLM)), "[]"},
		"OR across groups":          {query("or", group("and", cond("odd.road.context", "urban")), group("and", cond("odd.road.context", "rural"))), "[uid-a uid-b]"},
		"AND across groups":         {query("and", group("and", cond("odd.environment.sky", "clear")), group("or", notDusk)), "[uid-a]"},
		"value never labelled":      {query("and", group("and", cond("odd.road.context", "motorway"))), "[]"},
		"key absent from the scene": {query("and", group("and", cond("odd.weather.fog", "dense"))), "[]"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if got := fmt.Sprint(searchUIDs(t, service, tc.q)); got != tc.want {
				t.Fatalf("scenes = %s, want %s", got, tc.want)
			}
		})
	}
}

func TestMatchODDQueryLanesKeepOnlySatisfyingIntervals(t *testing.T) {
	service := newODDFixture(t)
	long := cond("odd.road.context", "urban")
	long.MinDurationS = 2
	notDusk := cond("odd.environment.day_phase", "dusk")
	notDusk.Not = true
	response, err := service.SearchODDScenes(
		context.Background(), "kitscenes", "v3.0",
		// The failed OR branch (rural) and the NOT condition add no lane.
		query("and", group("and", long, notDusk), group("or", cond("odd.road.context", "urban"), cond("odd.road.context", "rural"))),
		50, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Scenes) != 1 {
		t.Fatalf("scenes = %+v", response.Scenes)
	}
	lanes := response.Scenes[0].Lanes
	if len(lanes) != 1 || lanes[0].Key != "odd.road.context" || len(lanes[0].Intervals) != 2 {
		t.Fatalf("lanes = %+v, want both urban intervals once in one lane", lanes)
	}
}

func TestODDSceneReturnsEveryLabel(t *testing.T) {
	service := newODDFixture(t)
	response, err := service.ODDScene(context.Background(), "kitscenes", "v3.0", "uid-b")
	if err != nil {
		t.Fatal(err)
	}
	keys := make([]string, len(response.Scene.Lanes))
	for i, lane := range response.Scene.Lanes {
		keys[i] = lane.Key
	}
	if fmt.Sprint(keys) != "[odd.environment.day_phase odd.environment.sky odd.road.context]" ||
		len(response.Scene.Lanes[2].Intervals) != 1 {
		t.Fatalf("lanes = %+v, want sorted keys and no ambiguous interval", response.Scene.Lanes)
	}
	if _, err := service.ODDScene(context.Background(), "kitscenes", "v3.0", "uid-z"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unknown scene err = %v, want ErrNotFound", err)
	}
}

func TestValidateODDQuery(t *testing.T) {
	tooManyValues := cond("odd.road.context")
	for range maxODDValues + 1 {
		tooManyValues.Values = append(tooManyValues.Values, "urban")
	}
	negative := cond("odd.road.context", "urban")
	negative.MinDurationS = -1
	overConfident := cond("odd.road.context", "urban")
	overConfident.MinConfidence = 1.5
	badSource := cond("odd.road.context", "urban")
	badSource.Sources = []string{"VLM"}
	manyGroups := query("and")
	for range MaxODDGroups + 1 {
		manyGroups.Groups = append(manyGroups.Groups, group("and", cond("odd.road.context", "urban")))
	}
	manyConditions := query("and")
	for range 3 {
		g := group("and")
		for range 9 {
			g.Conditions = append(g.Conditions, cond("odd.road.context", "urban"))
		}
		manyConditions.Groups = append(manyConditions.Groups, g)
	}
	for name, q := range map[string]model.ODDQuery{
		"bad top op":      {Op: "xor"},
		"bad group op":    query("and", group("nand", cond("odd.road.context", "urban"))),
		"empty group":     query("and", group("and")),
		"bad key":         query("and", group("and", cond("road", "urban"))),
		"no values":       query("and", group("and", cond("odd.road.context"))),
		"bad value":       query("and", group("and", cond("odd.road.context", "Urban"))),
		"too many values": query("and", group("and", tooManyValues)),
		"negative min":    query("and", group("and", negative)),
		"confidence > 1":  query("and", group("and", overConfident)),
		"bad source":      query("and", group("and", badSource)),
		"too many groups": manyGroups,
		"too many in all": manyConditions,
	} {
		var invalid ErrInvalidODDQuery
		if err := ValidateODDQuery(q); !errors.As(err, &invalid) {
			t.Errorf("%s: err = %v, want ErrInvalidODDQuery", name, err)
		}
	}
	if err := ValidateODDQuery(query("or")); err != nil {
		t.Errorf("empty query: %v", err)
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
				context.Background(), "kitscenes", "v3.7", query("and"), 50, 0,
			)
			if !errors.Is(err, ErrNotFound) {
				t.Fatalf("err = %v, want ErrNotFound", err)
			}
		})
	}
}

func TestSearchODDScenesRejectsUnverifiedLabelSet(t *testing.T) {
	withRow := func(mutate func(map[string]any)) []any {
		scenes := testODDScenes()
		mutate(scenes[0].(map[string]any)["observations"].([]any)[0].(map[string]any))
		return scenes
	}
	cases := map[string]testLabelSet{
		"tampered scene index":           {tamperIndex: true},
		"missing scene index":            {omitSceneIndex: true},
		"tampered observations":          {tamperObservations: true},
		"row count differs":              {rowCountDelta: 1},
		"labels for another publication": {publicationSHA: strings.Repeat("e", 64)},
		"scene count differs from publication": {
			scenes: testODDScenes()[:1], manifestSceneCnt: 1,
		},
		"pointer for another version": {pointerVersion: "v2.9"},
		"malformed labelset id":       {pointerLabelSet: "oddls-../../x"},
		"value outside the search contract": {scenes: withRow(func(r map[string]any) {
			r["values"] = []string{"urban", "Urban"}
		})},
		"label before its scene starts": {scenes: withRow(func(r map[string]any) { r["start"] = int64(0) })},
		"label after its scene ends":    {scenes: withRow(func(r map[string]any) { r["end"] = 30 * sec })},
		"confidence above one":          {scenes: withRow(func(r map[string]any) { r["confidence"] = float32(1.5) })},
		"row for an unknown scene":      {scenes: withRow(func(r map[string]any) { r["scene_uid"] = "uid-z" })},
	}
	for name, spec := range cases {
		t.Run(name, func(t *testing.T) {
			service, client := newODDTestService()
			labelSHA := addTestPublication(t, client, "v3.0", [2]int{10, 5})
			spec.version = "v3.0"
			if spec.publicationSHA == "" {
				spec.publicationSHA = labelSHA
			}
			if spec.scenes == nil {
				spec.scenes = testODDScenes()
			}
			addTestLabelSet(t, client, spec)
			_, err := service.SearchODDScenes(
				context.Background(), "kitscenes", "v3.0", query("and"), 50, 0,
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
	observationsKey := fmt.Sprintf(
		"kitscenes/v3.0/odd/labelsets/%s/observations/part-00000.parquet", testLabelSetID,
	)
	now := time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC)
	restore := nowFunc
	nowFunc = func() time.Time { return now }
	t.Cleanup(func() { nowFunc = restore })
	addTestPublication(t, client, "v3.7", [2]int{10, 5})
	pointerKey := "kitscenes/v3.0/odd/latest.json"
	for _, version := range []string{"v3.0", "v3.0", "v3.7"} {
		if _, err := service.SearchODDScenes(
			context.Background(), "kitscenes", version, query("and"), 50, 0,
		); err != nil {
			t.Fatal(err)
		}
	}
	// v3.7 resolves its own pointer but reuses the parsed v3.0 LabelSet.
	if client.getCalls[observationsKey] != 1 || client.getCalls[pointerKey] != 2 {
		t.Fatalf(
			"observations GETs = %d, pointer GETs = %d, want 1 and 2",
			client.getCalls[observationsKey], client.getCalls[pointerKey],
		)
	}
	now = now.Add(versionTTL)
	if _, err := service.SearchODDScenes(
		context.Background(), "kitscenes", "v3.0", query("and"), 50, 0,
	); err != nil {
		t.Fatal(err)
	}
	if client.getCalls[pointerKey] != 3 || client.getCalls[observationsKey] != 1 {
		t.Fatalf(
			"after TTL pointer GETs = %d, observations GETs = %d, want 3 and 1",
			client.getCalls[pointerKey], client.getCalls[observationsKey],
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
			context.Background(), "kitscenes", "v3.0", query("and"), 50, 0,
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
			context.Background(), "kitscenes", "v3.0", query("and"), 50, 0,
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
			context.Background(), "kitscenes", "v3.0", query("and"), 50, 0,
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
