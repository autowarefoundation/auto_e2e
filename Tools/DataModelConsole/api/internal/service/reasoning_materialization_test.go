package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/store"
)

const (
	testGenerationA = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	testGenerationB = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
	testLeaseOwnerA = "1111111111111111111111111111111111111111111111111111111111111111"
	testLeaseOwnerB = "2222222222222222222222222222222222222222222222222222222222222222"
	testManifestSHA = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
)

type fakeReasoningStore struct {
	inventories map[string]model.ReasoningInventory
	stats       map[string]model.ReasoningStatsBlob
	scenes      map[string][]model.SceneRef
	lookups     map[string]model.ReasoningSampleLookup
	indices     map[string]*model.ShardIndex
	owners      map[string]string
	expires     map[string]int64
	operations  []string

	failStats         bool
	getShardIndexCall int
	maxSceneBatch     int
	maxLookupBatch    int
}

func reasoningStoreKey(parts ...string) string {
	key := ""
	for _, part := range parts {
		key += "\x00" + part
	}
	return key
}

func (f *fakeReasoningStore) GetShardIndex(
	_ context.Context,
	dataset, version, shard string,
) (*model.ShardIndex, error) {
	f.getShardIndexCall++
	index := f.indices[reasoningStoreKey(dataset, version, shard)]
	if index == nil {
		return nil, store.ErrNotFound
	}
	return index, nil
}

func (f *fakeReasoningStore) PutShardIndex(
	_ context.Context,
	dataset, version, shard string,
	index *model.ShardIndex,
) error {
	if f.indices == nil {
		f.indices = make(map[string]*model.ShardIndex)
	}
	f.indices[reasoningStoreKey(dataset, version, shard)] = index
	return nil
}

func (f *fakeReasoningStore) GetTeacherStats(
	_ context.Context,
	dataset, version, generation, teacher, prompt string,
) (model.ReasoningStatsBlob, string, error) {
	blob, ok := f.stats[reasoningStoreKey(
		dataset, version, generation, teacher, prompt,
	)]
	if !ok {
		return model.ReasoningStatsBlob{}, "", store.ErrNotFound
	}
	return blob, "2026-07-16T00:00:00Z", nil
}

func (f *fakeReasoningStore) PutTeacherStats(
	_ context.Context,
	dataset, version, generation, teacher, prompt string,
	blob model.ReasoningStatsBlob,
) (string, error) {
	f.operations = append(f.operations, "stats:"+generation)
	if f.failStats {
		return "", errors.New("stats failed")
	}
	if f.stats == nil {
		f.stats = make(map[string]model.ReasoningStatsBlob)
	}
	f.stats[reasoningStoreKey(
		dataset, version, generation, teacher, prompt,
	)] = blob
	return "2026-07-16T00:00:00Z", nil
}

func (f *fakeReasoningStore) GetReasoningInventory(
	_ context.Context,
	dataset, version string,
) (model.ReasoningInventory, string, error) {
	inventory, ok := f.inventories[reasoningStoreKey(dataset, version)]
	if !ok {
		return model.ReasoningInventory{}, "", store.ErrNotFound
	}
	return inventory, "2026-07-16T00:00:00Z", nil
}

func (f *fakeReasoningStore) PutReasoningInventory(
	_ context.Context,
	dataset, version, owner string, nowUnix int64,
	inventory model.ReasoningInventory,
) (string, error) {
	key := reasoningStoreKey(dataset, version)
	if f.owners[key] != owner || f.expires[key] < nowUnix {
		return "", errors.New("materialization owner was fenced")
	}
	f.operations = append(f.operations, "inventory:"+inventory.Generation)
	if f.inventories == nil {
		f.inventories = make(map[string]model.ReasoningInventory)
	}
	f.inventories[reasoningStoreKey(dataset, version)] = inventory
	delete(f.owners, key)
	delete(f.expires, key)
	return "2026-07-16T00:00:00Z", nil
}

func (f *fakeReasoningStore) BeginReasoningMaterialization(
	_ context.Context,
	dataset, version, owner string,
	nowUnix, expiresUnix int64,
) error {
	if f.owners == nil {
		f.owners = make(map[string]string)
	}
	if f.expires == nil {
		f.expires = make(map[string]int64)
	}
	key := reasoningStoreKey(dataset, version)
	if current := f.owners[key]; current != "" &&
		f.expires[key] >= nowUnix {
		return errors.New("materialization lease is held")
	}
	f.owners[key] = owner
	f.expires[key] = expiresUnix
	return nil
}

func (f *fakeReasoningStore) RenewReasoningMaterialization(
	_ context.Context,
	dataset, version, owner string,
	nowUnix, expiresUnix int64,
) error {
	key := reasoningStoreKey(dataset, version)
	if f.owners[key] != owner ||
		f.expires[key] < nowUnix ||
		expiresUnix <= nowUnix {
		return errors.New("materialization owner was fenced")
	}
	f.expires[key] = expiresUnix
	return nil
}

func (f *fakeReasoningStore) ReleaseReasoningMaterialization(
	_ context.Context,
	dataset, version, owner string,
) error {
	key := reasoningStoreKey(dataset, version)
	if f.owners[key] != owner {
		return errors.New("materialization owner was fenced")
	}
	delete(f.owners, key)
	delete(f.expires, key)
	return nil
}

func (f *fakeReasoningStore) PutReasoningSampleLookups(
	_ context.Context,
	dataset, version, generation string,
	lookups []model.ReasoningSampleLookup,
) (int, error) {
	f.operations = append(f.operations, "lookups:"+generation)
	f.maxLookupBatch = max(f.maxLookupBatch, len(lookups))
	if f.lookups == nil {
		f.lookups = make(map[string]model.ReasoningSampleLookup)
	}
	for _, lookup := range lookups {
		f.lookups[reasoningStoreKey(
			dataset, version, generation, lookup.SampleID,
		)] = lookup
	}
	return len(lookups), nil
}

func (f *fakeReasoningStore) GetReasoningSampleLookup(
	_ context.Context,
	dataset, version, generation, sampleID string,
) (model.ReasoningSampleLookup, error) {
	lookup, ok := f.lookups[reasoningStoreKey(
		dataset, version, generation, sampleID,
	)]
	if !ok {
		return model.ReasoningSampleLookup{}, store.ErrNotFound
	}
	return lookup, nil
}

func (f *fakeReasoningStore) PutReasoningSceneLabels(
	_ context.Context,
	dataset, version, generation, teacher, prompt string,
	rows []store.SceneLabelRow,
) (int, error) {
	f.operations = append(f.operations, "scenes:"+generation)
	f.maxSceneBatch = max(f.maxSceneBatch, len(rows))
	if f.scenes == nil {
		f.scenes = make(map[string][]model.SceneRef)
	}
	for _, row := range rows {
		key := reasoningStoreKey(
			dataset,
			version,
			generation,
			teacher,
			prompt,
			row.Field,
			row.Value,
		)
		f.scenes[key] = append(f.scenes[key], model.SceneRef{
			SampleID:  row.SampleID,
			Shard:     row.Shard,
			Available: row.Shard != "",
		})
	}
	return len(rows), nil
}

func (f *fakeReasoningStore) QueryReasoningScenes(
	_ context.Context,
	dataset, version, generation, teacher, prompt, field, value string,
	limit int,
) ([]model.SceneRef, error) {
	scenes := append([]model.SceneRef(nil), f.scenes[reasoningStoreKey(
		dataset, version, generation, teacher, prompt, field, value,
	)]...)
	if limit > 0 && len(scenes) > limit {
		scenes = scenes[:limit]
	}
	return scenes, nil
}

func (f *fakeReasoningStore) QueryReadyOverlayModels(
	context.Context, string, string, string, int, string, ...string,
) ([]model.OverlayModel, string, error) {
	panic("unexpected QueryReadyOverlayModels")
}

func (f *fakeReasoningStore) GetReadyOverlayPointer(
	context.Context, string, string, string, string, ...string,
) (*store.OverlayPointer, error) {
	panic("unexpected GetReadyOverlayPointer")
}

func (f *fakeReasoningStore) GetGeoRecord(
	context.Context, string, string,
) (*store.GeoRecord, error) {
	panic("unexpected GetGeoRecord")
}

type boundedReasoningS3 struct {
	*fakePublicationS3

	mu        sync.Mutex
	active    int
	maxActive int
	listCalls int
	delay     time.Duration
	ifMatches map[string][]string
}

func (s *boundedReasoningS3) GetObject(
	ctx context.Context,
	input *s3.GetObjectInput,
	options ...func(*s3.Options),
) (*s3.GetObjectOutput, error) {
	s.mu.Lock()
	s.active++
	s.maxActive = max(s.maxActive, s.active)
	key := aws.ToString(input.Key)
	if s.ifMatches == nil {
		s.ifMatches = make(map[string][]string)
	}
	s.ifMatches[key] = append(
		s.ifMatches[key], aws.ToString(input.IfMatch),
	)
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.active--
		s.mu.Unlock()
	}()
	if s.delay > 0 {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(s.delay):
		}
	}
	return s.fakePublicationS3.GetObject(ctx, input, options...)
}

func (s *boundedReasoningS3) ListObjectsV2(
	ctx context.Context,
	input *s3.ListObjectsV2Input,
	options ...func(*s3.Options),
) (*s3.ListObjectsV2Output, error) {
	s.mu.Lock()
	s.listCalls++
	s.mu.Unlock()
	return s.fakePublicationS3.ListObjectsV2(ctx, input, options...)
}

func (s *boundedReasoningS3) observations() (int, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxActive, s.listCalls
}

func (s *boundedReasoningS3) objectIfMatches(key string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.ifMatches[key]...)
}

func testReasoningEntry(prompt string, count int) model.ReasoningPromptVersion {
	label := testReasoningLabel("sample", prompt)
	return model.ReasoningPromptVersion{
		Teacher:         reasoningTeacher(label),
		TeacherProvider: label.TeacherProvider,
		TeacherModel:    label.TeacherModel,
		PromptVersion:   prompt,
		Count:           count,
	}
}

func testReasoningLabel(
	sampleID, prompt string,
) store.ReasoningLabel {
	horizons := make([]store.LabelHorizon, 5)
	for i := range horizons {
		horizon := float64(i)
		horizons[i] = store.LabelHorizon{
			HorizonSec:           &horizon,
			RelationToEgo:        "same_lane_ahead",
			HazardEvent:          []string{"no_hazard"},
			Cause:                []string{"lead_vehicle"},
			LongitudinalResponse: "slow_down",
			LateralResponse:      "keep_lane",
			TacticalResponse:     "proceed_with_caution",
			RuleResponse:         "none",
			Confidence:           0.9,
			Provenance:           "teacher_gt",
		}
	}
	return store.ReasoningLabel{
		SchemaVersion:   "reasoning_label_v2",
		SampleID:        sampleID,
		DatasetName:     "KIT-MRT/KITScenes-Multimodal",
		TeacherProvider: "teacher",
		TeacherModel:    "model",
		PromptVersion:   prompt,
		RequestMode:     "temporal_multi_frame",
		Provenance:      "teacher_gt",
		Horizons:        horizons,
	}
}

func testEmbeddedReasoning(
	sampleID, prompt, shard string,
	offset, size int64,
) embeddedReasoningLabel {
	return embeddedReasoningLabel{
		Label: testReasoningLabel(sampleID, prompt),
		Location: reasoningMemberLocation{
			Shard:     shard,
			SampleUID: sampleID,
			Range: model.MemberRange{
				Offset: offset,
				Size:   size,
			},
		},
	}
}

func testPublication(
	shard string,
	size int64,
	reasoningCount int,
) *publicationManifest {
	key := "kitscenes/v2.1/shards/" + shard
	entry := publicationShardEntry{
		Name:     shard,
		Key:      key,
		ByteSize: size,
		ETag:     `"test-etag"`,
	}
	return &publicationManifest{
		Dataset:             "kitscenes",
		Version:             "v2.1",
		SHA256:              testManifestSHA,
		ReasoningLabelCount: reasoningCount,
		ShardEntries:        []publicationShardEntry{entry},
		ShardByName: map[string]publicationShardEntry{
			shard: entry,
		},
	}
}

func testReasoningService(
	fakeStore *fakeReasoningStore,
	client *boundedReasoningS3,
	manifest *publicationManifest,
) *S3Service {
	return &S3Service{
		client:           client,
		bucket:           "datasets",
		store:            fakeStore,
		versionCache:     make(map[string]cachedVersion),
		publicationCache: map[string]*publicationManifest{"kitscenes/v2.1": manifest},
		indexSF:          make(map[string]*shardIndexBuild),
	}
}

func TestReasoningDiscoveryUsesMaterializedInventoryOnly(t *testing.T) {
	fakeStore := &fakeReasoningStore{
		inventories: make(map[string]model.ReasoningInventory),
		stats:       make(map[string]model.ReasoningStatsBlob),
	}
	service := &S3Service{
		store:            fakeStore,
		versionCache:     make(map[string]cachedVersion),
		publicationCache: make(map[string]*publicationManifest),
	}
	for _, dataset := range knownDatasets {
		entry := testReasoningEntry("prompt-"+dataset, 1)
		fakeStore.inventories[reasoningStoreKey(dataset, "v2.1")] =
			model.ReasoningInventory{
				Generation:            testGenerationA,
				DatasetManifestSHA256: testManifestSHA,
				PromptVersions:        []model.ReasoningPromptVersion{entry},
				Total:                 1,
			}
		service.versionCache[dataset] = cachedVersion{
			version: "v2.1",
			at:      time.Now(),
		}
		service.publicationCache[dataset+"/v2.1"] =
			&publicationManifest{
				Dataset: dataset,
				Version: "v2.1",
				SHA256:  testManifestSHA,
			}
		fakeStore.stats[reasoningStoreKey(
			dataset,
			"v2.1",
			testGenerationA,
			entry.Teacher,
			entry.PromptVersion,
		)] = model.ReasoningStatsBlob{NRecords: 1, NLabels: 1}
	}

	entries, total, err := service.ReasoningStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if total != len(knownDatasets) || len(entries) != len(knownDatasets) {
		t.Fatalf(
			"ReasoningStats = (%d entries, total %d)",
			len(entries), total,
		)
	}
	prompts, err := service.ReasoningPromptVersionsAtVersion(
		context.Background(), "kitscenes", "v2.1",
	)
	if err != nil || len(prompts) != 1 {
		t.Fatalf("ReasoningPromptVersionsAtVersion = %+v, %v", prompts, err)
	}
	detail, err := service.ReasoningStatsDetail(
		context.Background(),
		"kitscenes",
		"v2.1",
		prompts[0].PromptVersion,
		prompts[0].Teacher,
	)
	if err != nil || detail.Stats.NRecords != 1 {
		t.Fatalf("ReasoningStatsDetail = %+v, %v", detail, err)
	}
}

func TestReasoningStatsAllowsPartialDatasetMaterialization(t *testing.T) {
	fakeStore := &fakeReasoningStore{
		inventories: make(map[string]model.ReasoningInventory),
	}
	service := &S3Service{
		store:            fakeStore,
		versionCache:     make(map[string]cachedVersion),
		publicationCache: make(map[string]*publicationManifest),
	}
	for _, dataset := range knownDatasets {
		service.versionCache[dataset] = cachedVersion{
			version: "v2.1",
			at:      time.Now(),
		}
		service.publicationCache[dataset+"/v2.1"] =
			&publicationManifest{
				Dataset: dataset,
				Version: "v2.1",
				SHA256:  testManifestSHA,
			}
	}
	entry := testReasoningEntry("prompt-kitscenes", 1)
	key := reasoningStoreKey("kitscenes", "v2.1")
	fakeStore.inventories[key] = model.ReasoningInventory{
		Generation:            testGenerationA,
		DatasetManifestSHA256: testManifestSHA,
		PromptVersions:        []model.ReasoningPromptVersion{entry},
		Total:                 1,
	}

	entries, total, err := service.ReasoningStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 ||
		entries[0].Dataset != "kitscenes" ||
		total != 1 {
		t.Fatalf("partial ReasoningStats = (%+v, %d)", entries, total)
	}

	delete(fakeStore.inventories, key)
	if _, _, err := service.ReasoningStats(
		context.Background(),
	); !errors.Is(err, ErrReasoningUnavailable) {
		t.Fatalf("all-unmaterialized ReasoningStats error = %v", err)
	}

	fakeStore.inventories[key] = model.ReasoningInventory{
		Generation:            testGenerationA,
		DatasetManifestSHA256: testManifestSHA,
	}
	entries, total, err = service.ReasoningStats(context.Background())
	if err != nil || len(entries) != 0 || total != 0 {
		t.Fatalf(
			"empty materialized ReasoningStats = (%+v, %d, %v)",
			entries,
			total,
			err,
		)
	}
}

func TestReasoningReadErrorsSeparateAvailabilityAndIntegrity(t *testing.T) {
	const (
		dataset = "kitscenes"
		version = "v2.1"
		prompt  = "prompt-v1"
	)
	entry := testReasoningEntry(prompt, 1)
	fakeStore := &fakeReasoningStore{
		inventories: make(map[string]model.ReasoningInventory),
		stats:       make(map[string]model.ReasoningStatsBlob),
	}
	service := &S3Service{
		store: fakeStore,
		publicationCache: map[string]*publicationManifest{
			dataset + "/" + version: {
				Dataset: dataset,
				Version: version,
				SHA256:  testManifestSHA,
			},
		},
	}

	_, err := service.ReasoningStatsDetail(
		context.Background(),
		dataset,
		version,
		prompt,
		entry.Teacher,
	)
	if !errors.Is(err, ErrReasoningUnavailable) {
		t.Fatalf("unmaterialized inventory error = %v", err)
	}

	inventoryKey := reasoningStoreKey(dataset, version)
	fakeStore.inventories[inventoryKey] = model.ReasoningInventory{
		Generation:            testGenerationA,
		DatasetManifestSHA256: testManifestSHA,
		PromptVersions:        []model.ReasoningPromptVersion{entry},
		Total:                 1,
	}
	_, err = service.ReasoningStatsDetail(
		context.Background(),
		dataset,
		version,
		prompt,
		entry.Teacher,
	)
	if !errors.Is(err, ErrReasoningIntegrity) ||
		errors.Is(err, ErrNotFound) {
		t.Fatalf("advertised stats miss error = %v", err)
	}

	_, err = service.ReasoningStatsDetail(
		context.Background(),
		dataset,
		version,
		"absent-prompt",
		entry.Teacher,
	)
	if !errors.Is(err, ErrNotFound) ||
		errors.Is(err, ErrReasoningIntegrity) {
		t.Fatalf("absent partition error = %v", err)
	}

	fakeStore.inventories[inventoryKey] = model.ReasoningInventory{
		DatasetManifestSHA256: testManifestSHA,
		PromptVersions:        []model.ReasoningPromptVersion{entry},
		Total:                 1,
	}
	_, err = service.ReasoningStatsDetail(
		context.Background(),
		dataset,
		version,
		prompt,
		entry.Teacher,
	)
	if !errors.Is(err, ErrReasoningIntegrity) {
		t.Fatalf("invalid inventory error = %v", err)
	}
}

func TestMaterializeReasoningUsesBoundedBatchesAndPublishesLast(
	t *testing.T,
) {
	const (
		nRecords = 130
		shard    = "scene-a-train-000000.tar"
		prompt   = "prompt-v1"
	)
	var object bytes.Buffer
	index := &model.ShardIndex{
		Version: "v2.1",
		Shard:   shard,
		Samples: make([]model.IndexSample, 0, nRecords),
	}
	for i := 0; i < nRecords; i++ {
		sampleID := fmt.Sprintf("sample-%06d", i)
		body, err := json.Marshal(testReasoningLabel(sampleID, prompt))
		if err != nil {
			t.Fatal(err)
		}
		offset := int64(object.Len())
		if _, err := object.Write(body); err != nil {
			t.Fatal(err)
		}
		index.Samples = append(index.Samples, model.IndexSample{
			Key:          sampleID,
			SampleUID:    sampleID,
			HasReasoning: true,
			Members: map[string]model.MemberRange{
				"meta.json": {
					Offset: offset,
					Size:   1,
				},
				"reasoning.json": {
					Offset: offset,
					Size:   int64(len(body)),
				},
			},
		})
	}
	key := "kitscenes/v2.1/shards/" + shard
	baseClient := &fakePublicationS3{
		objects: map[string]fakePublicationObject{
			key: {body: object.Bytes(), etag: `"test-etag"`},
		},
		getCalls:  make(map[string]int),
		headCalls: make(map[string]int),
	}
	client := &boundedReasoningS3{
		fakePublicationS3: baseClient,
		delay:             time.Millisecond,
		ifMatches:         make(map[string][]string),
	}
	fakeStore := &fakeReasoningStore{
		indices: map[string]*model.ShardIndex{
			reasoningStoreKey("kitscenes", "v2.1", shard): index,
		},
	}
	service := testReasoningService(
		fakeStore,
		client,
		testPublication(shard, int64(object.Len()), nRecords),
	)

	response, err := service.MaterializeReasoning(
		context.Background(), "kitscenes", "v2.1", testManifestSHA,
	)
	if err != nil {
		t.Fatal(err)
	}
	if response.TotalRecords != nRecords || response.Partitions != 1 {
		t.Fatalf("materialization response = %+v", response)
	}
	inventory := fakeStore.inventories[reasoningStoreKey(
		"kitscenes", "v2.1",
	)]
	if inventory.Generation != reasoningGenerationID(testManifestSHA) ||
		inventory.Total != nRecords {
		t.Fatalf("published inventory = %+v", inventory)
	}
	if response.Generation != inventory.Generation ||
		response.DatasetManifestSHA256 != testManifestSHA {
		t.Fatalf("materialization identity = %+v", response)
	}
	if got := fakeStore.operations[len(fakeStore.operations)-1]; got != "inventory:"+inventory.Generation {
		t.Fatalf("last operation = %q, want inventory publish", got)
	}
	if fakeStore.maxLookupBatch > reasoningLookupBatchSize {
		t.Fatalf("lookup batch = %d", fakeStore.maxLookupBatch)
	}
	if fakeStore.maxSceneBatch > reasoningSceneBatchSize {
		t.Fatalf("scene batch = %d", fakeStore.maxSceneBatch)
	}
	maxActive, listCalls := client.observations()
	if maxActive > labelFetchConcurrency {
		t.Fatalf("S3 label concurrency = %d", maxActive)
	}
	if listCalls != 0 {
		t.Fatalf("materializer listed S3 shards despite cached manifest: %d", listCalls)
	}
	for _, ifMatch := range client.objectIfMatches(key) {
		if ifMatch != `"test-etag"` {
			t.Fatalf("reasoning range If-Match = %q", ifMatch)
		}
	}
	entry := inventory.PromptVersions[0]
	stats := fakeStore.stats[reasoningStoreKey(
		"kitscenes",
		"v2.1",
		inventory.Generation,
		entry.Teacher,
		entry.PromptVersion,
	)]
	if stats.NRecords != nRecords || stats.NLabels != nRecords {
		t.Fatalf("incremental stats = %+v", stats)
	}

	operations := len(fakeStore.operations)
	indexCalls := fakeStore.getShardIndexCall
	rangeCalls := len(client.objectIfMatches(key))
	reused, err := service.MaterializeReasoning(
		context.Background(), "kitscenes", "v2.1", testManifestSHA,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !reused.Reused ||
		reused.Generation != response.Generation ||
		reused.TotalRecords != response.TotalRecords ||
		reused.SceneRows != response.SceneRows {
		t.Fatalf("reused materialization = %+v, first = %+v", reused, response)
	}
	if len(fakeStore.operations) != operations ||
		fakeStore.getShardIndexCall != indexCalls ||
		len(client.objectIfMatches(key)) != rangeCalls {
		t.Fatal("same-generation retry rewrote or rescanned published data")
	}
}

func TestMaterializeReasoningRejectsUnpinnedOrLegacyPublication(t *testing.T) {
	t.Run("manifest mismatch", func(t *testing.T) {
		client := &boundedReasoningS3{
			fakePublicationS3: &fakePublicationS3{
				objects:   make(map[string]fakePublicationObject),
				getCalls:  make(map[string]int),
				headCalls: make(map[string]int),
			},
			ifMatches: make(map[string][]string),
		}
		fakeStore := &fakeReasoningStore{}
		service := testReasoningService(
			fakeStore, client, testPublication("scene-a.tar", 1, 0),
		)

		_, err := service.MaterializeReasoning(
			context.Background(),
			"kitscenes",
			"v2.1",
			testGenerationA,
		)
		if err == nil ||
			!strings.Contains(err.Error(), "differs from requested publication") {
			t.Fatalf("manifest mismatch error = %v", err)
		}
		if len(fakeStore.owners) != 0 ||
			len(fakeStore.operations) != 0 ||
			fakeStore.getShardIndexCall != 0 {
			t.Fatal("manifest mismatch acquired a lease or accessed serving data")
		}
	})

	t.Run("pre-v2.1 publication", func(t *testing.T) {
		const key = "kitscenes/v2.0/shards/train-000000.tar"
		client := &boundedReasoningS3{
			fakePublicationS3: &fakePublicationS3{
				objects: map[string]fakePublicationObject{
					key: {body: []byte("legacy")},
				},
				getCalls:  make(map[string]int),
				headCalls: make(map[string]int),
			},
			ifMatches: make(map[string][]string),
		}
		fakeStore := &fakeReasoningStore{}
		service := &S3Service{
			client:           client,
			bucket:           "datasets",
			store:            fakeStore,
			versionCache:     make(map[string]cachedVersion),
			publicationCache: make(map[string]*publicationManifest),
			indexSF:          make(map[string]*shardIndexBuild),
		}

		_, err := service.MaterializeReasoning(
			context.Background(),
			"kitscenes",
			"v2.0",
			testManifestSHA,
		)
		if err == nil ||
			!strings.Contains(err.Error(), "requires an immutable v2.1+") {
			t.Fatalf("legacy publication error = %v", err)
		}
		if len(fakeStore.owners) != 0 || len(fakeStore.operations) != 0 {
			t.Fatal("legacy publication acquired a lease or wrote serving data")
		}
	})
}

func TestReasoningMaterializationGenerationIsDeterministic(t *testing.T) {
	first := reasoningGenerationID(testManifestSHA)
	second := reasoningGenerationID(testManifestSHA)
	other := reasoningGenerationID(strings.Repeat("d", 64))
	if first != second {
		t.Fatalf("same publication generated %q and %q", first, second)
	}
	if first == other || !store.ValidReasoningGeneration(first) {
		t.Fatalf("generation identities = %q and %q", first, other)
	}
}

func TestMaterializeReasoningPublishesEmptyInventory(t *testing.T) {
	const shard = "scene-a.tar"
	fakeStore := &fakeReasoningStore{
		indices: map[string]*model.ShardIndex{
			reasoningStoreKey("kitscenes", "v2.1", shard): {
				Version: "v2.1",
				Shard:   shard,
				Samples: []model.IndexSample{{
					Key:       "sample-a",
					SampleUID: "sample-a",
					Members: map[string]model.MemberRange{
						"meta.json": {Offset: 0, Size: 1},
					},
				}},
			},
		},
	}
	client := &boundedReasoningS3{
		fakePublicationS3: &fakePublicationS3{
			objects: map[string]fakePublicationObject{
				"kitscenes/v2.1/shards/" + shard: {
					body: []byte{0},
					etag: `"test-etag"`,
				},
			},
			getCalls:  make(map[string]int),
			headCalls: make(map[string]int),
		},
		ifMatches: make(map[string][]string),
	}
	service := testReasoningService(
		fakeStore, client, testPublication(shard, 1, 0),
	)

	response, err := service.MaterializeReasoning(
		context.Background(), "kitscenes", "v2.1", testManifestSHA,
	)
	if err != nil {
		t.Fatal(err)
	}
	inventory := fakeStore.inventories[reasoningStoreKey(
		"kitscenes", "v2.1",
	)]
	if response.TotalRecords != 0 ||
		response.Partitions != 0 ||
		inventory.Total != 0 ||
		len(inventory.PromptVersions) != 0 {
		t.Fatalf("empty materialization = %+v / %+v", response, inventory)
	}
}

func TestReasoningMaterializationRejectsPartitionOverflow(t *testing.T) {
	materialization := &reasoningMaterialization{
		ctx:                   context.Background(),
		store:                 &fakeReasoningStore{},
		dataset:               "kitscenes",
		version:               "v2.1",
		generation:            testGenerationA,
		owner:                 testLeaseOwnerA,
		datasetManifestSHA256: testManifestSHA,
		partitions: make(
			map[reasoningPartitionKey]*reasoningPartitionAccumulator,
		),
	}
	for i := 0; i < maxReasoningPartitions; i++ {
		record := testEmbeddedReasoning(
			fmt.Sprintf("sample-%03d", i),
			fmt.Sprintf("prompt-%03d", i),
			"scene-a.tar",
			int64(i),
			1,
		)
		if err := materialization.add(record); err != nil {
			t.Fatalf("partition %d: %v", i, err)
		}
	}
	overflow := testEmbeddedReasoning(
		"sample-overflow",
		"prompt-overflow",
		"scene-a.tar",
		1000,
		1,
	)
	if err := materialization.add(overflow); err == nil ||
		!strings.Contains(err.Error(), "partition count exceeds") {
		t.Fatalf("partition overflow error = %v", err)
	}
}

func TestFailedRerunPreservesActiveGeneration(t *testing.T) {
	fakeStore := &fakeReasoningStore{}
	leaseNow := time.Now()
	oldRun := &reasoningMaterialization{
		ctx:                   context.Background(),
		store:                 fakeStore,
		dataset:               "kitscenes",
		version:               "v2.1",
		generation:            testGenerationA,
		owner:                 testLeaseOwnerA,
		datasetManifestSHA256: testManifestSHA,
		partitions: make(
			map[reasoningPartitionKey]*reasoningPartitionAccumulator,
		),
	}
	if err := fakeStore.BeginReasoningMaterialization(
		context.Background(),
		"kitscenes",
		"v2.1",
		testLeaseOwnerA,
		leaseNow.Unix(),
		leaseNow.Add(time.Hour).Unix(),
	); err != nil {
		t.Fatal(err)
	}
	if err := oldRun.add(testEmbeddedReasoning(
		"old-sample", "prompt-v1", "old-shard.tar", 512, 128,
	)); err != nil {
		t.Fatal(err)
	}
	if _, err := oldRun.publish(); err != nil {
		t.Fatal(err)
	}

	fakeStore.failStats = true
	newRun := &reasoningMaterialization{
		ctx:                   context.Background(),
		store:                 fakeStore,
		dataset:               "kitscenes",
		version:               "v2.1",
		generation:            testGenerationB,
		owner:                 testLeaseOwnerB,
		datasetManifestSHA256: testManifestSHA,
		partitions: make(
			map[reasoningPartitionKey]*reasoningPartitionAccumulator,
		),
	}
	if err := fakeStore.BeginReasoningMaterialization(
		context.Background(),
		"kitscenes",
		"v2.1",
		testLeaseOwnerB,
		leaseNow.Unix(),
		leaseNow.Add(time.Hour).Unix(),
	); err != nil {
		t.Fatal(err)
	}
	if err := newRun.add(testEmbeddedReasoning(
		"new-sample", "prompt-v1", "new-shard.tar", 1024, 128,
	)); err != nil {
		t.Fatal(err)
	}
	if _, err := newRun.publish(); err == nil {
		t.Fatal("failed rerun unexpectedly published")
	}
	inventory := fakeStore.inventories[reasoningStoreKey(
		"kitscenes", "v2.1",
	)]
	if inventory.Generation != testGenerationA {
		t.Fatalf(
			"active generation = %q, want old generation",
			inventory.Generation,
		)
	}

	client := &boundedReasoningS3{
		fakePublicationS3: &fakePublicationS3{
			objects:   make(map[string]fakePublicationObject),
			getCalls:  make(map[string]int),
			headCalls: make(map[string]int),
		},
		ifMatches: make(map[string][]string),
	}
	service := testReasoningService(
		fakeStore, client, testPublication("old-shard.tar", 1, 1),
	)
	entry := inventory.PromptVersions[0]
	scenes, _, err := service.SearchScenesByLabelForTeacherAtVersion(
		context.Background(),
		"kitscenes",
		"v2.1",
		entry.Teacher,
		entry.PromptVersion,
		store.FieldCause,
		"lead_vehicle",
		10,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(scenes) != 1 ||
		scenes[0].SampleID != "old-sample" ||
		scenes[0].Shard != "old-shard.tar" {
		t.Fatalf("active generation scenes = %+v", scenes)
	}
	if fakeStore.getShardIndexCall != 0 {
		t.Fatalf(
			"scene search called GetShardIndex %d times",
			fakeStore.getShardIndexCall,
		)
	}
	_, listCalls := client.observations()
	if listCalls != 0 {
		t.Fatalf("scene search listed shards %d times", listCalls)
	}
}

func TestMaterializeReasoningValidatesPublicationCountAndCap(t *testing.T) {
	t.Run("count mismatch stays unpublished", func(t *testing.T) {
		const shard = "scene-a.tar"
		label := testReasoningLabel("sample-a", "prompt-v1")
		body, err := json.Marshal(label)
		if err != nil {
			t.Fatal(err)
		}
		index := &model.ShardIndex{
			Version: "v2.1",
			Shard:   shard,
			Samples: []model.IndexSample{{
				Key:          label.SampleID,
				SampleUID:    label.SampleID,
				HasReasoning: true,
				Members: map[string]model.MemberRange{
					"meta.json": {
						Offset: 0,
						Size:   1,
					},
					"reasoning.json": {
						Offset: 0,
						Size:   int64(len(body)),
					},
				},
			}},
		}
		key := "kitscenes/v2.1/shards/" + shard
		client := &boundedReasoningS3{
			fakePublicationS3: &fakePublicationS3{
				objects: map[string]fakePublicationObject{
					key: {body: body, etag: `"test-etag"`},
				},
				getCalls:  make(map[string]int),
				headCalls: make(map[string]int),
			},
		}
		fakeStore := &fakeReasoningStore{
			indices: map[string]*model.ShardIndex{
				reasoningStoreKey("kitscenes", "v2.1", shard): index,
			},
		}
		service := testReasoningService(
			fakeStore,
			client,
			testPublication(shard, int64(len(body)), 2),
		)
		if _, err := service.MaterializeReasoning(
			context.Background(), "kitscenes", "v2.1", testManifestSHA,
		); err == nil {
			t.Fatal("publication count mismatch was accepted")
		}
		if len(fakeStore.inventories) != 0 {
			t.Fatal("count mismatch published an inventory")
		}
	})

	t.Run("publication above cap fails before shard access", func(t *testing.T) {
		client := &boundedReasoningS3{
			fakePublicationS3: &fakePublicationS3{
				objects:   make(map[string]fakePublicationObject),
				getCalls:  make(map[string]int),
				headCalls: make(map[string]int),
			},
		}
		fakeStore := &fakeReasoningStore{}
		service := testReasoningService(
			fakeStore,
			client,
			testPublication("scene-a.tar", 1, statsScanCap+1),
		)
		if _, err := service.MaterializeReasoning(
			context.Background(), "kitscenes", "v2.1", testManifestSHA,
		); err == nil {
			t.Fatal("publication above statsScanCap was accepted")
		}
		if fakeStore.getShardIndexCall != 0 ||
			len(fakeStore.operations) != 0 {
			t.Fatal("cap failure accessed shards or wrote serving data")
		}
	})
}

func TestGetReasoningLabelUsesDirectLookupWithoutShardScan(t *testing.T) {
	const (
		shard    = "scene-a.tar"
		sampleID = "sample-a"
		prompt   = "prompt-v1"
	)
	label := testReasoningLabel(sampleID, prompt)
	body, err := json.Marshal(label)
	if err != nil {
		t.Fatal(err)
	}
	entry := testReasoningEntry(prompt, 1)
	fakeStore := &fakeReasoningStore{
		inventories: map[string]model.ReasoningInventory{
			reasoningStoreKey("kitscenes", "v2.1"): {
				Generation:            testGenerationA,
				DatasetManifestSHA256: testManifestSHA,
				PromptVersions:        []model.ReasoningPromptVersion{entry},
				Total:                 1,
			},
		},
		lookups: map[string]model.ReasoningSampleLookup{
			reasoningStoreKey(
				"kitscenes", "v2.1", testGenerationA, sampleID,
			): {
				SampleID: sampleID,
				Shard:    shard,
				Offset:   0,
				Size:     int64(len(body)),
			},
		},
	}
	key := "kitscenes/v2.1/shards/" + shard
	client := &boundedReasoningS3{
		fakePublicationS3: &fakePublicationS3{
			objects: map[string]fakePublicationObject{
				key: {body: body, etag: `"test-etag"`},
			},
			getCalls:  make(map[string]int),
			headCalls: make(map[string]int),
		},
	}
	service := testReasoningService(
		fakeStore,
		client,
		testPublication(shard, int64(len(body)), 1),
	)

	got, source, err := service.GetReasoningLabelAtVersion(
		context.Background(),
		"kitscenes",
		"v2.1",
		sampleID,
		entry.Teacher,
		prompt,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, body) ||
		source != "kitscenes/v2.1/sample-a/reasoning.json" {
		t.Fatalf("label/source = %s / %q", got, source)
	}
	if fakeStore.getShardIndexCall != 0 {
		t.Fatalf(
			"direct label read called GetShardIndex %d times",
			fakeStore.getShardIndexCall,
		)
	}
	_, listCalls := client.observations()
	if listCalls != 0 {
		t.Fatalf("direct label read listed shards %d times", listCalls)
	}
	if got := client.objectIfMatches(key); len(got) != 1 ||
		got[0] != `"test-etag"` {
		t.Fatalf("direct label range If-Match values = %q", got)
	}

	if _, _, err := service.GetReasoningLabelAtVersion(
		context.Background(),
		"kitscenes",
		"v2.1",
		"absent-sample",
		entry.Teacher,
		prompt,
	); !errors.Is(err, ErrNotFound) ||
		errors.Is(err, ErrReasoningIntegrity) {
		t.Fatalf("absent sample error = %v", err)
	}

	if _, _, err := service.GetReasoningLabelAtVersion(
		context.Background(),
		"kitscenes",
		"v2.1",
		sampleID,
		entry.Teacher,
		"wrong-prompt",
	); !errors.Is(err, ErrNotFound) {
		t.Fatalf("prompt validation after body parse = %v", err)
	}

	delete(client.objects, key)
	if _, _, err := service.GetReasoningLabelAtVersion(
		context.Background(),
		"kitscenes",
		"v2.1",
		sampleID,
		entry.Teacher,
		prompt,
	); !errors.Is(err, ErrReasoningIntegrity) ||
		errors.Is(err, ErrNotFound) {
		t.Fatalf("advertised shard miss error = %v", err)
	}
}

func TestValidateReasoningInventoryRequiresGeneration(t *testing.T) {
	entry := testReasoningEntry("prompt-v1", 1)
	if err := validateReasoningInventory(model.ReasoningInventory{
		DatasetManifestSHA256: testManifestSHA,
		PromptVersions:        []model.ReasoningPromptVersion{entry},
		Total:                 1,
	}); err == nil {
		t.Fatal("inventory without generation was accepted")
	}
	entry.Count = statsScanCap + 1
	if err := validateReasoningInventory(model.ReasoningInventory{
		Generation:            testGenerationA,
		DatasetManifestSHA256: testManifestSHA,
		PromptVersions:        []model.ReasoningPromptVersion{entry},
		Total:                 statsScanCap + 1,
	}); err == nil {
		t.Fatal("inventory above statsScanCap was accepted")
	}
}
