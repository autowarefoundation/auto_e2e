package store

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	ddbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

// fakeDDB is an in-memory ddbAPI implementation for unit tests (no live AWS).
// It stores items keyed by (pk, sk) and supports GetItem, PutItem,
// BatchWriteItem, and a pk-only Query.
type fakeDDB struct {
	items               map[string]map[string]ddbtypes.AttributeValue
	batchCalls          int
	getCalls            int
	lastGetConsistent   bool
	lastQueryConsistent bool
	// forceUnprocessedOnce returns the first batch's items as unprocessed once,
	// to exercise the retry path.
	forceUnprocessedOnce bool
}

func newFakeDDB() *fakeDDB {
	return &fakeDDB{items: map[string]map[string]ddbtypes.AttributeValue{}}
}

func keyOf(item map[string]ddbtypes.AttributeValue) string {
	pk := item["pk"].(*ddbtypes.AttributeValueMemberS).Value
	sk := item["sk"].(*ddbtypes.AttributeValueMemberS).Value
	return pk + "\x00" + sk
}

func (f *fakeDDB) GetItem(_ context.Context, in *dynamodb.GetItemInput, _ ...func(*dynamodb.Options)) (*dynamodb.GetItemOutput, error) {
	f.getCalls++
	f.lastGetConsistent = in.ConsistentRead != nil && *in.ConsistentRead
	k := keyOf(in.Key)
	item, ok := f.items[k]
	if !ok {
		return &dynamodb.GetItemOutput{}, nil
	}
	return &dynamodb.GetItemOutput{Item: item}, nil
}

func (f *fakeDDB) PutItem(_ context.Context, in *dynamodb.PutItemInput, _ ...func(*dynamodb.Options)) (*dynamodb.PutItemOutput, error) {
	if in.ConditionExpression != nil &&
		!fakePutConditionMatches(f.items[keyOf(in.Item)], in) {
		return nil, &ddbtypes.ConditionalCheckFailedException{
			Message: aws.String("condition failed"),
		}
	}
	f.items[keyOf(in.Item)] = in.Item
	return &dynamodb.PutItemOutput{}, nil
}

func fakePutConditionMatches(
	existing map[string]ddbtypes.AttributeValue,
	in *dynamodb.PutItemInput,
) bool {
	condition := aws.ToString(in.ConditionExpression)
	ownerName := in.ExpressionAttributeNames["#owner"]
	if strings.HasPrefix(condition, "(attribute_not_exists(#owner)") {
		available := existing == nil
		if !available {
			currentOwner, hasOwner := existing[ownerName].(*ddbtypes.AttributeValueMemberS)
			available = !hasOwner
			if hasOwner {
				expectedOwner := in.ExpressionAttributeValues[":owner"].(*ddbtypes.AttributeValueMemberS)
				available = currentOwner.Value == expectedOwner.Value
				expires, ok := existing[in.ExpressionAttributeNames["#expires"]].(*ddbtypes.AttributeValueMemberN)
				if !ok {
					return false
				}
				now := in.ExpressionAttributeValues[":now"].(*ddbtypes.AttributeValueMemberN)
				expiresUnix, expiresErr := strconv.ParseInt(
					expires.Value, 10, 64,
				)
				nowUnix, nowErr := strconv.ParseInt(now.Value, 10, 64)
				available = available ||
					(expiresErr == nil &&
						nowErr == nil &&
						expiresUnix < nowUnix)
			}
		}
		if !available {
			return false
		}
		if strings.Contains(
			condition, "inventory_generation = :active",
		) {
			current, ok := existing["inventory_generation"].(*ddbtypes.AttributeValueMemberS)
			active := in.ExpressionAttributeValues[":active"].(*ddbtypes.AttributeValueMemberS)
			if !ok || current.Value != active.Value {
				return false
			}
		}
		if strings.Contains(
			condition,
			"attribute_not_exists(inventory_generation)",
		) {
			_, exists := existing["inventory_generation"]
			if exists {
				return false
			}
		}
		if strings.Contains(
			condition, "inventory_revision = :revision",
		) {
			current, ok := existing["inventory_revision"].(*ddbtypes.AttributeValueMemberS)
			revision := in.ExpressionAttributeValues[":revision"].(*ddbtypes.AttributeValueMemberS)
			if !ok || current.Value != revision.Value {
				return false
			}
		}
		if strings.Contains(
			condition,
			"attribute_not_exists(inventory_revision)",
		) {
			_, exists := existing["inventory_revision"]
			if exists {
				return false
			}
		}
		return true
	}
	if strings.HasPrefix(
		condition,
		"(#owner = :owner AND #expires >= :now) OR",
	) {
		expected := in.ExpressionAttributeValues[":owner"].(*ddbtypes.AttributeValueMemberS)
		if current, ok := existing[ownerName].(*ddbtypes.AttributeValueMemberS); ok &&
			current.Value == expected.Value {
			expires, ok := existing[in.ExpressionAttributeNames["#expires"]].(*ddbtypes.AttributeValueMemberN)
			if !ok {
				return false
			}
			now := in.ExpressionAttributeValues[":now"].(*ddbtypes.AttributeValueMemberN)
			expiresUnix, expiresErr := strconv.ParseInt(
				expires.Value, 10, 64,
			)
			nowUnix, nowErr := strconv.ParseInt(now.Value, 10, 64)
			if expiresErr == nil && nowErr == nil && expiresUnix >= nowUnix {
				return true
			}
		}
		if _, hasOwner := existing[ownerName]; hasOwner {
			return false
		}
		revision, ok := existing["inventory_revision"].(*ddbtypes.AttributeValueMemberS)
		return ok && revision.Value == expected.Value
	}
	if strings.HasPrefix(condition, "#owner = :owner") {
		current, ok := existing[ownerName].(*ddbtypes.AttributeValueMemberS)
		expected := in.ExpressionAttributeValues[":owner"].(*ddbtypes.AttributeValueMemberS)
		if !ok || current.Value != expected.Value {
			return false
		}
		if strings.Contains(condition, "#expires >= :now") {
			expires, ok := existing[in.ExpressionAttributeNames["#expires"]].(*ddbtypes.AttributeValueMemberN)
			if !ok {
				return false
			}
			now := in.ExpressionAttributeValues[":now"].(*ddbtypes.AttributeValueMemberN)
			expiresUnix, expiresErr := strconv.ParseInt(
				expires.Value, 10, 64,
			)
			nowUnix, nowErr := strconv.ParseInt(now.Value, 10, 64)
			return expiresErr == nil &&
				nowErr == nil &&
				expiresUnix >= nowUnix
		}
		return true
	}
	return false
}

func (f *fakeDDB) BatchWriteItem(_ context.Context, in *dynamodb.BatchWriteItemInput, _ ...func(*dynamodb.Options)) (*dynamodb.BatchWriteItemOutput, error) {
	f.batchCalls++
	for table, reqs := range in.RequestItems {
		if f.forceUnprocessedOnce {
			f.forceUnprocessedOnce = false
			return &dynamodb.BatchWriteItemOutput{UnprocessedItems: map[string][]ddbtypes.WriteRequest{table: reqs}}, nil
		}
		for _, r := range reqs {
			if r.PutRequest != nil {
				f.items[keyOf(r.PutRequest.Item)] = r.PutRequest.Item
			}
		}
	}
	return &dynamodb.BatchWriteItemOutput{}, nil
}

func (f *fakeDDB) Query(_ context.Context, in *dynamodb.QueryInput, _ ...func(*dynamodb.Options)) (*dynamodb.QueryOutput, error) {
	f.lastQueryConsistent = in.ConsistentRead != nil && *in.ConsistentRead
	pk := in.ExpressionAttributeValues[":pk"].(*ddbtypes.AttributeValueMemberS).Value
	prefix := ""
	if value, ok := in.ExpressionAttributeValues[":model"]; ok {
		prefix = value.(*ddbtypes.AttributeValueMemberS).Value
	}
	var items []map[string]ddbtypes.AttributeValue
	for _, item := range f.items {
		if item["pk"].(*ddbtypes.AttributeValueMemberS).Value == pk &&
			(prefix == "" || strings.HasPrefix(
				item["sk"].(*ddbtypes.AttributeValueMemberS).Value,
				prefix,
			)) {
			items = append(items, item)
		}
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i]["sk"].(*ddbtypes.AttributeValueMemberS).Value <
			items[j]["sk"].(*ddbtypes.AttributeValueMemberS).Value
	})
	start := 0
	if len(in.ExclusiveStartKey) > 0 {
		startSK := in.ExclusiveStartKey["sk"].(*ddbtypes.AttributeValueMemberS).Value
		start = sort.Search(len(items), func(i int) bool {
			return items[i]["sk"].(*ddbtypes.AttributeValueMemberS).Value >
				startSK
		})
	}
	end := len(items)
	limitReached := false
	if in.Limit != nil && int(*in.Limit) <= end-start {
		end = start + int(*in.Limit)
		limitReached = true
	}
	output := &dynamodb.QueryOutput{Items: items[start:end]}
	if limitReached && end > start {
		output.LastEvaluatedKey = map[string]ddbtypes.AttributeValue{
			"pk": items[end-1]["pk"],
			"sk": items[end-1]["sk"],
		}
	}
	return output, nil
}

func newTestStore() (*DynamoStore, *fakeDDB) {
	f := newFakeDDB()
	return &DynamoStore{client: f, table: "test-table"}, f
}

type blockingGetDDB struct {
	ddbAPI
	started   chan struct{}
	release   chan struct{}
	startOnce sync.Once
	getCalls  atomic.Int32
}

type beforeLeasePutDDB struct {
	ddbAPI
	once   sync.Once
	before func()
}

type ambiguousPutDDB struct {
	ddbAPI
	conditionPrefix string
	failed          bool
}

func (a *ambiguousPutDDB) PutItem(
	ctx context.Context,
	in *dynamodb.PutItemInput,
	opts ...func(*dynamodb.Options),
) (*dynamodb.PutItemOutput, error) {
	out, err := a.ddbAPI.PutItem(ctx, in, opts...)
	if err == nil &&
		!a.failed &&
		strings.HasPrefix(
			aws.ToString(in.ConditionExpression),
			a.conditionPrefix,
		) {
		a.failed = true
		return nil, errors.New("ambiguous response after committed write")
	}
	return out, err
}

func (b *beforeLeasePutDDB) PutItem(
	ctx context.Context,
	in *dynamodb.PutItemInput,
	opts ...func(*dynamodb.Options),
) (*dynamodb.PutItemOutput, error) {
	if strings.Contains(
		aws.ToString(in.ConditionExpression),
		"attribute_not_exists(#owner)",
	) {
		b.once.Do(b.before)
	}
	return b.ddbAPI.PutItem(ctx, in, opts...)
}

func (b *blockingGetDDB) GetItem(
	ctx context.Context,
	in *dynamodb.GetItemInput,
	opts ...func(*dynamodb.Options),
) (*dynamodb.GetItemOutput, error) {
	b.getCalls.Add(1)
	b.startOnce.Do(func() { close(b.started) })
	select {
	case <-b.release:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return b.ddbAPI.GetItem(ctx, in, opts...)
}

func TestDynamoStore_ShardIndexRoundTrip(t *testing.T) {
	s, _ := newTestStore()
	ctx := context.Background()

	idx := &model.ShardIndex{
		Fps: 10,
		Samples: []model.IndexSample{
			{Key: "ep0_000000", FrameIdx: 0, EgoNow: []float32{1, 2, 3, 4}, HasReasoning: true},
		},
	}
	if err := s.PutShardIndex(ctx, "l2d", "v2.0", "train-000000.tar", idx); err != nil {
		t.Fatalf("PutShardIndex: %v", err)
	}
	got, err := s.GetShardIndex(ctx, "l2d", "v2.0", "train-000000.tar")
	if err != nil {
		t.Fatalf("GetShardIndex: %v", err)
	}
	if got.Fps != 10 || len(got.Samples) != 1 || got.Samples[0].Key != "ep0_000000" {
		t.Errorf("round-trip mismatch: %+v", got)
	}
	if !got.Samples[0].HasReasoning {
		t.Errorf("HasReasoning lost in round-trip")
	}

	// Miss on a different shard.
	if _, err := s.GetShardIndex(ctx, "l2d", "v2.0", "train-000099.tar"); err != ErrNotFound {
		t.Errorf("expected ErrNotFound on miss, got %v", err)
	}
}

func TestDynamoStore_ShardIndexRejectsZipBomb(t *testing.T) {
	s, f := newTestStore()
	payload, err := gzipBytes(bytes.Repeat(
		[]byte("x"),
		maxShardIndexExpandedBytes+1,
	))
	if err != nil {
		t.Fatal(err)
	}
	if len(payload) >= maxShardIndexExpandedBytes {
		t.Fatalf("test payload did not compress: %d bytes", len(payload))
	}
	item := map[string]ddbtypes.AttributeValue{
		"pk": &ddbtypes.AttributeValueMemberS{
			Value: ShardIndexPK("l2d", "v2.0", "bomb.tar"),
		},
		"sk":      &ddbtypes.AttributeValueMemberS{Value: metaSK},
		"payload": &ddbtypes.AttributeValueMemberB{Value: payload},
	}
	f.items[keyOf(item)] = item

	if _, err := s.GetShardIndex(
		context.Background(), "l2d", "v2.0", "bomb.tar",
	); !errors.Is(err, ErrShardIndexTooLarge) {
		t.Fatalf("zip bomb error = %v, want ErrShardIndexTooLarge", err)
	}
}

func TestDynamoStore_ShardIndexRejectsOversizedCompressedPayload(t *testing.T) {
	s, fake := newTestStore()
	random := make([]byte, 700<<10)
	if _, err := rand.Read(random); err != nil {
		t.Fatal(err)
	}
	index := &model.ShardIndex{
		Samples: []model.IndexSample{{
			Key: base64.RawStdEncoding.EncodeToString(random),
		}},
	}

	err := s.PutShardIndex(
		context.Background(), "l2d", "v2.1", "large.tar", index,
	)
	if !errors.Is(err, ErrShardIndexTooLarge) {
		t.Fatalf("oversized compressed index error = %v", err)
	}
	if len(fake.items) != 0 {
		t.Fatal("oversized compressed index reached DynamoDB")
	}
}

func TestShardIndexDecodeSemaphoreIsProcessWide(t *testing.T) {
	releases := make([]func(), 0, maxConcurrentShardIndexDecodes)
	for range maxConcurrentShardIndexDecodes {
		release, err := acquireShardIndexDecode(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		releases = append(releases, release)
	}
	defer func() {
		for _, release := range releases {
			release()
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := acquireShardIndexDecode(ctx); !errors.Is(
		err, context.Canceled,
	) {
		t.Fatalf("saturated decode semaphore error = %v", err)
	}

	releases[0]()
	releases = releases[1:]
	release, err := acquireShardIndexDecode(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	release()
}

func TestDynamoStore_ShardIndexCacheHitSingleFlight64(t *testing.T) {
	const workers = 64

	base, _ := newTestStore()
	index := &model.ShardIndex{
		Fps:     10,
		Version: "v2.1",
		Shard:   "train-000000.tar",
		Samples: []model.IndexSample{{
			Key:       "l2d-v1-e000001-f000001",
			SampleUID: "l2d-v1-e000001-f000001",
			Members: map[string]model.MemberRange{
				"meta.json": {Offset: 512, Size: 16},
			},
		}},
	}
	if err := base.PutShardIndex(
		context.Background(),
		"l2d",
		"v2.1",
		"train-000000.tar",
		index,
	); err != nil {
		t.Fatal(err)
	}
	blocking := &blockingGetDDB{
		ddbAPI:  base.client,
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	s := &DynamoStore{
		client: blocking,
		table:  base.table,
	}

	results := make([]*model.ShardIndex, workers)
	errs := make([]error, workers)
	start := make(chan struct{})
	var ready sync.WaitGroup
	var done sync.WaitGroup
	ready.Add(workers)
	done.Add(workers)
	for i := range workers {
		go func() {
			defer done.Done()
			ready.Done()
			<-start
			results[i], errs[i] = s.GetShardIndex(
				context.Background(),
				"l2d",
				"v2.1",
				"train-000000.tar",
			)
		}()
	}
	ready.Wait()
	close(start)

	select {
	case <-blocking.started:
	case <-time.After(time.Second):
		close(blocking.release)
		done.Wait()
		t.Fatal("leader did not reach DynamoDB")
	}

	flightKey := ShardIndexPK("l2d", "v2.1", "train-000000.tar")
	deadline := time.Now().Add(time.Second)
	participants := 0
	for time.Now().Before(deadline) {
		s.shardIndexMu.Lock()
		if call := s.shardIndexCalls[flightKey]; call != nil {
			participants = call.participants
		}
		s.shardIndexMu.Unlock()
		if participants == workers {
			break
		}
		time.Sleep(time.Millisecond)
	}
	close(blocking.release)
	done.Wait()

	if participants != workers {
		t.Fatalf(
			"single-flight participants = %d, want %d",
			participants,
			workers,
		)
	}
	if got := blocking.getCalls.Load(); got != 1 {
		t.Fatalf("Dynamo Get/decode executions = %d, want 1", got)
	}
	for i := range workers {
		if errs[i] != nil {
			t.Fatalf("worker %d: %v", i, errs[i])
		}
		if results[i] != results[0] {
			t.Fatalf("worker %d received a separately decoded index", i)
		}
	}
}

func TestDynamoStore_StatsRoundTrip(t *testing.T) {
	s, _ := newTestStore()
	ctx := context.Background()

	blob := model.ReasoningStatsBlob{
		NLabels:      3,
		HorizonCount: 15,
		ByField:      map[string]map[string]int{"lateral_response": {"keep_lane": 10, "turn_left": 5}},
	}
	computedAt, err := s.PutStats(ctx, "l2d", "v2.0", "pv3", blob)
	if err != nil {
		t.Fatalf("PutStats: %v", err)
	}
	if computedAt == "" {
		t.Errorf("PutStats returned empty computed_at")
	}
	got, gotAt, err := s.GetStats(ctx, "l2d", "v2.0", "pv3")
	if err != nil {
		t.Fatalf("GetStats: %v", err)
	}
	if got.NLabels != 3 || got.ByField["lateral_response"]["turn_left"] != 5 {
		t.Errorf("stats round-trip mismatch: %+v", got)
	}
	if gotAt != computedAt {
		t.Errorf("computed_at mismatch: got %q, put %q", gotAt, computedAt)
	}

	if _, _, err := s.GetStats(ctx, "l2d", "v2.0", "absent"); err != ErrNotFound {
		t.Errorf("expected ErrNotFound on stats miss, got %v", err)
	}
}

func TestDynamoStore_ReasoningInventoryRoundTrip(t *testing.T) {
	s, _ := newTestStore()
	ctx := context.Background()
	owner := strings.Repeat("1", 64)
	manifestSHA := strings.Repeat("c", 64)
	inventory := model.ReasoningInventory{
		Generation:            testReasoningGeneration,
		DatasetManifestSHA256: manifestSHA,
		PromptVersions: []model.ReasoningPromptVersion{{
			Teacher:         "dGVhY2hlcgBtb2RlbA",
			TeacherProvider: "teacher",
			TeacherModel:    "model",
			PromptVersion:   "prompt-v1",
			Count:           7,
		}},
		Total: 7,
	}
	if err := s.BeginReasoningMaterialization(
		ctx, "kitscenes", "v2.1", owner, 100, 200,
	); err != nil {
		t.Fatalf("BeginReasoningMaterialization: %v", err)
	}
	computedAt, err := s.PutReasoningInventory(
		ctx, "kitscenes", "v2.1", owner, 150, inventory,
	)
	if err != nil {
		t.Fatalf("PutReasoningInventory: %v", err)
	}
	got, gotAt, err := s.GetReasoningInventory(
		ctx, "kitscenes", "v2.1",
	)
	if err != nil {
		t.Fatalf("GetReasoningInventory: %v", err)
	}
	if got.Total != 7 || len(got.PromptVersions) != 1 ||
		got.PromptVersions[0].PromptVersion != "prompt-v1" ||
		got.Generation != testReasoningGeneration ||
		got.DatasetManifestSHA256 != manifestSHA {
		t.Fatalf("reasoning inventory round-trip mismatch: %+v", got)
	}
	if gotAt != computedAt {
		t.Fatalf("computed_at = %q, want %q", gotAt, computedAt)
	}
	if !s.client.(*fakeDDB).lastGetConsistent {
		t.Fatal("reasoning inventory was not read consistently")
	}
	if _, _, err := s.GetReasoningInventory(
		ctx, "kitscenes", "v2.2",
	); err != ErrNotFound {
		t.Fatalf("missing reasoning inventory error = %v", err)
	}
}

func TestDynamoStore_ReasoningMaterializationLeaseFencesOldOwner(
	t *testing.T,
) {
	s, _ := newTestStore()
	ctx := context.Background()
	ownerA := strings.Repeat("1", 64)
	ownerB := strings.Repeat("2", 64)
	manifestSHA := strings.Repeat("c", 64)
	if err := s.BeginReasoningMaterialization(
		ctx, "kitscenes", "v2.1", ownerA, 100, 200,
	); err != nil {
		t.Fatal(err)
	}
	if err := s.BeginReasoningMaterialization(
		ctx, "kitscenes", "v2.1", ownerB, 150, 250,
	); err == nil {
		t.Fatal("second owner acquired an unexpired lease")
	}
	if err := s.RenewReasoningMaterialization(
		ctx, "kitscenes", "v2.1", ownerA, 150, 250,
	); err != nil {
		t.Fatal(err)
	}
	if err := s.BeginReasoningMaterialization(
		ctx, "kitscenes", "v2.1", ownerB, 251, 350,
	); err != nil {
		t.Fatalf("new owner did not acquire expired lease: %v", err)
	}
	if err := s.RenewReasoningMaterialization(
		ctx, "kitscenes", "v2.1", ownerA, 252, 400,
	); err == nil {
		t.Fatal("old owner renewed after being fenced")
	}
	stale := model.ReasoningInventory{
		Generation:            strings.Repeat("a", 64),
		DatasetManifestSHA256: manifestSHA,
	}
	if _, err := s.PutReasoningInventory(
		ctx, "kitscenes", "v2.1", ownerA, 300, stale,
	); err == nil {
		t.Fatal("old owner published after being fenced")
	}
	current := model.ReasoningInventory{
		Generation:            strings.Repeat("b", 64),
		DatasetManifestSHA256: manifestSHA,
	}
	if _, err := s.PutReasoningInventory(
		ctx, "kitscenes", "v2.1", ownerB, 300, current,
	); err != nil {
		t.Fatal(err)
	}
	got, _, err := s.GetReasoningInventory(
		ctx, "kitscenes", "v2.1",
	)
	if err != nil || got.Generation != current.Generation {
		t.Fatalf("published inventory = %+v, %v", got, err)
	}
}

func TestDynamoStore_ExpiredReasoningLeaseCannotRenewOrPublish(t *testing.T) {
	s, _ := newTestStore()
	ctx := context.Background()
	ownerA := strings.Repeat("1", 64)
	ownerB := strings.Repeat("2", 64)
	inventory := model.ReasoningInventory{
		Generation:            testReasoningGeneration,
		DatasetManifestSHA256: strings.Repeat("c", 64),
	}
	if err := s.BeginReasoningMaterialization(
		ctx, "kitscenes", "v2.1", ownerA, 100, 200,
	); err != nil {
		t.Fatal(err)
	}
	if err := s.RenewReasoningMaterialization(
		ctx, "kitscenes", "v2.1", ownerA, 201, 300,
	); err == nil {
		t.Fatal("expired owner renewed its lease")
	}
	if _, err := s.PutReasoningInventory(
		ctx, "kitscenes", "v2.1", ownerA, 201, inventory,
	); err == nil {
		t.Fatal("expired owner published an inventory")
	}
	if err := s.BeginReasoningMaterialization(
		ctx, "kitscenes", "v2.1", ownerB, 201, 300,
	); err != nil {
		t.Fatalf("new owner did not acquire expired lease: %v", err)
	}
}

func TestDynamoStore_LeaseCASIncludesPublicationRevision(t *testing.T) {
	baseStore, fake := newTestStore()
	ctx := context.Background()
	ownerA := strings.Repeat("1", 64)
	ownerB := strings.Repeat("2", 64)
	ownerC := strings.Repeat("3", 64)
	manifestSHA := strings.Repeat("c", 64)
	initial := model.ReasoningInventory{
		Generation:            testReasoningGeneration,
		DatasetManifestSHA256: manifestSHA,
		SceneRows:             1,
	}
	if err := baseStore.BeginReasoningMaterialization(
		ctx, "kitscenes", "v2.1", ownerA, 100, 200,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := baseStore.PutReasoningInventory(
		ctx, "kitscenes", "v2.1", ownerA, 150, initial,
	); err != nil {
		t.Fatal(err)
	}

	pk, err := ReasoningInventoryPK("kitscenes", "v2.1")
	if err != nil {
		t.Fatal(err)
	}
	key := pk + "\x00" + metaSK
	newer := initial
	newer.SceneRows = 99
	newerPayload, err := json.Marshal(newer)
	if err != nil {
		t.Fatal(err)
	}
	racingClient := &beforeLeasePutDDB{
		ddbAPI: fake,
		before: func() {
			item := cloneAttributeMap(fake.items[key])
			item["payload"] = &ddbtypes.AttributeValueMemberS{
				Value: string(newerPayload),
			}
			item["inventory_revision"] = &ddbtypes.AttributeValueMemberS{
				Value: ownerB,
			}
			fake.items[key] = item
		},
	}
	racingStore := &DynamoStore{
		client: racingClient,
		table:  baseStore.table,
	}

	if err := racingStore.BeginReasoningMaterialization(
		ctx, "kitscenes", "v2.1", ownerC, 300, 400,
	); err == nil {
		t.Fatal("stale lease acquisition replaced a newer publication revision")
	}
	got, _, err := baseStore.GetReasoningInventory(
		ctx, "kitscenes", "v2.1",
	)
	if err != nil {
		t.Fatal(err)
	}
	if got.SceneRows != newer.SceneRows {
		t.Fatalf("publication was rolled back to %+v", got)
	}
}

func TestDynamoStore_ReasoningLeaseAcquireIsRetrySafe(t *testing.T) {
	baseStore, fake := newTestStore()
	client := &ambiguousPutDDB{
		ddbAPI:          fake,
		conditionPrefix: "(attribute_not_exists(#owner)",
	}
	s := &DynamoStore{client: client, table: baseStore.table}
	owner := strings.Repeat("1", 64)

	if err := s.BeginReasoningMaterialization(
		context.Background(), "kitscenes", "v2.1", owner, 100, 200,
	); err == nil {
		t.Fatal("ambiguous committed acquisition did not report its lost response")
	}
	if err := s.BeginReasoningMaterialization(
		context.Background(), "kitscenes", "v2.1", owner, 101, 201,
	); err != nil {
		t.Fatalf("same owner could not retry committed acquisition: %v", err)
	}
}

func TestDynamoStore_ReasoningInventoryPublishIsRetrySafe(t *testing.T) {
	baseStore, fake := newTestStore()
	owner := strings.Repeat("1", 64)
	inventory := model.ReasoningInventory{
		Generation:            testReasoningGeneration,
		DatasetManifestSHA256: strings.Repeat("c", 64),
	}
	if err := baseStore.BeginReasoningMaterialization(
		context.Background(), "kitscenes", "v2.1", owner, 100, 200,
	); err != nil {
		t.Fatal(err)
	}
	client := &ambiguousPutDDB{
		ddbAPI:          fake,
		conditionPrefix: "(#owner = :owner AND #expires >= :now) OR",
	}
	s := &DynamoStore{client: client, table: baseStore.table}

	if _, err := s.PutReasoningInventory(
		context.Background(),
		"kitscenes",
		"v2.1",
		owner,
		150,
		inventory,
	); err == nil {
		t.Fatal("ambiguous committed publication did not report its lost response")
	}
	if _, err := s.PutReasoningInventory(
		context.Background(),
		"kitscenes",
		"v2.1",
		owner,
		151,
		inventory,
	); err != nil {
		t.Fatalf("same owner could not retry committed publication: %v", err)
	}
	got, _, err := s.GetReasoningInventory(
		context.Background(), "kitscenes", "v2.1",
	)
	if err != nil || got.Generation != inventory.Generation {
		t.Fatalf("retried publication = %+v, %v", got, err)
	}
}

func TestDynamoStore_TeacherStatsAndScenesAreIsolated(t *testing.T) {
	s, _ := newTestStore()
	ctx := context.Background()
	const (
		teacherA = "dGVhY2hlci1hAG1vZGVsLTE"
		teacherB = "dGVhY2hlci1iAG1vZGVsLTI"
	)
	blobA := model.ReasoningStatsBlob{NLabels: 1}
	blobB := model.ReasoningStatsBlob{NLabels: 2}
	if _, err := s.PutTeacherStats(
		ctx, "l2d", "v2.1", testReasoningGeneration,
		teacherA, "pv3", blobA,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := s.PutTeacherStats(
		ctx, "l2d", "v2.1", testReasoningGeneration,
		teacherB, "pv3", blobB,
	); err != nil {
		t.Fatal(err)
	}
	gotA, _, err := s.GetTeacherStats(
		ctx, "l2d", "v2.1", testReasoningGeneration, teacherA, "pv3",
	)
	if err != nil || gotA.NLabels != 1 {
		t.Fatalf("teacher A stats = %+v, %v", gotA, err)
	}
	gotB, _, err := s.GetTeacherStats(
		ctx, "l2d", "v2.1", testReasoningGeneration, teacherB, "pv3",
	)
	if err != nil || gotB.NLabels != 2 {
		t.Fatalf("teacher B stats = %+v, %v", gotB, err)
	}

	rowsA := []SceneLabelRow{{
		Field: FieldCause, Value: "lead_vehicle",
		SampleID: "sample-a", Shard: "shard-a.tar",
	}}
	rowsB := []SceneLabelRow{{
		Field: FieldCause, Value: "lead_vehicle",
		SampleID: "sample-b", Shard: "shard-b.tar",
	}}
	if _, err := s.PutReasoningSceneLabels(
		ctx, "l2d", "v2.1", testReasoningGeneration,
		teacherA, "pv3", rowsA,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := s.PutReasoningSceneLabels(
		ctx, "l2d", "v2.1", testReasoningGeneration,
		teacherB, "pv3", rowsB,
	); err != nil {
		t.Fatal(err)
	}
	scenesA, err := s.QueryReasoningScenes(
		ctx, "l2d", "v2.1", testReasoningGeneration, teacherA, "pv3",
		FieldCause, "lead_vehicle", 0,
	)
	if err != nil || len(scenesA) != 1 ||
		scenesA[0].SampleID != "sample-a" ||
		scenesA[0].Shard != "shard-a.tar" {
		t.Fatalf("teacher A scenes = %v, %v", scenesA, err)
	}
	scenesB, err := s.QueryReasoningScenes(
		ctx, "l2d", "v2.1", testReasoningGeneration, teacherB, "pv3",
		FieldCause, "lead_vehicle", 0,
	)
	if err != nil || len(scenesB) != 1 ||
		scenesB[0].SampleID != "sample-b" ||
		scenesB[0].Shard != "shard-b.tar" {
		t.Fatalf("teacher B scenes = %v, %v", scenesB, err)
	}
}

func TestDynamoStore_ReasoningGenerationsExcludeStaleScenes(t *testing.T) {
	s, _ := newTestStore()
	ctx := context.Background()
	const (
		oldGeneration = testReasoningGeneration
		newGeneration = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
		teacher       = "dGVhY2hlcgBtb2RlbA"
	)
	oldRows := []SceneLabelRow{{
		Field: FieldCause, Value: "lead_vehicle",
		SampleID: "stale-sample", Shard: "stale.tar",
	}}
	newRows := []SceneLabelRow{{
		Field: FieldCause, Value: "lead_vehicle",
		SampleID: "current-sample", Shard: "current.tar",
	}}
	if _, err := s.PutReasoningSceneLabels(
		ctx, "kitscenes", "v2.1", oldGeneration,
		teacher, "prompt-v1", oldRows,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := s.PutReasoningSceneLabels(
		ctx, "kitscenes", "v2.1", newGeneration,
		teacher, "prompt-v1", newRows,
	); err != nil {
		t.Fatal(err)
	}
	scenes, err := s.QueryReasoningScenes(
		ctx, "kitscenes", "v2.1", newGeneration,
		teacher, "prompt-v1", FieldCause, "lead_vehicle", 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(scenes) != 1 ||
		scenes[0].SampleID != "current-sample" ||
		scenes[0].Shard != "current.tar" {
		t.Fatalf("new generation scenes = %+v", scenes)
	}
	if !s.client.(*fakeDDB).lastQueryConsistent {
		t.Fatal("reasoning scenes were not queried consistently")
	}
}

func TestDynamoStore_ReasoningSampleLookupRoundTripAndRetry(t *testing.T) {
	s, fake := newTestStore()
	fake.forceUnprocessedOnce = true
	ctx := context.Background()
	lookups := []model.ReasoningSampleLookup{
		{
			SampleID: "sample-a",
			Shard:    "scene-a.tar",
			Offset:   512,
			Size:     1024,
		},
		{
			SampleID: "sample-b",
			Shard:    "scene-b.tar",
			Offset:   2048,
			Size:     512,
		},
	}
	n, err := s.PutReasoningSampleLookups(
		ctx,
		"kitscenes",
		"v2.1",
		testReasoningGeneration,
		lookups,
	)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(lookups) || fake.batchCalls != 2 {
		t.Fatalf("lookup writes=%d batch calls=%d", n, fake.batchCalls)
	}
	got, err := s.GetReasoningSampleLookup(
		ctx,
		"kitscenes",
		"v2.1",
		testReasoningGeneration,
		"sample-b",
	)
	if err != nil {
		t.Fatal(err)
	}
	if got != lookups[1] {
		t.Fatalf("lookup = %+v, want %+v", got, lookups[1])
	}
	if !fake.lastGetConsistent {
		t.Fatal("reasoning lookup was not read consistently")
	}
}

func TestDynamoStore_ReasoningWritesRejectInvalidNamespace(t *testing.T) {
	s, fake := newTestStore()
	if _, err := s.PutReasoningInventory(
		context.Background(),
		"kitscenes",
		"v2.1",
		strings.Repeat("1", 64),
		1,
		model.ReasoningInventory{},
	); err == nil {
		t.Fatal("inventory without generation was accepted")
	}
	if _, err := s.PutReasoningSampleLookups(
		context.Background(),
		"kitscenes#injected",
		"v2.1",
		testReasoningGeneration,
		[]model.ReasoningSampleLookup{{
			SampleID: "sample-a", Shard: "shard.tar", Size: 1,
		}},
	); err == nil {
		t.Fatal("delimiter injection was accepted")
	}
	if len(fake.items) != 0 {
		t.Fatalf("invalid reasoning writes reached DynamoDB: %d items", len(fake.items))
	}
}

func TestDynamoStore_SceneLabelsBatchAndQuery(t *testing.T) {
	s, f := newTestStore()
	ctx := context.Background()

	// 60 rows across two labels -> forces >2 batches of 25.
	var rows []SceneLabelRow
	for i := 0; i < 60; i++ {
		rows = append(rows, SceneLabelRow{Field: FieldLateralResponse, Value: "turn_left", SampleID: sampleID(i)})
	}
	n, err := s.PutSceneLabels(ctx, "l2d", "pv3", rows)
	if err != nil {
		t.Fatalf("PutSceneLabels: %v", err)
	}
	if n != 60 {
		t.Errorf("wrote %d rows, want 60", n)
	}
	if f.batchCalls < 3 {
		t.Errorf("expected >=3 batch calls for 60 rows (cap 25), got %d", f.batchCalls)
	}

	ids, err := s.QueryScenesByLabel(ctx, "l2d", "pv3", FieldLateralResponse, "turn_left", 0)
	if err != nil {
		t.Fatalf("QueryScenesByLabel: %v", err)
	}
	if len(ids) != 60 {
		t.Errorf("query returned %d ids, want 60", len(ids))
	}

	// A different (field,value) returns nothing.
	other, err := s.QueryScenesByLabel(ctx, "l2d", "pv3", FieldLateralResponse, "turn_right", 0)
	if err != nil {
		t.Fatalf("QueryScenesByLabel (other): %v", err)
	}
	if len(other) != 0 {
		t.Errorf("expected 0 scenes for absent label, got %d", len(other))
	}

	// limit caps results.
	limited, err := s.QueryScenesByLabel(ctx, "l2d", "pv3", FieldLateralResponse, "turn_left", 10)
	if err != nil {
		t.Fatalf("QueryScenesByLabel (limit): %v", err)
	}
	if len(limited) != 10 {
		t.Errorf("limit=10 returned %d ids, want 10", len(limited))
	}
}

func TestDynamoStore_VersionedSceneLabelsAreIsolated(t *testing.T) {
	s, _ := newTestStore()
	ctx := context.Background()
	legacy := []SceneLabelRow{{
		Field: FieldLateralResponse, Value: "turn_left", SampleID: "s00000042",
	}}
	current := []SceneLabelRow{{
		Field:    FieldLateralResponse,
		Value:    "turn_left",
		SampleID: "l2d-v1-e000003-f000042",
	}}
	if _, err := s.PutSceneLabels(ctx, "l2d", "pv3", legacy); err != nil {
		t.Fatal(err)
	}
	if _, err := s.PutSceneLabelsForVersion(
		ctx, "l2d", "v2.1", "pv3", current,
	); err != nil {
		t.Fatal(err)
	}

	ids, err := s.QueryScenesByLabelForVersion(
		ctx, "l2d", "v2.1", "pv3",
		FieldLateralResponse, "turn_left", 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 || ids[0] != current[0].SampleID {
		t.Fatalf("versioned scene ids = %v, want current sample_uid only", ids)
	}
	if _, err := s.QueryScenesByLabelForVersion(
		ctx, "l2d", "", "pv3",
		FieldLateralResponse, "turn_left", 0,
	); err == nil {
		t.Fatal("empty dataset version was accepted")
	}
}

func TestDynamoStore_BatchWriteRetriesUnprocessed(t *testing.T) {
	s, f := newTestStore()
	f.forceUnprocessedOnce = true
	ctx := context.Background()

	rows := []SceneLabelRow{{Field: FieldCause, Value: "lead_vehicle", SampleID: "s0"}}
	if _, err := s.PutSceneLabels(ctx, "l2d", "pv3", rows); err != nil {
		t.Fatalf("PutSceneLabels with retry: %v", err)
	}
	ids, err := s.QueryScenesByLabel(ctx, "l2d", "pv3", FieldCause, "lead_vehicle", 0)
	if err != nil {
		t.Fatalf("QueryScenesByLabel: %v", err)
	}
	if len(ids) != 1 || ids[0] != "s0" {
		t.Errorf("retry path lost the write: %v", ids)
	}
	if f.batchCalls != 2 {
		t.Errorf("expected 2 batch calls (1 unprocessed + 1 retry), got %d", f.batchCalls)
	}
}

func TestDynamoStore_OverlayReadinessGatesModelsAndBody(t *testing.T) {
	s, f := newTestStore()
	ctx := context.Background()
	modelID := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	datasetManifest := "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
	cacheIdentity := "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
	pointer := map[string]any{
		"pk":                      ShardModelPK("l2d", "v2.1", "train-000001.tar"),
		"sk":                      ModelSK(modelID),
		"s3_key":                  "overlays/schema=v1/body.gz",
		"sha256":                  "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		"byte_size":               1234,
		"sample_count":            42,
		"overlay_schema":          "v1",
		"dataset_manifest_sha256": datasetManifest,
		"cache_identity":          cacheIdentity,
		"status":                  "ready",
		"registered_model_name":   "auto-e2e-driving-policy",
		"model_version":           30,
		"run_id":                  "run-30",
		"model_name":              "swin_v2_tiny",
		"eval_ade":                1.25,
		"eval_fde":                2.5,
		"val_fraction":            0.1,
	}
	pointerItem, err := attributevalue.MarshalMap(pointer)
	if err != nil {
		t.Fatal(err)
	}
	f.items[keyOf(pointerItem)] = pointerItem

	if models, _, err := s.QueryReadyOverlayModels(
		ctx, "l2d", "v2.1", "train-000001.tar", 100, "",
	); err != nil || len(models) != 0 {
		t.Fatalf("building/missing set must not advertise models: models=%v err=%v", models, err)
	}
	if _, err := s.GetReadyOverlayPointer(ctx, "l2d", "v2.1", "train-000001.tar", modelID); err != ErrNotFound {
		t.Fatalf("building/missing set pointer error = %v, want ErrNotFound", err)
	}

	setItem, err := attributevalue.MarshalMap(map[string]any{
		"pk":                      OverlaySetPK(modelID, "l2d", "v2.1"),
		"sk":                      metaSK,
		"status":                  "ready",
		"dataset_manifest_sha256": datasetManifest,
		"request_identity":        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
		"cache_identity":          cacheIdentity,
		"manifest_key":            "overlays_manifest/model/manifest.json",
	})
	if err != nil {
		t.Fatal(err)
	}
	f.items[keyOf(setItem)] = setItem

	models, nextPageToken, err := s.QueryReadyOverlayModels(
		ctx,
		"l2d",
		"v2.1",
		"train-000001.tar",
		100,
		"",
		datasetManifest,
	)
	if err != nil {
		t.Fatal(err)
	}
	if nextPageToken != "" {
		t.Fatalf("next page token = %q, want empty", nextPageToken)
	}
	if len(models) != 1 || models[0].ModelArtifactID != modelID || models[0].ModelVersion != 30 {
		t.Fatalf("ready models = %+v", models)
	}
	got, err := s.GetReadyOverlayPointer(
		ctx, "l2d", "v2.1", "train-000001.tar", modelID, datasetManifest,
	)
	if err != nil {
		t.Fatal(err)
	}
	if got.S3Key != pointer["s3_key"] || got.ByteSize != 1234 || got.SampleCount != 42 {
		t.Errorf("overlay pointer = %+v", got)
	}
	if got.DatasetManifestSHA256 != datasetManifest || got.CacheIdentity != cacheIdentity {
		t.Errorf("overlay pointer identity = %+v", got)
	}

	staleManifest := "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
	models, _, err = s.QueryReadyOverlayModels(
		ctx,
		"l2d",
		"v2.1",
		"train-000001.tar",
		100,
		"",
		staleManifest,
	)
	if err != nil || len(models) != 0 {
		t.Fatalf("stale models must be hidden: models=%v err=%v", models, err)
	}
	if _, err := s.GetReadyOverlayPointer(
		ctx, "l2d", "v2.1", "train-000001.tar", modelID, staleManifest,
	); err != ErrNotFound {
		t.Fatalf("stale overlay pointer error = %v, want ErrNotFound", err)
	}
	if _, _, err := s.QueryReadyOverlayModels(
		ctx, "l2d", "v2.1", "train-000001.tar", 100, "", "invalid",
	); err == nil {
		t.Fatal("invalid expected manifest digest was accepted")
	}

	pointer["cache_identity"] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
	pointerItem, err = attributevalue.MarshalMap(pointer)
	if err != nil {
		t.Fatal(err)
	}
	f.items[keyOf(pointerItem)] = pointerItem
	if _, err := s.GetReadyOverlayPointer(
		ctx, "l2d", "v2.1", "train-000001.tar", modelID,
	); err == nil {
		t.Fatal("pointer/set cache identity mismatch was accepted")
	}
}

func TestDynamoStore_OverlayModelsPaginationBoundsReadinessLookups(
	t *testing.T,
) {
	s, f := newTestStore()
	ctx := context.Background()
	const (
		dataset = "l2d"
		version = "v2.1"
		shard   = "train-000001.tar"
	)
	datasetManifest := strings.Repeat("c", 64)
	cacheIdentity := strings.Repeat("d", 64)
	requestIdentity := strings.Repeat("e", 64)
	for i := 1; i <= 101; i++ {
		modelID := fmt.Sprintf("%064x", i)
		pointer, err := attributevalue.MarshalMap(map[string]any{
			"pk":                      ShardModelPK(dataset, version, shard),
			"sk":                      ModelSK(modelID),
			"status":                  "ready",
			"dataset_manifest_sha256": datasetManifest,
			"cache_identity":          cacheIdentity,
			"model_version":           i,
		})
		if err != nil {
			t.Fatal(err)
		}
		f.items[keyOf(pointer)] = pointer
		set, err := attributevalue.MarshalMap(map[string]any{
			"pk":                      OverlaySetPK(modelID, dataset, version),
			"sk":                      metaSK,
			"status":                  "ready",
			"dataset_manifest_sha256": datasetManifest,
			"request_identity":        requestIdentity,
			"cache_identity":          cacheIdentity,
			"manifest_key":            "overlays/manifest.json",
		})
		if err != nil {
			t.Fatal(err)
		}
		f.items[keyOf(set)] = set
	}

	first, token, err := s.QueryReadyOverlayModels(
		ctx, dataset, version, shard, 100, "", datasetManifest,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 100 {
		t.Fatalf("first page models = %d, want 100", len(first))
	}
	wantToken := fmt.Sprintf("%064x", 100)
	if token != wantToken {
		t.Fatalf("first page token = %q, want %q", token, wantToken)
	}
	if f.getCalls != 100 {
		t.Fatalf("first page readiness lookups = %d, want 100", f.getCalls)
	}

	f.getCalls = 0
	second, token, err := s.QueryReadyOverlayModels(
		ctx, dataset, version, shard, 100, token, datasetManifest,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(second) != 1 ||
		second[0].ModelArtifactID != fmt.Sprintf("%064x", 101) {
		t.Fatalf("second page models = %+v", second)
	}
	if token != "" {
		t.Fatalf("second page token = %q, want empty", token)
	}
	if f.getCalls != 1 {
		t.Fatalf("second page readiness lookups = %d, want 1", f.getCalls)
	}

	lastModelID := fmt.Sprintf("%064x", 101)
	delete(
		f.items,
		ShardModelPK(dataset, version, shard)+"\x00"+ModelSK(lastModelID),
	)
	exact, terminalToken, err := s.QueryReadyOverlayModels(
		ctx, dataset, version, shard, 100, "", datasetManifest,
	)
	if err != nil || len(exact) != 100 || terminalToken == "" {
		t.Fatalf(
			"exact terminal page = (%d, %q, %v)",
			len(exact),
			terminalToken,
			err,
		)
	}
	empty, terminalToken, err := s.QueryReadyOverlayModels(
		ctx,
		dataset,
		version,
		shard,
		100,
		terminalToken,
		datasetManifest,
	)
	if err != nil || len(empty) != 0 || terminalToken != "" {
		t.Fatalf(
			"terminal probe = (%d, %q, %v)",
			len(empty),
			terminalToken,
			err,
		)
	}

	for _, test := range []struct {
		limit int
		token string
	}{
		{limit: 0},
		{limit: 101},
		{limit: 100, token: "invalid"},
	} {
		if _, _, err := s.QueryReadyOverlayModels(
			ctx,
			dataset,
			version,
			shard,
			test.limit,
			test.token,
			datasetManifest,
		); err == nil {
			t.Fatalf(
				"limit=%d token=%q was accepted",
				test.limit,
				test.token,
			)
		}
	}
}

func TestDynamoStore_GeoRecord(t *testing.T) {
	s, f := newTestStore()
	item, err := attributevalue.MarshalMap(map[string]any{
		"pk":                      GeoPK("l2d", "v2.1"),
		"sk":                      metaSK,
		"summary":                 `{"bbox":[11,48,12,49]}`,
		"geojson_key":             "l2d/v2.1/geo/heatmap.geojson.gz",
		"n_samples":               123,
		"computed_at":             "2026-07-15T00:00:00Z",
		"dataset_manifest_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
	})
	if err != nil {
		t.Fatal(err)
	}
	f.items[keyOf(item)] = item

	got, err := s.GetGeoRecord(context.Background(), "l2d", "v2.1")
	if err != nil {
		t.Fatal(err)
	}
	if got.NSamples != 123 ||
		got.GeoJSONKey != "l2d/v2.1/geo/heatmap.geojson.gz" ||
		got.DatasetManifestSHA256 !=
			"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc" {
		t.Errorf("geo record = %+v", got)
	}
	item["dataset_manifest_sha256"] =
		&ddbtypes.AttributeValueMemberS{Value: "invalid"}
	f.items[keyOf(item)] = item
	if _, err := s.GetGeoRecord(
		context.Background(), "l2d", "v2.1",
	); err == nil {
		t.Fatal("invalid geo publication digest was accepted")
	}
	if _, err := s.GetGeoRecord(context.Background(), "l2d", "v9"); err != ErrNotFound {
		t.Errorf("missing geo record error = %v", err)
	}
}

func sampleID(i int) string {
	const digits = "0123456789"
	b := []byte("s00000000")
	n := i
	for pos := len(b) - 1; pos >= 1 && n > 0; pos-- {
		b[pos] = digits[n%10]
		n /= 10
	}
	return string(b)
}
