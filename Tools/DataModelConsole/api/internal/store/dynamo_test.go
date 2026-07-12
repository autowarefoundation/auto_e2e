package store

import (
	"context"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	ddbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

// fakeDDB is an in-memory ddbAPI implementation for unit tests (no live AWS).
// It stores items keyed by (pk, sk) and supports GetItem, PutItem,
// BatchWriteItem, and a pk-only Query.
type fakeDDB struct {
	items      map[string]map[string]ddbtypes.AttributeValue
	batchCalls int
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
	k := keyOf(in.Key)
	item, ok := f.items[k]
	if !ok {
		return &dynamodb.GetItemOutput{}, nil
	}
	return &dynamodb.GetItemOutput{Item: item}, nil
}

func (f *fakeDDB) PutItem(_ context.Context, in *dynamodb.PutItemInput, _ ...func(*dynamodb.Options)) (*dynamodb.PutItemOutput, error) {
	f.items[keyOf(in.Item)] = in.Item
	return &dynamodb.PutItemOutput{}, nil
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
	pk := in.ExpressionAttributeValues[":pk"].(*ddbtypes.AttributeValueMemberS).Value
	var items []map[string]ddbtypes.AttributeValue
	for _, item := range f.items {
		if item["pk"].(*ddbtypes.AttributeValueMemberS).Value == pk {
			items = append(items, item)
		}
	}
	return &dynamodb.QueryOutput{Items: items}, nil
}

func newTestStore() (*DynamoStore, *fakeDDB) {
	f := newFakeDDB()
	return &DynamoStore{client: f, table: "test-table"}, f
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
