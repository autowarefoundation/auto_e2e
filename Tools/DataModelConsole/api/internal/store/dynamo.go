package store

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	ddbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

// sceneItem is the projection of a scene-by-label row decoded via
// attributevalue (dynamodbav tags). Only sample_id is needed for search
// results; dataset/prompt_version are decoded for completeness/debugging.
type sceneItem struct {
	SampleID      string `dynamodbav:"sample_id"`
	Dataset       string `dynamodbav:"dataset"`
	PromptVersion string `dynamodbav:"prompt_version"`
}

// ErrNotFound is returned when a requested item is absent from the table.
var ErrNotFound = errors.New("store: not found")

// DefaultTable is the DynamoDB table name when DYNAMO_TABLE is unset.
const DefaultTable = "auto-e2e-console"

// batchWriteMax is DynamoDB's hard cap on items per BatchWriteItem request.
const batchWriteMax = 25

// ddbAPI is the subset of the DynamoDB client the store uses (an interface so
// unit tests can stub it without live AWS).
type ddbAPI interface {
	GetItem(ctx context.Context, in *dynamodb.GetItemInput, opts ...func(*dynamodb.Options)) (*dynamodb.GetItemOutput, error)
	PutItem(ctx context.Context, in *dynamodb.PutItemInput, opts ...func(*dynamodb.Options)) (*dynamodb.PutItemOutput, error)
	BatchWriteItem(ctx context.Context, in *dynamodb.BatchWriteItemInput, opts ...func(*dynamodb.Options)) (*dynamodb.BatchWriteItemOutput, error)
	Query(ctx context.Context, in *dynamodb.QueryInput, opts ...func(*dynamodb.Options)) (*dynamodb.QueryOutput, error)
}

// DynamoStore is the DynamoDB-backed cache: shard indexes, precomputed stats,
// and the scene-by-label search index. It is the source of truth for cached
// artifacts so a pod restart or a second replica reuses them (no unbounded
// in-memory map).
type DynamoStore struct {
	client ddbAPI
	table  string
}

// New builds a DynamoStore from the default AWS credential chain (Pod Identity
// in-cluster, profile/env locally).
func New(ctx context.Context, region, table string) (*DynamoStore, error) {
	if table == "" {
		table = DefaultTable
	}
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, awsconfig.WithRegion(region))
	if err != nil {
		return nil, fmt.Errorf("load aws config: %w", err)
	}
	return &DynamoStore{client: dynamodb.NewFromConfig(awsCfg), table: table}, nil
}

// Table returns the configured table name (for logging/diagnostics).
func (s *DynamoStore) Table() string { return s.table }

// ---------------------------------------------------------------------------
// Shard index (read-through cache; gzip-compressed payload).
//
// A shard index is large (a real shard's index is multi-MB of JSON), well over
// DynamoDB's 400 KB item limit, so the payload is gzip-compressed before store
// and inflated on read (l2d ~1.7MB->77KB, nvidia ~5.2MB->206KB, both fit).
// ---------------------------------------------------------------------------

// GetShardIndex returns a cached shard index, or ErrNotFound on a miss.
func (s *DynamoStore) GetShardIndex(ctx context.Context, dataset, version, shard string) (*model.ShardIndex, error) {
	out, err := s.client.GetItem(ctx, &dynamodb.GetItemInput{
		TableName: aws.String(s.table),
		Key: map[string]ddbtypes.AttributeValue{
			"pk": &ddbtypes.AttributeValueMemberS{Value: ShardIndexPK(dataset, version, shard)},
			"sk": &ddbtypes.AttributeValueMemberS{Value: metaSK},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("get shard index: %w", err)
	}
	if out.Item == nil {
		return nil, ErrNotFound
	}
	raw, err := binaryAttr(out.Item, "payload")
	if err != nil {
		return nil, err
	}
	plain, err := gunzip(raw)
	if err != nil {
		return nil, fmt.Errorf("inflate shard index payload: %w", err)
	}
	var idx model.ShardIndex
	if err := json.Unmarshal(plain, &idx); err != nil {
		return nil, fmt.Errorf("decode shard index payload: %w", err)
	}
	return &idx, nil
}

// PutShardIndex stores a shard index (gzip-compressed payload + built_at).
func (s *DynamoStore) PutShardIndex(ctx context.Context, dataset, version, shard string, idx *model.ShardIndex) error {
	plain, err := json.Marshal(idx)
	if err != nil {
		return fmt.Errorf("encode shard index: %w", err)
	}
	gz, err := gzipBytes(plain)
	if err != nil {
		return fmt.Errorf("compress shard index: %w", err)
	}
	_, err = s.client.PutItem(ctx, &dynamodb.PutItemInput{
		TableName: aws.String(s.table),
		Item: map[string]ddbtypes.AttributeValue{
			"pk":       &ddbtypes.AttributeValueMemberS{Value: ShardIndexPK(dataset, version, shard)},
			"sk":       &ddbtypes.AttributeValueMemberS{Value: metaSK},
			"payload":  &ddbtypes.AttributeValueMemberB{Value: gz},
			"built_at": &ddbtypes.AttributeValueMemberS{Value: nowRFC3339()},
		},
	})
	if err != nil {
		return fmt.Errorf("put shard index: %w", err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Precomputed reasoning stats.
// ---------------------------------------------------------------------------

// GetStats returns a cached stats blob and its computed_at, or ErrNotFound.
func (s *DynamoStore) GetStats(ctx context.Context, dataset, version, promptVersion string) (model.ReasoningStatsBlob, string, error) {
	out, err := s.client.GetItem(ctx, &dynamodb.GetItemInput{
		TableName: aws.String(s.table),
		Key: map[string]ddbtypes.AttributeValue{
			"pk": &ddbtypes.AttributeValueMemberS{Value: StatsPK(dataset, version, promptVersion)},
			"sk": &ddbtypes.AttributeValueMemberS{Value: metaSK},
		},
	})
	if err != nil {
		return model.ReasoningStatsBlob{}, "", fmt.Errorf("get stats: %w", err)
	}
	if out.Item == nil {
		return model.ReasoningStatsBlob{}, "", ErrNotFound
	}
	raw, err := stringAttr(out.Item, "payload")
	if err != nil {
		return model.ReasoningStatsBlob{}, "", err
	}
	var blob model.ReasoningStatsBlob
	if err := json.Unmarshal([]byte(raw), &blob); err != nil {
		return model.ReasoningStatsBlob{}, "", fmt.Errorf("decode stats payload: %w", err)
	}
	computedAt, _ := stringAttr(out.Item, "computed_at")
	return blob, computedAt, nil
}

// PutStats stores a stats blob with computed_at and n_labels. Returns the
// computed_at timestamp it wrote so the caller can echo it in the response.
func (s *DynamoStore) PutStats(ctx context.Context, dataset, version, promptVersion string, blob model.ReasoningStatsBlob) (string, error) {
	payload, err := json.Marshal(blob)
	if err != nil {
		return "", fmt.Errorf("encode stats: %w", err)
	}
	computedAt := nowRFC3339()
	_, err = s.client.PutItem(ctx, &dynamodb.PutItemInput{
		TableName: aws.String(s.table),
		Item: map[string]ddbtypes.AttributeValue{
			"pk":          &ddbtypes.AttributeValueMemberS{Value: StatsPK(dataset, version, promptVersion)},
			"sk":          &ddbtypes.AttributeValueMemberS{Value: metaSK},
			"payload":     &ddbtypes.AttributeValueMemberS{Value: string(payload)},
			"computed_at": &ddbtypes.AttributeValueMemberS{Value: computedAt},
			"n_labels":    &ddbtypes.AttributeValueMemberN{Value: fmt.Sprintf("%d", blob.NLabels)},
		},
	})
	if err != nil {
		return "", fmt.Errorf("put stats: %w", err)
	}
	return computedAt, nil
}

// ---------------------------------------------------------------------------
// Scene-by-label search index.
// ---------------------------------------------------------------------------

// PutSceneLabels writes the scene-by-label search rows for one (dataset,
// promptVersion) in batches of 25 (BatchWriteItem's cap), retrying any
// UnprocessedItems. Idempotent: re-writing the same (field,value,sample_id) is
// a harmless overwrite. Returns the number of rows written.
func (s *DynamoStore) PutSceneLabels(ctx context.Context, dataset, promptVersion string, rows []SceneLabelRow) (int, error) {
	written := 0
	for start := 0; start < len(rows); start += batchWriteMax {
		end := start + batchWriteMax
		if end > len(rows) {
			end = len(rows)
		}
		reqs := make([]ddbtypes.WriteRequest, 0, end-start)
		for _, row := range rows[start:end] {
			reqs = append(reqs, ddbtypes.WriteRequest{
				PutRequest: &ddbtypes.PutRequest{
					Item: map[string]ddbtypes.AttributeValue{
						"pk":             &ddbtypes.AttributeValueMemberS{Value: SceneLabelPK(dataset, promptVersion, row.Field, row.Value)},
						"sk":             &ddbtypes.AttributeValueMemberS{Value: SceneLabelSK(row.SampleID)},
						"sample_id":      &ddbtypes.AttributeValueMemberS{Value: row.SampleID},
						"dataset":        &ddbtypes.AttributeValueMemberS{Value: dataset},
						"prompt_version": &ddbtypes.AttributeValueMemberS{Value: promptVersion},
					},
				},
			})
		}
		if err := s.batchWriteWithRetry(ctx, reqs); err != nil {
			return written, err
		}
		written += len(reqs)
	}
	return written, nil
}

// batchWriteWithRetry issues one BatchWriteItem and retries UnprocessedItems
// with a short backoff (DynamoDB returns unprocessed writes under throttling
// rather than failing the whole batch).
func (s *DynamoStore) batchWriteWithRetry(ctx context.Context, reqs []ddbtypes.WriteRequest) error {
	pending := map[string][]ddbtypes.WriteRequest{s.table: reqs}
	for attempt := 0; attempt < 8; attempt++ {
		out, err := s.client.BatchWriteItem(ctx, &dynamodb.BatchWriteItemInput{RequestItems: pending})
		if err != nil {
			return fmt.Errorf("batch write scene labels: %w", err)
		}
		if len(out.UnprocessedItems) == 0 || len(out.UnprocessedItems[s.table]) == 0 {
			return nil
		}
		pending = out.UnprocessedItems
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Duration(attempt+1) * 50 * time.Millisecond):
		}
	}
	return fmt.Errorf("batch write scene labels: unprocessed items remain after retries")
}

// QueryScenesByLabel returns every sample_id carrying the (field,value) label
// for a (dataset, promptVersion), paginating the DynamoDB Query and capping the
// result at limit (limit<=0 means no cap).
func (s *DynamoStore) QueryScenesByLabel(ctx context.Context, dataset, promptVersion, field, value string, limit int) ([]string, error) {
	pk := SceneLabelPK(dataset, promptVersion, field, value)
	var ids []string
	var startKey map[string]ddbtypes.AttributeValue
	for {
		in := &dynamodb.QueryInput{
			TableName:              aws.String(s.table),
			KeyConditionExpression: aws.String("pk = :pk"),
			ExpressionAttributeValues: map[string]ddbtypes.AttributeValue{
				":pk": &ddbtypes.AttributeValueMemberS{Value: pk},
			},
			ExclusiveStartKey: startKey,
		}
		if limit > 0 {
			// Fetch at most the remaining needed rows this page.
			in.Limit = aws.Int32(int32(limit - len(ids)))
		}
		out, err := s.client.Query(ctx, in)
		if err != nil {
			return nil, fmt.Errorf("query scenes by label: %w", err)
		}
		var items []sceneItem
		if err := attributevalue.UnmarshalListOfMaps(out.Items, &items); err != nil {
			return nil, fmt.Errorf("decode scene items: %w", err)
		}
		for _, it := range items {
			if it.SampleID != "" {
				ids = append(ids, it.SampleID)
			}
		}
		if limit > 0 && len(ids) >= limit {
			return ids[:limit], nil
		}
		if len(out.LastEvaluatedKey) == 0 {
			return ids, nil
		}
		startKey = out.LastEvaluatedKey
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func stringAttr(item map[string]ddbtypes.AttributeValue, name string) (string, error) {
	av, ok := item[name]
	if !ok {
		return "", fmt.Errorf("attribute %q absent", name)
	}
	s, ok := av.(*ddbtypes.AttributeValueMemberS)
	if !ok {
		return "", fmt.Errorf("attribute %q is not a string", name)
	}
	return s.Value, nil
}

func binaryAttr(item map[string]ddbtypes.AttributeValue, name string) ([]byte, error) {
	av, ok := item[name]
	if !ok {
		return nil, fmt.Errorf("attribute %q absent", name)
	}
	b, ok := av.(*ddbtypes.AttributeValueMemberB)
	if !ok {
		return nil, fmt.Errorf("attribute %q is not binary", name)
	}
	return b.Value, nil
}

func gzipBytes(b []byte) ([]byte, error) {
	var buf bytes.Buffer
	w := gzip.NewWriter(&buf)
	if _, err := w.Write(b); err != nil {
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func gunzip(b []byte) ([]byte, error) {
	r, err := gzip.NewReader(bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	defer r.Close()
	return io.ReadAll(r)
}

// nowRFC3339 is time.Now indirected (kept trivial) so timestamps are UTC RFC3339.
func nowRFC3339() string { return time.Now().UTC().Format(time.RFC3339) }
