package store

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	ddbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

// sceneItem is the projection of a scene-by-label row decoded via
// attributevalue (dynamodbav tags). sample_id and shard are returned directly
// to search callers; dataset/prompt_version are decoded for diagnostics.
type sceneItem struct {
	SampleID      string `dynamodbav:"sample_id"`
	Shard         string `dynamodbav:"shard"`
	Dataset       string `dynamodbav:"dataset"`
	PromptVersion string `dynamodbav:"prompt_version"`
}

type reasoningSampleLookupItem struct {
	SampleID string `dynamodbav:"sample_id"`
	Shard    string `dynamodbav:"shard"`
	Offset   int64  `dynamodbav:"offset"`
	Size     int64  `dynamodbav:"size"`
}

type overlayPointerItem struct {
	S3Key                 string  `dynamodbav:"s3_key"`
	SHA256                string  `dynamodbav:"sha256"`
	ByteSize              int64   `dynamodbav:"byte_size"`
	SampleCount           int     `dynamodbav:"sample_count"`
	OverlaySchema         string  `dynamodbav:"overlay_schema"`
	DatasetManifestSHA256 string  `dynamodbav:"dataset_manifest_sha256"`
	CacheIdentity         string  `dynamodbav:"cache_identity"`
	Status                string  `dynamodbav:"status"`
	RegisteredModelName   string  `dynamodbav:"registered_model_name"`
	ModelVersion          int     `dynamodbav:"model_version"`
	RunID                 string  `dynamodbav:"run_id"`
	ModelName             string  `dynamodbav:"model_name"`
	EvalADE               float64 `dynamodbav:"eval_ade"`
	EvalFDE               float64 `dynamodbav:"eval_fde"`
	ValFraction           float64 `dynamodbav:"val_fraction"`
	SK                    string  `dynamodbav:"sk"`
}

type overlaySetItem struct {
	Status                string `dynamodbav:"status"`
	DatasetManifestSHA256 string `dynamodbav:"dataset_manifest_sha256"`
	RequestIdentity       string `dynamodbav:"request_identity"`
	CacheIdentity         string `dynamodbav:"cache_identity"`
	ManifestKey           string `dynamodbav:"manifest_key"`
}

// OverlayPointer is the validated S3 locator for one canonical shard body.
type OverlayPointer struct {
	ModelArtifactID       string
	S3Key                 string
	SHA256                string
	ByteSize              int64
	SampleCount           int
	OverlaySchema         string
	DatasetManifestSHA256 string
	CacheIdentity         string
}

// GeoRecord is the serving metadata for one privacy-filtered geo artifact set.
type GeoRecord struct {
	Summary               string `dynamodbav:"summary"`
	GeoJSONKey            string `dynamodbav:"geojson_key"`
	NSamples              int    `dynamodbav:"n_samples"`
	ComputedAt            string `dynamodbav:"computed_at"`
	DatasetManifestSHA256 string `dynamodbav:"dataset_manifest_sha256"`
}

// ErrNotFound is returned when a requested item is absent from the table.
var ErrNotFound = errors.New("store: not found")

// ErrShardIndexTooLarge is returned when a compressed shard index expands
// beyond the maximum payload the API is willing to hold in memory.
var ErrShardIndexTooLarge = errors.New("store: shard index exceeds expanded size limit")

// DefaultTable is the DynamoDB table name when DYNAMO_TABLE is unset.
const DefaultTable = "auto-e2e-console"

// batchWriteMax is DynamoDB's hard cap on items per BatchWriteItem request.
const batchWriteMax = 25

const (
	// One overlay-model page performs one readiness lookup per pointer.
	// Keeping this hard bound in the store protects non-HTTP callers too.
	maxOverlayModelPageSize = 100
	// Leave room below DynamoDB's 400 KiB item limit for attribute names and
	// lease metadata.
	maxReasoningInventoryPayloadBytes = 300 << 10
	maxShardIndexCompressedBytes      = 350 << 10

	// Real shard indexes are currently at most about 5.2 MiB of JSON. Keep
	// enough growth room without allowing a small DynamoDB value to inflate
	// until it exhausts the API process.
	maxShardIndexExpandedBytes = 16 << 20

	// Decoding retains both expanded JSON and the decoded index briefly. This
	// process-wide bound limits that peak across all DynamoStore instances.
	maxConcurrentShardIndexDecodes = 4
)

var shardIndexDecodeSem = make(
	chan struct{},
	maxConcurrentShardIndexDecodes,
)

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

	shardIndexMu    sync.Mutex
	shardIndexCalls map[string]*shardIndexCall
}

type shardIndexCall struct {
	done         chan struct{}
	index        *model.ShardIndex
	err          error
	participants int
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
	return &DynamoStore{
		client:          dynamodb.NewFromConfig(awsCfg),
		table:           table,
		shardIndexCalls: make(map[string]*shardIndexCall),
	}, nil
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
	flightKey := ShardIndexPK(dataset, version, shard)
	s.shardIndexMu.Lock()
	if s.shardIndexCalls == nil {
		s.shardIndexCalls = make(map[string]*shardIndexCall)
	}
	if call, ok := s.shardIndexCalls[flightKey]; ok {
		call.participants++
		s.shardIndexMu.Unlock()
		select {
		case <-call.done:
			return call.index, call.err
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	call := &shardIndexCall{
		done:         make(chan struct{}),
		participants: 1,
	}
	s.shardIndexCalls[flightKey] = call
	s.shardIndexMu.Unlock()

	defer func() {
		recovered := recover()
		if recovered != nil {
			call.err = errors.New("shard index decode panicked")
		}
		s.shardIndexMu.Lock()
		delete(s.shardIndexCalls, flightKey)
		close(call.done)
		s.shardIndexMu.Unlock()
		if recovered != nil {
			panic(recovered)
		}
	}()

	call.index, call.err = s.getShardIndex(
		ctx, version, shard, flightKey,
	)
	return call.index, call.err
}

func (s *DynamoStore) getShardIndex(
	ctx context.Context,
	version, shard, pk string,
) (*model.ShardIndex, error) {
	out, err := s.client.GetItem(ctx, &dynamodb.GetItemInput{
		TableName: aws.String(s.table),
		Key: map[string]ddbtypes.AttributeValue{
			"pk": &ddbtypes.AttributeValueMemberS{Value: pk},
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
	release, err := acquireShardIndexDecode(ctx)
	if err != nil {
		return nil, err
	}
	defer release()
	plain, err := gunzipLimited(raw, maxShardIndexExpandedBytes)
	if err != nil {
		return nil, fmt.Errorf("inflate shard index payload: %w", err)
	}
	var idx model.ShardIndex
	if err := json.Unmarshal(plain, &idx); err != nil {
		return nil, fmt.Errorf("decode shard index payload: %w", err)
	}
	// The DynamoDB key is authoritative. Older cache entries did not always
	// persist these response fields, so populate them before sharing the
	// decoded pointer with concurrent callers.
	idx.Version = version
	idx.Shard = shard
	return &idx, nil
}

// PutShardIndex stores a shard index (gzip-compressed payload + built_at).
func (s *DynamoStore) PutShardIndex(ctx context.Context, dataset, version, shard string, idx *model.ShardIndex) error {
	plain, err := json.Marshal(idx)
	if err != nil {
		return fmt.Errorf("encode shard index: %w", err)
	}
	if len(plain) > maxShardIndexExpandedBytes {
		return fmt.Errorf(
			"%w: encoded payload is %d bytes (limit %d)",
			ErrShardIndexTooLarge,
			len(plain),
			maxShardIndexExpandedBytes,
		)
	}
	gz, err := gzipBytes(plain)
	if err != nil {
		return fmt.Errorf("compress shard index: %w", err)
	}
	if len(gz) > maxShardIndexCompressedBytes {
		return fmt.Errorf(
			"%w: compressed payload is %d bytes (limit %d)",
			ErrShardIndexTooLarge,
			len(gz),
			maxShardIndexCompressedBytes,
		)
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
	return s.getStats(
		ctx, EmbeddedStatsPK(dataset, version, promptVersion),
	)
}

// GetTeacherStats returns only the exact provider/model/prompt partition.
func (s *DynamoStore) GetTeacherStats(
	ctx context.Context,
	dataset, version, generation, teacherID, promptVersion string,
) (model.ReasoningStatsBlob, string, error) {
	pk, err := EmbeddedTeacherStatsPK(
		dataset, version, generation, teacherID, promptVersion,
	)
	if err != nil {
		return model.ReasoningStatsBlob{}, "", err
	}
	return s.getStats(ctx, pk)
}

func (s *DynamoStore) getStats(
	ctx context.Context,
	pk string,
) (model.ReasoningStatsBlob, string, error) {
	out, err := s.client.GetItem(ctx, &dynamodb.GetItemInput{
		TableName:      aws.String(s.table),
		ConsistentRead: aws.Bool(true),
		Key: map[string]ddbtypes.AttributeValue{
			"pk": &ddbtypes.AttributeValueMemberS{Value: pk},
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
	return s.putStats(
		ctx, EmbeddedStatsPK(dataset, version, promptVersion), "", blob,
	)
}

// PutTeacherStats persists one exact provider/model/prompt aggregate.
func (s *DynamoStore) PutTeacherStats(
	ctx context.Context,
	dataset, version, generation, teacherID, promptVersion string,
	blob model.ReasoningStatsBlob,
) (string, error) {
	pk, err := EmbeddedTeacherStatsPK(
		dataset, version, generation, teacherID, promptVersion,
	)
	if err != nil {
		return "", err
	}
	return s.putStats(ctx, pk, teacherID, blob)
}

func (s *DynamoStore) putStats(
	ctx context.Context,
	pk, teacherID string,
	blob model.ReasoningStatsBlob,
) (string, error) {
	payload, err := json.Marshal(blob)
	if err != nil {
		return "", fmt.Errorf("encode stats: %w", err)
	}
	computedAt := nowRFC3339()
	item := map[string]ddbtypes.AttributeValue{
		"pk":          &ddbtypes.AttributeValueMemberS{Value: pk},
		"sk":          &ddbtypes.AttributeValueMemberS{Value: metaSK},
		"payload":     &ddbtypes.AttributeValueMemberS{Value: string(payload)},
		"computed_at": &ddbtypes.AttributeValueMemberS{Value: computedAt},
		"n_labels":    &ddbtypes.AttributeValueMemberN{Value: fmt.Sprintf("%d", blob.NLabels)},
	}
	if teacherID != "" {
		item["teacher_id"] = &ddbtypes.AttributeValueMemberS{Value: teacherID}
	}
	_, err = s.client.PutItem(ctx, &dynamodb.PutItemInput{
		TableName: aws.String(s.table),
		Item:      item,
	})
	if err != nil {
		return "", fmt.Errorf("put stats: %w", err)
	}
	return computedAt, nil
}

// GetReasoningInventory returns the all-partition publication gate for one
// immutable dataset version.
func (s *DynamoStore) GetReasoningInventory(
	ctx context.Context,
	dataset, version string,
) (model.ReasoningInventory, string, error) {
	pk, err := ReasoningInventoryPK(dataset, version)
	if err != nil {
		return model.ReasoningInventory{}, "", err
	}
	out, err := s.client.GetItem(ctx, &dynamodb.GetItemInput{
		TableName:      aws.String(s.table),
		ConsistentRead: aws.Bool(true),
		Key: map[string]ddbtypes.AttributeValue{
			"pk": &ddbtypes.AttributeValueMemberS{Value: pk},
			"sk": &ddbtypes.AttributeValueMemberS{Value: metaSK},
		},
	})
	if err != nil {
		return model.ReasoningInventory{}, "", fmt.Errorf(
			"get reasoning inventory: %w", err,
		)
	}
	if out.Item == nil {
		return model.ReasoningInventory{}, "", ErrNotFound
	}
	if _, ok := out.Item["payload"]; !ok {
		// A first materialization may hold the lease before any generation has
		// been published. Readers must continue to see "not materialized".
		return model.ReasoningInventory{}, "", ErrNotFound
	}
	raw, err := stringAttr(out.Item, "payload")
	if err != nil {
		return model.ReasoningInventory{}, "", err
	}
	var inventory model.ReasoningInventory
	if err := json.Unmarshal([]byte(raw), &inventory); err != nil {
		return model.ReasoningInventory{}, "", fmt.Errorf(
			"decode reasoning inventory: %w", err,
		)
	}
	if !ValidReasoningGeneration(inventory.Generation) {
		return model.ReasoningInventory{}, "", fmt.Errorf(
			"reasoning inventory has invalid generation",
		)
	}
	manifestDigest, err := stringAttr(
		out.Item, "dataset_manifest_sha256",
	)
	if err != nil ||
		manifestDigest != inventory.DatasetManifestSHA256 ||
		!validSHA256(manifestDigest) {
		return model.ReasoningInventory{}, "", fmt.Errorf(
			"reasoning inventory has invalid publication identity",
		)
	}
	activeGeneration, err := stringAttr(out.Item, "inventory_generation")
	if err != nil || activeGeneration != inventory.Generation {
		return model.ReasoningInventory{}, "", fmt.Errorf(
			"reasoning inventory generation metadata differs",
		)
	}
	computedAt, _ := stringAttr(out.Item, "computed_at")
	return inventory, computedAt, nil
}

// BeginReasoningMaterialization acquires a dataset/version lease while
// preserving the currently published payload. The active generation condition
// prevents a stale read from overwriting a publish that races this acquisition.
func (s *DynamoStore) BeginReasoningMaterialization(
	ctx context.Context,
	dataset, version, owner string,
	nowUnix, expiresUnix int64,
) error {
	if !ValidReasoningGeneration(owner) ||
		nowUnix <= 0 ||
		expiresUnix <= nowUnix {
		return fmt.Errorf("invalid reasoning materialization lease")
	}
	pk, err := ReasoningInventoryPK(dataset, version)
	if err != nil {
		return err
	}
	out, err := s.client.GetItem(ctx, &dynamodb.GetItemInput{
		TableName:      aws.String(s.table),
		ConsistentRead: aws.Bool(true),
		Key: map[string]ddbtypes.AttributeValue{
			"pk": &ddbtypes.AttributeValueMemberS{Value: pk},
			"sk": &ddbtypes.AttributeValueMemberS{Value: metaSK},
		},
	})
	if err != nil {
		return fmt.Errorf("read reasoning materialization lease: %w", err)
	}
	item := cloneAttributeMap(out.Item)
	if item == nil {
		item = make(map[string]ddbtypes.AttributeValue)
	}
	item["pk"] = &ddbtypes.AttributeValueMemberS{Value: pk}
	item["sk"] = &ddbtypes.AttributeValueMemberS{Value: metaSK}
	item["materialization_owner"] =
		&ddbtypes.AttributeValueMemberS{Value: owner}
	item["materialization_expires_at"] =
		&ddbtypes.AttributeValueMemberN{Value: strconv.FormatInt(expiresUnix, 10)}

	condition := "(attribute_not_exists(#owner) OR #expires < :now OR #owner = :owner)"
	values := map[string]ddbtypes.AttributeValue{
		":now": &ddbtypes.AttributeValueMemberN{
			Value: strconv.FormatInt(nowUnix, 10),
		},
		":owner": &ddbtypes.AttributeValueMemberS{Value: owner},
	}
	if active, ok := out.Item["inventory_generation"]; ok {
		condition += " AND inventory_generation = :active"
		values[":active"] = active
	} else {
		condition += " AND attribute_not_exists(inventory_generation)"
	}
	if revision, ok := out.Item["inventory_revision"]; ok {
		condition += " AND inventory_revision = :revision"
		values[":revision"] = revision
	} else {
		condition += " AND attribute_not_exists(inventory_revision)"
	}
	_, err = s.client.PutItem(ctx, &dynamodb.PutItemInput{
		TableName:           aws.String(s.table),
		Item:                item,
		ConditionExpression: aws.String(condition),
		ExpressionAttributeNames: map[string]string{
			"#owner":   "materialization_owner",
			"#expires": "materialization_expires_at",
		},
		ExpressionAttributeValues: values,
	})
	if err != nil {
		return fmt.Errorf("acquire reasoning materialization lease: %w", err)
	}
	return nil
}

// RenewReasoningMaterialization extends only the caller's own lease. If a
// newer owner fenced it out, the conditional write fails and the old run stops.
func (s *DynamoStore) RenewReasoningMaterialization(
	ctx context.Context,
	dataset, version, owner string,
	nowUnix, expiresUnix int64,
) error {
	return s.rewriteReasoningMaterializationLease(
		ctx, dataset, version, owner, nowUnix, expiresUnix, false,
	)
}

// ReleaseReasoningMaterialization removes only the caller's own lease while
// preserving the last published inventory.
func (s *DynamoStore) ReleaseReasoningMaterialization(
	ctx context.Context,
	dataset, version, owner string,
) error {
	return s.rewriteReasoningMaterializationLease(
		ctx, dataset, version, owner, 0, 0, true,
	)
}

func (s *DynamoStore) rewriteReasoningMaterializationLease(
	ctx context.Context,
	dataset, version, owner string,
	nowUnix, expiresUnix int64,
	release bool,
) error {
	if !ValidReasoningGeneration(owner) ||
		(!release && (nowUnix <= 0 || expiresUnix <= nowUnix)) {
		return fmt.Errorf("invalid reasoning materialization lease")
	}
	pk, err := ReasoningInventoryPK(dataset, version)
	if err != nil {
		return err
	}
	out, err := s.client.GetItem(ctx, &dynamodb.GetItemInput{
		TableName:      aws.String(s.table),
		ConsistentRead: aws.Bool(true),
		Key: map[string]ddbtypes.AttributeValue{
			"pk": &ddbtypes.AttributeValueMemberS{Value: pk},
			"sk": &ddbtypes.AttributeValueMemberS{Value: metaSK},
		},
	})
	if err != nil {
		return fmt.Errorf("read reasoning materialization lease: %w", err)
	}
	if out.Item == nil {
		return ErrNotFound
	}
	item := cloneAttributeMap(out.Item)
	if release {
		delete(item, "materialization_owner")
		delete(item, "materialization_expires_at")
	} else {
		item["materialization_expires_at"] =
			&ddbtypes.AttributeValueMemberN{
				Value: strconv.FormatInt(expiresUnix, 10),
			}
	}
	condition := "#owner = :owner"
	values := map[string]ddbtypes.AttributeValue{
		":owner": &ddbtypes.AttributeValueMemberS{Value: owner},
	}
	names := map[string]string{
		"#owner": "materialization_owner",
	}
	if !release {
		condition += " AND #expires >= :now"
		names["#expires"] = "materialization_expires_at"
		values[":now"] = &ddbtypes.AttributeValueMemberN{
			Value: strconv.FormatInt(nowUnix, 10),
		}
	}
	_, err = s.client.PutItem(ctx, &dynamodb.PutItemInput{
		TableName:                 aws.String(s.table),
		Item:                      item,
		ConditionExpression:       aws.String(condition),
		ExpressionAttributeNames:  names,
		ExpressionAttributeValues: values,
	})
	if err != nil {
		return fmt.Errorf("rewrite reasoning materialization lease: %w", err)
	}
	return nil
}

// PutReasoningInventory publishes discovery only after all partition writes
// succeed. The owner condition fences out a slow run whose lease was replaced.
func (s *DynamoStore) PutReasoningInventory(
	ctx context.Context,
	dataset, version string,
	owner string,
	nowUnix int64,
	inventory model.ReasoningInventory,
) (string, error) {
	pk, err := ReasoningInventoryPK(dataset, version)
	if err != nil {
		return "", err
	}
	if !ValidReasoningGeneration(owner) ||
		!ValidReasoningGeneration(inventory.Generation) ||
		nowUnix <= 0 {
		return "", fmt.Errorf("reasoning inventory has invalid generation")
	}
	if !validSHA256(inventory.DatasetManifestSHA256) {
		return "", fmt.Errorf(
			"reasoning inventory has invalid dataset manifest digest",
		)
	}
	payload, err := json.Marshal(inventory)
	if err != nil {
		return "", fmt.Errorf("encode reasoning inventory: %w", err)
	}
	if len(payload) > maxReasoningInventoryPayloadBytes {
		return "", fmt.Errorf(
			"reasoning inventory payload is %d bytes, limit is %d",
			len(payload),
			maxReasoningInventoryPayloadBytes,
		)
	}
	computedAt := nowRFC3339()
	_, err = s.client.PutItem(ctx, &dynamodb.PutItemInput{
		TableName: aws.String(s.table),
		ConditionExpression: aws.String(
			"(#owner = :owner AND #expires >= :now) OR " +
				"(attribute_not_exists(#owner) AND " +
				"inventory_revision = :owner)",
		),
		ExpressionAttributeNames: map[string]string{
			"#owner":   "materialization_owner",
			"#expires": "materialization_expires_at",
		},
		ExpressionAttributeValues: map[string]ddbtypes.AttributeValue{
			":owner": &ddbtypes.AttributeValueMemberS{
				Value: owner,
			},
			":now": &ddbtypes.AttributeValueMemberN{
				Value: strconv.FormatInt(nowUnix, 10),
			},
		},
		Item: map[string]ddbtypes.AttributeValue{
			"pk": &ddbtypes.AttributeValueMemberS{Value: pk},
			"sk": &ddbtypes.AttributeValueMemberS{Value: metaSK},
			"payload": &ddbtypes.AttributeValueMemberS{
				Value: string(payload),
			},
			"computed_at": &ddbtypes.AttributeValueMemberS{
				Value: computedAt,
			},
			"inventory_generation": &ddbtypes.AttributeValueMemberS{
				Value: inventory.Generation,
			},
			"inventory_revision": &ddbtypes.AttributeValueMemberS{
				Value: owner,
			},
			"dataset_manifest_sha256": &ddbtypes.AttributeValueMemberS{
				Value: inventory.DatasetManifestSHA256,
			},
			"total": &ddbtypes.AttributeValueMemberN{
				Value: strconv.Itoa(inventory.Total),
			},
		},
	})
	if err != nil {
		return "", fmt.Errorf("put reasoning inventory: %w", err)
	}
	return computedAt, nil
}

// PutReasoningSampleLookups stores direct sample_uid to shard/member-range
// pointers under one unpublished generation.
func (s *DynamoStore) PutReasoningSampleLookups(
	ctx context.Context,
	dataset, version, generation string,
	lookups []model.ReasoningSampleLookup,
) (int, error) {
	if err := validateReasoningKeyComponents(dataset, version); err != nil {
		return 0, err
	}
	if !ValidReasoningGeneration(generation) {
		return 0, fmt.Errorf("invalid reasoning generation")
	}
	written := 0
	for start := 0; start < len(lookups); start += batchWriteMax {
		end := min(start+batchWriteMax, len(lookups))
		reqs := make([]ddbtypes.WriteRequest, 0, end-start)
		for _, lookup := range lookups[start:end] {
			pk, err := ReasoningSampleLookupPK(
				dataset, version, generation, lookup.SampleID,
			)
			if err != nil {
				return written, err
			}
			if err := validateReasoningKeyComponents(lookup.Shard); err != nil {
				return written, err
			}
			if lookup.Offset < 0 || lookup.Size <= 0 {
				return written, fmt.Errorf(
					"invalid reasoning member range",
				)
			}
			reqs = append(reqs, ddbtypes.WriteRequest{
				PutRequest: &ddbtypes.PutRequest{
					Item: map[string]ddbtypes.AttributeValue{
						"pk": &ddbtypes.AttributeValueMemberS{
							Value: pk,
						},
						"sk": &ddbtypes.AttributeValueMemberS{
							Value: metaSK,
						},
						"sample_id": &ddbtypes.AttributeValueMemberS{
							Value: lookup.SampleID,
						},
						"shard": &ddbtypes.AttributeValueMemberS{
							Value: lookup.Shard,
						},
						"offset": &ddbtypes.AttributeValueMemberN{
							Value: strconv.FormatInt(lookup.Offset, 10),
						},
						"size": &ddbtypes.AttributeValueMemberN{
							Value: strconv.FormatInt(lookup.Size, 10),
						},
						"generation": &ddbtypes.AttributeValueMemberS{
							Value: generation,
						},
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

// GetReasoningSampleLookup resolves one direct member pointer from the active
// generation selected by the caller's already-validated inventory.
func (s *DynamoStore) GetReasoningSampleLookup(
	ctx context.Context,
	dataset, version, generation, sampleID string,
) (model.ReasoningSampleLookup, error) {
	pk, err := ReasoningSampleLookupPK(
		dataset, version, generation, sampleID,
	)
	if err != nil {
		return model.ReasoningSampleLookup{}, err
	}
	out, err := s.client.GetItem(ctx, &dynamodb.GetItemInput{
		TableName:      aws.String(s.table),
		ConsistentRead: aws.Bool(true),
		Key: map[string]ddbtypes.AttributeValue{
			"pk": &ddbtypes.AttributeValueMemberS{Value: pk},
			"sk": &ddbtypes.AttributeValueMemberS{Value: metaSK},
		},
	})
	if err != nil {
		return model.ReasoningSampleLookup{}, fmt.Errorf(
			"get reasoning sample lookup: %w", err,
		)
	}
	if out.Item == nil {
		return model.ReasoningSampleLookup{}, ErrNotFound
	}
	var item reasoningSampleLookupItem
	if err := attributevalue.UnmarshalMap(out.Item, &item); err != nil {
		return model.ReasoningSampleLookup{}, fmt.Errorf(
			"decode reasoning sample lookup: %w", err,
		)
	}
	if item.SampleID != sampleID || item.Shard == "" ||
		item.Offset < 0 || item.Size <= 0 {
		return model.ReasoningSampleLookup{}, fmt.Errorf(
			"reasoning sample lookup is incomplete",
		)
	}
	return model.ReasoningSampleLookup{
		SampleID: item.SampleID,
		Shard:    item.Shard,
		Offset:   item.Offset,
		Size:     item.Size,
	}, nil
}

// ---------------------------------------------------------------------------
// Scene-by-label search index.
// ---------------------------------------------------------------------------

// PutSceneLabels writes the scene-by-label search rows for one (dataset,
// promptVersion) in batches of 25 (BatchWriteItem's cap), retrying any
// UnprocessedItems. Idempotent: re-writing the same (field,value,sample_id) is
// a harmless overwrite. Returns the number of rows written.
func (s *DynamoStore) PutSceneLabels(ctx context.Context, dataset, promptVersion string, rows []SceneLabelRow) (int, error) {
	return s.putSceneLabels(
		ctx, dataset, "", "", "", promptVersion, rows,
	)
}

// PutSceneLabelsForVersion writes sample_uid rows into the versioned LBLV2
// namespace so legacy flat-id rows can never leak into current search.
func (s *DynamoStore) PutSceneLabelsForVersion(
	ctx context.Context,
	dataset, version, promptVersion string,
	rows []SceneLabelRow,
) (int, error) {
	if version == "" {
		return 0, fmt.Errorf("dataset version is required for scene labels")
	}
	return s.putSceneLabels(
		ctx, dataset, version, "", "", promptVersion, rows,
	)
}

// PutReasoningSceneLabels writes one exact generation/provider/model
// partition. Shard is persisted with each row so search is a single DynamoDB
// query rather than a follow-up scan of every shard index.
func (s *DynamoStore) PutReasoningSceneLabels(
	ctx context.Context,
	dataset, version, generation, teacherID, promptVersion string,
	rows []SceneLabelRow,
) (int, error) {
	if err := validateReasoningKeyComponents(
		dataset, version, teacherID, promptVersion,
	); err != nil {
		return 0, err
	}
	if !ValidReasoningGeneration(generation) {
		return 0, fmt.Errorf("invalid reasoning generation")
	}
	return s.putSceneLabels(
		ctx,
		dataset,
		version,
		generation,
		teacherID,
		promptVersion,
		rows,
	)
}

func (s *DynamoStore) putSceneLabels(
	ctx context.Context,
	dataset, version, generation, teacherID, promptVersion string,
	rows []SceneLabelRow,
) (int, error) {
	written := 0
	for start := 0; start < len(rows); start += batchWriteMax {
		end := start + batchWriteMax
		if end > len(rows) {
			end = len(rows)
		}
		reqs := make([]ddbtypes.WriteRequest, 0, end-start)
		for _, row := range rows[start:end] {
			if err := validateReasoningKeyComponents(row.SampleID); err != nil {
				return written, err
			}
			pk := SceneLabelPK(
				dataset, promptVersion, row.Field, row.Value,
			)
			if version != "" {
				if generation != "" {
					var err error
					pk, err = SceneLabelTeacherVersionPK(
						dataset,
						version,
						generation,
						teacherID,
						promptVersion,
						row.Field,
						row.Value,
					)
					if err != nil {
						return written, err
					}
					if err := validateReasoningKeyComponents(
						row.Shard,
					); err != nil {
						return written, err
					}
				} else if teacherID == "" {
					pk = SceneLabelVersionPK(
						dataset, version, promptVersion,
						row.Field, row.Value,
					)
				} else {
					return written, fmt.Errorf(
						"scene label generation is required",
					)
				}
			}
			item := map[string]ddbtypes.AttributeValue{
				"pk":             &ddbtypes.AttributeValueMemberS{Value: pk},
				"sk":             &ddbtypes.AttributeValueMemberS{Value: SceneLabelSK(row.SampleID)},
				"sample_id":      &ddbtypes.AttributeValueMemberS{Value: row.SampleID},
				"dataset":        &ddbtypes.AttributeValueMemberS{Value: dataset},
				"prompt_version": &ddbtypes.AttributeValueMemberS{Value: promptVersion},
			}
			if version != "" {
				item["version"] = &ddbtypes.AttributeValueMemberS{Value: version}
			}
			if teacherID != "" {
				item["teacher_id"] = &ddbtypes.AttributeValueMemberS{Value: teacherID}
			}
			if generation != "" {
				item["generation"] = &ddbtypes.AttributeValueMemberS{
					Value: generation,
				}
				item["shard"] = &ddbtypes.AttributeValueMemberS{
					Value: row.Shard,
				}
			}
			reqs = append(reqs, ddbtypes.WriteRequest{
				PutRequest: &ddbtypes.PutRequest{
					Item: item,
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
			return fmt.Errorf("batch write reasoning items: %w", err)
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
	return fmt.Errorf(
		"batch write reasoning items: unprocessed items remain after retries",
	)
}

// QueryScenesByLabel returns every sample_id carrying the (field,value) label
// for a (dataset, promptVersion), paginating the DynamoDB Query and capping the
// result at limit (limit<=0 means no cap).
func (s *DynamoStore) QueryScenesByLabel(ctx context.Context, dataset, promptVersion, field, value string, limit int) ([]string, error) {
	return s.queryScenesByLabel(
		ctx,
		SceneLabelPK(dataset, promptVersion, field, value),
		limit,
	)
}

// QueryScenesByLabelForVersion reads only the sample_uid-based LBLV2 index for
// one immutable dataset version.
func (s *DynamoStore) QueryScenesByLabelForVersion(
	ctx context.Context,
	dataset, version, promptVersion, field, value string,
	limit int,
) ([]string, error) {
	if version == "" {
		return nil, fmt.Errorf("dataset version is required for scene search")
	}
	return s.queryScenesByLabel(
		ctx,
		SceneLabelVersionPK(dataset, version, promptVersion, field, value),
		limit,
	)
}

// QueryReasoningScenes reads only one exact materialization generation and
// returns the shard stored with each sample.
func (s *DynamoStore) QueryReasoningScenes(
	ctx context.Context,
	dataset, version, generation, teacherID, promptVersion, field, value string,
	limit int,
) ([]model.SceneRef, error) {
	pk, err := SceneLabelTeacherVersionPK(
		dataset,
		version,
		generation,
		teacherID,
		promptVersion,
		field,
		value,
	)
	if err != nil {
		return nil, err
	}
	return s.queryReasoningScenesByLabel(ctx, pk, limit)
}

func (s *DynamoStore) queryScenesByLabel(
	ctx context.Context,
	pk string,
	limit int,
) ([]string, error) {
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

func (s *DynamoStore) queryReasoningScenesByLabel(
	ctx context.Context,
	pk string,
	limit int,
) ([]model.SceneRef, error) {
	var scenes []model.SceneRef
	var startKey map[string]ddbtypes.AttributeValue
	for {
		in := &dynamodb.QueryInput{
			TableName:              aws.String(s.table),
			ConsistentRead:         aws.Bool(true),
			KeyConditionExpression: aws.String("pk = :pk"),
			ExpressionAttributeValues: map[string]ddbtypes.AttributeValue{
				":pk": &ddbtypes.AttributeValueMemberS{Value: pk},
			},
			ExclusiveStartKey: startKey,
		}
		if limit > 0 {
			in.Limit = aws.Int32(int32(limit - len(scenes)))
		}
		out, err := s.client.Query(ctx, in)
		if err != nil {
			return nil, fmt.Errorf("query reasoning scenes: %w", err)
		}
		var items []sceneItem
		if err := attributevalue.UnmarshalListOfMaps(
			out.Items, &items,
		); err != nil {
			return nil, fmt.Errorf("decode reasoning scene items: %w", err)
		}
		for _, item := range items {
			if item.SampleID == "" || item.Shard == "" {
				return nil, fmt.Errorf("reasoning scene item is incomplete")
			}
			scenes = append(scenes, model.SceneRef{
				SampleID:  item.SampleID,
				Shard:     item.Shard,
				Available: true,
			})
		}
		if limit > 0 && len(scenes) >= limit {
			return scenes[:limit], nil
		}
		if len(out.LastEvaluatedKey) == 0 {
			return scenes, nil
		}
		startKey = out.LastEvaluatedKey
	}
}

// QueryReadyOverlayModels returns one DynamoDB page of models advertised for a
// shard only after both the pointer and its whole overlay-set gate are ready.
// pageToken is the previous page's model artifact ID.
func (s *DynamoStore) QueryReadyOverlayModels(
	ctx context.Context,
	dataset, version, shard string,
	limit int,
	pageToken string,
	expectedDatasetManifestSHA256 ...string,
) ([]model.OverlayModel, string, error) {
	if limit < 1 || limit > maxOverlayModelPageSize {
		return nil, "", fmt.Errorf(
			"overlay model limit must be between 1 and %d",
			maxOverlayModelPageSize,
		)
	}
	expectedDigest, err := optionalSHA256(expectedDatasetManifestSHA256)
	if err != nil {
		return nil, "", err
	}
	pk := ShardModelPK(dataset, version, shard)
	var startKey map[string]ddbtypes.AttributeValue
	if pageToken != "" {
		if !validSHA256(pageToken) {
			return nil, "", fmt.Errorf("invalid overlay model page token")
		}
		startKey = map[string]ddbtypes.AttributeValue{
			"pk": &ddbtypes.AttributeValueMemberS{Value: pk},
			"sk": &ddbtypes.AttributeValueMemberS{
				Value: ModelSK(pageToken),
			},
		}
	}
	out, err := s.client.Query(ctx, &dynamodb.QueryInput{
		TableName:              aws.String(s.table),
		KeyConditionExpression: aws.String("pk = :pk AND begins_with(sk, :model)"),
		ExpressionAttributeValues: map[string]ddbtypes.AttributeValue{
			":pk":    &ddbtypes.AttributeValueMemberS{Value: pk},
			":model": &ddbtypes.AttributeValueMemberS{Value: "MODEL#"},
		},
		ExclusiveStartKey: startKey,
		Limit:             aws.Int32(int32(limit)),
	})
	if err != nil {
		return nil, "", fmt.Errorf("query overlay models: %w", err)
	}
	var items []overlayPointerItem
	if err := attributevalue.UnmarshalListOfMaps(out.Items, &items); err != nil {
		return nil, "", fmt.Errorf("decode overlay models: %w", err)
	}
	models := make([]model.OverlayModel, 0, len(items))
	for _, item := range items {
		modelArtifactID := strings.TrimPrefix(item.SK, "MODEL#")
		if item.Status != "ready" || !validSHA256(modelArtifactID) {
			continue
		}
		set, err := s.readyOverlaySet(ctx, modelArtifactID, dataset, version)
		if err != nil {
			return nil, "", err
		}
		if set == nil {
			continue
		}
		if item.DatasetManifestSHA256 != set.DatasetManifestSHA256 ||
			item.CacheIdentity != set.CacheIdentity {
			continue
		}
		if expectedDigest != "" &&
			set.DatasetManifestSHA256 != expectedDigest {
			continue
		}
		models = append(models, model.OverlayModel{
			ModelArtifactID:     modelArtifactID,
			RegisteredModelName: item.RegisteredModelName,
			ModelVersion:        item.ModelVersion,
			RunID:               item.RunID,
			ModelName:           item.ModelName,
			EvalADE:             item.EvalADE,
			EvalFDE:             item.EvalFDE,
			ValFraction:         item.ValFraction,
			OverlaySchema:       item.OverlaySchema,
			SampleCount:         item.SampleCount,
		})
	}
	sort.Slice(models, func(i, j int) bool {
		if models[i].ModelVersion != models[j].ModelVersion {
			return models[i].ModelVersion > models[j].ModelVersion
		}
		return models[i].ModelArtifactID < models[j].ModelArtifactID
	})
	nextPageToken, err := overlayModelPageToken(out.LastEvaluatedKey, pk)
	if err != nil {
		return nil, "", err
	}
	return models, nextPageToken, nil
}

// GetReadyOverlayPointer resolves one pointer after enforcing the set-level
// publication gate. A building set is intentionally indistinguishable from a
// missing overlay to readers.
func (s *DynamoStore) GetReadyOverlayPointer(
	ctx context.Context,
	dataset, version, shard, modelArtifactID string,
	expectedDatasetManifestSHA256 ...string,
) (*OverlayPointer, error) {
	expectedDigest, err := optionalSHA256(expectedDatasetManifestSHA256)
	if err != nil {
		return nil, err
	}
	out, err := s.client.GetItem(ctx, &dynamodb.GetItemInput{
		TableName: aws.String(s.table),
		Key: map[string]ddbtypes.AttributeValue{
			"pk": &ddbtypes.AttributeValueMemberS{Value: ShardModelPK(dataset, version, shard)},
			"sk": &ddbtypes.AttributeValueMemberS{Value: ModelSK(modelArtifactID)},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("get overlay pointer: %w", err)
	}
	if out.Item == nil {
		return nil, ErrNotFound
	}
	var item overlayPointerItem
	if err := attributevalue.UnmarshalMap(out.Item, &item); err != nil {
		return nil, fmt.Errorf("decode overlay pointer: %w", err)
	}
	set, err := s.readyOverlaySet(ctx, modelArtifactID, dataset, version)
	if err != nil {
		return nil, err
	}
	if item.Status != "ready" || set == nil {
		return nil, ErrNotFound
	}
	if item.DatasetManifestSHA256 != set.DatasetManifestSHA256 ||
		item.CacheIdentity != set.CacheIdentity {
		return nil, fmt.Errorf("overlay pointer identity differs from ready set")
	}
	if expectedDigest != "" &&
		set.DatasetManifestSHA256 != expectedDigest {
		return nil, ErrNotFound
	}
	if item.S3Key == "" || item.SHA256 == "" || item.ByteSize <= 0 || item.SampleCount <= 0 {
		return nil, fmt.Errorf("overlay pointer is incomplete")
	}
	return &OverlayPointer{
		ModelArtifactID:       modelArtifactID,
		S3Key:                 item.S3Key,
		SHA256:                item.SHA256,
		ByteSize:              item.ByteSize,
		SampleCount:           item.SampleCount,
		OverlaySchema:         item.OverlaySchema,
		DatasetManifestSHA256: item.DatasetManifestSHA256,
		CacheIdentity:         item.CacheIdentity,
	}, nil
}

func (s *DynamoStore) readyOverlaySet(ctx context.Context, modelArtifactID, dataset, version string) (*overlaySetItem, error) {
	out, err := s.client.GetItem(ctx, &dynamodb.GetItemInput{
		TableName: aws.String(s.table),
		Key: map[string]ddbtypes.AttributeValue{
			"pk": &ddbtypes.AttributeValueMemberS{Value: OverlaySetPK(modelArtifactID, dataset, version)},
			"sk": &ddbtypes.AttributeValueMemberS{Value: metaSK},
		},
		ProjectionExpression: aws.String(
			"#status, dataset_manifest_sha256, request_identity, " +
				"cache_identity, manifest_key",
		),
		ExpressionAttributeNames: map[string]string{"#status": "status"},
	})
	if err != nil {
		return nil, fmt.Errorf("get overlay-set status: %w", err)
	}
	if out.Item == nil {
		return nil, nil
	}
	var item overlaySetItem
	if err := attributevalue.UnmarshalMap(out.Item, &item); err != nil {
		return nil, fmt.Errorf("decode overlay-set status: %w", err)
	}
	if item.Status != "ready" {
		return nil, nil
	}
	if len(item.DatasetManifestSHA256) != 64 ||
		len(item.RequestIdentity) != 64 ||
		len(item.CacheIdentity) != 64 ||
		item.ManifestKey == "" {
		return nil, fmt.Errorf("ready overlay set is incomplete")
	}
	return &item, nil
}

// GetGeoRecord returns the privacy-filtered summary and S3 heatmap pointer.
func (s *DynamoStore) GetGeoRecord(ctx context.Context, dataset, version string) (*GeoRecord, error) {
	out, err := s.client.GetItem(ctx, &dynamodb.GetItemInput{
		TableName: aws.String(s.table),
		Key: map[string]ddbtypes.AttributeValue{
			"pk": &ddbtypes.AttributeValueMemberS{Value: GeoPK(dataset, version)},
			"sk": &ddbtypes.AttributeValueMemberS{Value: metaSK},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("get geo stats: %w", err)
	}
	if out.Item == nil {
		return nil, ErrNotFound
	}
	var record GeoRecord
	if err := attributevalue.UnmarshalMap(out.Item, &record); err != nil {
		return nil, fmt.Errorf("decode geo stats: %w", err)
	}
	if !json.Valid([]byte(record.Summary)) || record.GeoJSONKey == "" {
		return nil, fmt.Errorf("geo stats item is incomplete")
	}
	if !validSHA256(record.DatasetManifestSHA256) {
		return nil, fmt.Errorf("geo stats item has invalid dataset manifest digest")
	}
	return &record, nil
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func optionalSHA256(values []string) (string, error) {
	if len(values) > 1 {
		return "", fmt.Errorf("expected at most one dataset manifest digest")
	}
	if len(values) == 0 || values[0] == "" {
		return "", nil
	}
	if !validSHA256(values[0]) {
		return "", fmt.Errorf("invalid dataset manifest digest")
	}
	return values[0], nil
}

func validSHA256(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, char := range value {
		if (char < '0' || char > '9') &&
			(char < 'a' || char > 'f') {
			return false
		}
	}
	return true
}

func overlayModelPageToken(
	key map[string]ddbtypes.AttributeValue,
	expectedPK string,
) (string, error) {
	if len(key) == 0 {
		return "", nil
	}
	pk, err := stringAttr(key, "pk")
	if err != nil || pk != expectedPK {
		return "", fmt.Errorf("invalid overlay model continuation key")
	}
	sk, err := stringAttr(key, "sk")
	if err != nil || !strings.HasPrefix(sk, "MODEL#") {
		return "", fmt.Errorf("invalid overlay model continuation key")
	}
	token := strings.TrimPrefix(sk, "MODEL#")
	if !validSHA256(token) {
		return "", fmt.Errorf("invalid overlay model continuation key")
	}
	return token, nil
}

func cloneAttributeMap(
	item map[string]ddbtypes.AttributeValue,
) map[string]ddbtypes.AttributeValue {
	if item == nil {
		return nil
	}
	cloned := make(map[string]ddbtypes.AttributeValue, len(item))
	for name, value := range item {
		cloned[name] = value
	}
	return cloned
}

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

func acquireShardIndexDecode(ctx context.Context) (func(), error) {
	select {
	case shardIndexDecodeSem <- struct{}{}:
		return func() { <-shardIndexDecodeSem }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func gunzipLimited(b []byte, limit int) ([]byte, error) {
	r, err := gzip.NewReader(bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	defer r.Close()
	plain, err := io.ReadAll(io.LimitReader(r, int64(limit)+1))
	if err != nil {
		return nil, err
	}
	if len(plain) > limit {
		return nil, fmt.Errorf(
			"%w: limit is %d bytes",
			ErrShardIndexTooLarge,
			limit,
		)
	}
	return plain, nil
}

// nowRFC3339 is time.Now indirected (kept trivial) so timestamps are UTC RFC3339.
func nowRFC3339() string { return time.Now().UTC().Format(time.RFC3339) }
