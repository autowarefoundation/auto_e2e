package service

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/store"
)

const (
	// statsScanCap fails loudly instead of silently truncating a full-dataset
	// aggregate if a corrupt index advertises an unreasonable label count.
	statsScanCap = 100000
	// reasoning.json is a compact five-horizon record. A larger member is not a
	// valid label and must not consume an API pod's memory.
	maxEmbeddedReasoningBytes = 1 << 20
	// Materialization keeps only these bounded working sets in addition to the
	// incremental per-partition aggregates.
	reasoningFetchBatchSize  = 32
	reasoningLookupBatchSize = 128
	reasoningSceneBatchSize  = 256
	// Bound provenance cardinality and the fixed inventory item well below
	// DynamoDB's item-size limit.
	maxReasoningPartitions     = 128
	maxReasoningInventoryBytes = 300 << 10
	// S3 range reads are process-wide across concurrent materializers.
	labelFetchConcurrency = 32
	// A materializer refreshes this DynamoDB fencing lease before each shard,
	// range batch, and publication phase.
	reasoningMaterializationLease  = 15 * time.Minute
	reasoningMaterializationSchema = "v1"
)

var reasoningLabelFetchSem = make(chan struct{}, labelFetchConcurrency)

// ReasoningStatsDetail returns one precomputed stats blob. Interactive reads
// are cache-only: materialization is an explicit pipeline/admin responsibility,
// never an unauthenticated browser-triggered S3 scan plus DynamoDB write.
func (s *S3Service) ReasoningStatsDetail(ctx context.Context, dataset, version, promptVersion, teacher string) (model.ReasoningStatsDetailResponse, error) {
	teacherProvider, teacherModel, ok := parseReasoningTeacherID(teacher)
	if !ok {
		return model.ReasoningStatsDetailResponse{}, fmt.Errorf(
			"reasoning teacher identity is required",
		)
	}
	var err error
	version, err = s.publishedVersion(ctx, dataset, version)
	if err != nil {
		return model.ReasoningStatsDetailResponse{}, err
	}
	resp := model.ReasoningStatsDetailResponse{
		Dataset:         dataset,
		Version:         version,
		PromptVersion:   promptVersion,
		Teacher:         teacher,
		TeacherProvider: teacherProvider,
		TeacherModel:    teacherModel,
	}

	if s.store == nil {
		return model.ReasoningStatsDetailResponse{}, fmt.Errorf(
			"reasoning stats require a configured dynamo store",
		)
	}
	inventory, _, _, err := s.reasoningInventory(
		ctx, dataset, version,
	)
	if err != nil {
		return model.ReasoningStatsDetailResponse{}, err
	}
	if !inventoryContainsPartition(inventory, teacher, promptVersion) {
		return model.ReasoningStatsDetailResponse{}, ErrNotFound
	}
	blob, computedAt, err := s.store.GetTeacherStats(
		ctx,
		dataset,
		version,
		inventory.Generation,
		teacher,
		promptVersion,
	)
	if err != nil {
		if isStoreNotFound(err) {
			return model.ReasoningStatsDetailResponse{}, fmt.Errorf(
				"%w: stats are missing for advertised partition",
				ErrReasoningIntegrity,
			)
		}
		return model.ReasoningStatsDetailResponse{}, err
	}
	resp.Stats = blob
	resp.ComputedAt = computedAt
	resp.Cached = true
	return resp, nil
}

// MaterializeReasoning scans shard indexes sequentially and fetches labels in
// bounded batches. Every serving row is written below a fresh generation; the
// fixed inventory pointer is replaced only after the complete generation has
// passed validation and all writes have succeeded.
func (s *S3Service) MaterializeReasoning(
	ctx context.Context,
	dataset, version, expectedManifestSHA256 string,
) (model.ReasoningMaterializationResponse, error) {
	if s.store == nil {
		return model.ReasoningMaterializationResponse{}, fmt.Errorf(
			"reasoning materialization requires a configured dynamo store",
		)
	}
	resolvedVersion, err := s.publishedVersion(ctx, dataset, version)
	if err != nil {
		return model.ReasoningMaterializationResponse{}, err
	}
	if !requiresPublicationManifest(resolvedVersion) {
		return model.ReasoningMaterializationResponse{}, fmt.Errorf(
			"reasoning materialization requires an immutable v2.1+ publication",
		)
	}
	if !isLowerHexDigest(expectedManifestSHA256) {
		return model.ReasoningMaterializationResponse{}, fmt.Errorf(
			"expected dataset manifest SHA-256 is required",
		)
	}
	manifest, err := s.loadPublicationManifest(
		ctx, dataset, resolvedVersion,
	)
	if err != nil {
		return model.ReasoningMaterializationResponse{}, err
	}
	if manifest.SHA256 != expectedManifestSHA256 {
		return model.ReasoningMaterializationResponse{}, fmt.Errorf(
			"dataset manifest SHA-256 differs from requested publication",
		)
	}
	expectedRecords := manifest.ReasoningLabelCount
	if expectedRecords < 0 || expectedRecords > statsScanCap {
		return model.ReasoningMaterializationResponse{}, fmt.Errorf(
			"publication reasoning label count %d is outside [0,%d]",
			expectedRecords, statsScanCap,
		)
	}

	generation := reasoningGenerationID(manifest.SHA256)
	owner, err := newReasoningLeaseOwner()
	if err != nil {
		return model.ReasoningMaterializationResponse{}, err
	}
	now := time.Now()
	if err := s.store.BeginReasoningMaterialization(
		ctx,
		dataset,
		resolvedVersion,
		owner,
		now.Unix(),
		now.Add(reasoningMaterializationLease).Unix(),
	); err != nil {
		return model.ReasoningMaterializationResponse{}, err
	}
	published := false
	defer func() {
		if published {
			return
		}
		releaseCtx, cancel := context.WithTimeout(
			context.WithoutCancel(ctx), 5*time.Second,
		)
		defer cancel()
		if err := s.store.ReleaseReasoningMaterialization(
			releaseCtx, dataset, resolvedVersion, owner,
		); err != nil {
			slog.Warn(
				"release reasoning materialization lease",
				"dataset", dataset,
				"version", resolvedVersion,
				"error", err,
			)
		}
	}()

	if inventory, computedAt, err := s.store.GetReasoningInventory(
		ctx, dataset, resolvedVersion,
	); err == nil {
		if validateReasoningInventory(inventory) == nil &&
			inventory.Generation == generation &&
			inventory.DatasetManifestSHA256 == manifest.SHA256 {
			return model.ReasoningMaterializationResponse{
				Dataset:               dataset,
				Version:               resolvedVersion,
				Generation:            generation,
				DatasetManifestSHA256: manifest.SHA256,
				ComputedAt:            computedAt,
				Partitions:            len(inventory.PromptVersions),
				TotalRecords:          inventory.Total,
				SceneRows:             inventory.SceneRows,
				Reused:                true,
			}, nil
		}
	} else if !isStoreNotFound(err) {
		return model.ReasoningMaterializationResponse{}, fmt.Errorf(
			"read current reasoning inventory: %w", err,
		)
	}

	shards, page, err := s.ListShards(
		ctx, dataset, resolvedVersion, int(^uint(0)>>1), 0,
	)
	if err != nil {
		return model.ReasoningMaterializationResponse{}, err
	}
	if page.More {
		return model.ReasoningMaterializationResponse{}, fmt.Errorf(
			"reasoning shard inventory was truncated",
		)
	}
	materialization := &reasoningMaterialization{
		ctx:                   ctx,
		store:                 s.store,
		dataset:               dataset,
		version:               resolvedVersion,
		generation:            generation,
		owner:                 owner,
		datasetManifestSHA256: manifest.SHA256,
		partitions: make(
			map[reasoningPartitionKey]*reasoningPartitionAccumulator,
		),
		seenSamples: make(map[string]struct{}),
	}

	discovered := 0
	for _, shard := range shards {
		if err := materialization.renewLease(); err != nil {
			return model.ReasoningMaterializationResponse{}, err
		}
		index, err := s.BuildShardIndex(
			ctx, dataset, resolvedVersion, shard.Name,
		)
		if err != nil {
			return model.ReasoningMaterializationResponse{}, fmt.Errorf(
				"index reasoning shard %s: %w", shard.Name, err,
			)
		}
		locations := make(
			[]reasoningMemberLocation, 0, reasoningFetchBatchSize,
		)
		flush := func() error {
			if len(locations) == 0 {
				return nil
			}
			if err := materialization.renewLease(); err != nil {
				return err
			}
			records, err := s.fetchEmbeddedReasoning(
				ctx, dataset, resolvedVersion, locations,
			)
			if err != nil {
				return err
			}
			for i := range records {
				if err := materialization.add(records[i]); err != nil {
					return err
				}
			}
			locations = locations[:0]
			return nil
		}

		for _, sample := range index.Samples {
			location, ok, err := reasoningLocation(
				shard.Name, sample,
			)
			if err != nil {
				return model.ReasoningMaterializationResponse{}, err
			}
			if !ok {
				continue
			}
			discovered++
			if discovered > statsScanCap {
				return model.ReasoningMaterializationResponse{}, fmt.Errorf(
					"reasoning label count exceeds cap %d", statsScanCap,
				)
			}
			if expectedRecords >= 0 && discovered > expectedRecords {
				return model.ReasoningMaterializationResponse{}, fmt.Errorf(
					"reasoning record count exceeds publication %d",
					expectedRecords,
				)
			}
			locations = append(locations, location)
			if len(locations) == reasoningFetchBatchSize {
				if err := flush(); err != nil {
					return model.ReasoningMaterializationResponse{}, err
				}
			}
		}
		if err := flush(); err != nil {
			return model.ReasoningMaterializationResponse{}, err
		}
	}
	if expectedRecords >= 0 && discovered != expectedRecords {
		return model.ReasoningMaterializationResponse{}, fmt.Errorf(
			"reasoning record count %d differs from publication %d",
			discovered, expectedRecords,
		)
	}
	if materialization.totalRecords != discovered {
		return model.ReasoningMaterializationResponse{}, fmt.Errorf(
			"reasoning materializer consumed %d of %d records",
			materialization.totalRecords, discovered,
		)
	}
	response, err := materialization.publish()
	if err == nil {
		published = true
	}
	return response, err
}

func newReasoningLeaseOwner() (string, error) {
	var raw [32]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", fmt.Errorf("generate reasoning lease owner: %w", err)
	}
	return hex.EncodeToString(raw[:]), nil
}

func reasoningGenerationID(datasetManifestSHA256 string) string {
	digest := sha256.Sum256([]byte(
		reasoningMaterializationSchema + "\x00" + datasetManifestSHA256,
	))
	return hex.EncodeToString(digest[:])
}

type reasoningPartitionKey struct {
	teacher string
	prompt  string
}

type reasoningPartitionAccumulator struct {
	entry            model.ReasoningPromptVersion
	stats            *store.ReasoningStatsAccumulator
	pendingSceneRows []store.SceneLabelRow
}

type reasoningMaterialization struct {
	ctx                   context.Context
	store                 consoleStore
	dataset               string
	version               string
	generation            string
	owner                 string
	datasetManifestSHA256 string

	partitions       map[reasoningPartitionKey]*reasoningPartitionAccumulator
	seenSamples      map[string]struct{}
	pendingLookups   []model.ReasoningSampleLookup
	pendingSceneRows int
	totalRecords     int
	sceneRows        int
}

func (m *reasoningMaterialization) renewLease() error {
	now := time.Now()
	if err := m.store.RenewReasoningMaterialization(
		m.ctx,
		m.dataset,
		m.version,
		m.owner,
		now.Unix(),
		now.Add(reasoningMaterializationLease).Unix(),
	); err != nil {
		return fmt.Errorf("renew reasoning materialization lease: %w", err)
	}
	return nil
}

func (m *reasoningMaterialization) add(
	record embeddedReasoningLabel,
) error {
	label := record.Label
	if m.seenSamples == nil {
		m.seenSamples = make(map[string]struct{})
	}
	if _, duplicate := m.seenSamples[label.SampleID]; duplicate {
		return fmt.Errorf(
			"duplicate reasoning sample id %q", label.SampleID,
		)
	}
	m.seenSamples[label.SampleID] = struct{}{}
	teacher := reasoningTeacher(label)
	if teacher == "" ||
		!store.ValidReasoningKeyComponent(teacher) ||
		!store.ValidReasoningKeyComponent(label.TeacherProvider) ||
		!store.ValidReasoningKeyComponent(label.TeacherModel) ||
		!store.ValidReasoningKeyComponent(label.PromptVersion) {
		return fmt.Errorf(
			"reasoning record %q has incomplete provenance", label.SampleID,
		)
	}
	key := reasoningPartitionKey{
		teacher: teacher,
		prompt:  label.PromptVersion,
	}
	partition := m.partitions[key]
	if partition == nil {
		if len(m.partitions) >= maxReasoningPartitions {
			return fmt.Errorf(
				"reasoning partition count exceeds %d",
				maxReasoningPartitions,
			)
		}
		partition = &reasoningPartitionAccumulator{
			entry: model.ReasoningPromptVersion{
				Teacher:         teacher,
				TeacherProvider: label.TeacherProvider,
				TeacherModel:    label.TeacherModel,
				PromptVersion:   label.PromptVersion,
			},
			stats: store.NewReasoningStatsAccumulator(),
		}
		m.partitions[key] = partition
	}
	if partition.entry.TeacherProvider != label.TeacherProvider ||
		partition.entry.TeacherModel != label.TeacherModel {
		return fmt.Errorf(
			"reasoning teacher identity collision for %q", teacher,
		)
	}

	partition.stats.Add(label)
	partition.entry.Count++
	m.totalRecords++

	rows := store.SceneLabelRows(label)
	if len(rows) > 0 &&
		m.pendingSceneRows+len(rows) > reasoningSceneBatchSize {
		if err := m.flushSceneRows(); err != nil {
			return err
		}
	}
	for i := range rows {
		rows[i].Shard = record.Location.Shard
	}
	partition.pendingSceneRows = append(
		partition.pendingSceneRows, rows...,
	)
	m.pendingSceneRows += len(rows)

	if len(m.pendingLookups) == reasoningLookupBatchSize {
		if err := m.flushLookups(); err != nil {
			return err
		}
	}
	m.pendingLookups = append(
		m.pendingLookups,
		model.ReasoningSampleLookup{
			SampleID: record.Location.SampleUID,
			Shard:    record.Location.Shard,
			Offset:   record.Location.Range.Offset,
			Size:     record.Location.Range.Size,
		},
	)
	return nil
}

func (m *reasoningMaterialization) flushLookups() error {
	if len(m.pendingLookups) == 0 {
		return nil
	}
	n, err := m.store.PutReasoningSampleLookups(
		m.ctx,
		m.dataset,
		m.version,
		m.generation,
		m.pendingLookups,
	)
	if err != nil {
		return fmt.Errorf("persist reasoning sample lookup: %w", err)
	}
	if n != len(m.pendingLookups) {
		return fmt.Errorf(
			"persisted %d of %d reasoning sample lookups",
			n, len(m.pendingLookups),
		)
	}
	m.pendingLookups = m.pendingLookups[:0]
	return nil
}

func (m *reasoningMaterialization) flushSceneRows() error {
	if m.pendingSceneRows == 0 {
		return nil
	}
	for _, partition := range m.sortedPartitions() {
		if len(partition.pendingSceneRows) == 0 {
			continue
		}
		n, err := m.store.PutReasoningSceneLabels(
			m.ctx,
			m.dataset,
			m.version,
			m.generation,
			partition.entry.Teacher,
			partition.entry.PromptVersion,
			partition.pendingSceneRows,
		)
		if err != nil {
			return fmt.Errorf("persist reasoning scene index: %w", err)
		}
		if n != len(partition.pendingSceneRows) {
			return fmt.Errorf(
				"persisted %d of %d reasoning scene rows",
				n, len(partition.pendingSceneRows),
			)
		}
		m.sceneRows += n
		m.pendingSceneRows -= len(partition.pendingSceneRows)
		partition.pendingSceneRows = nil
	}
	if m.pendingSceneRows != 0 {
		return fmt.Errorf(
			"reasoning scene buffer accounting mismatch: %d",
			m.pendingSceneRows,
		)
	}
	return nil
}

func (m *reasoningMaterialization) sortedPartitions() []*reasoningPartitionAccumulator {
	partitions := make(
		[]*reasoningPartitionAccumulator, 0, len(m.partitions),
	)
	for _, partition := range m.partitions {
		partitions = append(partitions, partition)
	}
	sort.Slice(partitions, func(i, j int) bool {
		a, b := partitions[i].entry, partitions[j].entry
		if a.TeacherProvider != b.TeacherProvider {
			return a.TeacherProvider < b.TeacherProvider
		}
		if a.TeacherModel != b.TeacherModel {
			return a.TeacherModel < b.TeacherModel
		}
		return a.PromptVersion < b.PromptVersion
	})
	return partitions
}

func (m *reasoningMaterialization) publish() (model.ReasoningMaterializationResponse, error) {
	if err := m.renewLease(); err != nil {
		return model.ReasoningMaterializationResponse{}, err
	}
	if err := m.flushLookups(); err != nil {
		return model.ReasoningMaterializationResponse{}, err
	}
	if err := m.flushSceneRows(); err != nil {
		return model.ReasoningMaterializationResponse{}, err
	}
	partitions := m.sortedPartitions()
	inventory := model.ReasoningInventory{
		Generation:            m.generation,
		DatasetManifestSHA256: m.datasetManifestSHA256,
		PromptVersions: make(
			[]model.ReasoningPromptVersion, len(partitions),
		),
		Total:     m.totalRecords,
		SceneRows: m.sceneRows,
	}
	for i, partition := range partitions {
		if err := m.renewLease(); err != nil {
			return model.ReasoningMaterializationResponse{}, err
		}
		stats := partition.stats.Snapshot()
		if stats.NRecords != partition.entry.Count {
			return model.ReasoningMaterializationResponse{}, fmt.Errorf(
				"reasoning stats count mismatch for %s",
				partition.entry.PromptVersion,
			)
		}
		if _, err := m.store.PutTeacherStats(
			m.ctx,
			m.dataset,
			m.version,
			m.generation,
			partition.entry.Teacher,
			partition.entry.PromptVersion,
			stats,
		); err != nil {
			return model.ReasoningMaterializationResponse{}, fmt.Errorf(
				"persist reasoning stats: %w", err,
			)
		}
		inventory.PromptVersions[i] = partition.entry
	}
	if err := validateReasoningInventory(inventory); err != nil {
		return model.ReasoningMaterializationResponse{}, err
	}
	computedAt, err := m.store.PutReasoningInventory(
		m.ctx,
		m.dataset,
		m.version,
		m.owner,
		time.Now().Unix(),
		inventory,
	)
	if err != nil {
		return model.ReasoningMaterializationResponse{}, fmt.Errorf(
			"publish reasoning inventory: %w", err,
		)
	}
	return model.ReasoningMaterializationResponse{
		Dataset:               m.dataset,
		Version:               m.version,
		Generation:            m.generation,
		DatasetManifestSHA256: m.datasetManifestSHA256,
		ComputedAt:            computedAt,
		Partitions:            len(partitions),
		TotalRecords:          inventory.Total,
		SceneRows:             m.sceneRows,
	}, nil
}

func (s *S3Service) reasoningInventory(
	ctx context.Context,
	dataset, version string,
) (model.ReasoningInventory, string, string, error) {
	if s.store == nil {
		return model.ReasoningInventory{}, "", "", fmt.Errorf(
			"reasoning inventory requires a configured dynamo store",
		)
	}
	resolvedVersion, err := s.publishedVersion(ctx, dataset, version)
	if err != nil {
		return model.ReasoningInventory{}, "", "", err
	}
	if !requiresPublicationManifest(resolvedVersion) {
		return model.ReasoningInventory{}, "", "",
			ErrReasoningUnavailable
	}
	manifest, err := s.loadPublicationManifest(
		ctx, dataset, resolvedVersion,
	)
	if err != nil {
		return model.ReasoningInventory{}, "", "", err
	}
	inventory, computedAt, err := s.store.GetReasoningInventory(
		ctx, dataset, resolvedVersion,
	)
	if err != nil {
		if isStoreNotFound(err) {
			return model.ReasoningInventory{}, "", "",
				ErrReasoningUnavailable
		}
		return model.ReasoningInventory{}, "", "", err
	}
	if err := validateReasoningInventory(inventory); err != nil {
		return model.ReasoningInventory{}, "", "", fmt.Errorf(
			"%w: %v", ErrReasoningIntegrity, err,
		)
	}
	if inventory.DatasetManifestSHA256 != manifest.SHA256 {
		return model.ReasoningInventory{}, "", "", fmt.Errorf(
			"%w: reasoning inventory does not match the active publication",
			ErrReasoningIntegrity,
		)
	}
	return inventory, resolvedVersion, computedAt, nil
}

func validateReasoningInventory(inventory model.ReasoningInventory) error {
	if !store.ValidReasoningGeneration(inventory.Generation) ||
		!isLowerHexDigest(inventory.DatasetManifestSHA256) {
		return fmt.Errorf(
			"reasoning inventory has an invalid publication identity",
		)
	}
	if len(inventory.PromptVersions) > maxReasoningPartitions {
		return fmt.Errorf("reasoning inventory has too many partitions")
	}
	total := 0
	seen := make(map[string]struct{}, len(inventory.PromptVersions))
	for _, entry := range inventory.PromptVersions {
		provider, modelName, ok := parseReasoningTeacherID(entry.Teacher)
		if !ok || provider != entry.TeacherProvider ||
			modelName != entry.TeacherModel ||
			!store.ValidReasoningKeyComponent(entry.PromptVersion) ||
			entry.Count <= 0 || entry.Count > statsScanCap-total {
			return fmt.Errorf("reasoning inventory contains an invalid partition")
		}
		key := entry.Teacher + "\x00" + entry.PromptVersion
		if _, exists := seen[key]; exists {
			return fmt.Errorf("reasoning inventory contains a duplicate partition")
		}
		seen[key] = struct{}{}
		total += entry.Count
	}
	if inventory.Total < 0 || inventory.Total > statsScanCap ||
		inventory.SceneRows < 0 ||
		total != inventory.Total {
		return fmt.Errorf(
			"reasoning inventory total mismatch: partitions=%d total=%d",
			total, inventory.Total,
		)
	}
	body, err := json.Marshal(inventory)
	if err != nil {
		return fmt.Errorf("encode reasoning inventory: %w", err)
	}
	if len(body) > maxReasoningInventoryBytes {
		return fmt.Errorf(
			"reasoning inventory is %d bytes, limit is %d",
			len(body),
			maxReasoningInventoryBytes,
		)
	}
	return nil
}

func inventoryContainsPartition(
	inventory model.ReasoningInventory,
	teacher, promptVersion string,
) bool {
	for _, entry := range inventory.PromptVersions {
		if entry.Teacher == teacher &&
			entry.PromptVersion == promptVersion {
			return true
		}
	}
	return false
}

// reasoningMemberLocation is a trusted byte range discovered from a canonical
// v2.1 shard index. SampleUID is the join key carried by meta.json.
type reasoningMemberLocation struct {
	Shard     string
	SampleUID string
	Range     model.MemberRange
}

type embeddedReasoningLabel struct {
	Body     []byte
	Label    store.ReasoningLabel
	Location reasoningMemberLocation
}

func reasoningLocation(
	shard string,
	sample model.IndexSample,
) (reasoningMemberLocation, bool, error) {
	if !store.ValidReasoningKeyComponent(shard) {
		return reasoningMemberLocation{}, false, fmt.Errorf(
			"invalid reasoning shard %q", shard,
		)
	}
	member, hasMember := sample.Members["reasoning.json"]
	if sample.HasReasoning != hasMember {
		return reasoningMemberLocation{}, false, fmt.Errorf(
			"reasoning index membership mismatch for %q", sample.Key,
		)
	}
	if !hasMember {
		return reasoningMemberLocation{}, false, nil
	}
	sampleUID := sample.SampleUID
	if sampleUID == "" {
		sampleUID = sample.Key
	}
	if !store.ValidReasoningKeyComponent(sampleUID) {
		return reasoningMemberLocation{}, false, fmt.Errorf(
			"invalid reasoning sample id %q", sampleUID,
		)
	}
	if member.Offset < 0 ||
		member.Size <= 0 ||
		member.Size > maxEmbeddedReasoningBytes {
		return reasoningMemberLocation{}, false, fmt.Errorf(
			"invalid reasoning member range %d:%d for %s",
			member.Offset, member.Size, sampleUID,
		)
	}
	return reasoningMemberLocation{
		Shard:     shard,
		SampleUID: sampleUID,
		Range:     member,
	}, true, nil
}

// fetchEmbeddedReasoning reads canonical tar members concurrently through
// bounded S3 Range GETs. Malformed records and identity mismatches fail loudly
// because silently dropping either would publish incomplete statistics.
func (s *S3Service) fetchEmbeddedReasoning(
	ctx context.Context,
	dataset, version string,
	locations []reasoningMemberLocation,
) ([]embeddedReasoningLabel, error) {
	type result struct {
		record embeddedReasoningLabel
		ok     bool
		err    error
	}
	results := make([]result, len(locations))

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	for i, location := range locations {
		wg.Add(1)
		go func(i int, location reasoningMemberLocation) {
			defer wg.Done()
			select {
			case reasoningLabelFetchSem <- struct{}{}:
				defer func() { <-reasoningLabelFetchSem }()
			case <-ctx.Done():
				results[i] = result{err: ctx.Err()}
				return
			}
			reader, closer, _, err := s.StreamTarMemberRange(
				ctx,
				dataset,
				version,
				location.Shard,
				location.Range.Offset,
				location.Range.Size,
			)
			if err != nil {
				results[i] = result{err: fmt.Errorf(
					"read reasoning label %s from %s: %w",
					location.SampleUID, location.Shard, err,
				)}
				cancel() // stop siblings on the first hard error
				return
			}
			body, readErr := io.ReadAll(io.LimitReader(
				reader, maxEmbeddedReasoningBytes+1,
			))
			closeErr := closer.Close()
			if readErr != nil {
				results[i] = result{err: readErr}
				cancel()
				return
			}
			if closeErr != nil {
				results[i] = result{err: closeErr}
				cancel()
				return
			}
			if len(body) > maxEmbeddedReasoningBytes {
				results[i] = result{err: fmt.Errorf(
					"reasoning label %s exceeds size cap", location.SampleUID,
				)}
				cancel()
				return
			}
			lbl, perr := store.ParseReasoningLabel(body)
			if perr != nil {
				results[i] = result{err: fmt.Errorf(
					"decode reasoning label %s from %s: %w",
					location.SampleUID, location.Shard, perr,
				)}
				cancel()
				return
			}
			if lbl.SampleID != location.SampleUID {
				results[i] = result{err: fmt.Errorf(
					"reasoning sample id mismatch: member=%s body=%s",
					location.SampleUID, lbl.SampleID,
				)}
				cancel()
				return
			}
			if !reasoningDatasetMatches(dataset, lbl.DatasetName) {
				results[i] = result{err: fmt.Errorf(
					"reasoning dataset mismatch: publication=%s body=%s",
					dataset, lbl.DatasetName,
				)}
				cancel()
				return
			}
			results[i] = result{
				record: embeddedReasoningLabel{
					Body:     body,
					Label:    lbl,
					Location: location,
				},
				ok: true,
			}
		}(i, location)
	}
	wg.Wait()

	records := make([]embeddedReasoningLabel, 0, len(locations))
	for _, r := range results {
		if r.err != nil {
			return nil, r.err
		}
		if r.ok {
			records = append(records, r.record)
		}
	}
	return records, nil
}

func reasoningDatasetMatches(dataset, labelDataset string) bool {
	switch {
	case dataset == "l2d":
		return labelDataset == "yaak-ai/L2D"
	case dataset == "nvidia_av":
		return labelDataset == "nvidia/PhysicalAI-Autonomous-Vehicles"
	case dataset == "kitscenes" || isSmokeDataset(dataset):
		return labelDataset == "KIT-MRT/KITScenes-Multimodal"
	default:
		return false
	}
}

func reasoningTeacher(label store.ReasoningLabel) string {
	if label.TeacherProvider == "" && label.TeacherModel == "" {
		return ""
	}
	identity := label.TeacherProvider + "\x00" + label.TeacherModel
	return base64.RawURLEncoding.EncodeToString([]byte(identity))
}

func parseReasoningTeacherID(teacher string) (string, string, bool) {
	raw, err := base64.RawURLEncoding.DecodeString(teacher)
	if err != nil {
		return "", "", false
	}
	provider, modelName, found := strings.Cut(string(raw), "\x00")
	if !found || strings.Contains(modelName, "\x00") ||
		(provider == "" && modelName == "") {
		return "", "", false
	}
	return provider, modelName, true
}

// ValidReasoningTeacherID reports whether teacher is the canonical opaque
// provider/model identity accepted by stats, scene search, and label reads.
func ValidReasoningTeacherID(teacher string) bool {
	provider, modelName, ok := parseReasoningTeacherID(teacher)
	return ok &&
		store.ValidReasoningKeyComponent(teacher) &&
		store.ValidReasoningKeyComponent(provider) &&
		store.ValidReasoningKeyComponent(modelName)
}

func reasoningTeacherMatches(label store.ReasoningLabel, requested string) bool {
	return requested == "" || requested == reasoningTeacher(label)
}

// SearchScenesByLabelForTeacherAtVersion reads the exact immutable
// dataset/teacher/prompt partition materialized during stats computation.
func (s *S3Service) SearchScenesByLabelForTeacherAtVersion(
	ctx context.Context,
	dataset, version, teacher, promptVersion, field, value string,
	limit int,
) ([]model.SceneRef, string, error) {
	if s.store == nil {
		return nil, "", fmt.Errorf("scene search requires a configured dynamo store")
	}
	if _, _, ok := parseReasoningTeacherID(teacher); !ok {
		return nil, "", fmt.Errorf("scene search requires a teacher identity")
	}
	inventory, resolvedVersion, _, err := s.reasoningInventory(
		ctx, dataset, version,
	)
	if err != nil {
		return nil, "", err
	}
	if !inventoryContainsPartition(inventory, teacher, promptVersion) {
		return nil, resolvedVersion, ErrNotFound
	}
	scenes, err := s.store.QueryReasoningScenes(
		ctx,
		dataset,
		resolvedVersion,
		inventory.Generation,
		teacher,
		promptVersion,
		field,
		value,
		limit,
	)
	return scenes, resolvedVersion, err
}

// isStoreNotFound reports whether err is the store's not-found sentinel.
func isStoreNotFound(err error) bool { return errors.Is(err, store.ErrNotFound) }
