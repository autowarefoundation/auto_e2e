package service

import (
	"context"
	"errors"
	"fmt"
	"path"
	"strings"
	"sync"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/store"
)

// statsScanCap bounds how many label objects a single stats computation reads
// from S3, so a runaway partition cannot turn one request into an unbounded
// scan. The current partitions are ~1k labels; this leaves ample headroom.
const statsScanCap = 20000

// ReasoningStatsDetail returns the precomputed stats blob for a
// (dataset, version, promptVersion), read-through DynamoDB: a hit is returned
// directly (cached=true); on a miss the labels are scanned from S3, aggregated,
// the scene-by-label index is populated, the blob is persisted, and it is
// returned (cached=false). teacher, when non-empty, pins the cache partition;
// otherwise the first teacher partition carrying the promptVersion is used.
func (s *S3Service) ReasoningStatsDetail(ctx context.Context, dataset, version, promptVersion, teacher string) (model.ReasoningStatsDetailResponse, error) {
	resp := model.ReasoningStatsDetailResponse{
		Dataset:       dataset,
		Version:       version,
		PromptVersion: promptVersion,
		Teacher:       teacher,
	}

	if s.store != nil {
		if blob, computedAt, err := s.store.GetStats(ctx, dataset, version, promptVersion); err == nil {
			resp.Stats = blob
			resp.ComputedAt = computedAt
			resp.Cached = true
			return resp, nil
		} else if !isStoreNotFound(err) {
			// A Dynamo read error must not fail the endpoint; fall through to
			// recompute from S3 (logged by the caller on the recompute path).
			resp.Cached = false
		}
	}

	blob, resolvedTeacher, _, err := s.computeAndPersistStats(ctx, dataset, version, promptVersion, teacher)
	if err != nil {
		return model.ReasoningStatsDetailResponse{}, err
	}
	resp.Stats = blob
	resp.Teacher = resolvedTeacher
	resp.Cached = false
	// ComputedAt is set by the persist step (echoed via a fresh Get would be an
	// extra round-trip); leave it to the freshly-written value.
	if s.store != nil {
		if _, computedAt, gerr := s.store.GetStats(ctx, dataset, version, promptVersion); gerr == nil {
			resp.ComputedAt = computedAt
		}
	}
	return resp, nil
}

// ComputeReasoningStats force-(re)computes the stats blob AND repopulates the
// scene-by-label index for a (dataset, promptVersion), returning the blob plus
// the number of scene rows written. Idempotent. version scopes the stats item's
// key only (labels are not shard-versioned in S3); it defaults to the resolved
// newest version when empty.
func (s *S3Service) ComputeReasoningStats(ctx context.Context, dataset, version, promptVersion, teacher string) (model.ComputeStatsResponse, error) {
	if !isVersionDir(version) {
		version = s.resolveVersion(ctx, dataset)
	}
	blob, resolvedTeacher, sceneRows, err := s.computeAndPersistStats(ctx, dataset, version, promptVersion, teacher)
	if err != nil {
		return model.ComputeStatsResponse{}, err
	}
	resp := model.ComputeStatsResponse{
		Dataset:       dataset,
		Version:       version,
		PromptVersion: promptVersion,
		Teacher:       resolvedTeacher,
		SceneRows:     sceneRows,
		Stats:         blob,
	}
	if s.store != nil {
		if _, computedAt, gerr := s.store.GetStats(ctx, dataset, version, promptVersion); gerr == nil {
			resp.ComputedAt = computedAt
		}
	}
	return resp, nil
}

// computeAndPersistStats scans the label partition from S3, aggregates the
// stats blob, populates the scene-by-label index, and persists the blob. It is
// the shared body of the stats-detail miss path and the force-compute endpoint.
// Returns the blob, the resolved teacher, and the count of scene rows written.
func (s *S3Service) computeAndPersistStats(ctx context.Context, dataset, version, promptVersion, teacher string) (model.ReasoningStatsBlob, string, int, error) {
	labels, resolvedTeacher, err := s.scanReasoningLabels(ctx, dataset, promptVersion, teacher)
	if err != nil {
		return model.ReasoningStatsBlob{}, "", 0, err
	}
	blob := store.AggregateStats(labels)

	sceneRows := 0
	if s.store != nil {
		var rows []store.SceneLabelRow
		for _, lbl := range labels {
			rows = append(rows, store.SceneLabelRows(lbl)...)
		}
		if n, werr := s.store.PutSceneLabels(ctx, dataset, promptVersion, rows); werr != nil {
			return model.ReasoningStatsBlob{}, "", 0, fmt.Errorf("populate scene-by-label index: %w", werr)
		} else {
			sceneRows = n
		}
		if _, perr := s.store.PutStats(ctx, dataset, version, promptVersion, blob); perr != nil {
			return model.ReasoningStatsBlob{}, "", 0, fmt.Errorf("persist stats: %w", perr)
		}
	}
	return blob, resolvedTeacher, sceneRows, nil
}

// scanReasoningLabels reads (up to statsScanCap) reasoning-label objects for a
// (dataset, promptVersion) from S3 and parses them. When teacher is empty the
// first teacher partition carrying the promptVersion is used (there is one
// teacher per prompt_version in practice); the resolved teacher is returned.
func (s *S3Service) scanReasoningLabels(ctx context.Context, dataset, promptVersion, teacher string) ([]store.ReasoningLabel, string, error) {
	partition := cacheDataset(dataset)
	prefix := fmt.Sprintf("%sdataset=%s/", reasoningCachePrefix, partition)
	resolvedTeacher := teacher

	var keys []string
	p := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
		Bucket: aws.String(s.bucket),
		Prefix: aws.String(prefix),
	})
	for p.HasMorePages() {
		page, err := p.NextPage(ctx)
		if err != nil {
			return nil, "", fmt.Errorf("list reasoning labels for %s/%s: %w", dataset, promptVersion, err)
		}
		for _, obj := range page.Contents {
			key := aws.ToString(obj.Key)
			if !strings.HasSuffix(key, ".json") {
				continue
			}
			_, t, pv, ok := parseReasoningKey(key)
			if !ok || pv != promptVersion {
				continue
			}
			if teacher != "" && t != teacher {
				continue
			}
			if resolvedTeacher == "" {
				resolvedTeacher = t
			}
			// Guard against mixing two teachers under one prompt_version: once a
			// teacher is resolved, only that partition contributes.
			if t != resolvedTeacher {
				continue
			}
			keys = append(keys, key)
			if len(keys) >= statsScanCap {
				break
			}
		}
		if len(keys) >= statsScanCap {
			break
		}
	}

	labels, err := s.fetchAndParseLabels(ctx, keys)
	if err != nil {
		return nil, "", err
	}
	return labels, resolvedTeacher, nil
}

// labelFetchConcurrency bounds parallel S3 GetObject calls when materialising a
// label partition. Sequential fetches of ~1k small objects blow the request
// timeout (each GET is a full round-trip); a bounded pool keeps latency low
// without exhausting connections/file descriptors.
const labelFetchConcurrency = 32

// fetchAndParseLabels reads and parses each label object concurrently (bounded
// pool). A single malformed label is skipped (a bad teacher output must not
// blank the whole ODD view); any S3 error fails the scan. Order is not
// preserved — the aggregation is order-independent.
func (s *S3Service) fetchAndParseLabels(ctx context.Context, keys []string) ([]store.ReasoningLabel, error) {
	type result struct {
		lbl store.ReasoningLabel
		ok  bool
		err error
	}
	results := make([]result, len(keys))

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	sem := make(chan struct{}, labelFetchConcurrency)
	var wg sync.WaitGroup
	for i, key := range keys {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, key string) {
			defer wg.Done()
			defer func() { <-sem }()
			body, err := s.getObjectBytes(ctx, key)
			if err != nil {
				results[i] = result{err: fmt.Errorf("read reasoning label %s: %w", path.Base(key), err)}
				cancel() // stop siblings on the first hard error
				return
			}
			lbl, perr := store.ParseReasoningLabel(body)
			if perr != nil {
				// Skip a malformed label (not a hard error).
				return
			}
			results[i] = result{lbl: lbl, ok: true}
		}(i, key)
	}
	wg.Wait()

	labels := make([]store.ReasoningLabel, 0, len(keys))
	for _, r := range results {
		if r.err != nil {
			return nil, r.err
		}
		if r.ok {
			labels = append(labels, r.lbl)
		}
	}
	return labels, nil
}

// SearchScenesByLabel returns the sample ids carrying a (field,value) reasoning
// label for a (dataset, promptVersion), from the DynamoDB scene-by-label index.
func (s *S3Service) SearchScenesByLabel(ctx context.Context, dataset, promptVersion, field, value string, limit int) ([]string, error) {
	if s.store == nil {
		return nil, fmt.Errorf("scene search requires a configured dynamo store")
	}
	return s.store.QueryScenesByLabel(ctx, dataset, promptVersion, field, value, limit)
}

// ResolveSampleShards maps each sample id to the published shard (for the given
// version) that actually contains it, by consulting the shard indexes (cached,
// so repeated calls are cheap). A sample id that no published shard holds maps
// to "" — its reasoning label exists but the frame was not packed into this
// version, so the UI must not synthesise a shard name that would 404. Returns a
// map sampleID -> shard ("" when absent). Never errors on a single bad shard;
// shards that fail to index are skipped (best-effort resolution).
func (s *S3Service) ResolveSampleShards(ctx context.Context, dataset, version string, sampleIDs []string) map[string]string {
	out := make(map[string]string, len(sampleIDs))
	want := make(map[string]struct{}, len(sampleIDs))
	for _, id := range sampleIDs {
		want[id] = struct{}{}
		out[id] = "" // default: not present in this version
	}
	if len(want) == 0 {
		return out
	}

	// List this version's shards and probe each index for the wanted keys until
	// all are resolved. Shard indexes are cached (single-flighted), so this is a
	// map lookup after the first warm-up.
	shards, _, err := s.ListShards(ctx, dataset, version, 100000, 0)
	if err != nil {
		return out
	}
	remaining := len(want)
	for _, sh := range shards {
		if remaining == 0 {
			break
		}
		idx, err := s.BuildShardIndex(ctx, dataset, version, sh.Name)
		if err != nil {
			continue // best-effort: skip a shard that won't index
		}
		for _, smp := range idx.Samples {
			if _, ok := want[smp.Key]; ok && out[smp.Key] == "" {
				out[smp.Key] = sh.Name
				remaining--
			}
		}
	}
	return out
}

// isStoreNotFound reports whether err is the store's not-found sentinel.
func isStoreNotFound(err error) bool { return errors.Is(err, store.ErrNotFound) }
