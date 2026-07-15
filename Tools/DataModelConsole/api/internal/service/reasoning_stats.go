package service

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"

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
)

// ReasoningStatsDetail returns the precomputed stats blob for a
// (dataset, version, teacher, promptVersion), read-through DynamoDB: a hit is
// returned directly (cached=true); on a miss the exact teacher partition is
// scanned from S3, aggregated, indexed, persisted, and returned (cached=false).
func (s *S3Service) ReasoningStatsDetail(ctx context.Context, dataset, version, promptVersion, teacher string) (model.ReasoningStatsDetailResponse, error) {
	teacherProvider, teacherModel, ok := parseReasoningTeacherID(teacher)
	if !ok {
		return model.ReasoningStatsDetailResponse{}, fmt.Errorf(
			"reasoning teacher identity is required",
		)
	}
	// Resolve the version BEFORE touching the store, mirroring
	// ComputeReasoningStats. Otherwise the read-through path stores/reads under
	// an empty-version key while force-compute writes under the resolved
	// version, so the two never share a cache entry (a permanent miss here).
	if !isVersionDir(version) {
		version = s.resolveVersion(ctx, dataset)
	}
	resp := model.ReasoningStatsDetailResponse{
		Dataset:         dataset,
		Version:         version,
		PromptVersion:   promptVersion,
		Teacher:         teacher,
		TeacherProvider: teacherProvider,
		TeacherModel:    teacherModel,
	}

	if s.store != nil {
		if blob, computedAt, err := s.store.GetTeacherStats(
			ctx, dataset, version, teacher, promptVersion,
		); err == nil {
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
		if _, computedAt, gerr := s.store.GetTeacherStats(
			ctx, dataset, version, teacher, promptVersion,
		); gerr == nil {
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
	teacherProvider, teacherModel, ok := parseReasoningTeacherID(teacher)
	if !ok {
		return model.ComputeStatsResponse{}, fmt.Errorf(
			"reasoning teacher identity is required",
		)
	}
	if !isVersionDir(version) {
		version = s.resolveVersion(ctx, dataset)
	}
	blob, resolvedTeacher, sceneRows, err := s.computeAndPersistStats(ctx, dataset, version, promptVersion, teacher)
	if err != nil {
		return model.ComputeStatsResponse{}, err
	}
	resp := model.ComputeStatsResponse{
		Dataset:         dataset,
		Version:         version,
		PromptVersion:   promptVersion,
		Teacher:         resolvedTeacher,
		TeacherProvider: teacherProvider,
		TeacherModel:    teacherModel,
		SceneRows:       sceneRows,
		Stats:           blob,
	}
	if s.store != nil {
		if _, computedAt, gerr := s.store.GetTeacherStats(
			ctx, dataset, version, teacher, promptVersion,
		); gerr == nil {
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
	labels, resolvedTeacher, err := s.scanReasoningLabels(
		ctx, dataset, version, promptVersion, teacher,
	)
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
		if n, werr := s.store.PutSceneLabelsForTeacherVersion(
			ctx, dataset, version, resolvedTeacher, promptVersion, rows,
		); werr != nil {
			return model.ReasoningStatsBlob{}, "", 0, fmt.Errorf("populate scene-by-label index: %w", werr)
		} else {
			sceneRows = n
		}
		if _, perr := s.store.PutTeacherStats(
			ctx, dataset, version, resolvedTeacher, promptVersion, blob,
		); perr != nil {
			return model.ReasoningStatsBlob{}, "", 0, fmt.Errorf("persist stats: %w", perr)
		}
	}
	return blob, resolvedTeacher, sceneRows, nil
}

// reasoningMemberLocation is a trusted byte range discovered from a canonical
// v2.1 shard index. SampleUID is the join key carried by meta.json.
type reasoningMemberLocation struct {
	Shard     string
	SampleUID string
	Range     model.MemberRange
}

type embeddedReasoningLabel struct {
	Body  []byte
	Label store.ReasoningLabel
}

// reasoningMemberLocations resolves embedded labels without consulting the
// obsolete per-sample cache. targetSampleUID narrows the search for playback;
// empty means every embedded record for stats/index materialization.
func (s *S3Service) reasoningMemberLocations(
	ctx context.Context,
	dataset, version, targetSampleUID string,
) ([]reasoningMemberLocation, string, error) {
	version = s.versionOrResolve(ctx, dataset, version)
	shards, _, err := s.ListShards(ctx, dataset, version, 100000, 0)
	if err != nil {
		return nil, version, err
	}

	locations := make([]reasoningMemberLocation, 0)
	for _, shard := range shards {
		index, err := s.BuildShardIndex(ctx, dataset, version, shard.Name)
		if err != nil {
			return nil, version, fmt.Errorf(
				"index reasoning shard %s: %w", shard.Name, err,
			)
		}
		for _, sample := range index.Samples {
			sampleUID := sample.SampleUID
			if sampleUID == "" {
				sampleUID = sample.Key
			}
			if targetSampleUID != "" &&
				targetSampleUID != sampleUID &&
				targetSampleUID != sample.Key {
				continue
			}
			member, ok := sample.Members["reasoning.json"]
			if !ok || !sample.HasReasoning {
				continue
			}
			if member.Size <= 0 || member.Size > maxEmbeddedReasoningBytes {
				return nil, version, fmt.Errorf(
					"invalid reasoning member size %d for %s",
					member.Size, sampleUID,
				)
			}
			locations = append(locations, reasoningMemberLocation{
				Shard:     shard.Name,
				SampleUID: sampleUID,
				Range:     member,
			})
			if targetSampleUID != "" {
				return locations, version, nil
			}
			if len(locations) > statsScanCap {
				return nil, version, fmt.Errorf(
					"reasoning label count exceeds cap %d", statsScanCap,
				)
			}
		}
	}
	return locations, version, nil
}

// labelFetchConcurrency bounds parallel S3 GetObject calls when materialising a
// label partition. Sequential fetches of ~1k small objects blow the request
// timeout (each GET is a full round-trip); a bounded pool keeps latency low
// without exhausting connections/file descriptors.
const labelFetchConcurrency = 32

// fetchEmbeddedReasoning reads canonical tar members concurrently through
// bounded S3 Range GETs. A malformed JSON record is skipped, while a sample-id
// mismatch fails loudly because it means the shard join is corrupt.
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

	sem := make(chan struct{}, labelFetchConcurrency)
	var wg sync.WaitGroup
	for i, location := range locations {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, location reasoningMemberLocation) {
			defer wg.Done()
			defer func() { <-sem }()
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
				// Skip a malformed label (not a hard error).
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
			results[i] = result{
				record: embeddedReasoningLabel{Body: body, Label: lbl},
				ok:     true,
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

// scanReasoningLabels reads embedded v2.1 labels and selects one explicit
// prompt/teacher partition from provenance carried inside each JSON record.
func (s *S3Service) scanReasoningLabels(
	ctx context.Context,
	dataset, version, promptVersion, teacher string,
) ([]store.ReasoningLabel, string, error) {
	if _, _, ok := parseReasoningTeacherID(teacher); !ok {
		return nil, "", fmt.Errorf("reasoning teacher identity is required")
	}
	locations, resolvedVersion, err := s.reasoningMemberLocations(
		ctx, dataset, version, "",
	)
	if err != nil {
		return nil, "", err
	}
	records, err := s.fetchEmbeddedReasoning(
		ctx, dataset, resolvedVersion, locations,
	)
	if err != nil {
		return nil, "", err
	}

	labels := make([]store.ReasoningLabel, 0, len(records))
	for _, record := range records {
		label := record.Label
		if label.PromptVersion != promptVersion ||
			!reasoningTeacherMatches(label, teacher) {
			continue
		}
		labels = append(labels, label)
	}
	if len(labels) == 0 {
		return nil, "", ErrNotFound
	}
	return labels, teacher, nil
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

func reasoningTeacherMatches(label store.ReasoningLabel, requested string) bool {
	return requested == "" || requested == reasoningTeacher(label)
}

// SearchScenesByLabel returns the sample ids carrying a (field,value) reasoning
// label for a (dataset, promptVersion), from the DynamoDB scene-by-label index.
func (s *S3Service) SearchScenesByLabel(ctx context.Context, dataset, promptVersion, field, value string, limit int) ([]string, error) {
	ids, _, err := s.SearchScenesByLabelAtVersion(
		ctx, dataset, "", promptVersion, field, value, limit,
	)
	return ids, err
}

// SearchScenesByLabelAtVersion queries only the sample_uid index materialized
// from the same immutable shard version used for playback.
func (s *S3Service) SearchScenesByLabelAtVersion(
	ctx context.Context,
	dataset, version, promptVersion, field, value string,
	limit int,
) ([]string, string, error) {
	if s.store == nil {
		return nil, "", fmt.Errorf("scene search requires a configured dynamo store")
	}
	version = s.versionOrResolve(ctx, dataset, version)
	ids, err := s.store.QueryScenesByLabelForVersion(
		ctx, dataset, version, promptVersion, field, value, limit,
	)
	return ids, version, err
}

// SearchScenesByLabelForTeacherAtVersion reads the exact immutable
// dataset/teacher/prompt partition materialized during stats computation.
func (s *S3Service) SearchScenesByLabelForTeacherAtVersion(
	ctx context.Context,
	dataset, version, teacher, promptVersion, field, value string,
	limit int,
) ([]string, string, error) {
	if s.store == nil {
		return nil, "", fmt.Errorf("scene search requires a configured dynamo store")
	}
	if _, _, ok := parseReasoningTeacherID(teacher); !ok {
		return nil, "", fmt.Errorf("scene search requires a teacher identity")
	}
	version = s.versionOrResolve(ctx, dataset, version)
	ids, err := s.store.QueryScenesByLabelForTeacherVersion(
		ctx, dataset, version, teacher, promptVersion, field, value, limit,
	)
	return ids, version, err
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
			for _, sampleID := range []string{smp.SampleUID, smp.Key} {
				if _, ok := want[sampleID]; ok && out[sampleID] == "" {
					out[sampleID] = sh.Name
					remaining--
				}
			}
		}
	}
	return out
}

// isStoreNotFound reports whether err is the store's not-found sentinel.
func isStoreNotFound(err error) bool { return errors.Is(err, store.ErrNotFound) }
