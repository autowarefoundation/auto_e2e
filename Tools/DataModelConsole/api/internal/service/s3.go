// Package service implements the data-access layer: S3 (datasets, reasoning
// labels) and HTTP proxies to MLflow / Flyte Admin.
package service

import (
	"archive/tar"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"path"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

// ErrNotFound is returned when a requested S3 object / tar member is absent.
var ErrNotFound = errors.New("not found")

// datasetVersion is the only version published in Phase 1.
const datasetVersion = "v1.0"

// knownDatasets are the dataset prefixes exposed by the console.
var knownDatasets = []string{"l2d", "nvidia_av"}

// reasoningCachePrefix is the label cache layout written by
// Platform/pipelines/workflows.py generate_reasoning_labels:
// reasoning_labels_cache/dataset=<d>/teacher=<t>/prompt_version=<p>/<sample_id>.json
const reasoningCachePrefix = "reasoning_labels_cache/"

// reasoningDatasetAlias maps console dataset ids to the reasoning cache
// partition names actually written by the pipeline. The console browses
// datasets as "nvidia_av"/"l2d"; the label cache partitions them by their
// source dataset name.
var reasoningDatasetAlias = map[string]string{
	"nvidia_av": "nvidia_PhysicalAI-Autonomous-Vehicles",
	"l2d":       "yaak-ai_L2D",
}

// cacheDataset resolves a console dataset id to its reasoning cache partition
// name. Passing a cache partition name through (e.g. from the Inspector) is a
// no-op.
func cacheDataset(d string) string {
	if a, ok := reasoningDatasetAlias[d]; ok {
		return a
	}
	return d
}

// cacheSampleID resolves a console sample key to the reasoning cache's flat
// s%08d index. A console key ("25cd4769_000000" / "ep0_000000") maps by its
// frame index; a key already in flat "s<digits>" form passes through.
//
// NOTE (single-shard assumption): s%08d(frameIdx) is exact only because
// train-000000 is the first (and currently only) shard of each dataset, so the
// per-shard frame index coincides with the dataset-global sample index. Revisit
// when a second shard lands (the global index must then offset by prior shards'
// sample counts).
func cacheSampleID(sampleID string) (string, bool) {
	if rest, ok := strings.CutPrefix(sampleID, "s"); ok && isDigits(rest) {
		return sampleID, true
	}
	_, idx, ok := parseSampleKey(sampleID)
	if !ok {
		return "", false
	}
	return fmt.Sprintf("s%08d", idx), true
}

// S3Service provides read-only access to the datasets bucket.
type S3Service struct {
	client        *s3.Client
	presigner     *s3.PresignClient
	bucket        string
	presignExpiry time.Duration
}

// NewS3Service builds the S3 client from the default AWS credential chain
// (Pod Identity in-cluster, profile/env locally).
func NewS3Service(ctx context.Context, region, bucket string, presignExpiry time.Duration) (*S3Service, error) {
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, awsconfig.WithRegion(region))
	if err != nil {
		return nil, fmt.Errorf("load aws config: %w", err)
	}
	client := s3.NewFromConfig(awsCfg)
	return &S3Service{
		client:        client,
		presigner:     s3.NewPresignClient(client),
		bucket:        bucket,
		presignExpiry: presignExpiry,
	}, nil
}

// Ping checks S3 reachability for /readyz (HeadBucket, read-only).
func (s *S3Service) Ping(ctx context.Context) error {
	_, err := s.client.HeadBucket(ctx, &s3.HeadBucketInput{Bucket: aws.String(s.bucket)})
	return err
}

// ListDatasets returns the known datasets. Phase 1 uses a static list matching
// the ingest pipeline output prefixes.
func (s *S3Service) ListDatasets() []model.Dataset {
	out := make([]model.Dataset, 0, len(knownDatasets))
	for _, name := range knownDatasets {
		out = append(out, model.Dataset{
			Name:    name,
			Version: datasetVersion,
			Prefix:  shardPrefix(name),
		})
	}
	return out
}

// ValidDataset reports whether name is an exposed dataset.
func (s *S3Service) ValidDataset(name string) bool {
	for _, d := range knownDatasets {
		if d == name {
			return true
		}
	}
	return false
}

func shardPrefix(dataset string) string {
	return fmt.Sprintf("%s/%s/shards/", dataset, datasetVersion)
}

// ListShards lists .tar objects under <dataset>/v1.0/shards/ with pagination.
func (s *S3Service) ListShards(ctx context.Context, dataset string, limit, offset int) ([]model.Shard, model.Page, error) {
	prefix := shardPrefix(dataset)
	var all []model.Shard

	p := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
		Bucket: aws.String(s.bucket),
		Prefix: aws.String(prefix),
	})
	for p.HasMorePages() {
		page, err := p.NextPage(ctx)
		if err != nil {
			return nil, model.Page{}, fmt.Errorf("list shards: %w", err)
		}
		for _, obj := range page.Contents {
			key := aws.ToString(obj.Key)
			if !strings.HasSuffix(key, ".tar") {
				continue
			}
			all = append(all, model.Shard{
				Name:         path.Base(key),
				Key:          key,
				SizeBytes:    aws.ToInt64(obj.Size),
				LastModified: aws.ToTime(obj.LastModified),
			})
		}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].Name < all[j].Name })

	total := len(all)
	pageItems, pg := paginate(all, limit, offset, total)
	return pageItems, pg, nil
}

// ListSamples streams the tar from S3 reading headers only (tar.Next skips
// content without buffering it) and groups members by WebDataset sample key
// (member name up to the first dot).
func (s *S3Service) ListSamples(ctx context.Context, dataset, shard string, limit, offset int) ([]model.Sample, model.Page, error) {
	key := shardPrefix(dataset) + shard
	obj, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		if isS3NotFound(err) {
			return nil, model.Page{}, ErrNotFound
		}
		return nil, model.Page{}, fmt.Errorf("get shard %s: %w", key, err)
	}
	defer obj.Body.Close()

	// Counting reader lets us record each member's data offset so future
	// range-GET extraction (Phase 2 tar index) is possible from this listing.
	cr := &countingReader{r: obj.Body}
	tr := tar.NewReader(cr)

	order := []string{}
	groups := map[string][]model.TarMember{}
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, model.Page{}, fmt.Errorf("read tar %s: %w", key, err)
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		sampleKey := sampleKeyOf(hdr.Name)
		if _, ok := groups[sampleKey]; !ok {
			order = append(order, sampleKey)
		}
		groups[sampleKey] = append(groups[sampleKey], model.TarMember{
			Name:      hdr.Name,
			SizeBytes: hdr.Size,
			Offset:    cr.n, // header already consumed: n is at data start
		})
	}

	samples := make([]model.Sample, 0, len(order))
	for _, k := range order {
		samples = append(samples, model.Sample{Key: k, Members: groups[k]})
	}

	total := len(samples)
	pageItems, pg := paginate(samples, limit, offset, total)
	return pageItems, pg, nil
}

// StreamTarMember streams the tar from S3 until the requested member is found
// and returns a reader over that member's content (Phase 1: no tar index, so
// worst case reads the whole shard; headers of non-matching members are
// skipped without buffering). Caller must Close the returned closer.
//
// memberName is matched as "<sampleKey>.<suffix>", e.g. ep0_000064.cam_0.jpg.
func (s *S3Service) StreamTarMember(ctx context.Context, dataset, shard, memberName string) (io.Reader, io.Closer, int64, error) {
	key := shardPrefix(dataset) + shard
	obj, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		if isS3NotFound(err) {
			return nil, nil, 0, ErrNotFound
		}
		return nil, nil, 0, fmt.Errorf("get shard %s: %w", key, err)
	}

	tr := tar.NewReader(obj.Body)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			obj.Body.Close()
			return nil, nil, 0, ErrNotFound
		}
		if err != nil {
			obj.Body.Close()
			return nil, nil, 0, fmt.Errorf("read tar %s: %w", key, err)
		}
		if hdr.Typeflag == tar.TypeReg && hdr.Name == memberName {
			return tr, obj.Body, hdr.Size, nil
		}
	}
}

// StreamTarMemberRange fetches exactly one tar member's raw bytes via an S3
// byte-range GET, using the (offset, size) recorded in the shard index
// (BuildShardIndex). This turns each image fetch from an O(shard) linear tar
// scan into a bounded few-KB read. Caller must Close the returned closer.
func (s *S3Service) StreamTarMemberRange(ctx context.Context, dataset, shard string, offset, size int64) (io.Reader, io.Closer, int64, error) {
	if offset < 0 || size <= 0 {
		return nil, nil, 0, fmt.Errorf("invalid range offset=%d size=%d", offset, size)
	}
	key := shardPrefix(dataset) + shard
	rng := fmt.Sprintf("bytes=%d-%d", offset, offset+size-1)
	obj, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
		Range:  aws.String(rng),
	})
	if err != nil {
		if isS3NotFound(err) {
			return nil, nil, 0, ErrNotFound
		}
		return nil, nil, 0, fmt.Errorf("get shard range %s %s: %w", key, rng, err)
	}
	// Return the actual body length from the range GET, not the client-supplied
	// size: a stale index must not advertise a Content-Length longer than the
	// body (which would hang the client waiting for bytes that never arrive).
	actual := aws.ToInt64(obj.ContentLength)
	if actual <= 0 {
		actual = size
	}
	return obj.Body, obj.Body, actual, nil
}

// Ego layout constants: ego.npy is raw little-endian float32 packed with
// numpy tobytes() (no npy header). 384 floats = 1536 bytes: the first 256
// are history (64 steps x 4 signals: speed, accel, yaw_rate, curvature), the
// last 128 are future (64 steps x 2 signals).
const (
	egoHistoryFloats = 256
	egoFutureFloats  = 128
	egoTotalFloats   = egoHistoryFloats + egoFutureFloats
	egoNowSignals    = 4 // one history row: [speed, accel, yaw_rate, curvature]

	// indexFps is the frame rate the ADAS player renders shards at.
	indexFps = 10

	// maxInlineMemberBytes caps how much of a small metadata member (meta.json,
	// ego.npy) is buffered during a tar scan, guarding against oversized or
	// corrupt members.
	maxInlineMemberBytes = 1 << 20 // 1 MiB
)

// GetSampleDetail streams the shard tar once and assembles the detail view of
// a single sample: its member list (for Cameras), raw meta.json bytes and the
// decoded ego.npy history/future arrays.
func (s *S3Service) GetSampleDetail(ctx context.Context, dataset, shard, sampleKey string) (*model.SampleDetail, error) {
	key := shardPrefix(dataset) + shard
	obj, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		if isS3NotFound(err) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("get shard %s: %w", key, err)
	}
	defer obj.Body.Close()

	detail := &model.SampleDetail{
		Key:     sampleKey,
		Cameras: []string{},
	}
	detail.EpisodeID, detail.FrameIdx, _ = parseSampleKey(sampleKey)

	found := false
	tr := tar.NewReader(obj.Body)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read tar %s: %w", key, err)
		}
		if hdr.Typeflag != tar.TypeReg || sampleKeyOf(hdr.Name) != sampleKey {
			continue
		}
		found = true
		switch suffix := memberSuffixOf(hdr.Name); {
		case strings.HasPrefix(suffix, "cam_") && strings.HasSuffix(suffix, ".jpg"):
			detail.Cameras = append(detail.Cameras, strings.TrimSuffix(suffix, ".jpg"))
		case suffix == "meta.json":
			body, err := readMemberBytes(tr, hdr.Size)
			if err != nil {
				return nil, fmt.Errorf("read %s from %s: %w", hdr.Name, key, err)
			}
			detail.Meta = json.RawMessage(body)
		case suffix == "ego.npy":
			body, err := readMemberBytes(tr, hdr.Size)
			if err != nil {
				return nil, fmt.Errorf("read %s from %s: %w", hdr.Name, key, err)
			}
			floats := decodeFloat32LE(body)
			if len(floats) >= egoTotalFloats {
				detail.EgoHistory = floats[:egoHistoryFloats]
				detail.EgoFuture = floats[egoHistoryFloats:egoTotalFloats]
			} else if len(floats) >= egoHistoryFloats {
				detail.EgoHistory = floats[:egoHistoryFloats]
				detail.EgoFuture = floats[egoHistoryFloats:]
			} else {
				detail.EgoHistory = floats
			}
		}
	}
	if !found {
		return nil, ErrNotFound
	}
	sort.Strings(detail.Cameras)
	if detail.EgoHistory == nil {
		detail.EgoHistory = []float32{}
	}
	if detail.EgoFuture == nil {
		detail.EgoFuture = []float32{}
	}
	return detail, nil
}

// BuildShardIndex streams the shard tar once and builds the playback index
// for the ADAS player: per-member byte ranges (tar DATA offsets, same
// countingReader accounting as ListSamples) plus the current ego state and
// future plan per sample. Frames are fetched member-by-member through the
// image endpoint, so no whole-shard presigned URL is emitted.
func (s *S3Service) BuildShardIndex(ctx context.Context, dataset, shard string) (*model.ShardIndex, error) {
	key := shardPrefix(dataset) + shard
	obj, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		if isS3NotFound(err) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("get shard %s: %w", key, err)
	}
	defer obj.Body.Close()

	cr := &countingReader{r: obj.Body}
	tr := tar.NewReader(cr)

	order := []string{}
	byKey := map[string]*model.IndexSample{}
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read tar %s: %w", key, err)
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		sk := sampleKeyOf(hdr.Name)
		entry, ok := byKey[sk]
		if !ok {
			episodeID, frameIdx, _ := parseSampleKey(sk)
			entry = &model.IndexSample{
				Key:       sk,
				EpisodeID: episodeID,
				FrameIdx:  frameIdx,
				TripFrame: -1, // set from meta.json below when present
				Members:   map[string]model.MemberRange{},
			}
			byKey[sk] = entry
			order = append(order, sk)
		}
		suffix := memberSuffixOf(hdr.Name)
		entry.Members[suffix] = model.MemberRange{
			Offset: cr.n, // header already consumed: n is at data start
			Size:   hdr.Size,
		}
		switch suffix {
		case "reasoning.json":
			entry.HasReasoning = true
		case "meta.json":
			body, err := readMemberBytes(tr, hdr.Size)
			if err != nil {
				return nil, fmt.Errorf("read %s from %s: %w", hdr.Name, key, err)
			}
			// The intra-shard playback ordinal (FrameIdx, key suffix) is not the
			// trip-global frame; meta.json carries the true trip frame index.
			if tf, ok := tripFrameFromMeta(body); ok {
				entry.TripFrame = tf
			}
		case "ego.npy":
			body, err := readMemberBytes(tr, hdr.Size)
			if err != nil {
				return nil, fmt.Errorf("read %s from %s: %w", hdr.Name, key, err)
			}
			floats := decodeFloat32LE(body)
			// EgoNow = last history row (row 63 of 64x4): floats[252:256].
			if len(floats) >= egoHistoryFloats {
				entry.EgoNow = floats[egoHistoryFloats-egoNowSignals : egoHistoryFloats]
				// EgoHistory = the full 256-float past window (64 steps x
				// [speed, accel, yaw_rate, curvature]); the BEV draws the
				// trailing driven path from it, meaningful mid-clip without
				// cross-shard stitching.
				entry.EgoHistory = floats[:egoHistoryFloats]
			}
			// EgoFuture = the 128-float future plan (64 steps x [accel,
			// curvature]); the BEV renders this directly instead of chaining
			// the per-frame ego_now of subsequent samples.
			if len(floats) >= egoTotalFloats {
				entry.EgoFuture = floats[egoHistoryFloats:egoTotalFloats]
			}
		}
	}

	// Set HasReasoning by listing the aliased label partition once (cheap): the
	// player's amber ticks / reasoning panel then light up even though Phase-1
	// shards embed no reasoning.json member.
	labelIDs, err := s.reasoningSampleIDs(ctx, dataset)
	if err != nil {
		return nil, err
	}

	samples := make([]model.IndexSample, 0, len(order))
	for _, sk := range order {
		e := byKey[sk]
		if e.EgoNow == nil {
			e.EgoNow = []float32{}
		}
		if e.EgoHistory == nil {
			e.EgoHistory = []float32{}
		}
		if e.EgoFuture == nil {
			e.EgoFuture = []float32{}
		}
		if id, ok := cacheSampleID(sk); ok {
			if _, has := labelIDs[id]; has {
				e.HasReasoning = true
			}
		}
		samples = append(samples, *e)
	}
	return &model.ShardIndex{
		Fps:     indexFps,
		Samples: samples,
	}, nil
}

// tripFrameFromMeta extracts the trip-global frame index from a sample's
// meta.json bytes. ok is false when the JSON is malformed or carries no
// frame_idx, so the caller keeps TripFrame = -1 (absent).
func tripFrameFromMeta(body []byte) (int, bool) {
	var m struct {
		FrameIdx *int `json:"frame_idx"`
	}
	if json.Unmarshal(body, &m) == nil && m.FrameIdx != nil {
		return *m.FrameIdx, true
	}
	return 0, false
}

// readMemberBytes buffers a tar member's content with a sanity cap so a
// corrupt or oversized member cannot exhaust memory.
func readMemberBytes(tr *tar.Reader, size int64) ([]byte, error) {
	if size < 0 || size > maxInlineMemberBytes {
		return nil, fmt.Errorf("member size %d exceeds %d byte cap", size, maxInlineMemberBytes)
	}
	return io.ReadAll(io.LimitReader(tr, size))
}

// decodeFloat32LE decodes raw little-endian float32 bytes (numpy tobytes(),
// no npy header). Trailing bytes that do not form a full float are ignored.
func decodeFloat32LE(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

// parseSampleKey extracts the episode id and frame index from a WebDataset
// sample key. Handles both packer conventions:
//   - "ep0_000064"        -> ("0", 64)          (L2D episode-prefixed)
//   - "25cd4769_000064"   -> ("25cd4769", 64)   (nvidia hash-prefixed)
//   - "s00000064"         -> ("", 64)           (flat s%08d global index)
//
// The flat s%08d form MUST yield distinct frame indices per sample, otherwise
// the player keys every frame to 0 and collides (renders one frame for the
// whole shard).
//
// ok reports whether the key carried a recognizable frame index: either an
// "_<digits>" suffix or the flat "s<digits>" form. Keys with neither (e.g. a
// garbage sample_id from the URL) return ok=false so callers can 404 instead
// of silently defaulting to frame 0.
func parseSampleKey(key string) (episodeID string, frameIdx int, ok bool) {
	i := strings.LastIndexByte(key, '_')
	if i < 0 {
		// No underscore: accept the flat "s<digits>" index form.
		if rest, ok := strings.CutPrefix(key, "s"); ok && isDigits(rest) {
			if n, err := strconv.Atoi(rest); err == nil {
				return "", n, true
			}
		}
		return "", 0, false
	}
	if n, err := strconv.Atoi(key[i+1:]); err == nil {
		frameIdx = n
	} else {
		// An underscore with a non-numeric suffix is not a valid frame key.
		return "", 0, false
	}
	episodeID = key[:i]
	// L2D keys use an "ep<N>" episode prefix; nvidia keys are hex hashes
	// (which cannot start with "ep": 'p' is not a hex digit).
	if rest, ok := strings.CutPrefix(episodeID, "ep"); ok && isDigits(rest) {
		episodeID = rest
	}
	return episodeID, frameIdx, true
}

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

// memberSuffixOf returns the member name after the sample key, e.g.
// "ep0_000064.cam_0.jpg" -> "cam_0.jpg" (base name past the first dot).
func memberSuffixOf(name string) string {
	base := path.Base(name)
	if i := strings.IndexByte(base, '.'); i > 0 {
		return base[i+1:]
	}
	return ""
}

// ReasoningStats walks reasoning_labels_cache/ and counts label objects per
// dataset/teacher/prompt_version partition.
func (s *S3Service) ReasoningStats(ctx context.Context) ([]model.ReasoningStatsEntry, int, error) {
	counts := map[[3]string]int{}
	order := [][3]string{}

	p := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
		Bucket: aws.String(s.bucket),
		Prefix: aws.String(reasoningCachePrefix),
	})
	total := 0
	for p.HasMorePages() {
		page, err := p.NextPage(ctx)
		if err != nil {
			return nil, 0, fmt.Errorf("list reasoning labels: %w", err)
		}
		for _, obj := range page.Contents {
			key := aws.ToString(obj.Key)
			if !strings.HasSuffix(key, ".json") {
				continue
			}
			ds, teacher, pv, ok := parseReasoningKey(key)
			if !ok {
				slog.Debug("skipping unparseable reasoning label key", "key", key)
				continue
			}
			k := [3]string{ds, teacher, pv}
			if _, seen := counts[k]; !seen {
				order = append(order, k)
			}
			counts[k]++
			total++
		}
	}

	entries := make([]model.ReasoningStatsEntry, 0, len(order))
	for _, k := range order {
		entries = append(entries, model.ReasoningStatsEntry{
			Dataset:       k[0],
			Teacher:       k[1],
			PromptVersion: k[2],
			Count:         counts[k],
		})
	}
	sort.Slice(entries, func(i, j int) bool {
		a, b := entries[i], entries[j]
		if a.Dataset != b.Dataset {
			return a.Dataset < b.Dataset
		}
		if a.Teacher != b.Teacher {
			return a.Teacher < b.Teacher
		}
		return a.PromptVersion < b.PromptVersion
	})
	return entries, total, nil
}

// reasoningSampleIDs lists every reasoning label present for a console dataset
// and returns the set of flat s%08d sample ids (label object base names, minus
// the .json suffix), aggregated across all teacher/prompt_version partitions.
// Used by BuildShardIndex to set HasReasoning without depending on a
// reasoning.json tar member (Phase-1 shards embed none).
func (s *S3Service) reasoningSampleIDs(ctx context.Context, dataset string) (map[string]struct{}, error) {
	prefix := fmt.Sprintf("%sdataset=%s/", reasoningCachePrefix, cacheDataset(dataset))
	ids := map[string]struct{}{}
	p := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
		Bucket: aws.String(s.bucket),
		Prefix: aws.String(prefix),
	})
	for p.HasMorePages() {
		page, err := p.NextPage(ctx)
		if err != nil {
			return nil, fmt.Errorf("list reasoning sample ids for %s: %w", dataset, err)
		}
		for _, obj := range page.Contents {
			key := aws.ToString(obj.Key)
			if !strings.HasSuffix(key, ".json") {
				continue
			}
			ids[strings.TrimSuffix(path.Base(key), ".json")] = struct{}{}
		}
	}
	return ids, nil
}

// GetReasoningLabel fetches the raw JSON label for (dataset, sampleID). The
// cache is partitioned by teacher/prompt_version, which the caller usually
// does not know, so we list the dataset partition and pick the first (or the
// requested teacher/promptVersion when provided) match.
func (s *S3Service) GetReasoningLabel(ctx context.Context, dataset, sampleID, teacher, promptVersion string) ([]byte, string, error) {
	// Console dataset ids / sample keys -> cache partition + flat s%08d id.
	// An unparseable sample_id (no frame index) has no label: 404, don't
	// silently resolve to s00000000 and serve frame 0's label.
	dataset = cacheDataset(dataset)
	resolvedID, ok := cacheSampleID(sampleID)
	if !ok {
		return nil, "", ErrNotFound
	}
	sampleID = resolvedID

	// Fast path: fully-qualified key.
	if teacher != "" && promptVersion != "" {
		key := fmt.Sprintf("%sdataset=%s/teacher=%s/prompt_version=%s/%s.json",
			reasoningCachePrefix, dataset, teacher, promptVersion, sampleID)
		body, err := s.getObjectBytes(ctx, key)
		if err != nil {
			return nil, "", err
		}
		return body, key, nil
	}

	// Discover partitions for the dataset, then probe each for the sample.
	prefix := fmt.Sprintf("%sdataset=%s/", reasoningCachePrefix, dataset)
	suffix := "/" + sampleID + ".json"

	p := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
		Bucket: aws.String(s.bucket),
		Prefix: aws.String(prefix),
	})
	for p.HasMorePages() {
		page, err := p.NextPage(ctx)
		if err != nil {
			return nil, "", fmt.Errorf("list reasoning labels for %s: %w", dataset, err)
		}
		for _, obj := range page.Contents {
			key := aws.ToString(obj.Key)
			if !strings.HasSuffix(key, suffix) {
				continue
			}
			if teacher != "" && !strings.Contains(key, "/teacher="+teacher+"/") {
				continue
			}
			if promptVersion != "" && !strings.Contains(key, "/prompt_version="+promptVersion+"/") {
				continue
			}
			body, err := s.getObjectBytes(ctx, key)
			if err != nil {
				return nil, "", err
			}
			return body, key, nil
		}
	}
	return nil, "", ErrNotFound
}

func (s *S3Service) getObjectBytes(ctx context.Context, key string) ([]byte, error) {
	obj, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		if isS3NotFound(err) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("get %s: %w", key, err)
	}
	defer obj.Body.Close()
	return io.ReadAll(obj.Body)
}

// TotalSamples returns the aggregate sample count across all known datasets.
// Preferred source is the pipeline-written manifest.json (total_samples);
// when absent it estimates as samples(first shard) x shard count, which is
// exact for uniformly packed shards and cheap enough for a dashboard KPI.
func (s *S3Service) TotalSamples(ctx context.Context) (int, error) {
	total := 0
	for _, dataset := range knownDatasets {
		n, err := s.datasetSampleCount(ctx, dataset)
		if err != nil {
			return 0, fmt.Errorf("sample count for %s: %w", dataset, err)
		}
		total += n
	}
	return total, nil
}

func (s *S3Service) datasetSampleCount(ctx context.Context, dataset string) (int, error) {
	// Preferred: pipeline manifest next to the shards.
	for _, key := range []string{
		shardPrefix(dataset) + "manifest.json",
		fmt.Sprintf("%s/%s/manifest.json", dataset, datasetVersion),
	} {
		body, err := s.getObjectBytes(ctx, key)
		if err != nil {
			if errors.Is(err, ErrNotFound) {
				continue
			}
			return 0, err
		}
		var m struct {
			TotalSamples int `json:"total_samples"`
		}
		if json.Unmarshal(body, &m) == nil && m.TotalSamples > 0 {
			return m.TotalSamples, nil
		}
	}

	// Fallback: estimate from the first shard's sample count x shard count.
	shards, page, err := s.ListShards(ctx, dataset, 1, 0)
	if err != nil {
		return 0, err
	}
	if len(shards) == 0 {
		return 0, nil
	}
	_, spg, err := s.ListSamples(ctx, dataset, shards[0].Name, 1, 0)
	if err != nil {
		return 0, err
	}
	return spg.Total * page.Total, nil
}

// CountReasoningLabels returns the total number of label JSON objects under
// reasoning_labels_cache/ without materialising per-partition stats.
func (s *S3Service) CountReasoningLabels(ctx context.Context) (int, error) {
	total := 0
	p := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
		Bucket: aws.String(s.bucket),
		Prefix: aws.String(reasoningCachePrefix),
	})
	for p.HasMorePages() {
		page, err := p.NextPage(ctx)
		if err != nil {
			return 0, fmt.Errorf("count reasoning labels: %w", err)
		}
		for _, obj := range page.Contents {
			if strings.HasSuffix(aws.ToString(obj.Key), ".json") {
				total++
			}
		}
	}
	return total, nil
}

// sampleKeyOf implements the WebDataset grouping convention: the sample key
// is the member name up to the first dot (ep0_000064.cam_0.jpg → ep0_000064).
func sampleKeyOf(name string) string {
	base := path.Base(name)
	if i := strings.IndexByte(base, '.'); i > 0 {
		return base[:i]
	}
	return base
}

// parseReasoningKey extracts (dataset, teacher, prompt_version) from a cache
// key like reasoning_labels_cache/dataset=l2d/teacher=mock/prompt_version=v3/x.json.
func parseReasoningKey(key string) (dataset, teacher, promptVersion string, ok bool) {
	rest := strings.TrimPrefix(key, reasoningCachePrefix)
	parts := strings.Split(rest, "/")
	if len(parts) < 4 {
		return "", "", "", false
	}
	dataset, ok1 := strings.CutPrefix(parts[0], "dataset=")
	teacher, ok2 := strings.CutPrefix(parts[1], "teacher=")
	promptVersion, ok3 := strings.CutPrefix(parts[2], "prompt_version=")
	if !ok1 || !ok2 || !ok3 {
		return "", "", "", false
	}
	return dataset, teacher, promptVersion, true
}

func isS3NotFound(err error) bool {
	var apiErr interface{ ErrorCode() string }
	if errors.As(err, &apiErr) {
		code := apiErr.ErrorCode()
		return code == "NoSuchKey" || code == "NotFound" || code == "NoSuchBucket"
	}
	return false
}

// paginate slices items by limit/offset and builds Page metadata.
func paginate[T any](items []T, limit, offset, total int) ([]T, model.Page) {
	if offset < 0 {
		offset = 0
	}
	if limit <= 0 {
		limit = 50
	}
	// Clamp offset BEFORE computing end: a remotely-supplied offset near
	// MaxInt would otherwise overflow end and panic on items[offset:end].
	if offset > total {
		offset = total
	}
	end := offset + limit
	if end > total || end < offset { // end < offset catches int overflow
		end = total
	}
	return items[offset:end], model.Page{
		Limit:  limit,
		Offset: offset,
		Total:  total,
		More:   end < total,
	}
}

// countingReader tracks bytes consumed from the underlying stream so tar
// member data offsets can be recorded during header-only listing.
type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}
