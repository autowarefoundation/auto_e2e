// Package service implements the data-access layer: S3 (datasets, reasoning
// labels) and HTTP proxies to MLflow / Flyte Admin.
package service

import (
	"archive/tar"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"path"
	"sort"
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

// PresignShard returns a short-lived presigned GET URL for the whole shard
// tar. Combined with the Offset/SizeBytes from ListSamples, a client can
// range-GET a single member.
func (s *S3Service) PresignShard(ctx context.Context, dataset, shard string) (string, error) {
	key := shardPrefix(dataset) + shard
	req, err := s.presigner.PresignGetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	}, s3.WithPresignExpires(s.presignExpiry))
	if err != nil {
		return "", fmt.Errorf("presign %s: %w", key, err)
	}
	return req.URL, nil
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

// GetReasoningLabel fetches the raw JSON label for (dataset, sampleID). The
// cache is partitioned by teacher/prompt_version, which the caller usually
// does not know, so we list the dataset partition and pick the first (or the
// requested teacher/promptVersion when provided) match.
func (s *S3Service) GetReasoningLabel(ctx context.Context, dataset, sampleID, teacher, promptVersion string) ([]byte, string, error) {
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
	end := offset + limit
	if offset > total {
		offset = total
	}
	if end > total {
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
