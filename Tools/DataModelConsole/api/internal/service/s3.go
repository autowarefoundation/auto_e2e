// Package service implements the data-access layer: S3 (datasets, reasoning
// labels) and HTTP proxies to MLflow / Flyte Admin.
package service

import (
	"archive/tar"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
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
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/store"
)

// ErrNotFound is returned when a requested S3 object / tar member is absent.
var ErrNotFound = errors.New("not found")

// ErrRangeTooLarge is returned when a requested byte range exceeds MaxRangeBytes.
var ErrRangeTooLarge = errors.New("requested range too large")

// MaxRangeBytes bounds a single range GET against a shard. Both the per-image
// endpoint and the windowed /blob endpoint stream through StreamTarMemberRange,
// so enforcing the cap here (not only at the /blob handler) means a crafted
// offset/size on EITHER endpoint cannot ask the origin to stream a whole
// multi-hundred-MB / GB shard. A generous multi-frame camera window is well
// under this; a single JPEG is KB.
const MaxRangeBytes = 32 << 20 // 32 MiB

// MaxOverlayBytes bounds pointer-following artifact reads. A v1 overlay is
// roughly 0.5 MiB per 1000 samples and seed; 16 MiB leaves ample fan-out room
// while preventing a corrupt pointer from exhausting the API pod.
const MaxOverlayBytes = 16 << 20

// fallbackVersion is used when no vX.Y/ prefix with shards can be resolved.
const fallbackVersion = "v1.0"

// versionTTL bounds how long a resolved dataset version is cached; the
// published version set changes rarely (a new pipeline run), so a few minutes
// avoids a per-request ListObjects while still picking up new versions.
const versionTTL = 2 * time.Minute

// knownDatasets are the dataset prefixes exposed by the console.
var knownDatasets = []string{"kitscenes", "l2d", "nvidia_av"}

// S3Service provides read-only access to the datasets bucket.
type S3Service struct {
	client          *s3.Client
	presigner       *s3.PresignClient
	bucket          string
	artifactsBucket string
	presignExpiry   time.Duration

	// store is the DynamoDB-backed cache: the shard-index source of truth
	// (read-through), plus precomputed stats and the scene-by-label index.
	// May be nil in tests / a Dynamo-less deployment, in which case shard
	// indexes are built fresh from S3 on every request (single-flighted).
	store *store.DynamoStore

	// versionCache memoizes the resolved newest version per dataset (see
	// resolveVersion). Guarded by versionMu.
	versionMu    sync.Mutex
	versionCache map[string]cachedVersion

	// indexSF single-flights concurrent shard-index builds so a large shard is
	// scanned from S3 only once even under many simultaneous players. The built
	// index is NOT held in an in-memory map: those indexes are multi-MB each (a
	// nvidia shard index is ~5MB of JSON), so caching dozens of them was the OOM
	// risk this backend removes — DynamoDB is now the cache/source of truth, so
	// waiters re-read the freshly-written Dynamo item instead. Guarded by
	// indexMu; keyed by "<dataset>/<version>/<shard>".
	indexMu sync.Mutex
	indexSF map[string]*sync.WaitGroup // single-flight in-progress builds
}

type cachedVersion struct {
	version string
	at      time.Time
}

// NewS3Service builds the S3 client from the default AWS credential chain
// (Pod Identity in-cluster, profile/env locally). st is the DynamoDB-backed
// cache used as the shard-index source of truth; it may be nil (tests /
// Dynamo-less deployment), in which case indexes are always built from S3.
func NewS3Service(ctx context.Context, region, bucket string, presignExpiry time.Duration, st *store.DynamoStore, artifactsBucket ...string) (*S3Service, error) {
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, awsconfig.WithRegion(region))
	if err != nil {
		return nil, fmt.Errorf("load aws config: %w", err)
	}
	client := s3.NewFromConfig(awsCfg)
	artifactBucket := bucket
	if len(artifactsBucket) > 0 && artifactsBucket[0] != "" {
		artifactBucket = artifactsBucket[0]
	}
	return &S3Service{
		client:          client,
		presigner:       s3.NewPresignClient(client),
		bucket:          bucket,
		artifactsBucket: artifactBucket,
		presignExpiry:   presignExpiry,
		store:           st,
		versionCache:    make(map[string]cachedVersion),
		indexSF:         make(map[string]*sync.WaitGroup),
	}, nil
}

// Ping checks S3 reachability for /readyz (HeadBucket, read-only).
func (s *S3Service) Ping(ctx context.Context) error {
	_, err := s.client.HeadBucket(ctx, &s3.HeadBucketInput{Bucket: aws.String(s.bucket)})
	return err
}

// ListDatasets returns the known datasets, each reporting the newest version
// resolved from S3 (see resolveVersion).
func (s *S3Service) ListDatasets(ctx context.Context) []model.Dataset {
	out := make([]model.Dataset, 0, len(knownDatasets))
	for _, name := range knownDatasets {
		version := s.resolveVersion(ctx, name)
		out = append(out, model.Dataset{
			Name:    name,
			Version: version,
			Prefix:  fmt.Sprintf("%s/%s/shards/", name, version),
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

// ResolvedVersion returns the explicit version coordinate used by a request.
func (s *S3Service) ResolvedVersion(ctx context.Context, dataset, requested string) string {
	return s.versionOrResolve(ctx, dataset, requested)
}

// OverlayBody is a verified canonical binary overlay and its public metadata.
type OverlayBody struct {
	Descriptor model.OverlayDescriptor
	Payload    []byte
}

// ListOverlayModels returns only completely published model overlays for one
// immutable shard.
func (s *S3Service) ListOverlayModels(ctx context.Context, dataset, version, shard string) ([]model.OverlayModel, string, error) {
	if s.store == nil {
		return nil, "", fmt.Errorf("overlay lookup requires a configured dynamo store")
	}
	version = s.versionOrResolve(ctx, dataset, version)
	models, err := s.store.QueryReadyOverlayModels(ctx, dataset, version, shard)
	return models, version, err
}

// GetOverlayBody follows a ready Dynamo pointer, constrains it to the canonical
// model/dataset/version/shard prefix, and verifies size and digest before
// exposing bytes to the browser.
func (s *S3Service) GetOverlayBody(ctx context.Context, dataset, version, shard, modelArtifactID string) (*OverlayBody, string, error) {
	if s.store == nil {
		return nil, "", fmt.Errorf("overlay lookup requires a configured dynamo store")
	}
	version = s.versionOrResolve(ctx, dataset, version)
	pointer, err := s.store.GetReadyOverlayPointer(
		ctx, dataset, version, shard, modelArtifactID,
	)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil, version, ErrNotFound
		}
		return nil, version, err
	}
	expectedPrefix := fmt.Sprintf(
		"overlays/schema=%s/model=%s/dataset=%s/version=%s/shard=%s/",
		pointer.OverlaySchema, modelArtifactID, dataset, version, shard,
	)
	if !strings.HasPrefix(pointer.S3Key, expectedPrefix) ||
		path.Base(pointer.S3Key) != "overlay.bin.gz" {
		return nil, version, fmt.Errorf("overlay pointer escapes canonical prefix")
	}
	if pointer.ByteSize > MaxOverlayBytes {
		return nil, version, ErrRangeTooLarge
	}
	payload, err := s.getObjectBytesFromBucket(
		ctx, s.artifactsBucket, pointer.S3Key, MaxOverlayBytes,
	)
	if err != nil {
		return nil, version, err
	}
	if int64(len(payload)) != pointer.ByteSize {
		return nil, version, fmt.Errorf(
			"overlay size mismatch: pointer=%d body=%d",
			pointer.ByteSize, len(payload),
		)
	}
	digest := sha256.Sum256(payload)
	if hex.EncodeToString(digest[:]) != pointer.SHA256 {
		return nil, version, fmt.Errorf("overlay SHA-256 mismatch")
	}
	if len(payload) < 2 || payload[0] != 0x1f || payload[1] != 0x8b {
		return nil, version, fmt.Errorf("overlay body is not gzip")
	}
	return &OverlayBody{
		Descriptor: model.OverlayDescriptor{
			ModelArtifactID: modelArtifactID,
			OverlaySchema:   pointer.OverlaySchema,
			SHA256:          pointer.SHA256,
			ByteSize:        pointer.ByteSize,
			SampleCount:     pointer.SampleCount,
		},
		Payload: payload,
	}, version, nil
}

// GeoStats returns a small Dynamo summary and a same-origin heatmap URL. The
// raw S3 key remains server-side.
func (s *S3Service) GeoStats(ctx context.Context, dataset, version string) (*model.GeoStatsResponse, error) {
	if s.store == nil {
		return nil, fmt.Errorf("geo stats require a configured dynamo store")
	}
	version = s.versionOrResolve(ctx, dataset, version)
	record, err := s.store.GetGeoRecord(ctx, dataset, version)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	return &model.GeoStatsResponse{
		Dataset: dataset,
		Version: version,
		Summary: json.RawMessage(record.Summary),
		HeatmapURL: fmt.Sprintf(
			"/api/v1/datasets/%s/geo/heatmap?version=%s",
			dataset, version,
		),
		NSamples:   record.NSamples,
		ComputedAt: record.ComputedAt,
	}, nil
}

// GeoHeatmap returns the k-anonymized, endpoint-trimmed aggregate referenced by
// Dynamo. It never derives an arbitrary S3 key from a request.
func (s *S3Service) GeoHeatmap(ctx context.Context, dataset, version string) ([]byte, string, error) {
	if s.store == nil {
		return nil, "", fmt.Errorf("geo heatmap requires a configured dynamo store")
	}
	version = s.versionOrResolve(ctx, dataset, version)
	record, err := s.store.GetGeoRecord(ctx, dataset, version)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil, version, ErrNotFound
		}
		return nil, version, err
	}
	expectedPrefix := fmt.Sprintf("%s/%s/geo/", dataset, version)
	if !strings.HasPrefix(record.GeoJSONKey, expectedPrefix) {
		return nil, version, fmt.Errorf("geo heatmap pointer escapes dataset prefix")
	}
	body, err := s.getObjectBytesFromBucket(
		ctx, s.bucket, record.GeoJSONKey, MaxRangeBytes,
	)
	return body, version, err
}

// RigProjection returns the dataset-level projection artifact emitted by the
// v2.1 repack.
func (s *S3Service) RigProjection(ctx context.Context, dataset, version string) ([]byte, string, error) {
	version = s.versionOrResolve(ctx, dataset, version)
	key := fmt.Sprintf("%s/%s/rig/projection.json", dataset, version)
	body, err := s.getObjectBytesFromBucket(ctx, s.bucket, key, 1<<20)
	return body, version, err
}

// EpisodePath returns one exact route as packed little-endian float64 rows
// [lat, lon, heading, timestamp]. Authorization is enforced by the handler.
func (s *S3Service) EpisodePath(ctx context.Context, dataset, version, episode string) ([]byte, string, error) {
	if episode == "" || len(episode) > 128 {
		return nil, "", ErrNotFound
	}
	for _, char := range episode {
		if (char < 'a' || char > 'z') &&
			(char < 'A' || char > 'Z') &&
			(char < '0' || char > '9') &&
			char != '-' && char != '_' {
			return nil, "", ErrNotFound
		}
	}
	version = s.versionOrResolve(ctx, dataset, version)
	stem := episodePathStem(dataset, episode)
	key := fmt.Sprintf(
		"%s/%s/geo/episode_paths/%s.f64", dataset, version, stem,
	)
	body, err := s.getObjectBytesFromBucket(ctx, s.bucket, key, MaxRangeBytes)
	return body, version, err
}

func episodePathStem(dataset, episode string) string {
	if dataset != "l2d" {
		return episode
	}
	numeric, err := strconv.ParseUint(episode, 10, 64)
	if err != nil {
		return episode
	}
	return fmt.Sprintf("%06d", numeric)
}

// resolveVersion returns the newest published version for a dataset: the
// lexicographically-greatest "vX.Y/" prefix under "<dataset>/" that passes the
// versionHasShards publication gate. Result is cached for versionTTL.
// Falls back to fallbackVersion when nothing resolves (or S3 errors), so the
// console still serves the historical path.
func (s *S3Service) resolveVersion(ctx context.Context, dataset string) string {
	s.versionMu.Lock()
	if c, ok := s.versionCache[dataset]; ok && time.Since(c.at) < versionTTL {
		s.versionMu.Unlock()
		return c.version
	}
	s.versionMu.Unlock()

	version := s.discoverNewestVersion(ctx, dataset)

	s.versionMu.Lock()
	s.versionCache[dataset] = cachedVersion{version: version, at: nowFunc()}
	s.versionMu.Unlock()
	return version
}

// nowFunc is time.Now indirected so tests can avoid the clock; kept trivial.
var nowFunc = time.Now

// discoverNewestVersion lists "<dataset>/" version prefixes and returns the
// newest that has passed the shard publication gate. Uncached.
func (s *S3Service) discoverNewestVersion(ctx context.Context, dataset string) string {
	prefix := dataset + "/"
	out, err := s.client.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
		Bucket:    aws.String(s.bucket),
		Prefix:    aws.String(prefix),
		Delimiter: aws.String("/"),
	})
	if err != nil {
		slog.Warn("resolve version: list versions failed, using fallback",
			"dataset", dataset, "error", err)
		return fallbackVersion
	}
	versions := make([]string, 0, len(out.CommonPrefixes))
	for _, cp := range out.CommonPrefixes {
		// cp.Prefix is "<dataset>/vX.Y/"; extract "vX.Y".
		v := strings.TrimSuffix(strings.TrimPrefix(aws.ToString(cp.Prefix), prefix), "/")
		if isVersionDir(v) {
			versions = append(versions, v)
		}
	}
	// Newest first (version-aware, not raw lexical, so v10 > v9).
	sortVersionsNewestFirst(versions)
	for _, v := range versions {
		if s.versionHasShards(ctx, dataset, v) {
			return v
		}
	}
	return fallbackVersion
}

// versionHasShards reports whether <dataset>/<version>/shards/ has a .tar.
// Canonical v2.1+ publication writes manifest.json last, so those versions stay
// invisible while partition copies are incomplete. Older, already-published
// snapshots retain their historical tar-only discovery behavior.
func (s *S3Service) versionHasShards(ctx context.Context, dataset, version string) bool {
	if requiresPublicationManifest(version) {
		_, err := s.client.HeadObject(ctx, &s3.HeadObjectInput{
			Bucket: aws.String(s.bucket),
			Key:    aws.String(shardsPrefix(dataset, version) + "manifest.json"),
		})
		if err != nil {
			return false
		}
	}
	out, err := s.client.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
		Bucket:  aws.String(s.bucket),
		Prefix:  aws.String(fmt.Sprintf("%s/%s/shards/", dataset, version)),
		MaxKeys: aws.Int32(50),
	})
	if err != nil {
		return false
	}
	for _, obj := range out.Contents {
		if strings.HasSuffix(aws.ToString(obj.Key), ".tar") {
			return true
		}
	}
	return false
}

func requiresPublicationManifest(version string) bool {
	return !versionLess(version, "v2.1")
}

// ValidVersion reports whether v is a well-formed version dir ("vN"/"vN.M").
// Exported so handlers can 400 on a garbage ?version= before it reaches S3.
func ValidVersion(v string) bool { return isVersionDir(v) }

// versionOrResolve returns requested when it is a well-formed version dir,
// otherwise the auto-resolved newest version. This is the single seam the
// per-request ?version= override flows through: an empty or malformed value
// falls back to resolveVersion (the historical auto-newest behavior), a valid
// value pins that exact version for the request.
func (s *S3Service) versionOrResolve(ctx context.Context, dataset, requested string) string {
	if isVersionDir(requested) {
		return requested
	}
	return s.resolveVersion(ctx, dataset)
}

// sortVersionsNewestFirst sorts version dirs in place newest-first
// (version-aware, not raw lexical, so v10 > v9 and v2.0 > v1.10).
func sortVersionsNewestFirst(versions []string) {
	sort.Slice(versions, func(i, j int) bool { return versionLess(versions[j], versions[i]) })
}

// shardManifest is the pipeline-written shards/manifest.json shape (see
// Platform/pipelines/workflows.py data_processing). Fields absent in a given
// manifest decode to their zero value.
type shardManifest struct {
	TotalSamples  int  `json:"total_samples"`
	Shards        int  `json:"shards"`
	Episodes      int  `json:"episodes"`
	NumViews      int  `json:"num_views"`
	HasMap        bool `json:"has_map"`
	HasWorldModel bool `json:"has_world_model"`
	HasGPS        bool `json:"has_gps"`
}

// ListDatasetVersions lists every published version under <dataset>/ that has
// shards, newest-first, summarising each version's WHOLE training composition
// from its shards/manifest.json plus a ListObjects sum of shard .tar sizes.
// A historical version without a manifest still reports Shards/SizeBytes (the
// manifest-derived counts are then zero, HasManifest=false).
func (s *S3Service) ListDatasetVersions(ctx context.Context, dataset string) ([]model.DatasetVersion, error) {
	prefix := dataset + "/"
	out, err := s.client.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
		Bucket:    aws.String(s.bucket),
		Prefix:    aws.String(prefix),
		Delimiter: aws.String("/"),
	})
	if err != nil {
		return nil, fmt.Errorf("list versions for %s: %w", dataset, err)
	}
	versions := make([]string, 0, len(out.CommonPrefixes))
	for _, cp := range out.CommonPrefixes {
		v := strings.TrimSuffix(strings.TrimPrefix(aws.ToString(cp.Prefix), prefix), "/")
		if isVersionDir(v) && s.versionHasShards(ctx, dataset, v) {
			versions = append(versions, v)
		}
	}
	sortVersionsNewestFirst(versions)

	entries := make([]model.DatasetVersion, 0, len(versions))
	for _, v := range versions {
		dv, err := s.datasetVersionSummary(ctx, dataset, v)
		if err != nil {
			return nil, err
		}
		entries = append(entries, dv)
	}
	return entries, nil
}

// datasetVersionSummary builds one DatasetVersion: manifest fields (when a
// shards/manifest.json is present) plus a ListObjects tally of shard .tar
// count and total size.
func (s *S3Service) datasetVersionSummary(ctx context.Context, dataset, version string) (model.DatasetVersion, error) {
	dv := model.DatasetVersion{Version: version}

	if body, err := s.getObjectBytes(ctx, shardsPrefix(dataset, version)+"manifest.json"); err == nil {
		var m shardManifest
		if json.Unmarshal(body, &m) == nil {
			dv.TotalSamples = m.TotalSamples
			dv.Episodes = m.Episodes
			dv.NumViews = m.NumViews
			dv.HasMap = m.HasMap
			dv.HasWorldModel = m.HasWorldModel
			dv.HasGPS = m.HasGPS
			dv.HasManifest = true
		}
	} else if !errors.Is(err, ErrNotFound) {
		return model.DatasetVersion{}, fmt.Errorf("read manifest for %s/%s: %w", dataset, version, err)
	}

	// Tally .tar objects (count + summed size) regardless of manifest presence,
	// so a manifest-less version still reports real shard/size numbers and the
	// manifest's shards count is cross-checked against reality.
	var count int
	var size int64
	p := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
		Bucket: aws.String(s.bucket),
		Prefix: aws.String(shardsPrefix(dataset, version)),
	})
	for p.HasMorePages() {
		page, err := p.NextPage(ctx)
		if err != nil {
			return model.DatasetVersion{}, fmt.Errorf("list shards for %s/%s: %w", dataset, version, err)
		}
		for _, obj := range page.Contents {
			if strings.HasSuffix(aws.ToString(obj.Key), ".tar") {
				count++
				size += aws.ToInt64(obj.Size)
			}
		}
	}
	dv.Shards = count
	dv.SizeBytes = size
	return dv, nil
}

// isVersionDir reports whether s looks like "vN" or "vN.M" (digits only).
func isVersionDir(s string) bool {
	if !strings.HasPrefix(s, "v") || len(s) < 2 {
		return false
	}
	for _, part := range strings.Split(s[1:], ".") {
		if part == "" || !isDigits(part) {
			return false
		}
	}
	return true
}

// versionLess compares "vN.M" version dirs numerically per component so that
// v10 > v9 and v1.10 > v1.2. Non-version strings sort before versions.
func versionLess(a, b string) bool {
	pa, pb := versionParts(a), versionParts(b)
	for i := 0; i < len(pa) && i < len(pb); i++ {
		if pa[i] != pb[i] {
			return pa[i] < pb[i]
		}
	}
	return len(pa) < len(pb)
}

func versionParts(v string) []int {
	v = strings.TrimPrefix(v, "v")
	fields := strings.Split(v, ".")
	out := make([]int, 0, len(fields))
	for _, f := range fields {
		n, _ := strconv.Atoi(f)
		out = append(out, n)
	}
	return out
}

// shardsPrefix returns the shards/ prefix for an explicit version (no resolve).
func shardsPrefix(dataset, version string) string {
	return fmt.Sprintf("%s/%s/shards/", dataset, version)
}

// ListShards lists .tar objects under <dataset>/<version>/shards/ with
// pagination. An empty version auto-resolves to the newest (versionOrResolve).
func (s *S3Service) ListShards(ctx context.Context, dataset, version string, limit, offset int) ([]model.Shard, model.Page, error) {
	prefix := shardsPrefix(dataset, s.versionOrResolve(ctx, dataset, version))
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
func (s *S3Service) ListSamples(ctx context.Context, dataset, version, shard string, limit, offset int) ([]model.Sample, model.Page, error) {
	key := shardsPrefix(dataset, s.versionOrResolve(ctx, dataset, version)) + shard
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
func (s *S3Service) StreamTarMember(ctx context.Context, dataset, version, shard, memberName string) (io.Reader, io.Closer, int64, error) {
	key := shardsPrefix(dataset, s.versionOrResolve(ctx, dataset, version)) + shard
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
func (s *S3Service) StreamTarMemberRange(ctx context.Context, dataset, version, shard string, offset, size int64) (io.Reader, io.Closer, int64, error) {
	if offset < 0 || size <= 0 {
		return nil, nil, 0, fmt.Errorf("invalid range offset=%d size=%d", offset, size)
	}
	if size > MaxRangeBytes {
		return nil, nil, 0, ErrRangeTooLarge
	}
	// Guard the offset+size-1 arithmetic against int64 overflow before it forms
	// the Range header (a huge offset near MaxInt64 would wrap to a negative end).
	if offset > math.MaxInt64-size {
		return nil, nil, 0, ErrNotFound
	}
	key := shardsPrefix(dataset, s.versionOrResolve(ctx, dataset, version)) + shard
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
		// A stale index whose offset now lies past EOF yields S3 416
		// InvalidRange; surface it as not-found (a 4xx) rather than a 502.
		if isS3InvalidRange(err) {
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
func (s *S3Service) GetSampleDetail(ctx context.Context, dataset, version, shard, sampleKey string) (*model.SampleDetail, error) {
	key := shardsPrefix(dataset, s.versionOrResolve(ctx, dataset, version)) + shard
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

// BuildShardIndex returns the playback index for a shard, read-through
// DynamoDB: a Dynamo hit is returned directly; on a miss the index is built
// from the (immutable) shard tar, written to Dynamo, and returned. Concurrent
// builds are single-flighted so a large shard is scanned from S3 only once even
// under many simultaneous players; waiters re-check Dynamo after the owner
// finishes rather than sharing an in-memory copy (those indexes are multi-MB,
// so holding them in a process map was the OOM risk this backend removes).
func (s *S3Service) BuildShardIndex(ctx context.Context, dataset, version, shard string) (*model.ShardIndex, error) {
	version = s.versionOrResolve(ctx, dataset, version)
	cacheKey := fmt.Sprintf("%s/%s/%s", dataset, version, shard)

	for {
		// Dynamo is the source of truth: check it before (and after) taking the
		// single-flight slot so a build by another request/replica is reused.
		if idx, ok := s.shardIndexFromStore(ctx, dataset, version, shard); ok {
			return idx, nil
		}

		s.indexMu.Lock()
		if wg, building := s.indexSF[cacheKey]; building {
			// Another goroutine is building this index; wait, then re-check
			// Dynamo (the owner will have written it).
			s.indexMu.Unlock()
			wg.Wait()
			continue
		}
		// We own the build.
		wg := &sync.WaitGroup{}
		wg.Add(1)
		s.indexSF[cacheKey] = wg
		s.indexMu.Unlock()

		var idx *model.ShardIndex
		var err error
		func() {
			// Deferred cleanup so a panic in buildShardIndexUncached can't leave
			// indexSF wedged (waiters would block forever). On panic idx stays
			// nil, the single-flight slot is cleared, waiters wake and retry,
			// and the panic still unwinds to middleware.Recoverer.
			defer func() {
				s.indexMu.Lock()
				delete(s.indexSF, cacheKey)
				s.indexMu.Unlock()
				wg.Done()
			}()
			idx, err = s.buildShardIndexUncached(ctx, dataset, version, shard)
			if err == nil && idx != nil && s.store != nil {
				// Persist for this and future requests / replicas. A write
				// failure must not fail the request: the index is already built,
				// so log and serve it (the next request rebuilds + retries).
				if perr := s.store.PutShardIndex(ctx, dataset, version, shard, idx); perr != nil {
					slog.Warn("persist shard index to dynamo failed; serving without cache",
						"dataset", dataset, "version", version, "shard", shard, "error", perr)
				}
			}
		}()
		return idx, err
	}
}

// shardIndexFromStore reads a shard index from DynamoDB. ok is false on a miss,
// on a read error (logged; treated as a miss so the build path still serves),
// or when no store is configured.
func (s *S3Service) shardIndexFromStore(ctx context.Context, dataset, version, shard string) (*model.ShardIndex, bool) {
	if s.store == nil {
		return nil, false
	}
	idx, err := s.store.GetShardIndex(ctx, dataset, version, shard)
	if err != nil {
		if !errors.Is(err, store.ErrNotFound) {
			slog.Warn("read shard index from dynamo failed; will rebuild",
				"dataset", dataset, "version", version, "shard", shard, "error", err)
		}
		return nil, false
	}
	idx.Version = version
	idx.Shard = shard
	for i := range idx.Samples {
		if idx.Samples[i].SampleUID == "" {
			idx.Samples[i].SampleUID = idx.Samples[i].Key
		}
	}
	return idx, true
}

// buildShardIndexUncached streams the shard tar once and builds the playback
// index for the ADAS player: per-member byte ranges (tar DATA offsets, same
// countingReader accounting as ListSamples) plus the current ego state and
// future plan per sample. Frames are fetched member-by-member through the
// image endpoint, so no whole-shard presigned URL is emitted.
func (s *S3Service) buildShardIndexUncached(ctx context.Context, dataset, version, shard string) (*model.ShardIndex, error) {
	key := shardsPrefix(dataset, version) + shard
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
			applyPackedMeta(entry, body)
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
		case "pose.npy":
			body, err := readMemberBytes(tr, hdr.Size)
			if err != nil {
				return nil, fmt.Errorf("read %s from %s: %w", hdr.Name, key, err)
			}
			pose, err := decodeGeoPose(body)
			if err != nil {
				return nil, fmt.Errorf("decode %s from %s: %w", hdr.Name, key, err)
			}
			entry.PoseCurrent = pose
		}
	}

	samples := make([]model.IndexSample, 0, len(order))
	for _, sk := range order {
		e := byKey[sk]
		if e.SampleUID == "" {
			e.SampleUID = sk
		}
		if e.EgoNow == nil {
			e.EgoNow = []float32{}
		}
		if e.EgoHistory == nil {
			e.EgoHistory = []float32{}
		}
		if e.EgoFuture == nil {
			e.EgoFuture = []float32{}
		}
		samples = append(samples, *e)
	}
	return &model.ShardIndex{
		Fps:     indexFps,
		Version: version,
		Shard:   shard,
		Samples: samples,
	}, nil
}

// applyPackedMeta copies the v2.1 identity and split contract onto one index
// entry. Malformed optional metadata leaves its zero/default values.
func applyPackedMeta(entry *model.IndexSample, body []byte) {
	var m struct {
		FrameIdx      *int   `json:"frame_idx"`
		SampleUID     string `json:"sample_uid"`
		SplitGroupUID string `json:"split_group_uid"`
		SplitBucket   *int   `json:"split_bucket"`
	}
	if json.Unmarshal(body, &m) != nil {
		return
	}
	if m.FrameIdx != nil {
		entry.TripFrame = *m.FrameIdx
	}
	if m.SampleUID != "" {
		entry.SampleUID = m.SampleUID
	}
	entry.SplitGroupUID = m.SplitGroupUID
	if m.SplitBucket != nil {
		entry.SplitBucket = *m.SplitBucket
	}
}

func tripFrameFromMeta(body []byte) (int, bool) {
	entry := model.IndexSample{TripFrame: -1}
	applyPackedMeta(&entry, body)
	if entry.TripFrame < 0 {
		return 0, false
	}
	return entry.TripFrame, true
}

// decodeGeoPose decodes the fixed v1 pose layout:
// latitude:f64, longitude:f64, heading:f64, timestamp:i64, accuracy:f32.
func decodeGeoPose(body []byte) (*model.GeoPose, error) {
	const poseBytes = 8 + 8 + 8 + 8 + 4
	if len(body) != poseBytes {
		return nil, fmt.Errorf("pose payload must be %d bytes, got %d", poseBytes, len(body))
	}
	latitude := math.Float64frombits(binary.LittleEndian.Uint64(body[0:8]))
	longitude := math.Float64frombits(binary.LittleEndian.Uint64(body[8:16]))
	heading := math.Float64frombits(binary.LittleEndian.Uint64(body[16:24]))
	timestamp := int64(binary.LittleEndian.Uint64(body[24:32]))
	accuracy := math.Float32frombits(binary.LittleEndian.Uint32(body[32:36]))
	if !isFinite(latitude) || !isFinite(longitude) || !isFinite(heading) ||
		latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180 {
		return nil, fmt.Errorf("pose contains invalid coordinates")
	}
	var accuracyPtr *float32
	if !math.IsNaN(float64(accuracy)) && !math.IsInf(float64(accuracy), 0) {
		accuracyCopy := accuracy
		accuracyPtr = &accuracyCopy
	}
	return &model.GeoPose{
		LatitudeDeg:           latitude,
		LongitudeDeg:          longitude,
		HeadingDegCWFromNorth: heading,
		TimestampNS:           timestamp,
		GPSAccuracyM:          accuracyPtr,
	}, nil
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
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
// sample key. Handles current content-addressed ids and historical conventions:
//   - "l2d-v1-e000012-f000064" -> ("12", 64)
//   - "nv-v1-<uuid>-f000064"   -> ("<uuid>", 64)
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
	if i := strings.LastIndex(key, "-f"); i > 0 {
		framePart := key[i+2:]
		if isDigits(framePart) {
			frame, err := strconv.Atoi(framePart)
			if err == nil {
				identity := key[:i]
				switch {
				case strings.HasPrefix(identity, "l2d-"):
					if e := strings.LastIndex(identity, "-e"); e >= 0 {
						episode := strings.TrimLeft(identity[e+2:], "0")
						if episode == "" {
							episode = "0"
						}
						return episode, frame, true
					}
				case strings.HasPrefix(identity, "nv-"):
					parts := strings.SplitN(identity, "-", 3)
					if len(parts) == 3 {
						return parts[2], frame, true
					}
				case strings.HasPrefix(identity, "kitscenes-"):
					parts := strings.SplitN(identity, "-", 3)
					if len(parts) == 3 {
						return parts[2], frame, true
					}
				}
			}
		}
	}
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

// ReasoningStats counts embedded labels in each dataset's newest immutable
// shard version, grouped by provenance carried in reasoning.json.
func (s *S3Service) ReasoningStats(ctx context.Context) ([]model.ReasoningStatsEntry, int, error) {
	counts := map[[3]string]int{}
	order := [][3]string{}

	total := 0
	for _, dataset := range knownDatasets {
		locations, version, err := s.reasoningMemberLocations(
			ctx, dataset, "", "",
		)
		if err != nil {
			return nil, 0, fmt.Errorf(
				"list embedded reasoning labels for %s: %w", dataset, err,
			)
		}
		records, err := s.fetchEmbeddedReasoning(
			ctx, dataset, version, locations,
		)
		if err != nil {
			return nil, 0, err
		}
		for _, record := range records {
			label := record.Label
			teacher := reasoningTeacher(label)
			if teacher == "" || label.PromptVersion == "" {
				continue
			}
			k := [3]string{dataset, teacher, label.PromptVersion}
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

// ReasoningPromptVersions lists embedded label provenance for one dataset's
// newest published version, sorted by (teacher, prompt_version).
func (s *S3Service) ReasoningPromptVersions(ctx context.Context, dataset string) ([]model.ReasoningPromptVersion, error) {
	counts := map[[2]string]int{}
	order := [][2]string{}

	locations, version, err := s.reasoningMemberLocations(
		ctx, dataset, "", "",
	)
	if err != nil {
		return nil, err
	}
	records, err := s.fetchEmbeddedReasoning(
		ctx, dataset, version, locations,
	)
	if err != nil {
		return nil, err
	}
	for _, record := range records {
		label := record.Label
		teacher := reasoningTeacher(label)
		if teacher == "" || label.PromptVersion == "" {
			continue
		}
		k := [2]string{teacher, label.PromptVersion}
		if _, seen := counts[k]; !seen {
			order = append(order, k)
		}
		counts[k]++
	}

	entries := make([]model.ReasoningPromptVersion, 0, len(order))
	for _, k := range order {
		entries = append(entries, model.ReasoningPromptVersion{
			Teacher:       k[0],
			PromptVersion: k[1],
			Count:         counts[k],
		})
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Teacher != entries[j].Teacher {
			return entries[i].Teacher < entries[j].Teacher
		}
		return entries[i].PromptVersion < entries[j].PromptVersion
	})
	return entries, nil
}

// reasoningSampleIDs returns canonical sample_uid values carrying an embedded
// reasoning.json member in the newest dataset version.
func (s *S3Service) reasoningSampleIDs(ctx context.Context, dataset string) (map[string]struct{}, error) {
	ids := map[string]struct{}{}
	locations, _, err := s.reasoningMemberLocations(ctx, dataset, "", "")
	if err != nil {
		return nil, err
	}
	for _, location := range locations {
		ids[location.SampleUID] = struct{}{}
	}
	return ids, nil
}

// GetReasoningLabel fetches the canonical embedded JSON label from the newest
// published dataset version.
func (s *S3Service) GetReasoningLabel(ctx context.Context, dataset, sampleID, teacher, promptVersion string) ([]byte, string, error) {
	return s.GetReasoningLabelAtVersion(
		ctx, dataset, "", sampleID, teacher, promptVersion,
	)
}

// GetReasoningLabelAtVersion fetches one reasoning.json member by sample_uid
// and returns a logical source coordinate without disclosing the bucket.
func (s *S3Service) GetReasoningLabelAtVersion(
	ctx context.Context,
	dataset, version, sampleID, teacher, promptVersion string,
) ([]byte, string, error) {
	locations, resolvedVersion, err := s.reasoningMemberLocations(
		ctx, dataset, version, sampleID,
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
	for _, record := range records {
		label := record.Label
		if promptVersion != "" && label.PromptVersion != promptVersion {
			continue
		}
		if !reasoningTeacherMatches(label, teacher) {
			continue
		}
		source := fmt.Sprintf(
			"%s/%s/%s/reasoning.json",
			dataset, resolvedVersion, sampleID,
		)
		return record.Body, source, nil
	}
	return nil, "", ErrNotFound
}

func (s *S3Service) getObjectBytes(ctx context.Context, key string) ([]byte, error) {
	return s.getObjectBytesFromBucket(ctx, s.bucket, key, 0)
}

func (s *S3Service) getObjectBytesFromBucket(ctx context.Context, bucket, key string, maxBytes int64) ([]byte, error) {
	obj, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		if isS3NotFound(err) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("get %s: %w", key, err)
	}
	defer obj.Body.Close()
	if maxBytes > 0 && aws.ToInt64(obj.ContentLength) > maxBytes {
		return nil, ErrRangeTooLarge
	}
	reader := io.Reader(obj.Body)
	if maxBytes > 0 {
		reader = io.LimitReader(obj.Body, maxBytes+1)
	}
	body, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	if maxBytes > 0 && int64(len(body)) > maxBytes {
		return nil, ErrRangeTooLarge
	}
	return body, nil
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
	// Preferred: pipeline manifest next to the shards (resolved version), then
	// the manifest one level up from shards/.
	version := s.resolveVersion(ctx, dataset)
	for _, key := range []string{
		fmt.Sprintf("%s/%s/shards/manifest.json", dataset, version),
		fmt.Sprintf("%s/%s/manifest.json", dataset, version),
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
	shards, page, err := s.ListShards(ctx, dataset, version, 1, 0)
	if err != nil {
		return 0, err
	}
	if len(shards) == 0 {
		return 0, nil
	}
	_, spg, err := s.ListSamples(ctx, dataset, version, shards[0].Name, 1, 0)
	if err != nil {
		return 0, err
	}
	return spg.Total * page.Total, nil
}

// CountReasoningLabels counts canonical embedded labels without fetching their
// JSON bodies; shard indexes already carry HasReasoning and member ranges.
func (s *S3Service) CountReasoningLabels(ctx context.Context) (int, error) {
	total := 0
	for _, dataset := range knownDatasets {
		locations, _, err := s.reasoningMemberLocations(
			ctx, dataset, "", "",
		)
		if err != nil {
			return 0, fmt.Errorf(
				"count embedded reasoning labels for %s: %w", dataset, err,
			)
		}
		total += len(locations)
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

func isS3NotFound(err error) bool {
	var apiErr interface{ ErrorCode() string }
	if errors.As(err, &apiErr) {
		code := apiErr.ErrorCode()
		return code == "NoSuchKey" || code == "NotFound" || code == "NoSuchBucket"
	}
	return false
}

// isS3InvalidRange reports whether err is S3's 416 InvalidRange (a Range whose
// start is at/after the object end — e.g. from a stale shard index offset).
func isS3InvalidRange(err error) bool {
	var apiErr interface{ ErrorCode() string }
	if errors.As(err, &apiErr) {
		return apiErr.ErrorCode() == "InvalidRange"
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
