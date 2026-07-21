// Package service implements the data-access layer: S3 (datasets, reasoning
// labels) and HTTP proxies to MLflow / Flyte Admin.
package service

import (
	"archive/tar"
	"bytes"
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

// ErrReasoningUnavailable means a published dataset has not had its reasoning
// serving inventory materialized yet.
var ErrReasoningUnavailable = errors.New("reasoning inventory unavailable")

// ErrReasoningIntegrity means a materialized reasoning inventory points to
// serving data that is missing, stale, or invalid.
var ErrReasoningIntegrity = errors.New("reasoning publication integrity failure")

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

// maxConcurrentFullTarScans bounds full-shard streams across index, listing,
// detail, and legacy member reads. A package-global semaphore makes the limit
// process-wide even if more than one S3Service is constructed.
const maxConcurrentFullTarScans = 4

var fullTarScanSem = make(chan struct{}, maxConcurrentFullTarScans)

// fallbackVersion is used when no vX.Y/ prefix with shards can be resolved.
const fallbackVersion = "v1.0"

// versionTTL bounds how long a resolved dataset version is cached; the
// published version set changes rarely (a new pipeline run), so a few minutes
// avoids a per-request ListObjects while still picking up new versions.
const versionTTL = 2 * time.Minute

// knownDatasets is the production dataset allowlist exposed by the console.
var knownDatasets = []string{"kitscenes"}

const kitScenesSmokePrefix = "kitscenes-smoke-"

type s3API interface {
	GetObject(
		context.Context,
		*s3.GetObjectInput,
		...func(*s3.Options),
	) (*s3.GetObjectOutput, error)
	HeadBucket(
		context.Context,
		*s3.HeadBucketInput,
		...func(*s3.Options),
	) (*s3.HeadBucketOutput, error)
	HeadObject(
		context.Context,
		*s3.HeadObjectInput,
		...func(*s3.Options),
	) (*s3.HeadObjectOutput, error)
	ListObjectsV2(
		context.Context,
		*s3.ListObjectsV2Input,
		...func(*s3.Options),
	) (*s3.ListObjectsV2Output, error)
}

type consoleStore interface {
	GetShardIndex(
		context.Context, string, string, string,
	) (*model.ShardIndex, error)
	PutShardIndex(
		context.Context, string, string, string, *model.ShardIndex,
	) error
	GetTeacherStats(
		context.Context, string, string, string, string, string,
	) (model.ReasoningStatsBlob, string, error)
	PutTeacherStats(
		context.Context, string, string, string, string, string,
		model.ReasoningStatsBlob,
	) (string, error)
	GetReasoningInventory(
		context.Context, string, string,
	) (model.ReasoningInventory, string, error)
	BeginReasoningMaterialization(
		context.Context, string, string, string, int64, int64,
	) error
	RenewReasoningMaterialization(
		context.Context, string, string, string, int64, int64,
	) error
	ReleaseReasoningMaterialization(
		context.Context, string, string, string,
	) error
	PutReasoningInventory(
		context.Context, string, string, string, int64,
		model.ReasoningInventory,
	) (string, error)
	PutReasoningSampleLookups(
		context.Context, string, string, string,
		[]model.ReasoningSampleLookup,
	) (int, error)
	GetReasoningSampleLookup(
		context.Context, string, string, string, string,
	) (model.ReasoningSampleLookup, error)
	PutReasoningSceneLabels(
		context.Context, string, string, string, string, string,
		[]store.SceneLabelRow,
	) (int, error)
	QueryReasoningScenes(
		context.Context,
		string, string, string, string, string, string, string,
		int,
	) ([]model.SceneRef, error)
	QueryReadyOverlayModels(
		context.Context, string, string, string, int, string, ...string,
	) ([]model.OverlayModel, string, error)
	GetReadyOverlayPointer(
		context.Context, string, string, string, string, ...string,
	) (*store.OverlayPointer, error)
	GetGeoRecord(
		context.Context, string, string,
	) (*store.GeoRecord, error)
}

// S3Service provides read-only access to the datasets bucket.
type S3Service struct {
	client          s3API
	presigner       *s3.PresignClient
	bucket          string
	artifactsBucket string
	presignExpiry   time.Duration

	// store is the DynamoDB-backed cache: the shard-index source of truth
	// (read-through), plus precomputed stats and the scene-by-label index.
	// May be nil in tests / a Dynamo-less deployment, in which case shard
	// indexes are built fresh from S3 on every request (single-flighted).
	store consoleStore

	// versionCache memoizes the resolved newest version per dataset (see
	// resolveVersion). Guarded by versionMu.
	versionMu    sync.Mutex
	versionCache map[string]cachedVersion

	// publicationCache contains only fully validated immutable v2.1+
	// manifests. Failures are never cached, so a publication becomes visible
	// immediately after its final manifest write succeeds.
	publicationMu    sync.Mutex
	publicationCache map[string]*publicationManifest

	// indexSF single-flights concurrent shard-index builds so a large shard is
	// scanned from S3 only once even under many simultaneous players. The built
	// index is NOT held in an in-memory map: those indexes are multi-MB each (a
	// nvidia shard index is ~5MB of JSON), so caching dozens of them was the OOM
	// risk this backend removes — DynamoDB is now the cache/source of truth, so
	// waiters re-read the freshly-written Dynamo item instead. Guarded by
	// indexMu; keyed by "<dataset>/<version>/<shard>".
	indexMu sync.Mutex
	indexSF map[string]*shardIndexBuild // single-flight in-progress builds
}

type shardIndexBuild struct {
	done chan struct{}
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
		client:           client,
		presigner:        s3.NewPresignClient(client),
		bucket:           bucket,
		artifactsBucket:  artifactBucket,
		presignExpiry:    presignExpiry,
		store:            st,
		versionCache:     make(map[string]cachedVersion),
		publicationCache: make(map[string]*publicationManifest),
		indexSF:          make(map[string]*shardIndexBuild),
	}, nil
}

// Ping checks S3 reachability for /readyz (HeadBucket, read-only).
func (s *S3Service) Ping(ctx context.Context) error {
	_, err := s.client.HeadBucket(ctx, &s3.HeadBucketInput{Bucket: aws.String(s.bucket)})
	return err
}

// ListDatasets returns only production datasets with a completed publication.
func (s *S3Service) ListDatasets(ctx context.Context) []model.Dataset {
	out := make([]model.Dataset, 0, len(knownDatasets))
	for _, name := range knownDatasets {
		version := s.resolveVersion(ctx, name)
		if version == fallbackVersion {
			continue
		}
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
	return name == "kitscenes"
}

func smokeDatasetNameFromPrefix(prefix string) (string, bool) {
	name, ok := strings.CutSuffix(prefix, "/")
	if !ok || strings.Contains(name, "/") || !isSmokeDataset(name) {
		return "", false
	}
	return name, true
}

func isSmokeDataset(name string) bool {
	digest, ok := strings.CutPrefix(name, kitScenesSmokePrefix)
	if !ok || len(digest) != 12 {
		return false
	}
	for _, char := range digest {
		if (char < '0' || char > '9') && (char < 'a' || char > 'f') {
			return false
		}
	}
	return true
}

// ResolvedVersion returns a published version coordinate used by a request.
func (s *S3Service) ResolvedVersion(
	ctx context.Context,
	dataset, requested string,
) (string, error) {
	return s.publishedVersion(ctx, dataset, requested)
}

// OverlayBody is a verified canonical binary overlay and its public metadata.
type OverlayBody struct {
	Descriptor model.OverlayDescriptor
	Payload    []byte
}

// ListOverlayModels returns one bounded page of completely published model
// overlays for an immutable shard.
func (s *S3Service) ListOverlayModels(
	ctx context.Context,
	dataset, version, shard string,
	limit int,
	pageToken string,
) ([]model.OverlayModel, string, string, error) {
	if s.store == nil {
		return nil, "", "", fmt.Errorf(
			"overlay lookup requires a configured dynamo store",
		)
	}
	var err error
	expectedManifestDigest := ""
	version, err = s.publishedVersion(ctx, dataset, version)
	if err != nil {
		return nil, "", "", err
	}
	if requiresPublicationManifest(version) {
		if _, err := s.publishedShard(ctx, dataset, version, shard); err != nil {
			return nil, version, "", err
		}
		manifest, err := s.loadPublicationManifest(ctx, dataset, version)
		if err != nil {
			return nil, version, "", err
		}
		expectedManifestDigest = manifest.SHA256
	}
	models, nextPageToken, err := s.store.QueryReadyOverlayModels(
		ctx,
		dataset,
		version,
		shard,
		limit,
		pageToken,
		expectedManifestDigest,
	)
	return models, version, nextPageToken, err
}

// GetOverlayBody follows a ready Dynamo pointer, constrains it to the canonical
// model/dataset/version/shard prefix, and verifies size and digest before
// exposing bytes to the browser.
func (s *S3Service) GetOverlayBody(ctx context.Context, dataset, version, shard, modelArtifactID string) (*OverlayBody, string, error) {
	if s.store == nil {
		return nil, "", fmt.Errorf("overlay lookup requires a configured dynamo store")
	}
	var err error
	expectedManifestDigest := ""
	version, err = s.publishedVersion(ctx, dataset, version)
	if err != nil {
		return nil, "", err
	}
	if requiresPublicationManifest(version) {
		if _, err := s.publishedShard(ctx, dataset, version, shard); err != nil {
			return nil, version, err
		}
		manifest, err := s.loadPublicationManifest(ctx, dataset, version)
		if err != nil {
			return nil, version, err
		}
		expectedManifestDigest = manifest.SHA256
	}
	pointer, err := s.store.GetReadyOverlayPointer(
		ctx, dataset, version, shard, modelArtifactID,
		expectedManifestDigest,
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
	var err error
	version, err = s.publishedVersion(ctx, dataset, version)
	if err != nil {
		return nil, err
	}
	expectedManifestDigest := ""
	if requiresPublicationManifest(version) {
		manifest, err := s.loadPublicationManifest(ctx, dataset, version)
		if err != nil {
			return nil, err
		}
		expectedManifestDigest = manifest.SHA256
	}
	record, err := s.store.GetGeoRecord(ctx, dataset, version)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	if expectedManifestDigest != "" &&
		record.DatasetManifestSHA256 != expectedManifestDigest {
		return nil, ErrNotFound
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
	var err error
	version, err = s.publishedVersion(ctx, dataset, version)
	if err != nil {
		return nil, "", err
	}
	var manifest *publicationManifest
	if requiresPublicationManifest(version) {
		manifest, err = s.loadPublicationManifest(ctx, dataset, version)
		if err != nil {
			return nil, version, err
		}
	}
	record, err := s.store.GetGeoRecord(ctx, dataset, version)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil, version, ErrNotFound
		}
		return nil, version, err
	}
	if manifest != nil &&
		record.DatasetManifestSHA256 != manifest.SHA256 {
		return nil, version, ErrNotFound
	}
	expectedKey := fmt.Sprintf(
		"%s/%s/geo/heatmap.geojson.gz", dataset, version,
	)
	if record.GeoJSONKey != expectedKey {
		return nil, version, fmt.Errorf("geo heatmap pointer is not canonical")
	}
	body, err := s.getObjectBytesFromBucket(
		ctx, s.bucket, record.GeoJSONKey, MaxRangeBytes,
	)
	if err != nil {
		return nil, version, err
	}
	if manifest != nil {
		if manifest.GeoArtifacts == nil ||
			manifest.GeoArtifacts.HeatmapKey != record.GeoJSONKey {
			return nil, version, fmt.Errorf(
				"geo heatmap pointer differs from publication",
			)
		}
		digest := sha256.Sum256(body)
		if hex.EncodeToString(digest[:]) !=
			manifest.GeoArtifacts.HeatmapSHA256 {
			return nil, version, fmt.Errorf("geo heatmap SHA-256 mismatch")
		}
	}
	return body, version, err
}

// ShardRigProjection returns the projection artifact bound to one immutable shard.
func (s *S3Service) ShardRigProjection(
	ctx context.Context,
	dataset, version, shard string,
) ([]byte, string, error) {
	var err error
	version, err = s.publishedVersion(ctx, dataset, version)
	if err != nil {
		return nil, "", err
	}
	if !requiresPublicationManifest(version) {
		return nil, version, ErrNotFound
	}
	entry, err := s.publishedShard(ctx, dataset, version, shard)
	if err != nil {
		return nil, version, err
	}
	body, err := s.getObjectBytesFromBucket(
		ctx, s.bucket, entry.Rig.Key, 1<<20,
	)
	if err == nil {
		digest := sha256.Sum256(body)
		if hex.EncodeToString(digest[:]) != entry.Rig.SHA256 {
			return nil, version, fmt.Errorf("rig projection SHA-256 mismatch")
		}
	}
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
	var err error
	version, err = s.publishedVersion(ctx, dataset, version)
	if err != nil {
		return nil, "", err
	}
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
		_, err := s.loadPublicationManifest(ctx, dataset, version)
		return err == nil
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

	if requiresPublicationManifest(version) {
		manifest, err := s.loadPublicationManifest(ctx, dataset, version)
		if err != nil {
			return model.DatasetVersion{}, err
		}
		dv.TotalSamples = manifest.TotalSamples
		dv.Episodes = manifest.Episodes
		dv.NumViews = manifest.NumViews
		dv.HasMap = manifest.HasMap
		dv.HasWorldModel = manifest.HasWorldModel
		dv.HasGPS = manifest.HasGPS
		dv.HasManifest = true
		dv.Shards = len(manifest.ShardEntries)
		for _, entry := range manifest.ShardEntries {
			dv.SizeBytes += entry.ByteSize
		}
		return dv, nil
	}

	if body, err := s.getObjectBytesFromBucket(
		ctx,
		s.bucket,
		shardsPrefix(dataset, version)+"manifest.json",
		maxPublicationManifestBytes,
	); err == nil {
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

// ListShards lists published shard entries with pagination. For v2.1+ the
// final manifest is the only inventory; orphan objects under the prefix are
// never advertised.
func (s *S3Service) ListShards(ctx context.Context, dataset, version string, limit, offset int) ([]model.Shard, model.Page, error) {
	resolvedVersion, err := s.publishedVersion(ctx, dataset, version)
	if err != nil {
		return nil, model.Page{}, err
	}
	var all []model.Shard
	if requiresPublicationManifest(resolvedVersion) {
		manifest, err := s.loadPublicationManifest(
			ctx, dataset, resolvedVersion,
		)
		if err != nil {
			return nil, model.Page{}, err
		}
		all = make([]model.Shard, 0, len(manifest.ShardEntries))
		for _, entry := range manifest.ShardEntries {
			all = append(all, model.Shard{
				Name:         entry.Name,
				Key:          entry.Key,
				SizeBytes:    entry.ByteSize,
				LastModified: entry.LastModified,
			})
		}
		total := len(all)
		pageItems, pg := paginate(all, limit, offset, total)
		return pageItems, pg, nil
	}

	prefix := shardsPrefix(dataset, resolvedVersion)
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

// ListSamples uses the shard index as the canonical member inventory. A cache
// hit avoids opening the shard; a miss follows the same single-flighted,
// process-bounded tar scan as the playback index endpoint.
func (s *S3Service) ListSamples(ctx context.Context, dataset, version, shard string, limit, offset int) ([]model.Sample, model.Page, error) {
	index, err := s.BuildShardIndex(ctx, dataset, version, shard)
	if err != nil {
		return nil, model.Page{}, err
	}
	samples := samplesFromShardIndex(index)
	total := len(samples)
	pageItems, pg := paginate(samples, limit, offset, total)
	return pageItems, pg, nil
}

func samplesFromShardIndex(index *model.ShardIndex) []model.Sample {
	samples := make([]model.Sample, 0, len(index.Samples))
	for _, indexed := range index.Samples {
		members := make([]model.TarMember, 0, len(indexed.Members))
		for suffix, member := range indexed.Members {
			members = append(members, model.TarMember{
				Name:      indexed.Key + "." + suffix,
				SizeBytes: member.Size,
				Offset:    member.Offset,
			})
		}
		// Map iteration is unordered. Data offsets recover the original tar
		// member order and therefore preserve the list endpoint's response.
		sort.Slice(members, func(i, j int) bool {
			return members[i].Offset < members[j].Offset
		})
		samples = append(samples, model.Sample{
			Key:     indexed.Key,
			Members: members,
		})
	}
	return samples
}

// StreamTarMember streams the tar from S3 until the requested member is found
// and returns a reader over that member's content (Phase 1: no tar index, so
// worst case reads the whole shard; headers of non-matching members are
// skipped without buffering). Caller must Close the returned closer.
//
// memberName is matched as "<sampleKey>.<suffix>", e.g. ep0_000064.cam_0.jpg.
func (s *S3Service) StreamTarMember(ctx context.Context, dataset, version, shard, memberName string) (io.Reader, io.Closer, int64, error) {
	_, key, _, etag, err := s.publishedShardObject(
		ctx, dataset, version, shard,
	)
	if err != nil {
		return nil, nil, 0, err
	}
	release, err := acquireFullTarScan(ctx)
	if err != nil {
		return nil, nil, 0, err
	}
	obj, err := s.client.GetObject(
		ctx, shardGetObjectInput(s.bucket, key, etag),
	)
	if err != nil {
		release()
		if isS3NotFound(err) {
			return nil, nil, 0, ErrNotFound
		}
		return nil, nil, 0, fmt.Errorf("get shard %s: %w", key, err)
	}
	closer := &scanReadCloser{
		Closer:  obj.Body,
		release: release,
	}

	tr := tar.NewReader(obj.Body)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			closer.Close()
			return nil, nil, 0, ErrNotFound
		}
		if err != nil {
			closer.Close()
			return nil, nil, 0, fmt.Errorf("read tar %s: %w", key, err)
		}
		if hdr.Typeflag == tar.TypeReg && hdr.Name == memberName {
			return tr, closer, hdr.Size, nil
		}
	}
}

type scanReadCloser struct {
	io.Closer
	release  func()
	once     sync.Once
	closeErr error
}

func (c *scanReadCloser) Close() error {
	c.once.Do(func() {
		defer c.release()
		c.closeErr = c.Closer.Close()
	})
	return c.closeErr
}

func shardGetObjectInput(bucket, key, etag string) *s3.GetObjectInput {
	input := &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	}
	if etag != "" {
		input.IfMatch = aws.String(etag)
	}
	return input
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
	_, key, shardSize, etag, err := s.publishedShardObject(
		ctx, dataset, version, shard,
	)
	if err != nil {
		return nil, nil, 0, err
	}
	if shardSize > 0 && offset > shardSize-size {
		return nil, nil, 0, ErrNotFound
	}
	rng := fmt.Sprintf("bytes=%d-%d", offset, offset+size-1)
	input := shardGetObjectInput(s.bucket, key, etag)
	input.Range = aws.String(rng)
	obj, err := s.client.GetObject(ctx, input)
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
	expectedContentRange := fmt.Sprintf(
		"bytes %d-%d/%d", offset, offset+size-1, shardSize,
	)
	if actual != size ||
		(shardSize > 0 &&
			aws.ToString(obj.ContentRange) != expectedContentRange) {
		obj.Body.Close()
		return nil, nil, 0, ErrNotFound
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
	egoPayloadBytes  = egoTotalFloats * 4
	egoNowSignals    = 4 // one history row: [speed, accel, yaw_rate, curvature]

	// indexFps is the frame rate the ADAS player renders shards at.
	indexFps = 10

	// maxInlineMemberBytes caps how much of a small metadata member (meta.json,
	// ego.npy) is buffered during a tar scan, guarding against oversized or
	// corrupt members.
	maxInlineMemberBytes = 1 << 20 // 1 MiB

	// Bound index construction before maps, metadata arrays, and the final JSON
	// encoding can exhaust an API pod. Production shards currently hold about
	// 1,000 samples and fewer than 16 members per sample.
	maxShardIndexSamples  = 4 << 10
	maxShardIndexMembers  = 64 << 10
	maxTarMemberNameBytes = 1 << 10
)

// GetSampleDetail streams the shard tar once and assembles the detail view of
// a single sample: its member list (for Cameras), raw meta.json bytes and the
// decoded ego.npy history/future arrays.
func (s *S3Service) GetSampleDetail(ctx context.Context, dataset, version, shard, sampleKey string) (*model.SampleDetail, error) {
	_, key, _, etag, err := s.publishedShardObject(
		ctx, dataset, version, shard,
	)
	if err != nil {
		return nil, err
	}
	release, err := acquireFullTarScan(ctx)
	if err != nil {
		return nil, err
	}
	defer release()
	obj, err := s.client.GetObject(
		ctx, shardGetObjectInput(s.bucket, key, etag),
	)
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
			floats, err := decodeEgoPayload(body)
			if err != nil {
				return nil, fmt.Errorf(
					"decode %s from %s: %w", hdr.Name, key, err,
				)
			}
			detail.EgoHistory = floats[:egoHistoryFloats]
			detail.EgoFuture = floats[egoHistoryFloats:]
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
	var err error
	version, _, _, err = s.publishedShardKey(
		ctx, dataset, version, shard,
	)
	if err != nil {
		return nil, err
	}
	cacheKey := fmt.Sprintf("%s/%s/%s", dataset, version, shard)

	for {
		// Dynamo is the source of truth: check it before (and after) taking the
		// single-flight slot so a build by another request/replica is reused.
		if idx, ok := s.shardIndexFromStore(ctx, dataset, version, shard); ok {
			return idx, nil
		}

		s.indexMu.Lock()
		if build, building := s.indexSF[cacheKey]; building {
			// Another goroutine is building this index; wait, then re-check
			// Dynamo (the owner will have written it).
			s.indexMu.Unlock()
			select {
			case <-build.done:
				continue
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		// We own the build.
		build := &shardIndexBuild{done: make(chan struct{})}
		s.indexSF[cacheKey] = build
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
				close(build.done)
				s.indexMu.Unlock()
			}()
			release, acquireErr := acquireFullTarScan(ctx)
			if acquireErr != nil {
				err = acquireErr
				return
			}
			defer release()
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

func acquireFullTarScan(ctx context.Context) (func(), error) {
	select {
	case fullTarScanSem <- struct{}{}:
		return func() { <-fullTarScanSem }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
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
	if idx == nil {
		slog.Warn(
			"read nil shard index from dynamo; will rebuild",
			"dataset", dataset,
			"version", version,
			"shard", shard,
		)
		return nil, false
	}
	normalized := *idx
	normalized.Version = version
	normalized.Shard = shard
	normalized.BlobRangesAllowed = true
	samplesCopied := false
	for i := range normalized.Samples {
		if normalized.Samples[i].SampleUID == "" {
			if requiresPublicationManifest(version) {
				return nil, false
			}
			if !samplesCopied {
				normalized.Samples = append(
					[]model.IndexSample(nil),
					normalized.Samples...,
				)
				samplesCopied = true
			}
			normalized.Samples[i].SampleUID = normalized.Samples[i].Key
		}
	}
	idx = &normalized
	if err := validateShardIndex(idx); err != nil {
		slog.Warn(
			"cached shard index failed validation; will rebuild",
			"dataset", dataset,
			"version", version,
			"shard", shard,
			"error", err,
		)
		return nil, false
	}
	return idx, true
}

func validateShardIndex(idx *model.ShardIndex) error {
	if idx == nil || idx.Version == "" || idx.Shard == "" {
		return fmt.Errorf("shard index identity is incomplete")
	}
	strictMeta := requiresPublicationManifest(idx.Version)
	keys := make(map[string]struct{}, len(idx.Samples))
	sampleUIDs := make(map[string]string, len(idx.Samples))
	for _, sample := range idx.Samples {
		if sample.Key == "" || sample.SampleUID == "" {
			return fmt.Errorf("shard index has an empty sample identity")
		}
		if _, exists := keys[sample.Key]; exists {
			return fmt.Errorf("duplicate sample key %q", sample.Key)
		}
		keys[sample.Key] = struct{}{}
		if previousKey, exists := sampleUIDs[sample.SampleUID]; exists {
			return fmt.Errorf(
				"duplicate sample_uid %q for %q and %q",
				sample.SampleUID, previousKey, sample.Key,
			)
		}
		sampleUIDs[sample.SampleUID] = sample.Key
		if strictMeta && sample.SampleUID != sample.Key {
			return fmt.Errorf(
				"sample key %q differs from sample_uid %q",
				sample.Key, sample.SampleUID,
			)
		}
		if strictMeta {
			if _, exists := sample.Members["meta.json"]; !exists {
				return fmt.Errorf("sample %q has no meta.json", sample.Key)
			}
		}
		for suffix, member := range sample.Members {
			if suffix == "" || member.Offset < 0 || member.Size <= 0 {
				return fmt.Errorf(
					"sample %q has invalid member %q", sample.Key, suffix,
				)
			}
		}
	}
	return nil
}

// buildShardIndexUncached streams the shard tar once and builds the playback
// index for the ADAS player: per-member byte ranges (tar DATA offsets, same
// countingReader accounting as ListSamples) plus the current ego state and
// future plan per sample. Frames are fetched member-by-member through the
// image endpoint, so no whole-shard presigned URL is emitted.
func (s *S3Service) buildShardIndexUncached(ctx context.Context, dataset, version, shard string) (*model.ShardIndex, error) {
	resolvedVersion, key, _, etag, err := s.publishedShardObject(
		ctx, dataset, version, shard,
	)
	if err != nil {
		return nil, err
	}
	obj, err := s.client.GetObject(
		ctx, shardGetObjectInput(s.bucket, key, etag),
	)
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
	metaSeen := map[string]bool{}
	memberCount := 0
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
		if len(hdr.Name) > maxTarMemberNameBytes {
			return nil, fmt.Errorf(
				"tar member name exceeds %d bytes in %s",
				maxTarMemberNameBytes, key,
			)
		}
		memberCount++
		if memberCount > maxShardIndexMembers {
			return nil, fmt.Errorf(
				"shard %s exceeds %d regular members",
				key, maxShardIndexMembers,
			)
		}
		sk := sampleKeyOf(hdr.Name)
		suffix := memberSuffixOf(hdr.Name)
		if sk == "" || suffix == "" ||
			hdr.Name != path.Base(hdr.Name) ||
			hdr.Name != sk+"."+suffix {
			return nil, fmt.Errorf(
				"non-canonical tar member %q in %s", hdr.Name, key,
			)
		}
		entry, ok := byKey[sk]
		if !ok {
			if len(byKey) >= maxShardIndexSamples {
				return nil, fmt.Errorf(
					"shard %s exceeds %d samples",
					key, maxShardIndexSamples,
				)
			}
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
		if _, exists := entry.Members[suffix]; exists {
			return nil, fmt.Errorf(
				"duplicate member suffix %q for sample %q in %s",
				suffix, sk, key,
			)
		}
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
			if err := applyPackedMeta(
				entry, body, requiresPublicationManifest(resolvedVersion),
			); err != nil {
				return nil, fmt.Errorf(
					"decode %s from %s: %w", hdr.Name, key, err,
				)
			}
			metaSeen[sk] = true
		case "ego.npy":
			body, err := readMemberBytes(tr, hdr.Size)
			if err != nil {
				return nil, fmt.Errorf("read %s from %s: %w", hdr.Name, key, err)
			}
			floats, err := decodeEgoPayload(body)
			if err != nil {
				return nil, fmt.Errorf(
					"decode %s from %s: %w", hdr.Name, key, err,
				)
			}
			// EgoNow = last history row (row 63 of 64x4): floats[252:256].
			entry.EgoNow = floats[egoHistoryFloats-egoNowSignals : egoHistoryFloats]
			// EgoHistory = the full 256-float past window (64 steps x
			// [speed, accel, yaw_rate, curvature]); the BEV draws the
			// trailing driven path from it, meaningful mid-clip without
			// cross-shard stitching.
			entry.EgoHistory = floats[:egoHistoryFloats]
			// EgoFuture = the 128-float future plan (64 steps x [accel,
			// curvature]); the BEV renders this directly instead of chaining
			// the per-frame ego_now of subsequent samples.
			entry.EgoFuture = floats[egoHistoryFloats:]
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
		if requiresPublicationManifest(resolvedVersion) && !metaSeen[sk] {
			return nil, fmt.Errorf(
				"sample %q in %s has no meta.json", sk, key,
			)
		}
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
	idx := &model.ShardIndex{
		Fps:               indexFps,
		Version:           resolvedVersion,
		Shard:             shard,
		BlobRangesAllowed: true,
		Samples:           samples,
	}
	if err := validateShardIndex(idx); err != nil {
		return nil, fmt.Errorf("validate shard index %s: %w", key, err)
	}
	return idx, nil
}

// applyPackedMeta copies the v2.1 identity and split contract onto one index
// entry. Strict mode requires the complete current identity contract.
func applyPackedMeta(
	entry *model.IndexSample,
	body []byte,
	strict ...bool,
) error {
	var m struct {
		FrameIdx      *int   `json:"frame_idx"`
		SampleUID     string `json:"sample_uid"`
		SplitGroupUID string `json:"split_group_uid"`
		SplitBucket   *int   `json:"split_bucket"`
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	if err := decoder.Decode(&m); err != nil {
		return fmt.Errorf("decode meta JSON: %w", err)
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("meta JSON contains multiple values")
		}
		return fmt.Errorf("decode trailing meta JSON: %w", err)
	}
	if m.FrameIdx != nil && *m.FrameIdx < 0 {
		return fmt.Errorf("frame_idx must be non-negative")
	}
	if m.SplitBucket != nil &&
		(*m.SplitBucket < 0 || *m.SplitBucket >= 10) {
		return fmt.Errorf("split_bucket must be in [0, 10)")
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
	if entry.Key != "" && m.SampleUID != "" && m.SampleUID != entry.Key {
		return fmt.Errorf(
			"sample_uid %q differs from tar key %q",
			m.SampleUID, entry.Key,
		)
	}
	if len(strict) > 0 && strict[0] &&
		(m.FrameIdx == nil || m.SampleUID == "" ||
			m.SplitGroupUID == "" || m.SplitBucket == nil) {
		return fmt.Errorf("meta JSON is missing the v2.1 identity contract")
	}
	return nil
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
	body, err := io.ReadAll(io.LimitReader(tr, size))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) != size {
		return nil, io.ErrUnexpectedEOF
	}
	return body, nil
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

func decodeEgoPayload(b []byte) ([]float32, error) {
	if len(b) != egoPayloadBytes {
		return nil, fmt.Errorf(
			"ego payload must be %d bytes, got %d",
			egoPayloadBytes,
			len(b),
		)
	}
	floats := decodeFloat32LE(b)
	for i, value := range floats {
		if !isFinite(float64(value)) {
			return nil, fmt.Errorf(
				"ego payload contains non-finite float at index %d",
				i,
			)
		}
	}
	return floats, nil
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

// ReasoningStats reads materialized discovery inventories only. Interactive
// requests never scan reasoning.json bodies.
func (s *S3Service) ReasoningStats(ctx context.Context) ([]model.ReasoningStatsEntry, int, error) {
	entries := make([]model.ReasoningStatsEntry, 0)
	total := 0
	foundInventory := false
	for _, dataset := range knownDatasets {
		inventory, _, _, err := s.reasoningInventory(ctx, dataset, "")
		if err != nil {
			if errors.Is(err, ErrReasoningUnavailable) {
				continue
			}
			return nil, 0, fmt.Errorf(
				"read reasoning inventory for %s: %w", dataset, err,
			)
		}
		foundInventory = true
		for _, partition := range inventory.PromptVersions {
			entries = append(entries, model.ReasoningStatsEntry{
				Dataset:         dataset,
				Teacher:         partition.Teacher,
				TeacherProvider: partition.TeacherProvider,
				TeacherModel:    partition.TeacherModel,
				PromptVersion:   partition.PromptVersion,
				Count:           partition.Count,
			})
		}
		total += inventory.Total
	}
	if !foundInventory {
		return nil, 0, ErrReasoningUnavailable
	}
	sort.Slice(entries, func(i, j int) bool {
		a, b := entries[i], entries[j]
		if a.Dataset != b.Dataset {
			return a.Dataset < b.Dataset
		}
		if a.TeacherProvider != b.TeacherProvider {
			return a.TeacherProvider < b.TeacherProvider
		}
		if a.TeacherModel != b.TeacherModel {
			return a.TeacherModel < b.TeacherModel
		}
		return a.PromptVersion < b.PromptVersion
	})
	return entries, total, nil
}

// ReasoningPromptVersions lists materialized provenance for a dataset's newest
// published version.
func (s *S3Service) ReasoningPromptVersions(ctx context.Context, dataset string) ([]model.ReasoningPromptVersion, error) {
	return s.ReasoningPromptVersionsAtVersion(ctx, dataset, "")
}

// ReasoningPromptVersionsAtVersion lists the cache-only inventory for one
// immutable dataset version. An empty version resolves to the newest
// publication.
func (s *S3Service) ReasoningPromptVersionsAtVersion(
	ctx context.Context,
	dataset, version string,
) ([]model.ReasoningPromptVersion, error) {
	inventory, _, _, err := s.reasoningInventory(ctx, dataset, version)
	if err != nil {
		return nil, err
	}
	entries := append(
		[]model.ReasoningPromptVersion(nil),
		inventory.PromptVersions...,
	)
	return entries, nil
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
	inventory, resolvedVersion, _, err := s.reasoningInventory(
		ctx, dataset, version,
	)
	if err != nil {
		return nil, "", err
	}
	lookup, err := s.store.GetReasoningSampleLookup(
		ctx,
		dataset,
		resolvedVersion,
		inventory.Generation,
		sampleID,
	)
	if err != nil {
		if isStoreNotFound(err) {
			return nil, "", ErrNotFound
		}
		return nil, "", err
	}
	if lookup.SampleID != sampleID ||
		lookup.Offset < 0 ||
		lookup.Size <= 0 ||
		lookup.Size > maxEmbeddedReasoningBytes {
		return nil, "", fmt.Errorf(
			"%w: invalid reasoning sample lookup",
			ErrReasoningIntegrity,
		)
	}
	records, err := s.fetchEmbeddedReasoning(
		ctx,
		dataset,
		resolvedVersion,
		[]reasoningMemberLocation{{
			Shard:     lookup.Shard,
			SampleUID: lookup.SampleID,
			Range: model.MemberRange{
				Offset: lookup.Offset,
				Size:   lookup.Size,
			},
		}},
	)
	if err != nil {
		return nil, "", fmt.Errorf(
			"%w: fetch embedded reasoning label: %v",
			ErrReasoningIntegrity,
			err,
		)
	}
	if len(records) != 1 {
		return nil, "", fmt.Errorf(
			"%w: reasoning lookup returned %d records",
			ErrReasoningIntegrity,
			len(records),
		)
	}
	record := records[0]
	if promptVersion != "" &&
		record.Label.PromptVersion != promptVersion {
		return nil, "", ErrNotFound
	}
	if !reasoningTeacherMatches(record.Label, teacher) {
		return nil, "", ErrNotFound
	}
	source := fmt.Sprintf(
		"%s/%s/%s/reasoning.json",
		dataset, resolvedVersion, sampleID,
	)
	return record.Body, source, nil
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

// CountReasoningLabels reads only small publication manifests. It never warms
// shard indexes or fetches reasoning.json bodies from an interactive request.
func (s *S3Service) CountReasoningLabels(ctx context.Context) (int, error) {
	total := 0
	for _, dataset := range knownDatasets {
		version, err := s.publishedVersion(ctx, dataset, "")
		if err != nil {
			return 0, fmt.Errorf(
				"resolve reasoning publication for %s: %w", dataset, err,
			)
		}
		if requiresPublicationManifest(version) {
			manifest, err := s.loadPublicationManifest(
				ctx, dataset, version,
			)
			if err != nil {
				return 0, err
			}
			total += manifest.ReasoningLabelCount
			continue
		}
		body, err := s.getObjectBytesFromBucket(
			ctx,
			s.bucket,
			fmt.Sprintf(
				"%s/%s/shards/manifest.json", dataset, version,
			),
			maxPublicationManifestBytes,
		)
		if err != nil {
			return 0, fmt.Errorf(
				"read reasoning count for %s/%s: %w",
				dataset, version, err,
			)
		}
		var manifest struct {
			ReasoningLabelCount int `json:"reasoning_label_count"`
		}
		if err := json.Unmarshal(body, &manifest); err != nil ||
			manifest.ReasoningLabelCount < 0 {
			return 0, fmt.Errorf(
				"decode reasoning count for %s/%s", dataset, version,
			)
		}
		total += manifest.ReasoningLabelCount
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
