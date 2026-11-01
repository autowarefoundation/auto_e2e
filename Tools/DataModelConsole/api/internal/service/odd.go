package service

// ODD scene search. An ODD LabelSet is scene-level catalog metadata written by
// the ODD labeler under <dataset>/<version>/odd/; it is neither a training
// input nor a target. Each label is an interval inside a scene. The Console
// reads the LabelSet's scene index (scene to shard) and its observations
// table (the labelled intervals), after checking the chain ready pointer ->
// manifest -> artifacts by SHA-256 and binding the manifest to the dataset
// publication it was made for.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/parquet-go/parquet-go"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

const (
	oddPointerSchema        = "odd_labelset_pointer_v1"
	oddManifestSchema       = "odd_labelset_manifest_v1"
	oddSceneIndexSchema     = "odd_scene_index_v2"
	maxODDPointerBytes      = 64 << 10
	maxODDManifestBytes     = 4 << 20
	maxODDSceneIndexBytes   = 64 << 20
	maxODDObservationsBytes = 64 << 20
	// Decoded-size bounds for one LabelSet, about 10x the KITScenes one, so a
	// malformed table fails instead of exhausting the 512 MiB pod.
	maxODDObservationRows = 2_000_000
	maxODDRowValues       = 64
)

var (
	oddLabelSetIDPattern = regexp.MustCompile(`^oddls-[0-9a-f]{32}$`)
	oddSceneUIDPattern   = regexp.MustCompile(`^[0-9a-f][0-9a-f-]{0,63}$`)
	oddKeyPattern        = regexp.MustCompile(`^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$`)
	oddValuePattern      = regexp.MustCompile(`^[a-z0-9][a-z0-9_]*$`)
)

// ValidODDKey reports whether key has the form of an ODD key such as
// "odd.road.context". Search terms and loaded labels share this contract, so
// every facet value the API returns can be searched.
func ValidODDKey(key string) bool {
	return len(key) <= 128 && oddKeyPattern.MatchString(key)
}

func invalidODDValue(value string) bool { return !ValidODDValue(value) }

// ValidODDSceneUID reports whether uid has the form of a KITScenes scene UID
// (lowercase hex and dashes). Loaded scenes and scene requests share it.
func ValidODDSceneUID(uid string) bool { return oddSceneUIDPattern.MatchString(uid) }

// ValidODDValue reports whether value has the form of an ODD value such as
// "urban".
func ValidODDValue(value string) bool {
	return len(value) <= 64 && oddValuePattern.MatchString(value)
}

type oddPointer struct {
	SchemaVersion  string `json:"schema_version"`
	Status         string `json:"status"`
	DatasetName    string `json:"dataset_name"`
	DatasetVersion string `json:"dataset_version"`
	LabelSetID     string `json:"labelset_id"`
	ManifestKey    string `json:"manifest_key"`
	ManifestSHA256 string `json:"manifest_sha256"`
}

type oddArtifact struct {
	Key      string `json:"key"`
	SHA256   string `json:"sha256"`
	ByteSize int64  `json:"byte_size"`
	RowCount int64  `json:"row_count"`
}

type oddManifest struct {
	SchemaVersion         string                 `json:"schema_version"`
	Status                string                 `json:"status"`
	LabelSetID            string                 `json:"labelset_id"`
	DatasetName           string                 `json:"dataset_name"`
	DatasetVersion        string                 `json:"dataset_version"`
	DatasetManifestSHA256 string                 `json:"dataset_manifest_sha256"`
	SceneCount            int                    `json:"scene_count"`
	Quality               model.ODDQuality       `json:"quality"`
	Artifacts             map[string]oddArtifact `json:"artifacts"`
}

// oddInterval is one valid labelled interval, in absolute timestamps.
type oddInterval struct {
	values   []string
	source   string
	cameraID string
	// confidence stays float32 as stored, so a threshold compares exactly.
	confidence float32
	start, end int64
}

type oddScene struct {
	SceneUID         string  `json:"scene_uid"`
	ShardName        string  `json:"shard_name"`
	StartTimestampNS int64   `json:"start_timestamp_ns"`
	EndTimestampNS   int64   `json:"end_timestamp_ns"`
	DistanceM        float64 `json:"distance_m"`
	// labels holds the valid intervals per key, in start order.
	labels map[string][]oddInterval
}

type oddSceneIndex struct {
	SchemaVersion string     `json:"schema_version"`
	LabelSetID    string     `json:"labelset_id"`
	Scenes        []oddScene `json:"scenes"`
}

// oddObservationRow is the subset of the labeler's observations table the
// Console reads (Model/data_processing/odd_labeling/parquet.py).
type oddObservationRow struct {
	SceneUID   string   `parquet:"scene_uid"`
	Key        string   `parquet:"label_key"`
	Status     string   `parquet:"status"`
	Values     []string `parquet:"values,list"`
	Confidence float32  `parquet:"confidence"`
	Source     string   `parquet:"source"`
	Start      int64    `parquet:"start_timestamp_ns"`
	End        int64    `parquet:"end_timestamp_ns"`
	CameraID   *string  `parquet:"camera_id,optional"`
}

// oddLabelSet is a verified LabelSet. Scenes keep only valid intervals, the
// only ones search reads.
type oddLabelSet struct {
	labelVersion   string
	labelSetID     string
	manifestSHA256 string
	quality        model.ODDQuality
	scenes         []oddScene
	sceneByUID     map[string]int
	// vocabulary is the facets over every scene, computed once.
	vocabulary []model.ODDFacet
}

// cachedODDLabelSet records which LabelSet, by manifest SHA-256, applies to a
// dataset version; an empty manifestSHA256 records that none does.
type cachedODDLabelSet struct {
	manifestSHA256 string
	at             time.Time
}

func (s *S3Service) oddLabelSetFor(
	ctx context.Context,
	dataset, version string,
) (*oddLabelSet, error) {
	cacheKey := dataset + "/" + version
	s.oddMu.Lock()
	defer s.oddMu.Unlock()
	if cached, ok := s.oddCache[cacheKey]; ok &&
		nowFunc().Sub(cached.at) < versionTTL {
		if cached.manifestSHA256 == "" {
			return nil, ErrNotFound
		}
		return s.oddLabelSets[cached.manifestSHA256], nil
	}
	if s.oddCache == nil {
		s.oddCache = make(map[string]cachedODDLabelSet)
		s.oddLabelSets = make(map[string]*oddLabelSet)
	}
	pointer, source, err := s.resolveODDPointer(ctx, dataset, version)
	if errors.Is(err, ErrNotFound) {
		s.oddCache[cacheKey] = cachedODDLabelSet{at: nowFunc()}
		s.pruneODDLabelSets()
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	// A manifest digest names an immutable LabelSet, so a parsed one is kept
	// while any version resolves to it; only the pointer resolution expires.
	labels := s.oddLabelSets[pointer.ManifestSHA256]
	if labels == nil {
		labels, err = s.loadODDLabelSet(ctx, pointer, source)
		if err != nil {
			return nil, err
		}
		s.oddLabelSets[pointer.ManifestSHA256] = labels
	}
	s.oddCache[cacheKey] = cachedODDLabelSet{
		manifestSHA256: pointer.ManifestSHA256, at: nowFunc(),
	}
	s.pruneODDLabelSets()
	return labels, nil
}

// pruneODDLabelSets drops parsed LabelSets that no version resolves to any
// more, e.g. after a pointer moved to a new LabelSet. Callers hold oddMu.
func (s *S3Service) pruneODDLabelSets() {
	referenced := make(map[string]bool, len(s.oddCache))
	for _, cached := range s.oddCache {
		referenced[cached.manifestSHA256] = true
	}
	for digest := range s.oddLabelSets {
		if !referenced[digest] {
			delete(s.oddLabelSets, digest)
		}
	}
}

// resolveODDPointer returns the ODD pointer of the newest version, at or
// before the requested one, whose publication has the same scene inventory,
// and that publication. Repacking a dataset (new geometry or shard schema)
// keeps the scenes, so their scene-level labels still apply. The labeler
// writes a pointer only once its LabelSet is ready, so an invalid pointer is
// an error rather than a reason to fall back to an older version.
func (s *S3Service) resolveODDPointer(
	ctx context.Context,
	dataset, version string,
) (oddPointer, *publicationManifest, error) {
	target, err := s.loadPublicationManifest(ctx, dataset, version)
	if err != nil {
		return oddPointer{}, nil, err
	}
	versions, err := s.versionDirs(ctx, dataset)
	if err != nil {
		return oddPointer{}, nil, err
	}
	for _, candidate := range versions {
		if versionLess(version, candidate) ||
			!requiresPublicationManifest(candidate) {
			continue
		}
		pointer, err := s.readODDPointer(ctx, dataset, candidate)
		if errors.Is(err, ErrNotFound) {
			continue
		}
		if err != nil {
			return oddPointer{}, nil, err
		}
		source := target
		if candidate != version {
			source, err = s.loadPublicationManifest(ctx, dataset, candidate)
			if err != nil {
				// %v: a broken source publication is an error, not "no labels".
				return oddPointer{}, nil, fmt.Errorf(
					"ODD labels at %s/%s: %v", dataset, candidate, err,
				)
			}
			if !sameSceneInventory(target, source) {
				continue
			}
		}
		return pointer, source, nil
	}
	return oddPointer{}, nil, ErrNotFound
}

// sameSceneInventory reports whether two publications pack the same scenes:
// the same shard names and the same samples per source partition. A partition
// ID is a digest of the raw scene units it packs (shard_partition_id in
// Model/data_processing/dataset_snapshot.py), so equal IDs mean equal scenes.
func sameSceneInventory(left, right *publicationManifest) bool {
	if left.TotalSamples != right.TotalSamples ||
		len(left.ShardEntries) != len(right.ShardEntries) ||
		len(left.Partitions) == 0 ||
		len(left.Partitions) != len(right.Partitions) {
		return false
	}
	// decodePublicationManifest rejects unsorted shard entries.
	for i := range left.ShardEntries {
		if left.ShardEntries[i].Name != right.ShardEntries[i].Name {
			return false
		}
	}
	samples := make(map[string]int, len(left.Partitions))
	total := 0
	for _, partition := range left.Partitions {
		if partition.PartitionID == "" || partition.SampleCount < 0 {
			return false
		}
		samples[partition.PartitionID] = partition.SampleCount
		total += partition.SampleCount
	}
	if len(samples) != len(left.Partitions) || total != left.TotalSamples {
		return false
	}
	for _, partition := range right.Partitions {
		count, ok := samples[partition.PartitionID]
		if !ok || count != partition.SampleCount {
			return false
		}
		delete(samples, partition.PartitionID)
	}
	return len(samples) == 0
}

func (s *S3Service) readODDPointer(
	ctx context.Context,
	dataset, version string,
) (oddPointer, error) {
	var pointer oddPointer
	key := fmt.Sprintf("%s/%s/odd/latest.json", dataset, version)
	body, err := s.getObjectBytesFromBucket(
		ctx, s.bucket, key, maxODDPointerBytes,
	)
	if err != nil {
		return pointer, err
	}
	if err := json.Unmarshal(body, &pointer); err != nil {
		return pointer, fmt.Errorf("decode %s: %w", key, err)
	}
	expectedManifest := fmt.Sprintf(
		"%s/%s/odd/labelsets/%s/manifest.json",
		dataset, version, pointer.LabelSetID,
	)
	if pointer.SchemaVersion != oddPointerSchema ||
		pointer.Status != "ready" ||
		pointer.DatasetName != dataset ||
		pointer.DatasetVersion != version ||
		!oddLabelSetIDPattern.MatchString(pointer.LabelSetID) ||
		pointer.ManifestKey != expectedManifest ||
		!isLowerHexDigest(pointer.ManifestSHA256) {
		return pointer, fmt.Errorf("invalid ODD ready pointer %s", key)
	}
	return pointer, nil
}

func (s *S3Service) loadODDLabelSet(
	ctx context.Context,
	pointer oddPointer,
	source *publicationManifest,
) (*oddLabelSet, error) {
	body, err := s.verifiedODDObject(
		ctx, pointer.ManifestKey, pointer.ManifestSHA256, 0,
		maxODDManifestBytes,
	)
	if err != nil {
		return nil, err
	}
	var manifest oddManifest
	if err := json.Unmarshal(body, &manifest); err != nil {
		return nil, fmt.Errorf("decode ODD manifest: %w", err)
	}
	root := strings.TrimSuffix(pointer.ManifestKey, "manifest.json")
	index := manifest.Artifacts["scene_index"]
	observations := manifest.Artifacts["observations_parquet"]
	if manifest.SchemaVersion != oddManifestSchema ||
		manifest.Status != "ready" ||
		manifest.LabelSetID != pointer.LabelSetID ||
		manifest.DatasetName != pointer.DatasetName ||
		manifest.DatasetVersion != pointer.DatasetVersion ||
		manifest.DatasetManifestSHA256 != source.SHA256 ||
		manifest.SceneCount != len(source.ShardEntries) ||
		!validODDArtifact(index, root+"scene_index.json", maxODDSceneIndexBytes) ||
		!validODDArtifact(
			observations, root+"observations/part-00000.parquet",
			maxODDObservationsBytes,
		) ||
		observations.RowCount <= 0 {
		return nil, fmt.Errorf(
			"ODD manifest %s does not match its pointer or publication",
			pointer.ManifestKey,
		)
	}
	body, err = s.verifiedODDObject(
		ctx, index.Key, index.SHA256, index.ByteSize, maxODDSceneIndexBytes,
	)
	if err != nil {
		return nil, err
	}
	labels, err := decodeODDSceneIndex(body, pointer.LabelSetID, manifest.SceneCount, source)
	if err != nil {
		return nil, fmt.Errorf("ODD scene index %s: %w", index.Key, err)
	}
	body, err = s.verifiedODDObject(
		ctx, observations.Key, observations.SHA256, observations.ByteSize,
		maxODDObservationsBytes,
	)
	if err != nil {
		return nil, err
	}
	if err := labels.decodeObservations(body, observations.RowCount); err != nil {
		return nil, fmt.Errorf("ODD observations %s: %w", observations.Key, err)
	}
	labels.labelVersion = pointer.DatasetVersion
	labels.labelSetID = pointer.LabelSetID
	labels.manifestSHA256 = pointer.ManifestSHA256
	labels.quality = manifest.Quality
	labels.vocabulary = oddFacets(labels.scenes)
	return labels, nil
}

func validODDArtifact(artifact oddArtifact, key string, maxBytes int64) bool {
	return artifact.Key == key &&
		isLowerHexDigest(artifact.SHA256) &&
		artifact.ByteSize > 0 &&
		artifact.ByteSize <= maxBytes
}

// decodeODDSceneIndex reads the scenes of a LabelSet and checks that they map
// one to one onto the shards of the publication it labels.
func decodeODDSceneIndex(
	body []byte,
	labelSetID string,
	sceneCount int,
	source *publicationManifest,
) (*oddLabelSet, error) {
	var index oddSceneIndex
	if err := json.Unmarshal(body, &index); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	if index.SchemaVersion != oddSceneIndexSchema ||
		index.LabelSetID != labelSetID ||
		len(index.Scenes) != sceneCount {
		return nil, fmt.Errorf("schema, LabelSet or scene count differs")
	}
	labels := &oddLabelSet{
		scenes:     index.Scenes,
		sceneByUID: make(map[string]int, len(index.Scenes)),
	}
	shards := make(map[string]bool, len(index.Scenes))
	for i := range labels.scenes {
		scene := &labels.scenes[i]
		_, published := source.ShardByName[scene.ShardName]
		_, seen := labels.sceneByUID[scene.SceneUID]
		if !published || !ValidODDSceneUID(scene.SceneUID) || seen ||
			shards[scene.ShardName] ||
			scene.StartTimestampNS < 0 ||
			scene.EndTimestampNS <= scene.StartTimestampNS {
			return nil, fmt.Errorf("invalid scene %q", scene.SceneUID)
		}
		labels.sceneByUID[scene.SceneUID] = i
		shards[scene.ShardName] = true
		scene.labels = make(map[string][]oddInterval)
	}
	return labels, nil
}

// decodeObservations reads the observations table and keeps the valid
// intervals. Every row must belong to a scene and every valid one must be
// searchable and lie inside its scene; otherwise the LabelSet is rejected.
func (labels *oddLabelSet) decodeObservations(body []byte, rowCount int64) error {
	file, err := parquet.OpenFile(bytes.NewReader(body), int64(len(body)))
	if err != nil {
		return fmt.Errorf("open: %w", err)
	}
	if file.NumRows() != rowCount {
		return fmt.Errorf("has %d rows, manifest says %d", file.NumRows(), rowCount)
	}
	if rowCount > maxODDObservationRows {
		return fmt.Errorf("has %d rows, more than %d", rowCount, maxODDObservationRows)
	}
	reader := parquet.NewGenericReader[oddObservationRow](file)
	defer reader.Close()
	// Keys, values and sources repeat across 10^5 rows; share one copy.
	interned := make(map[string]string)
	intern := func(value string) string {
		if shared, ok := interned[value]; ok {
			return shared
		}
		value = strings.Clone(value)
		interned[value] = value
		return value
	}
	rows := make([]oddObservationRow, 1024)
	read := int64(0)
	for {
		n, err := reader.Read(rows)
		for _, row := range rows[:n] {
			read++
			index, ok := labels.sceneByUID[row.SceneUID]
			if !ok {
				return fmt.Errorf("row for unknown scene %q", row.SceneUID)
			}
			if row.Status != "valid" {
				continue
			}
			scene := &labels.scenes[index]
			confidence := row.Confidence
			cameraID := ""
			if row.CameraID != nil {
				cameraID = *row.CameraID
			}
			// A valid label without a value would make NOT conditions hold
			// for a scene that does carry the key.
			if !ValidODDKey(row.Key) ||
				len(row.Values) == 0 || len(row.Values) > maxODDRowValues ||
				slices.ContainsFunc(row.Values, invalidODDValue) ||
				!ValidODDValue(row.Source) ||
				(cameraID != "" && !ValidODDValue(cameraID)) ||
				math.IsNaN(float64(confidence)) || confidence < 0 || confidence > 1 ||
				row.Start < scene.StartTimestampNS ||
				row.End > scene.EndTimestampNS ||
				row.End < row.Start {
				return fmt.Errorf(
					"invalid %s label in scene %q", row.Key, scene.SceneUID,
				)
			}
			values := make([]string, len(row.Values))
			for i, value := range row.Values {
				values[i] = intern(value)
			}
			key := intern(row.Key)
			scene.labels[key] = append(scene.labels[key], oddInterval{
				values:     values,
				source:     intern(row.Source),
				cameraID:   intern(cameraID),
				confidence: confidence,
				start:      row.Start,
				end:        row.End,
			})
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("read: %w", err)
		}
	}
	if read != rowCount {
		return fmt.Errorf("read %d rows, manifest says %d", read, rowCount)
	}
	for i := range labels.scenes {
		for _, intervals := range labels.scenes[i].labels {
			// Camera breaks start ties so per-camera rows keep one order.
			sort.Slice(intervals, func(a, b int) bool {
				if intervals[a].start != intervals[b].start {
					return intervals[a].start < intervals[b].start
				}
				return intervals[a].cameraID < intervals[b].cameraID
			})
		}
	}
	return nil
}

// verifiedODDObject reads a datasets-bucket object and checks its SHA-256 and,
// when expectedSize > 0, its size.
func (s *S3Service) verifiedODDObject(
	ctx context.Context,
	key, expectedSHA256 string,
	expectedSize, maxBytes int64,
) ([]byte, error) {
	body, err := s.getObjectBytesFromBucket(ctx, s.bucket, key, maxBytes)
	if err != nil {
		// %v: a LabelSet object missing behind a ready pointer is an error,
		// not "no labels".
		return nil, fmt.Errorf("read %s: %v", key, err)
	}
	digest := sha256.Sum256(body)
	if hex.EncodeToString(digest[:]) != expectedSHA256 ||
		(expectedSize > 0 && int64(len(body)) != expectedSize) {
		return nil, fmt.Errorf("%s does not match its recorded digest", key)
	}
	return body, nil
}
