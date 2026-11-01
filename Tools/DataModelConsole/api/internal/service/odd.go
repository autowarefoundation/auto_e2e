package service

// ODD scene search. An ODD LabelSet is scene-level catalog metadata written by
// the ODD labeler under <dataset>/<version>/odd/; it is neither a training
// input nor a target. The Console reads only the LabelSet's scene index, after
// checking the chain ready pointer -> manifest -> scene index by SHA-256 and
// binding the manifest to the dataset publication it was made for.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

const (
	oddPointerSchema      = "odd_labelset_pointer_v1"
	oddManifestSchema     = "odd_labelset_manifest_v1"
	oddSceneIndexSchema   = "odd_scene_index_v2"
	maxODDPointerBytes    = 64 << 10
	maxODDManifestBytes   = 4 << 20
	maxODDSceneIndexBytes = 64 << 20
)

var (
	oddLabelSetIDPattern = regexp.MustCompile(`^oddls-[0-9a-f]{32}$`)
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

type oddObservation struct {
	Key              string   `json:"key"`
	Status           string   `json:"status"`
	Values           []string `json:"values"`
	FirstTimestampNS int64    `json:"first_timestamp_ns"`
}

type oddScene struct {
	SceneUID         string           `json:"scene_uid"`
	ShardName        string           `json:"shard_name"`
	StartTimestampNS int64            `json:"start_timestamp_ns"`
	EndTimestampNS   int64            `json:"end_timestamp_ns"`
	DistanceM        float64          `json:"distance_m"`
	Observations     []oddObservation `json:"observations"`
}

type oddSceneIndex struct {
	SchemaVersion string     `json:"schema_version"`
	LabelSetID    string     `json:"labelset_id"`
	Scenes        []oddScene `json:"scenes"`
}

// oddLabelSet is a verified scene index. Scenes keep only valid observations,
// the only ones search and facets read.
type oddLabelSet struct {
	labelVersion   string
	labelSetID     string
	manifestSHA256 string
	quality        model.ODDQuality
	scenes         []oddScene
}

// cachedODDLabelSet records which LabelSet, by manifest SHA-256, applies to a
// dataset version; an empty manifestSHA256 records that none does.
type cachedODDLabelSet struct {
	manifestSHA256 string
	at             time.Time
}

// SearchODDScenes returns the scenes of dataset/version whose ODD labels
// satisfy every condition, the value counts over those scenes, and one page
// of them. ErrNotFound means the version has no applicable ODD LabelSet.
func (s *S3Service) SearchODDScenes(
	ctx context.Context,
	dataset, requestedVersion string,
	where []model.ODDCondition,
	limit, offset int,
) (*model.ODDSearchResponse, error) {
	version, err := s.publishedVersion(ctx, dataset, requestedVersion)
	if err != nil {
		return nil, err
	}
	if !requiresPublicationManifest(version) {
		return nil, ErrNotFound
	}
	labels, err := s.oddLabelSetFor(ctx, dataset, version)
	if err != nil {
		return nil, err
	}

	matched := make([]model.ODDScene, 0)
	matchedScenes := make([]oddScene, 0)
	for _, scene := range labels.scenes {
		matches, ok := matchODDScene(scene, where)
		if !ok {
			continue
		}
		matchedScenes = append(matchedScenes, scene)
		matched = append(matched, model.ODDScene{
			SceneUID:   scene.SceneUID,
			Shard:      scene.ShardName,
			DurationNS: scene.EndTimestampNS - scene.StartTimestampNS,
			DistanceM:  scene.DistanceM,
			Matches:    matches,
		})
	}
	page, pageInfo := paginate(matched, limit, offset, len(matched))
	return &model.ODDSearchResponse{
		Dataset:        dataset,
		Version:        version,
		LabelVersion:   labels.labelVersion,
		LabelSetID:     labels.labelSetID,
		ManifestSHA256: labels.manifestSHA256,
		Quality:        labels.quality,
		LabeledScenes:  len(labels.scenes),
		Where:          where,
		Facets:         oddFacets(matchedScenes),
		Scenes:         page,
		Page:           pageInfo,
	}, nil
}

// matchODDScene reports whether the scene satisfies every condition and, for
// each, the earliest offset from the scene start where it holds.
func matchODDScene(
	scene oddScene,
	where []model.ODDCondition,
) ([]model.ODDSceneMatch, bool) {
	matches := make([]model.ODDSceneMatch, 0, len(where))
	for _, condition := range where {
		found := false
		var first int64
		for _, observation := range scene.Observations {
			if observation.Key != condition.Key ||
				!slices.Contains(observation.Values, condition.Value) {
				continue
			}
			if !found || observation.FirstTimestampNS < first {
				first = observation.FirstTimestampNS
			}
			found = true
		}
		if !found {
			return nil, false
		}
		matches = append(matches, model.ODDSceneMatch{
			Key:           condition.Key,
			Value:         condition.Value,
			FirstOffsetNS: first - scene.StartTimestampNS,
		})
	}
	return matches, true
}

// oddFacets counts, per key and value, the scenes carrying that value. Keys
// are sorted by name; values by scene count, then name.
func oddFacets(scenes []oddScene) []model.ODDFacet {
	counts := make(map[string]map[string]int)
	for _, scene := range scenes {
		seen := make(map[[2]string]bool)
		for _, observation := range scene.Observations {
			for _, value := range observation.Values {
				pair := [2]string{observation.Key, value}
				if seen[pair] {
					continue
				}
				seen[pair] = true
				if counts[observation.Key] == nil {
					counts[observation.Key] = make(map[string]int)
				}
				counts[observation.Key][value]++
			}
		}
	}
	facets := make([]model.ODDFacet, 0, len(counts))
	for key, values := range counts {
		facet := model.ODDFacet{
			Key:    key,
			Values: make([]model.ODDFacetValue, 0, len(values)),
		}
		for value, count := range values {
			facet.Values = append(facet.Values, model.ODDFacetValue{
				Value: value, Scenes: count,
			})
		}
		sort.Slice(facet.Values, func(i, j int) bool {
			left, right := facet.Values[i], facet.Values[j]
			if left.Scenes != right.Scenes {
				return left.Scenes > right.Scenes
			}
			return left.Value < right.Value
		})
		facets = append(facets, facet)
	}
	sort.Slice(facets, func(i, j int) bool {
		return facets[i].Key < facets[j].Key
	})
	return facets
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
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	// A manifest digest names an immutable LabelSet, so a parsed one is kept
	// for the process lifetime; only the pointer resolution expires.
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
	return labels, nil
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
// the same shard names and the same samples per source partition.
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
	for _, partition := range left.Partitions {
		samples[partition.PartitionID] = partition.SampleCount
	}
	if len(samples) != len(left.Partitions) {
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
	artifact := manifest.Artifacts["scene_index"]
	if manifest.SchemaVersion != oddManifestSchema ||
		manifest.Status != "ready" ||
		manifest.LabelSetID != pointer.LabelSetID ||
		manifest.DatasetName != pointer.DatasetName ||
		manifest.DatasetVersion != pointer.DatasetVersion ||
		manifest.DatasetManifestSHA256 != source.SHA256 ||
		manifest.SceneCount != len(source.ShardEntries) ||
		artifact.Key != root+"scene_index.json" ||
		!isLowerHexDigest(artifact.SHA256) ||
		artifact.ByteSize <= 0 ||
		artifact.ByteSize > maxODDSceneIndexBytes {
		return nil, fmt.Errorf(
			"ODD manifest %s does not match its pointer or publication",
			pointer.ManifestKey,
		)
	}
	body, err = s.verifiedODDObject(
		ctx, artifact.Key, artifact.SHA256, artifact.ByteSize,
		maxODDSceneIndexBytes,
	)
	if err != nil {
		return nil, err
	}
	var index oddSceneIndex
	if err := json.Unmarshal(body, &index); err != nil {
		return nil, fmt.Errorf("decode ODD scene index: %w", err)
	}
	if index.SchemaVersion != oddSceneIndexSchema ||
		index.LabelSetID != pointer.LabelSetID ||
		len(index.Scenes) != manifest.SceneCount {
		return nil, fmt.Errorf("ODD scene index %s is invalid", artifact.Key)
	}
	sceneUIDs := make(map[string]bool, len(index.Scenes))
	shards := make(map[string]bool, len(index.Scenes))
	for i := range index.Scenes {
		scene := &index.Scenes[i]
		if _, ok := source.ShardByName[scene.ShardName]; !ok ||
			scene.SceneUID == "" ||
			sceneUIDs[scene.SceneUID] ||
			shards[scene.ShardName] ||
			scene.EndTimestampNS < scene.StartTimestampNS {
			return nil, fmt.Errorf(
				"ODD scene index %s has an invalid scene %q",
				artifact.Key, scene.SceneUID,
			)
		}
		sceneUIDs[scene.SceneUID] = true
		shards[scene.ShardName] = true
		valid := scene.Observations[:0]
		for _, observation := range scene.Observations {
			if observation.Status != "valid" || len(observation.Values) == 0 {
				continue
			}
			if !ValidODDKey(observation.Key) ||
				slices.ContainsFunc(observation.Values, invalidODDValue) ||
				observation.FirstTimestampNS < scene.StartTimestampNS ||
				observation.FirstTimestampNS > scene.EndTimestampNS {
				return nil, fmt.Errorf(
					"ODD scene index %s has an invalid %s label in scene %q",
					artifact.Key, observation.Key, scene.SceneUID,
				)
			}
			valid = append(valid, observation)
		}
		scene.Observations = valid
	}
	return &oddLabelSet{
		labelVersion:   pointer.DatasetVersion,
		labelSetID:     pointer.LabelSetID,
		manifestSHA256: pointer.ManifestSHA256,
		quality:        manifest.Quality,
		scenes:         index.Scenes,
	}, nil
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
