package service

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sort"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

// Limits on one ODD query, so a request does bounded work.
const (
	MaxODDGroups     = 8
	MaxODDConditions = 24
	maxODDValues     = 32
	maxODDSources    = 8
	maxODDDurationS  = 3600
)

// ErrInvalidODDQuery wraps a reason an ODD query is not executable.
type ErrInvalidODDQuery struct{ Reason string }

func (e ErrInvalidODDQuery) Error() string { return "invalid ODD query: " + e.Reason }

func validODDOp(op string) bool { return op == "and" || op == "or" }

// ValidateODDQuery checks an ODD query's shape and bounds. An empty query
// (no groups) selects every scene.
func ValidateODDQuery(query model.ODDQuery) error {
	invalid := func(format string, args ...any) error {
		return ErrInvalidODDQuery{Reason: fmt.Sprintf(format, args...)}
	}
	if !validODDOp(query.Op) {
		return invalid("op must be and or or")
	}
	if len(query.Groups) > MaxODDGroups {
		return invalid("at most %d groups", MaxODDGroups)
	}
	conditions := 0
	for g, group := range query.Groups {
		if !validODDOp(group.Op) {
			return invalid("group %d: op must be and or or", g+1)
		}
		if len(group.Conditions) == 0 {
			return invalid("group %d has no conditions", g+1)
		}
		conditions += len(group.Conditions)
		for c, condition := range group.Conditions {
			at := fmt.Sprintf("group %d condition %d", g+1, c+1)
			switch {
			case !ValidODDKey(condition.Key):
				return invalid("%s: invalid key", at)
			case len(condition.Values) == 0 || len(condition.Values) > maxODDValues:
				return invalid("%s: give 1 to %d values", at, maxODDValues)
			case slices.ContainsFunc(condition.Values, invalidODDValue):
				return invalid("%s: invalid value", at)
			case len(condition.Sources) > maxODDSources ||
				slices.ContainsFunc(condition.Sources, invalidODDValue):
				return invalid("%s: invalid sources", at)
			case math.IsNaN(condition.MinDurationS) ||
				condition.MinDurationS < 0 ||
				condition.MinDurationS > maxODDDurationS:
				return invalid("%s: min_duration_s must be 0 to %d", at, maxODDDurationS)
			case math.IsNaN(condition.MinConfidence) ||
				condition.MinConfidence < 0 || condition.MinConfidence > 1:
				return invalid("%s: min_confidence must be 0 to 1", at)
			}
		}
	}
	if conditions > MaxODDConditions {
		return invalid("at most %d conditions", MaxODDConditions)
	}
	return nil
}

// SearchODDScenes returns the scenes of dataset/version whose ODD labels
// satisfy the query, with the intervals that satisfied it, value counts over
// the matched and over all scenes, and one page of results. ErrNotFound
// means the version has no applicable ODD LabelSet.
func (s *S3Service) SearchODDScenes(
	ctx context.Context,
	dataset, version string,
	query model.ODDQuery,
	limit, offset int,
) (*model.ODDSearchResponse, error) {
	if err := ValidateODDQuery(query); err != nil {
		return nil, err
	}
	labels, info, err := s.oddLabels(ctx, dataset, version)
	if err != nil {
		return nil, err
	}
	compiled := compileODDQuery(query)
	empty := len(query.Groups) == 0
	type hit struct {
		scene oddScene
		lanes []oddLane
	}
	hits := make([]hit, 0)
	matchedScenes := make([]oddScene, 0)
	for _, scene := range labels.scenes {
		lanes, ok := matchODDQuery(scene, compiled)
		if !ok {
			continue
		}
		matchedScenes = append(matchedScenes, scene)
		hits = append(hits, hit{scene, lanes})
	}
	// Every scene matches an empty query, so its facets are the vocabulary.
	facets := labels.vocabulary
	if !empty {
		facets = oddFacets(matchedScenes)
	}
	// Lanes are converted to the response shape for one page only.
	pageHits, pageInfo := paginate(hits, limit, offset, len(hits))
	page := make([]model.ODDScene, len(pageHits))
	for i, h := range pageHits {
		page[i] = oddSceneResult(h.scene, h.lanes)
	}
	return &model.ODDSearchResponse{
		ODDLabelSetInfo: info,
		Vocabulary:      labels.vocabulary,
		Facets:          facets,
		Scenes:          page,
		Page:            pageInfo,
	}, nil
}

// ODDScene returns every valid label of one scene.
func (s *S3Service) ODDScene(
	ctx context.Context,
	dataset, version, sceneUID string,
) (*model.ODDSceneResponse, error) {
	labels, info, err := s.oddLabels(ctx, dataset, version)
	if err != nil {
		return nil, err
	}
	index, ok := labels.sceneByUID[sceneUID]
	if !ok {
		return nil, ErrNotFound
	}
	scene := labels.scenes[index]
	keys := make([]string, 0, len(scene.labels))
	for key := range scene.labels {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	lanes := make([]oddLane, len(keys))
	for i, key := range keys {
		lanes[i] = oddLane{key: key, intervals: scene.labels[key]}
	}
	return &model.ODDSceneResponse{
		ODDLabelSetInfo: info,
		Scene:           oddSceneResult(scene, lanes),
	}, nil
}

func (s *S3Service) oddLabels(
	ctx context.Context,
	dataset, version string,
) (*oddLabelSet, model.ODDLabelSetInfo, error) {
	// ODD LabelSets bind to v2.1+ publication manifests; older versions have
	// none, so skip their S3 shard probe.
	if !requiresPublicationManifest(version) {
		return nil, model.ODDLabelSetInfo{}, ErrNotFound
	}
	resolved, err := s.publishedVersion(ctx, dataset, version)
	if err != nil {
		return nil, model.ODDLabelSetInfo{}, err
	}
	labels, err := s.oddLabelSetFor(ctx, dataset, resolved)
	if err != nil {
		return nil, model.ODDLabelSetInfo{}, err
	}
	return labels, model.ODDLabelSetInfo{
		Dataset:        dataset,
		Version:        resolved,
		LabelVersion:   labels.labelVersion,
		LabelSetID:     labels.labelSetID,
		ManifestSHA256: labels.manifestSHA256,
		Quality:        labels.quality,
		LabeledScenes:  len(labels.scenes),
	}, nil
}

type oddLane struct {
	key       string
	intervals []oddInterval
}

// oddCondition is a validated query condition with its values and sources
// as sets, so matching an interval is a few map lookups.
type oddCondition struct {
	key           string
	values        map[string]bool
	sources       map[string]bool
	not           bool
	minDurationNS int64
	minConfidence float32
}

type oddGroup struct {
	and        bool
	conditions []oddCondition
}

type oddCompiledQuery struct {
	and    bool
	groups []oddGroup
}

func setOf(items []string) map[string]bool {
	out := make(map[string]bool, len(items))
	for _, item := range items {
		out[item] = true
	}
	return out
}

// durationNS rounds seconds to the nearest nanosecond, so 8.3 s is exactly
// 8_300_000_000 ns; a positive value stays at least 1 ns so it never admits
// zero-length intervals.
func durationNS(seconds float64) int64 {
	ns := int64(math.Round(seconds * 1e9))
	if seconds > 0 && ns == 0 {
		return 1
	}
	return ns
}

func compileODDQuery(query model.ODDQuery) oddCompiledQuery {
	compiled := oddCompiledQuery{and: query.Op == "and"}
	for _, group := range query.Groups {
		g := oddGroup{and: group.Op == "and"}
		for _, c := range group.Conditions {
			g.conditions = append(g.conditions, oddCondition{
				key:           c.Key,
				values:        setOf(c.Values),
				sources:       setOf(c.Sources),
				not:           c.Not,
				minDurationNS: durationNS(c.MinDurationS),
				minConfidence: float32(c.MinConfidence),
			})
		}
		compiled.groups = append(compiled.groups, g)
	}
	return compiled
}

// matchODDQuery reports whether the scene satisfies the query and, if so, the
// intervals of the satisfied positive conditions, one lane per key in query
// order. A NOT condition contributes no intervals.
func matchODDQuery(scene oddScene, query oddCompiledQuery) ([]oddLane, bool) {
	order := make([]string, 0)
	hits := make(map[string]map[int]bool)
	collect := func(key string, indexes []int) {
		if hits[key] == nil {
			hits[key] = make(map[int]bool)
			order = append(order, key)
		}
		for _, index := range indexes {
			hits[key][index] = true
		}
	}
	satisfied := query.and
	for _, group := range query.groups {
		groupHolds := group.and
		type hit struct {
			key     string
			indexes []int
		}
		groupHits := make([]hit, 0, len(group.conditions))
		for _, condition := range group.conditions {
			indexes := matchingODDIntervals(scene.labels[condition.key], condition)
			holds := (len(indexes) > 0) != condition.not
			if holds && !condition.not {
				groupHits = append(groupHits, hit{condition.key, indexes})
			}
			if group.and {
				groupHolds = groupHolds && holds
			} else {
				groupHolds = groupHolds || holds
			}
		}
		if groupHolds {
			for _, h := range groupHits {
				collect(h.key, h.indexes)
			}
		}
		if query.and {
			satisfied = satisfied && groupHolds
		} else {
			satisfied = satisfied || groupHolds
		}
	}
	if len(query.groups) == 0 {
		satisfied = true
	}
	if !satisfied {
		return nil, false
	}
	lanes := make([]oddLane, 0, len(order))
	for _, key := range order {
		all := scene.labels[key]
		intervals := make([]oddInterval, 0, len(hits[key]))
		for index := range all {
			if hits[key][index] {
				intervals = append(intervals, all[index])
			}
		}
		lanes = append(lanes, oddLane{key: key, intervals: intervals})
	}
	return lanes, true
}

// matchingODDIntervals returns the indexes of the intervals that satisfy the
// condition, ignoring Not.
func matchingODDIntervals(intervals []oddInterval, condition oddCondition) []int {
	out := make([]int, 0)
	for index, interval := range intervals {
		if interval.end-interval.start < condition.minDurationNS ||
			interval.confidence < condition.minConfidence ||
			(len(condition.sources) > 0 && !condition.sources[interval.source]) ||
			!slices.ContainsFunc(interval.values, func(value string) bool {
				return condition.values[value]
			}) {
			continue
		}
		out = append(out, index)
	}
	return out
}

func oddSceneResult(scene oddScene, lanes []oddLane) model.ODDScene {
	result := model.ODDScene{
		SceneUID:   scene.SceneUID,
		Shard:      scene.ShardName,
		DurationNS: scene.EndTimestampNS - scene.StartTimestampNS,
		DistanceM:  scene.DistanceM,
		Lanes:      make([]model.ODDLane, len(lanes)),
	}
	for i, lane := range lanes {
		intervals := make([]model.ODDInterval, len(lane.intervals))
		for j, interval := range lane.intervals {
			intervals[j] = model.ODDInterval{
				StartNS:    interval.start - scene.StartTimestampNS,
				EndNS:      interval.end - scene.StartTimestampNS,
				Values:     interval.values,
				Source:     interval.source,
				Confidence: float64(interval.confidence),
				CameraID:   interval.cameraID,
			}
		}
		result.Lanes[i] = model.ODDLane{Key: lane.key, Intervals: intervals}
	}
	return result
}

// oddFacets counts, per key and value, the scenes carrying that value, and
// lists the sources of each key. Keys are sorted by name; values by scene
// count, then name.
func oddFacets(scenes []oddScene) []model.ODDFacet {
	counts := make(map[string]map[string]int)
	sources := make(map[string]map[string]bool)
	for _, scene := range scenes {
		for key, intervals := range scene.labels {
			if counts[key] == nil {
				counts[key] = make(map[string]int)
				sources[key] = make(map[string]bool)
			}
			seen := make(map[string]bool)
			for _, interval := range intervals {
				sources[key][interval.source] = true
				for _, value := range interval.values {
					if !seen[value] {
						seen[value] = true
						counts[key][value]++
					}
				}
			}
		}
	}
	facets := make([]model.ODDFacet, 0, len(counts))
	for key, values := range counts {
		facet := model.ODDFacet{
			Key:     key,
			Values:  make([]model.ODDFacetValue, 0, len(values)),
			Sources: make([]string, 0, len(sources[key])),
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
		for source := range sources[key] {
			facet.Sources = append(facet.Sources, source)
		}
		sort.Strings(facet.Sources)
		facets = append(facets, facet)
	}
	sort.Slice(facets, func(i, j int) bool {
		return facets[i].Key < facets[j].Key
	})
	return facets
}
