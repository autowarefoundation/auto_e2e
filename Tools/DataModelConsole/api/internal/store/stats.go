package store

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

// ReasoningLabel is the subset of a reasoning_label_v2 JSON object the console
// aggregates. Only the fields that actually carry values in the data are
// decoded; the v2 context axes (weather, geo, road topology, actor_*, timing)
// are ALL null in this dataset and are intentionally not modelled here.
type ReasoningLabel struct {
	SampleID string         `json:"sample_id"`
	Horizons []LabelHorizon `json:"horizons"`
}

// LabelHorizon is one horizon (now/+1s/.../+4s) of a reasoning label. Single-
// label axes are scalars; hazard_event and cause are multi-label lists.
type LabelHorizon struct {
	RelationToEgo        string   `json:"relation_to_ego"`
	HazardEvent          []string `json:"hazard_event"`
	Cause                []string `json:"cause"`
	LongitudinalResponse string   `json:"longitudinal_response"`
	LateralResponse      string   `json:"lateral_response"`
	TacticalResponse     string   `json:"tactical_response"`
	RuleResponse         string   `json:"rule_response"`
	Confidence           float64  `json:"confidence"`
}

// statFields is the fixed set of categorical taxonomy axes aggregated into
// by_field. The scalar (single-label) axes contribute one value per horizon;
// the two list (multi-label) axes contribute each member.
const (
	FieldRelationToEgo        = "relation_to_ego"
	FieldHazardEvent          = "hazard_event"
	FieldCause                = "cause"
	FieldLongitudinalResponse = "longitudinal_response"
	FieldLateralResponse      = "lateral_response"
	FieldTacticalResponse     = "tactical_response"
	FieldRuleResponse         = "rule_response"
)

// StatFields lists every searchable/aggregatable taxonomy axis in a stable
// order. Used to validate ?field= on the scene-search endpoint and to iterate
// the by_field map deterministically.
var StatFields = []string{
	FieldRelationToEgo,
	FieldHazardEvent,
	FieldCause,
	FieldLongitudinalResponse,
	FieldLateralResponse,
	FieldTacticalResponse,
	FieldRuleResponse,
}

// IsStatField reports whether f is a known aggregatable/searchable axis.
func IsStatField(f string) bool {
	for _, s := range StatFields {
		if s == f {
			return true
		}
	}
	return false
}

// ParseReasoningLabel decodes one reasoning-label JSON body into a
// ReasoningLabel (pure; no AWS). Unknown/null fields decode to zero values.
func ParseReasoningLabel(body []byte) (ReasoningLabel, error) {
	var lbl ReasoningLabel
	if err := json.Unmarshal(body, &lbl); err != nil {
		return ReasoningLabel{}, fmt.Errorf("decode reasoning label: %w", err)
	}
	return lbl, nil
}

// AggregateStats builds the precomputed stats blob from a slice of parsed
// labels (pure; no AWS). Every horizon of every label contributes to by_field
// and the confidence histogram, so the distribution answers "which ODD does
// this label set cover" across the full 5-horizon window.
//
// Empty/blank categorical values are skipped (they carry no ODD signal); the
// taxonomy's own abstain labels (no_hazard, unknown_*, none) are real values
// and ARE counted so an all-nominal set still reports its dominant class.
func AggregateStats(labels []ReasoningLabel) model.ReasoningStatsBlob {
	byField := map[string]map[string]int{}
	for _, f := range StatFields {
		byField[f] = map[string]int{}
	}
	// 10 fixed confidence buckets [0.0-0.1) .. [0.9-1.0]; the top bucket is
	// closed so confidence==1.0 lands in "0.9-1.0" rather than overflowing.
	confCounts := make([]int, 10)

	horizonCount := 0
	for _, lbl := range labels {
		for _, h := range lbl.Horizons {
			horizonCount++
			addScalar(byField[FieldRelationToEgo], h.RelationToEgo)
			addList(byField[FieldHazardEvent], h.HazardEvent)
			addList(byField[FieldCause], h.Cause)
			addScalar(byField[FieldLongitudinalResponse], h.LongitudinalResponse)
			addScalar(byField[FieldLateralResponse], h.LateralResponse)
			addScalar(byField[FieldTacticalResponse], h.TacticalResponse)
			addScalar(byField[FieldRuleResponse], h.RuleResponse)
			confCounts[confBucket(h.Confidence)]++
		}
	}

	return model.ReasoningStatsBlob{
		NLabels:             len(labels),
		HorizonCount:        horizonCount,
		ByField:             byField,
		ConfidenceHistogram: confHistogram(confCounts),
	}
}

func addScalar(m map[string]int, v string) {
	if v == "" {
		return
	}
	m[v]++
}

func addList(m map[string]int, vs []string) {
	for _, v := range vs {
		if v == "" {
			continue
		}
		m[v]++
	}
}

// confBucket maps a confidence in [0,1] to one of 10 buckets. Values outside
// [0,1] clamp to the nearest edge so a malformed teacher output cannot panic
// the aggregation.
func confBucket(c float64) int {
	if c < 0 {
		c = 0
	}
	if c >= 1 {
		return 9
	}
	return int(c * 10)
}

func confHistogram(counts []int) []model.HistogramBucket {
	labels := []string{
		"0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
		"0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0",
	}
	out := make([]model.HistogramBucket, len(counts))
	for i, c := range counts {
		out[i] = model.HistogramBucket{Bucket: labels[i], Count: c}
	}
	return out
}

// SceneLabelRow is one (field,value)->sample_id pairing to persist in the
// scene-by-label search index. It is derived from a single label's horizon-0
// state plus every hazard/cause list member (also at horizon 0): those are the
// labels a user searches "show me scenes that are X".
type SceneLabelRow struct {
	Field    string
	Value    string
	SampleID string
}

// SceneLabelRows extracts the searchable (field,value) pairs of one label over
// ALL horizons (pure; no AWS), de-duplicated per (field,value) so a sample is
// indexed once per value it carries anywhere in its horizon window.
//
// This MUST match the horizon window AggregateStats counts (every horizon): the
// ODD bar charts are built from all 5 horizons, and each bar is click-through
// to this scene index. Indexing only horizon 0 (the old behavior) meant a value
// that appears only at a future horizon (+1s..+4s) rendered a nonzero, clickable
// bar that opened an empty "No matching scenes" drawer. Multi-label axes
// contribute every member; blank values are skipped.
func SceneLabelRows(lbl ReasoningLabel) []SceneLabelRow {
	if len(lbl.Horizons) == 0 || lbl.SampleID == "" {
		return nil
	}
	seen := map[[2]string]struct{}{}
	var rows []SceneLabelRow
	add := func(field, value string) {
		if value == "" {
			return
		}
		k := [2]string{field, value}
		if _, dup := seen[k]; dup {
			return
		}
		seen[k] = struct{}{}
		rows = append(rows, SceneLabelRow{Field: field, Value: value, SampleID: lbl.SampleID})
	}
	for _, h := range lbl.Horizons {
		add(FieldRelationToEgo, h.RelationToEgo)
		for _, v := range h.HazardEvent {
			add(FieldHazardEvent, v)
		}
		for _, v := range h.Cause {
			add(FieldCause, v)
		}
		add(FieldLongitudinalResponse, h.LongitudinalResponse)
		add(FieldLateralResponse, h.LateralResponse)
		add(FieldTacticalResponse, h.TacticalResponse)
		add(FieldRuleResponse, h.RuleResponse)
	}
	return rows
}

// SortedByField returns a field's value->count map as (value,count) buckets
// sorted by descending count then value, for a stable API response ordering.
// Kept here (pure) so handlers/tests share one ordering rule.
func SortedByField(m map[string]int) []model.HistogramBucket {
	out := make([]model.HistogramBucket, 0, len(m))
	for v, c := range m {
		out = append(out, model.HistogramBucket{Bucket: v, Count: c})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Bucket < out[j].Bucket
	})
	return out
}
