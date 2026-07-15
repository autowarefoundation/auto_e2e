package store

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

// ReasoningLabel is the subset of a reasoning_label_v2 JSON object the console
// aggregates. Only the fields that actually carry values in the data are
// decoded; the v2 context axes (weather, geo, road topology, actor_*, timing)
// are ALL null in this dataset and are intentionally not modelled here.
type ReasoningLabel struct {
	SchemaVersion   string         `json:"schema_version"`
	SampleID        string         `json:"sample_id"`
	DatasetName     string         `json:"dataset_name"`
	TeacherProvider string         `json:"teacher_provider"`
	TeacherModel    string         `json:"teacher_model"`
	PromptVersion   string         `json:"prompt_version"`
	RequestMode     string         `json:"request_mode"`
	Provenance      string         `json:"provenance"`
	Abstained       bool           `json:"abstained"`
	TeacherError    string         `json:"teacher_error"`
	Horizons        []LabelHorizon `json:"horizons"`
}

// LabelHorizon is one horizon (now/+1s/.../+4s) of a reasoning label. Single-
// label axes are scalars; hazard_event and cause are multi-label lists.
type LabelHorizon struct {
	HorizonSec           *float64 `json:"horizon_sec"`
	RelationToEgo        string   `json:"relation_to_ego"`
	HazardEvent          []string `json:"hazard_event"`
	Cause                []string `json:"cause"`
	LongitudinalResponse string   `json:"longitudinal_response"`
	LateralResponse      string   `json:"lateral_response"`
	TacticalResponse     string   `json:"tactical_response"`
	RuleResponse         string   `json:"rule_response"`
	Confidence           float64  `json:"confidence"`
	Provenance           string   `json:"provenance"`
}

func (horizon *LabelHorizon) UnmarshalJSON(body []byte) error {
	type plainHorizon LabelHorizon
	var decoded plainHorizon
	if err := json.Unmarshal(body, &decoded); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return err
	}
	confidence, exists := fields["confidence"]
	if !exists || bytes.Equal(bytes.TrimSpace(confidence), []byte("null")) {
		return fmt.Errorf("reasoning horizon confidence is required")
	}
	*horizon = LabelHorizon(decoded)
	return nil
}

const reasoningLabelSchema = "reasoning_label_v2"

var validReasoningProvenance = stringSet(
	"audited_gt",
	"direct_gt",
	"derived_gt",
	"teacher_gt",
	"weak_gt",
	"counterfactual_gt",
	"teacher_error",
)

var reasoningTaxonomy = map[string]map[string]struct{}{
	FieldRelationToEgo: stringSet(
		"same_lane_ahead", "same_lane_behind", "left_adjacent",
		"right_adjacent", "crossing_path", "about_to_cross_path",
		"merging_into_ego_path", "cutting_into_ego_path",
		"oncoming_conflict", "intersection_conflict",
		"blocking_current_lane", "blocking_target_lane", "blocking_route",
		"occluded_near_path", "outside_path", "behind_ego",
		"unknown_relation",
	),
	FieldHazardEvent: stringSet(
		"no_hazard", "collision_risk", "vru_collision_risk",
		"cut_in_risk", "merge_conflict_risk",
		"right_of_way_violation_risk", "red_light_violation_risk",
		"blocked_route_risk", "occlusion_risk", "low_friction_risk",
		"emergency_vehicle_risk", "unknown_hazard",
	),
	FieldCause: stringSet(
		"lead_vehicle", "slow_lead_vehicle", "stopped_lead_vehicle",
		"cut_in_vehicle", "cross_traffic", "oncoming_vehicle",
		"pedestrian_crossing", "pedestrian_about_to_cross", "vru_conflict",
		"red_light", "yellow_light", "stop_sign", "yield_sign",
		"human_direction", "route_turn", "route_merge",
		"route_lane_change", "lane_ending", "object_blocking_path",
		"blocked_lane", "road_closed", "construction_blocking_path",
		"occlusion", "poor_visibility", "slippery_road",
		"uncertainty_high", "unknown_cause",
	),
	FieldLongitudinalResponse: stringSet(
		"keep_speed", "accelerate", "coast", "slow_down", "prepare_stop",
		"stop", "stay_stopped", "creep", "yield", "follow_lead_vehicle",
		"increase_gap", "emergency_brake", "unknown_longitudinal",
	),
	FieldLateralResponse: stringSet(
		"keep_lane", "nudge_left", "nudge_right", "shift_left_within_lane",
		"shift_right_within_lane", "lane_change_left", "lane_change_right",
		"avoid_left", "avoid_right", "return_to_lane", "pull_over",
		"reverse", "unknown_lateral",
	),
	FieldTacticalResponse: stringSet(
		"proceed", "proceed_with_caution", "wait", "wait_for_gap",
		"wait_for_actor", "wait_for_signal", "creep_for_visibility",
		"negotiate_merge", "negotiate_unprotected_turn",
		"yield_then_proceed", "stop_then_proceed", "reroute_or_wait",
		"unknown_tactical",
	),
	FieldRuleResponse: stringSet(
		"none", "wait_for_green", "stop_at_stop_line",
		"stop_before_crosswalk", "yield_to_vru", "yield_to_oncoming",
		"yield_to_cross_traffic", "yield_to_emergency_vehicle",
		"obey_human_direction", "respect_speed_limit",
		"slow_for_school_zone", "slow_for_construction_zone",
		"do_not_enter", "do_not_turn", "unknown_rule",
	),
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

// ParseReasoningLabel decodes and validates one reasoning-label JSON body.
// Optional context fields remain forward-compatible, while the current
// identity, horizon, provenance, taxonomy, and confidence contracts are strict.
func ParseReasoningLabel(body []byte) (ReasoningLabel, error) {
	var lbl ReasoningLabel
	if err := json.Unmarshal(body, &lbl); err != nil {
		return ReasoningLabel{}, fmt.Errorf("decode reasoning label: %w", err)
	}
	if err := validateReasoningLabel(lbl); err != nil {
		return ReasoningLabel{}, err
	}
	return lbl, nil
}

func validateReasoningLabel(label ReasoningLabel) error {
	if label.SchemaVersion != reasoningLabelSchema {
		return fmt.Errorf(
			"reasoning label schema is %q, want %q",
			label.SchemaVersion, reasoningLabelSchema,
		)
	}
	for name, value := range map[string]string{
		"sample_id":        label.SampleID,
		"dataset_name":     label.DatasetName,
		"teacher_provider": label.TeacherProvider,
		"teacher_model":    label.TeacherModel,
		"prompt_version":   label.PromptVersion,
		"request_mode":     label.RequestMode,
		"provenance":       label.Provenance,
	} {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("reasoning label %s is required", name)
		}
	}
	if _, ok := validReasoningProvenance[label.Provenance]; !ok {
		return fmt.Errorf(
			"reasoning label has invalid provenance %q", label.Provenance,
		)
	}
	if label.Abstained {
		if strings.TrimSpace(label.TeacherError) == "" ||
			len(label.Horizons) != 0 ||
			label.Provenance != "teacher_error" {
			return fmt.Errorf("abstained reasoning label is inconsistent")
		}
		return nil
	}
	if label.TeacherError != "" || label.Provenance == "teacher_error" {
		return fmt.Errorf("successful reasoning label has teacher error provenance")
	}
	if len(label.Horizons) != 5 {
		return fmt.Errorf(
			"reasoning label has %d horizons, want 5", len(label.Horizons),
		)
	}
	for i, horizon := range label.Horizons {
		if horizon.HorizonSec == nil || *horizon.HorizonSec != float64(i) {
			return fmt.Errorf(
				"reasoning horizon %d has invalid horizon_sec", i,
			)
		}
		if _, ok := validReasoningProvenance[horizon.Provenance]; !ok ||
			horizon.Provenance == "teacher_error" {
			return fmt.Errorf(
				"reasoning horizon %d has invalid provenance %q",
				i, horizon.Provenance,
			)
		}
		if math.IsNaN(horizon.Confidence) ||
			math.IsInf(horizon.Confidence, 0) ||
			horizon.Confidence < 0 ||
			horizon.Confidence > 1 {
			return fmt.Errorf(
				"reasoning horizon %d has invalid confidence %v",
				i, horizon.Confidence,
			)
		}
		if err := validateTaxonomyValue(
			FieldRelationToEgo, horizon.RelationToEgo,
		); err != nil {
			return fmt.Errorf("reasoning horizon %d: %w", i, err)
		}
		if err := validateTaxonomyValues(
			FieldHazardEvent, horizon.HazardEvent,
		); err != nil {
			return fmt.Errorf("reasoning horizon %d: %w", i, err)
		}
		if err := validateTaxonomyValues(
			FieldCause, horizon.Cause,
		); err != nil {
			return fmt.Errorf("reasoning horizon %d: %w", i, err)
		}
		for field, value := range map[string]string{
			FieldLongitudinalResponse: horizon.LongitudinalResponse,
			FieldLateralResponse:      horizon.LateralResponse,
			FieldTacticalResponse:     horizon.TacticalResponse,
			FieldRuleResponse:         horizon.RuleResponse,
		} {
			if err := validateTaxonomyValue(field, value); err != nil {
				return fmt.Errorf("reasoning horizon %d: %w", i, err)
			}
		}
	}
	return nil
}

func validateTaxonomyValue(field, value string) error {
	// The Python contract permits a null single-label value as a per-axis
	// abstention. JSON null decodes to the empty string here.
	if value == "" {
		return nil
	}
	if _, ok := reasoningTaxonomy[field][value]; !ok {
		return fmt.Errorf("unknown %s label %q", field, value)
	}
	return nil
}

func validateTaxonomyValues(field string, values []string) error {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if _, ok := reasoningTaxonomy[field][value]; !ok {
			return fmt.Errorf("unknown %s label %q", field, value)
		}
		if _, duplicate := seen[value]; duplicate {
			return fmt.Errorf("duplicate %s label %q", field, value)
		}
		seen[value] = struct{}{}
	}
	return nil
}

func stringSet(values ...string) map[string]struct{} {
	set := make(map[string]struct{}, len(values))
	for _, value := range values {
		set[value] = struct{}{}
	}
	return set
}

// ReasoningStatsAccumulator incrementally aggregates parsed labels without
// retaining the label slice or allocating a new histogram for every record.
type ReasoningStatsAccumulator struct {
	blob             model.ReasoningStatsBlob
	confidenceCounts [10]int
}

// NewReasoningStatsAccumulator initializes all stable taxonomy fields.
func NewReasoningStatsAccumulator() *ReasoningStatsAccumulator {
	byField := make(map[string]map[string]int, len(StatFields))
	for _, f := range StatFields {
		byField[f] = map[string]int{}
	}
	return &ReasoningStatsAccumulator{
		blob: model.ReasoningStatsBlob{ByField: byField},
	}
}

// Add incorporates one parsed label.
func (a *ReasoningStatsAccumulator) Add(label ReasoningLabel) {
	a.blob.NRecords++
	if label.Abstained {
		a.blob.NAbstained++
		return
	}
	a.blob.NLabels++
	for _, horizon := range label.Horizons {
		a.blob.HorizonCount++
		addScalar(
			a.blob.ByField[FieldRelationToEgo],
			horizon.RelationToEgo,
		)
		addList(
			a.blob.ByField[FieldHazardEvent],
			horizon.HazardEvent,
		)
		addList(a.blob.ByField[FieldCause], horizon.Cause)
		addScalar(
			a.blob.ByField[FieldLongitudinalResponse],
			horizon.LongitudinalResponse,
		)
		addScalar(
			a.blob.ByField[FieldLateralResponse],
			horizon.LateralResponse,
		)
		addScalar(
			a.blob.ByField[FieldTacticalResponse],
			horizon.TacticalResponse,
		)
		addScalar(
			a.blob.ByField[FieldRuleResponse],
			horizon.RuleResponse,
		)
		a.confidenceCounts[confBucket(horizon.Confidence)]++
	}
}

// Snapshot returns the current aggregate. Callers must treat the returned
// ByField maps as read-only while further Add calls are possible.
func (a *ReasoningStatsAccumulator) Snapshot() model.ReasoningStatsBlob {
	blob := a.blob
	blob.ConfidenceHistogram = confHistogram(a.confidenceCounts[:])
	return blob
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
	accumulator := NewReasoningStatsAccumulator()
	for _, label := range labels {
		accumulator.Add(label)
	}
	return accumulator.Snapshot()
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
// scene-by-label search index. Shard is attached by the materializer after the
// pure label-to-row expansion so scene search never has to scan shard indexes.
type SceneLabelRow struct {
	Field    string
	Value    string
	SampleID string
	Shard    string
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
	if lbl.Abstained || len(lbl.Horizons) == 0 || lbl.SampleID == "" {
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
