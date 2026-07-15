package store

import (
	"encoding/json"
	"reflect"
	"testing"
)

// sampleLabelJSON mirrors the real reasoning_label_v2 shape written to S3
// (nulls for every v2 context axis, which the console does not aggregate).
const sampleLabelJSON = `{
  "schema_version": "reasoning_label_v2",
  "sample_id": "l2d-v1-e000003-f000126",
  "dataset_name": "yaak-ai/L2D",
  "teacher_provider": "openai_compatible",
  "teacher_model": "nvidia/Cosmos3-Nano",
  "prompt_version": "action_relevant_reasoning_v3_temporal_front256",
  "request_mode": "temporal_front_clip",
  "provenance": "teacher_gt",
  "horizons": [
    {"horizon_sec": 0.0, "relation_to_ego": "same_lane_ahead", "hazard_event": ["no_hazard"], "cause": ["lead_vehicle"], "longitudinal_response": "slow_down", "lateral_response": "keep_lane", "tactical_response": "proceed_with_caution", "rule_response": "none", "confidence": 0.99, "provenance": "teacher_gt", "global_scene_context": null, "road_topology": null},
    {"horizon_sec": 1.0, "relation_to_ego": "same_lane_ahead", "hazard_event": ["no_hazard"], "cause": ["lead_vehicle"], "longitudinal_response": "slow_down", "lateral_response": "keep_lane", "tactical_response": "proceed_with_caution", "rule_response": "none", "confidence": 0.95, "provenance": "teacher_gt"},
    {"horizon_sec": 2.0, "relation_to_ego": "same_lane_ahead", "hazard_event": ["no_hazard"], "cause": ["lead_vehicle"], "longitudinal_response": "slow_down", "lateral_response": "keep_lane", "tactical_response": "proceed_with_caution", "rule_response": "none", "confidence": 0.90, "provenance": "teacher_gt"},
    {"horizon_sec": 3.0, "relation_to_ego": "same_lane_ahead", "hazard_event": ["no_hazard"], "cause": ["lead_vehicle"], "longitudinal_response": "slow_down", "lateral_response": "keep_lane", "tactical_response": "proceed_with_caution", "rule_response": "none", "confidence": 0.85, "provenance": "teacher_gt"},
    {"horizon_sec": 4.0, "relation_to_ego": "same_lane_ahead", "hazard_event": ["no_hazard"], "cause": ["lead_vehicle"], "longitudinal_response": "slow_down", "lateral_response": "keep_lane", "tactical_response": "proceed_with_caution", "rule_response": "none", "confidence": 0.80, "provenance": "teacher_gt"}
  ]
}`

func TestParseReasoningLabel(t *testing.T) {
	lbl, err := ParseReasoningLabel([]byte(sampleLabelJSON))
	if err != nil {
		t.Fatalf("ParseReasoningLabel error: %v", err)
	}
	if lbl.SampleID != "l2d-v1-e000003-f000126" {
		t.Errorf("SampleID = %q", lbl.SampleID)
	}
	if lbl.TeacherProvider != "openai_compatible" ||
		lbl.TeacherModel != "nvidia/Cosmos3-Nano" ||
		lbl.PromptVersion != "action_relevant_reasoning_v3_temporal_front256" {
		t.Errorf("provenance was not decoded: %+v", lbl)
	}
	if len(lbl.Horizons) != 5 {
		t.Fatalf("Horizons len = %d, want 5", len(lbl.Horizons))
	}
	h := lbl.Horizons[0]
	if h.RelationToEgo != "same_lane_ahead" || h.LongitudinalResponse != "slow_down" {
		t.Errorf("unexpected horizon 0: %+v", h)
	}
	if !reflect.DeepEqual(h.HazardEvent, []string{"no_hazard"}) {
		t.Errorf("HazardEvent = %v, want [no_hazard]", h.HazardEvent)
	}
	if !reflect.DeepEqual(h.Cause, []string{"lead_vehicle"}) {
		t.Errorf("Cause = %v, want [lead_vehicle]", h.Cause)
	}
	if h.Confidence != 0.99 {
		t.Errorf("Confidence = %v, want 0.99", h.Confidence)
	}
}

func TestParseReasoningLabelRejectsContractViolations(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(map[string]any)
	}{
		{
			name: "wrong schema",
			mutate: func(label map[string]any) {
				label["schema_version"] = "reasoning_label_v1"
			},
		},
		{
			name: "missing teacher provenance",
			mutate: func(label map[string]any) {
				delete(label, "teacher_model")
			},
		},
		{
			name: "four horizons",
			mutate: func(label map[string]any) {
				label["horizons"] = label["horizons"].([]any)[:4]
			},
		},
		{
			name: "unordered horizon",
			mutate: func(label map[string]any) {
				horizons := label["horizons"].([]any)
				horizons[2].(map[string]any)["horizon_sec"] = 3.0
			},
		},
		{
			name: "unknown taxonomy label",
			mutate: func(label map[string]any) {
				horizons := label["horizons"].([]any)
				horizons[0].(map[string]any)["lateral_response"] = "turn_left"
			},
		},
		{
			name: "duplicate multi label",
			mutate: func(label map[string]any) {
				horizons := label["horizons"].([]any)
				horizons[0].(map[string]any)["hazard_event"] =
					[]any{"no_hazard", "no_hazard"}
			},
		},
		{
			name: "confidence above one",
			mutate: func(label map[string]any) {
				horizons := label["horizons"].([]any)
				horizons[0].(map[string]any)["confidence"] = 1.1
			},
		},
		{
			name: "null confidence",
			mutate: func(label map[string]any) {
				horizons := label["horizons"].([]any)
				horizons[0].(map[string]any)["confidence"] = nil
			},
		},
		{
			name: "missing confidence",
			mutate: func(label map[string]any) {
				horizons := label["horizons"].([]any)
				delete(horizons[0].(map[string]any), "confidence")
			},
		},
		{
			name: "missing horizon provenance",
			mutate: func(label map[string]any) {
				horizons := label["horizons"].([]any)
				delete(horizons[0].(map[string]any), "provenance")
			},
		},
		{
			name: "inconsistent abstention",
			mutate: func(label map[string]any) {
				label["abstained"] = true
				label["provenance"] = "teacher_error"
				label["teacher_error"] = "timeout"
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var label map[string]any
			if err := json.Unmarshal([]byte(sampleLabelJSON), &label); err != nil {
				t.Fatal(err)
			}
			test.mutate(label)
			body, err := json.Marshal(label)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := ParseReasoningLabel(body); err == nil {
				t.Fatal("invalid reasoning label was accepted")
			}
		})
	}
	if _, err := ParseReasoningLabel([]byte(`{"schema_version":`)); err == nil {
		t.Fatal("malformed reasoning JSON was accepted")
	}

	for _, confidence := range []float64{0, 1} {
		var label map[string]any
		if err := json.Unmarshal([]byte(sampleLabelJSON), &label); err != nil {
			t.Fatal(err)
		}
		label["horizons"].([]any)[0].(map[string]any)["confidence"] =
			confidence
		body, err := json.Marshal(label)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := ParseReasoningLabel(body); err != nil {
			t.Fatalf("boundary confidence %v was rejected: %v", confidence, err)
		}
	}
}

func TestParseReasoningLabelAcceptsExplicitAbstention(t *testing.T) {
	body := []byte(`{
		"schema_version": "reasoning_label_v2",
		"sample_id": "l2d-v1-e000003-f000126",
		"dataset_name": "yaak-ai/L2D",
		"teacher_provider": "openai_compatible",
		"teacher_model": "nvidia/Cosmos3-Nano",
		"prompt_version": "action_relevant_reasoning_v3_temporal_front256",
		"request_mode": "temporal_front_clip",
		"provenance": "teacher_error",
		"abstained": true,
		"teacher_error": "timeout",
		"horizons": []
	}`)
	label, err := ParseReasoningLabel(body)
	if err != nil {
		t.Fatal(err)
	}
	if !label.Abstained || label.TeacherError != "timeout" {
		t.Fatalf("abstained label = %+v", label)
	}
}

func TestAggregateStats_AcrossHorizons(t *testing.T) {
	labels := []ReasoningLabel{
		{
			SampleID: "s0",
			Horizons: []LabelHorizon{
				{RelationToEgo: "same_lane_ahead", HazardEvent: []string{"no_hazard"}, Cause: []string{"lead_vehicle"}, LongitudinalResponse: "slow_down", LateralResponse: "keep_lane", TacticalResponse: "proceed", RuleResponse: "none", Confidence: 0.99},
				{RelationToEgo: "same_lane_ahead", HazardEvent: []string{"no_hazard", "cut_in_risk"}, Cause: []string{"lead_vehicle", "cut_in_vehicle"}, LongitudinalResponse: "slow_down", LateralResponse: "turn_left", TacticalResponse: "wait", RuleResponse: "none", Confidence: 0.5},
			},
		},
		{
			SampleID: "s1",
			Horizons: []LabelHorizon{
				{RelationToEgo: "crossing_path", HazardEvent: []string{"vru_collision_risk"}, Cause: []string{"pedestrian_crossing"}, LongitudinalResponse: "stop", LateralResponse: "keep_lane", TacticalResponse: "wait_for_actor", RuleResponse: "yield_to_vru", Confidence: 0.05},
			},
		},
	}

	blob := AggregateStats(labels)

	if blob.NLabels != 2 {
		t.Errorf("NLabels = %d, want 2", blob.NLabels)
	}
	if blob.HorizonCount != 3 {
		t.Errorf("HorizonCount = %d, want 3 (2 + 1)", blob.HorizonCount)
	}

	// relation_to_ego: same_lane_ahead x2 (from s0's two horizons), crossing_path x1.
	rel := blob.ByField[FieldRelationToEgo]
	if rel["same_lane_ahead"] != 2 || rel["crossing_path"] != 1 {
		t.Errorf("relation_to_ego = %v, want same_lane_ahead:2 crossing_path:1", rel)
	}

	// lateral_response (the turn/steer axis): keep_lane x2, turn_left x1.
	lat := blob.ByField[FieldLateralResponse]
	if lat["keep_lane"] != 2 || lat["turn_left"] != 1 {
		t.Errorf("lateral_response = %v, want keep_lane:2 turn_left:1", lat)
	}

	// hazard_event (multi-label): no_hazard x2, cut_in_risk x1, vru_collision_risk x1.
	hz := blob.ByField[FieldHazardEvent]
	if hz["no_hazard"] != 2 || hz["cut_in_risk"] != 1 || hz["vru_collision_risk"] != 1 {
		t.Errorf("hazard_event = %v", hz)
	}

	// cause (multi-label): lead_vehicle x2, cut_in_vehicle x1, pedestrian_crossing x1.
	cause := blob.ByField[FieldCause]
	if cause["lead_vehicle"] != 2 || cause["cut_in_vehicle"] != 1 || cause["pedestrian_crossing"] != 1 {
		t.Errorf("cause = %v", cause)
	}

	// confidence histogram: 0.99 -> bucket "0.9-1.0", 0.5 -> "0.5-0.6", 0.05 -> "0.0-0.1".
	byBucket := map[string]int{}
	for _, b := range blob.ConfidenceHistogram {
		byBucket[b.Bucket] = b.Count
	}
	if byBucket["0.9-1.0"] != 1 || byBucket["0.5-0.6"] != 1 || byBucket["0.0-0.1"] != 1 {
		t.Errorf("confidence histogram = %v", byBucket)
	}
	if len(blob.ConfidenceHistogram) != 10 {
		t.Errorf("confidence histogram len = %d, want 10 fixed buckets", len(blob.ConfidenceHistogram))
	}
}

func TestAggregateStats_SkipsEmptyValues(t *testing.T) {
	labels := []ReasoningLabel{
		{SampleID: "s0", Horizons: []LabelHorizon{
			{RelationToEgo: "", HazardEvent: []string{"", "no_hazard"}, Cause: nil, LongitudinalResponse: "keep_speed", Confidence: 0.8},
		}},
	}
	blob := AggregateStats(labels)
	if len(blob.ByField[FieldRelationToEgo]) != 0 {
		t.Errorf("empty relation_to_ego should be skipped, got %v", blob.ByField[FieldRelationToEgo])
	}
	if blob.ByField[FieldHazardEvent]["no_hazard"] != 1 || blob.ByField[FieldHazardEvent][""] != 0 {
		t.Errorf("hazard_event should count only non-empty: %v", blob.ByField[FieldHazardEvent])
	}
	if blob.ByField[FieldLongitudinalResponse]["keep_speed"] != 1 {
		t.Errorf("longitudinal_response = %v", blob.ByField[FieldLongitudinalResponse])
	}
}

func TestConfBucket_Boundaries(t *testing.T) {
	cases := map[float64]int{
		-0.5: 0, 0.0: 0, 0.09: 0, 0.1: 1, 0.55: 5, 0.9: 9, 0.99: 9, 1.0: 9, 2.0: 9,
	}
	for c, want := range cases {
		if got := confBucket(c); got != want {
			t.Errorf("confBucket(%v) = %d, want %d", c, got, want)
		}
	}
}

func TestSceneLabelRows_AllHorizons(t *testing.T) {
	// The scene index MUST cover every horizon so it matches AggregateStats
	// (which counts all horizons): a value that appears only at a future horizon
	// still renders a clickable bar, so it must be searchable. A value shared
	// across horizons is de-duplicated to a single row.
	lbl := ReasoningLabel{
		SampleID: "s00000010",
		Horizons: []LabelHorizon{
			{RelationToEgo: "same_lane_ahead", HazardEvent: []string{"cut_in_risk", "collision_risk"}, Cause: []string{"cut_in_vehicle"}, LongitudinalResponse: "slow_down", LateralResponse: "turn_right", TacticalResponse: "wait", RuleResponse: "none", Confidence: 0.9},
			// A distinct horizon-1 state MUST also contribute (the bar counts it).
			{RelationToEgo: "crossing_path", HazardEvent: []string{"vru_collision_risk"}, Cause: []string{"pedestrian_crossing"}, LongitudinalResponse: "stop", LateralResponse: "keep_lane"},
		},
	}
	rows := SceneLabelRows(lbl)

	got := map[[2]string]int{}
	for _, r := range rows {
		if r.SampleID != "s00000010" {
			t.Errorf("row sample id = %q, want s00000010", r.SampleID)
		}
		got[[2]string{r.Field, r.Value}]++
	}
	want := [][2]string{
		// horizon 0
		{FieldRelationToEgo, "same_lane_ahead"},
		{FieldHazardEvent, "cut_in_risk"},
		{FieldHazardEvent, "collision_risk"},
		{FieldCause, "cut_in_vehicle"},
		{FieldLongitudinalResponse, "slow_down"},
		{FieldLateralResponse, "turn_right"},
		{FieldTacticalResponse, "wait"},
		{FieldRuleResponse, "none"},
		// horizon 1 (future-only values must be present now)
		{FieldRelationToEgo, "crossing_path"},
		{FieldHazardEvent, "vru_collision_risk"},
		{FieldCause, "pedestrian_crossing"},
		{FieldLongitudinalResponse, "stop"},
		{FieldLateralResponse, "keep_lane"},
	}
	for _, w := range want {
		if got[w] != 1 {
			t.Errorf("scene row %v: got %d, want exactly 1", w, got[w])
		}
	}
	if len(rows) != len(want) {
		t.Errorf("SceneLabelRows produced %d rows, want %d: %+v", len(rows), len(want), rows)
	}
}

func TestSceneLabelRows_Dedup(t *testing.T) {
	// A hazard list with a repeated member must yield one row for it.
	lbl := ReasoningLabel{
		SampleID: "s0",
		Horizons: []LabelHorizon{{HazardEvent: []string{"no_hazard", "no_hazard"}, RelationToEgo: "outside_path"}},
	}
	rows := SceneLabelRows(lbl)
	count := 0
	for _, r := range rows {
		if r.Field == FieldHazardEvent && r.Value == "no_hazard" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("duplicate hazard member produced %d rows, want 1", count)
	}
}

func TestSceneLabelRows_EmptyLabel(t *testing.T) {
	if rows := SceneLabelRows(ReasoningLabel{SampleID: "s0"}); rows != nil {
		t.Errorf("no horizons should yield no rows, got %v", rows)
	}
	if rows := SceneLabelRows(ReasoningLabel{Horizons: []LabelHorizon{{RelationToEgo: "x"}}}); rows != nil {
		t.Errorf("no sample id should yield no rows, got %v", rows)
	}
}

func TestSortedByField(t *testing.T) {
	m := map[string]int{"a": 1, "b": 3, "c": 3, "d": 2}
	got := SortedByField(m)
	// Descending count, then ascending value: b(3), c(3), d(2), a(1).
	wantOrder := []string{"b", "c", "d", "a"}
	if len(got) != len(wantOrder) {
		t.Fatalf("len = %d, want %d", len(got), len(wantOrder))
	}
	for i, w := range wantOrder {
		if got[i].Bucket != w {
			t.Errorf("SortedByField[%d].Bucket = %q, want %q (full: %+v)", i, got[i].Bucket, w, got)
		}
	}
}
