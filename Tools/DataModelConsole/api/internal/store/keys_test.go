package store

import (
	"strings"
	"testing"
)

const testReasoningGeneration = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func TestShardIndexPK(t *testing.T) {
	got := ShardIndexPK("l2d", "v2.0", "train-000000.tar")
	want := "IDX#l2d#v2.0#train-000000.tar"
	if got != want {
		t.Errorf("ShardIndexPK = %q, want %q", got, want)
	}
}

func TestStatsPK(t *testing.T) {
	got := StatsPK("l2d", "v2.0", "action_relevant_reasoning_v3_temporal_front256")
	want := "STATS#l2d#v2.0#action_relevant_reasoning_v3_temporal_front256"
	if got != want {
		t.Errorf("StatsPK = %q, want %q", got, want)
	}
}

func TestEmbeddedStatsPK(t *testing.T) {
	got := EmbeddedStatsPK(
		"l2d", "v2.1",
		"action_relevant_reasoning_v3_temporal_front256",
	)
	want := "STATSV2#l2d#v2.1#action_relevant_reasoning_v3_temporal_front256"
	if got != want {
		t.Errorf("EmbeddedStatsPK = %q, want %q", got, want)
	}
}

func TestTeacherScopedEmbeddedStatsPK(t *testing.T) {
	got, err := EmbeddedTeacherStatsPK(
		"l2d",
		"v2.1",
		testReasoningGeneration,
		"b3BlbmFpX2NvbXBhdGlibGUAbnZpZGlhL0Nvc21vczMtTmFubw",
		"action_relevant_reasoning_v3_temporal_front256",
	)
	if err != nil {
		t.Fatal(err)
	}
	want := "STATSV5#l2d#v2.1#" + testReasoningGeneration + "#b3BlbmFpX2NvbXBhdGlibGUAbnZpZGlhL0Nvc21vczMtTmFubw#action_relevant_reasoning_v3_temporal_front256"
	if got != want {
		t.Errorf("EmbeddedTeacherStatsPK = %q, want %q", got, want)
	}
}

func TestReasoningInventoryPK(t *testing.T) {
	got, err := ReasoningInventoryPK("kitscenes", "v2.1")
	if err != nil {
		t.Fatal(err)
	}
	want := "RINV5#kitscenes#v2.1"
	if got != want {
		t.Errorf("ReasoningInventoryPK = %q, want %q", got, want)
	}
}

func TestSceneLabelKeys(t *testing.T) {
	pk := SceneLabelPK("l2d", "action_relevant_reasoning_v3_temporal_front256", "lateral_response", "turn_left")
	wantPK := "LBL#l2d#action_relevant_reasoning_v3_temporal_front256#lateral_response#turn_left"
	if pk != wantPK {
		t.Errorf("SceneLabelPK = %q, want %q", pk, wantPK)
	}
	sk := SceneLabelSK("s00000123")
	if sk != "SCENE#s00000123" {
		t.Errorf("SceneLabelSK = %q, want %q", sk, "SCENE#s00000123")
	}
}

func TestSceneLabelVersionKey(t *testing.T) {
	got := SceneLabelVersionPK(
		"l2d",
		"v2.1",
		"action_relevant_reasoning_v3_temporal_front256",
		"lateral_response",
		"turn_left",
	)
	want := "LBLV2#l2d#v2.1#action_relevant_reasoning_v3_temporal_front256#lateral_response#turn_left"
	if got != want {
		t.Errorf("SceneLabelVersionPK = %q, want %q", got, want)
	}
}

func TestTeacherScopedSceneLabelVersionKey(t *testing.T) {
	got, err := SceneLabelTeacherVersionPK(
		"l2d",
		"v2.1",
		testReasoningGeneration,
		"b3BlbmFpX2NvbXBhdGlibGUAbnZpZGlhL0Nvc21vczMtTmFubw",
		"action_relevant_reasoning_v3_temporal_front256",
		"lateral_response",
		"turn_left",
	)
	if err != nil {
		t.Fatal(err)
	}
	want := "LBLV5#l2d#v2.1#" + testReasoningGeneration + "#b3BlbmFpX2NvbXBhdGlibGUAbnZpZGlhL0Nvc21vczMtTmFubw#action_relevant_reasoning_v3_temporal_front256#lateral_response#turn_left"
	if got != want {
		t.Errorf("SceneLabelTeacherVersionPK = %q, want %q", got, want)
	}
}

func TestReasoningSampleLookupKeys(t *testing.T) {
	pk, err := ReasoningSampleLookupPK(
		"kitscenes",
		"v2.1",
		testReasoningGeneration,
		"kitscenes-v1-scene-a-f000001",
	)
	if err != nil {
		t.Fatal(err)
	}
	if want := "RLOOKUP#kitscenes#v2.1#" + testReasoningGeneration +
		"#kitscenes-v1-scene-a-f000001"; pk != want {
		t.Fatalf("ReasoningSampleLookupPK = %q, want %q", pk, want)
	}
}

func TestReasoningKeysRejectInvalidComponents(t *testing.T) {
	badGenerations := []string{
		"",
		"generation#injected",
		strings.Repeat("A", 64),
		strings.Repeat("a", 63),
	}
	for _, generation := range badGenerations {
		if _, err := ReasoningSampleLookupPK(
			"kitscenes", "v2.1", generation, "sample-a",
		); err == nil {
			t.Errorf("invalid generation %q was accepted", generation)
		}
	}
	for _, component := range []string{"", "bad#value", "bad\x00value", " padded"} {
		if _, err := ReasoningSampleLookupPK(
			"kitscenes",
			"v2.1",
			testReasoningGeneration,
			component,
		); err == nil {
			t.Errorf("invalid key component %q was accepted", component)
		}
	}
	if _, err := ReasoningInventoryPK("kitscenes#other", "v2.1"); err == nil {
		t.Fatal("delimiter injection in inventory key was accepted")
	}
	long := strings.Repeat("x", 512)
	if _, err := SceneLabelTeacherVersionPK(
		long,
		long,
		testReasoningGeneration,
		long,
		long,
		long,
		long,
	); err == nil {
		t.Fatal("oversized DynamoDB partition key was accepted")
	}
}

// TestSceneLabelPK_DistinctPerField guards the invariant that different
// (field,value) pairs never collide into one partition (which would return
// scenes for the wrong label on search).
func TestSceneLabelPK_DistinctPerField(t *testing.T) {
	a := SceneLabelPK("l2d", "pv", "hazard_event", "collision_risk")
	b := SceneLabelPK("l2d", "pv", "cause", "collision_risk")
	if a == b {
		t.Errorf("distinct fields collapsed to same pk: %q", a)
	}
	c := SceneLabelPK("nvidia_av", "pv", "hazard_event", "collision_risk")
	if a == c {
		t.Errorf("distinct datasets collapsed to same pk: %q", a)
	}
}

func TestOverlayAndGeoKeys(t *testing.T) {
	modelID := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	if got := ShardModelPK("l2d", "v2.1", "train-000001.tar"); got != "SHARD#l2d#v2.1#train-000001.tar" {
		t.Errorf("ShardModelPK = %q", got)
	}
	if got := ModelSK(modelID); got != "MODEL#"+modelID {
		t.Errorf("ModelSK = %q", got)
	}
	if got := OverlaySetPK(modelID, "l2d", "v2.1"); got != "OVLSET#"+modelID+"#l2d#v2.1" {
		t.Errorf("OverlaySetPK = %q", got)
	}
	if got := GeoPK("l2d", "v2.1"); got != "GEO#l2d#v2.1" {
		t.Errorf("GeoPK = %q", got)
	}
}
