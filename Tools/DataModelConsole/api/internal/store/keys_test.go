package store

import "testing"

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
	got := EmbeddedTeacherStatsPK(
		"l2d",
		"v2.1",
		"b3BlbmFpX2NvbXBhdGlibGUAbnZpZGlhL0Nvc21vczMtTmFubw",
		"action_relevant_reasoning_v3_temporal_front256",
	)
	want := "STATSV3#l2d#v2.1#b3BlbmFpX2NvbXBhdGlibGUAbnZpZGlhL0Nvc21vczMtTmFubw#action_relevant_reasoning_v3_temporal_front256"
	if got != want {
		t.Errorf("EmbeddedTeacherStatsPK = %q, want %q", got, want)
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
	got := SceneLabelTeacherVersionPK(
		"l2d",
		"v2.1",
		"b3BlbmFpX2NvbXBhdGlibGUAbnZpZGlhL0Nvc21vczMtTmFubw",
		"action_relevant_reasoning_v3_temporal_front256",
		"lateral_response",
		"turn_left",
	)
	want := "LBLV3#l2d#v2.1#b3BlbmFpX2NvbXBhdGlibGUAbnZpZGlhL0Nvc21vczMtTmFubw#action_relevant_reasoning_v3_temporal_front256#lateral_response#turn_left"
	if got != want {
		t.Errorf("SceneLabelTeacherVersionPK = %q, want %q", got, want)
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
