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
