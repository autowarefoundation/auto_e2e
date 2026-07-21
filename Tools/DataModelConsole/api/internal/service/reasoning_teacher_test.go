package service

import (
	"strings"
	"testing"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/store"
)

func TestReasoningTeacherIdentityRoundTrip(t *testing.T) {
	label := store.ReasoningLabel{
		TeacherProvider: "openai_compatible",
		TeacherModel:    "nvidia/Cosmos3-Nano",
	}
	id := reasoningTeacher(label)
	const want = "b3BlbmFpX2NvbXBhdGlibGUAbnZpZGlhL0Nvc21vczMtTmFubw"
	if id != want {
		t.Fatalf("reasoningTeacher = %q, want %q", id, want)
	}
	provider, modelName, ok := parseReasoningTeacherID(id)
	if !ok || provider != label.TeacherProvider || modelName != label.TeacherModel {
		t.Fatalf(
			"parseReasoningTeacherID = (%q, %q, %v)",
			provider, modelName, ok,
		)
	}
}

func TestReasoningTeacherIdentityIsInjective(t *testing.T) {
	labels := []store.ReasoningLabel{
		{TeacherProvider: "teacher", TeacherModel: "model-a"},
		{TeacherProvider: "teacher", TeacherModel: "model-b"},
		{TeacherProvider: "teacher-a", TeacherModel: "model"},
		{TeacherProvider: "teacher", TeacherModel: "a-model"},
	}
	seen := map[string]struct{}{}
	for _, label := range labels {
		id := reasoningTeacher(label)
		if _, exists := seen[id]; exists {
			t.Fatalf("teacher identity collision for %+v", label)
		}
		seen[id] = struct{}{}
	}
}

func TestReasoningTeacherMatchesOnlyCanonicalIdentity(t *testing.T) {
	label := store.ReasoningLabel{
		TeacherProvider: "provider",
		TeacherModel:    "model",
	}
	if !reasoningTeacherMatches(label, reasoningTeacher(label)) {
		t.Fatal("canonical teacher identity did not match")
	}
	if reasoningTeacherMatches(label, label.TeacherProvider) {
		t.Fatal("provider-only teacher selector matched")
	}
	if reasoningTeacherMatches(label, label.TeacherModel) {
		t.Fatal("model-only teacher selector matched")
	}
}

func TestParseReasoningTeacherIDRejectsAmbiguousValues(t *testing.T) {
	for _, teacher := range []string{
		"",
		"not-base64!",
		"cHJvdmlkZXI",
		"AA",
		"cHJvdmlkZXIAbW9kZWwAZXh0cmE",
	} {
		if provider, modelName, ok := parseReasoningTeacherID(teacher); ok {
			t.Errorf(
				"parseReasoningTeacherID(%q) = (%q, %q, true)",
				teacher, provider, modelName,
			)
		}
	}
}

func TestValidReasoningTeacherIDAppliesDynamoKeyLimits(t *testing.T) {
	for _, label := range []store.ReasoningLabel{
		{TeacherProvider: "provider#suffix", TeacherModel: "model"},
		{TeacherProvider: strings.Repeat("p", 513), TeacherModel: "model"},
	} {
		if teacher := reasoningTeacher(label); ValidReasoningTeacherID(teacher) {
			t.Errorf("ValidReasoningTeacherID(%q) = true", teacher)
		}
	}
}
