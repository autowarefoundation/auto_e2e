package handler

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

func TestParseODDConditions(t *testing.T) {
	conditions, ok := parseODDConditions([]string{
		"odd.road.context=urban", "event.ego.maneuver=turn_left",
	})
	if !ok || len(conditions) != 2 ||
		conditions[1].Key != "event.ego.maneuver" ||
		conditions[1].Value != "turn_left" {
		t.Fatalf("conditions = %+v ok = %v", conditions, ok)
	}
	if conditions, ok := parseODDConditions(nil); !ok || len(conditions) != 0 {
		t.Fatalf("no terms = %+v ok = %v, want empty search", conditions, ok)
	}
	for _, term := range []string{
		"odd.road.context",
		"context=urban",
		"odd.road.context=",
		"odd.road.context=Urban",
		"odd..context=urban",
		"odd.road.context=urban=x",
		"odd.road/context=urban",
		"odd.road.context=" + strings.Repeat("a", 65),
	} {
		if _, ok := parseODDConditions([]string{term}); ok {
			t.Errorf("term %q accepted", term)
		}
	}
	tooMany := make([]string, maxODDConditions+1)
	for i := range tooMany {
		tooMany[i] = "odd.road.context=urban"
	}
	if _, ok := parseODDConditions(tooMany); ok {
		t.Error("more than maxODDConditions terms accepted")
	}
}

func TestODDSearchRejectsInvalidRequestBeforeS3(t *testing.T) {
	handler := NewODDHandler(&service.S3Service{})
	for target, want := range map[string]int{
		"/api/v1/odd/scenes?dataset=l2d":                                 http.StatusNotFound,
		"/api/v1/odd/scenes?dataset=kitscenes&version=latest":            http.StatusBadRequest,
		"/api/v1/odd/scenes?dataset=kitscenes&where=odd.road.context%3D": http.StatusBadRequest,
	} {
		recorder := httptest.NewRecorder()
		handler.Search(recorder, httptest.NewRequest(http.MethodGet, target, nil))
		if recorder.Code != want {
			t.Errorf("%s status = %d, want %d", target, recorder.Code, want)
		}
	}
}
