package handler

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

func TestParseODDQuery(t *testing.T) {
	query, err := parseODDQuery(`{"op":"or","groups":[{"op":"and","conditions":[` +
		`{"key":"odd.road.context","values":["urban","rural"],"not":true,` +
		`"min_duration_s":2,"min_confidence":0.8,"sources":["vlm"]}]}]}`)
	if err != nil {
		t.Fatal(err)
	}
	condition := query.Groups[0].Conditions[0]
	if query.Op != "or" || !condition.Not || len(condition.Values) != 2 ||
		condition.MinDurationS != 2 || condition.MinConfidence != 0.8 ||
		condition.Sources[0] != "vlm" {
		t.Fatalf("query = %+v", query)
	}
	if query, err := parseODDQuery(""); err != nil || query.Op != "and" || query.Groups == nil {
		t.Fatalf("empty q = %+v, %v; want every scene", query, err)
	}
	for name, raw := range map[string]string{
		"not JSON":       `odd.road.context=urban`,
		"unknown field":  `{"op":"and","groups":[],"limit":5}`,
		"two objects":    `{"op":"and","groups":[]}{"op":"and","groups":[]}`,
		"invalid query":  `{"op":"and","groups":[{"op":"and","conditions":[]}]}`,
		"too long":       `{"op":"and","groups":[],"x":"` + strings.Repeat("a", maxODDQueryBytes) + `"}`,
		"wrong type":     `{"op":"and","groups":{}}`,
		"invalid source": `{"op":"and","groups":[{"op":"and","conditions":[{"key":"odd.road.context","values":["urban"],"sources":["../x"]}]}]}`,
	} {
		var invalid service.ErrInvalidODDQuery
		if _, err := parseODDQuery(raw); !errors.As(err, &invalid) {
			t.Errorf("%s: err = %v, want ErrInvalidODDQuery", name, err)
		}
	}
}

func TestODDHandlersRejectInvalidRequestsBeforeS3(t *testing.T) {
	handler := NewODDHandler(&service.S3Service{})
	badQuery := url.QueryEscape(`{"op":"xor","groups":[]}`)
	for target, want := range map[string]int{
		"/api/v1/odd/scenes?dataset=l2d&version=v3.7":                     http.StatusNotFound,
		"/api/v1/odd/scenes?dataset=kitscenes":                            http.StatusBadRequest,
		"/api/v1/odd/scenes?dataset=kitscenes&version=latest":             http.StatusBadRequest,
		"/api/v1/odd/scenes?dataset=kitscenes&version=v3.7&q=" + badQuery: http.StatusBadRequest,
	} {
		recorder := httptest.NewRecorder()
		handler.Search(recorder, httptest.NewRequest(http.MethodGet, target, nil))
		if recorder.Code != want {
			t.Errorf("%s status = %d, want %d", target, recorder.Code, want)
		}
	}
	for uid, want := range map[string]int{
		"UID-A":    http.StatusBadRequest,
		"..%2Fx":   http.StatusBadRequest,
		"uid-a":    http.StatusBadRequest, // not hex
		"0123-abc": http.StatusBadRequest, // valid uid, but no version
	} {
		request := httptest.NewRequest(http.MethodGet, "/api/v1/odd/scenes/x?dataset=kitscenes", nil)
		routeContext := chi.NewRouteContext()
		routeContext.URLParams.Add("scene_uid", uid)
		request = request.WithContext(context.WithValue(request.Context(), chi.RouteCtxKey, routeContext))
		recorder := httptest.NewRecorder()
		handler.Scene(recorder, request)
		if recorder.Code != want {
			t.Errorf("scene %s status = %d, want %d", uid, recorder.Code, want)
		}
	}
}
