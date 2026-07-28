package handler

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

func TestDecodeODDSearchRequestAcceptsNestedQuery(t *testing.T) {
	body := `{
		"query": {
			"logic": "and",
			"predicates": [{
				"key": "odd.road.context",
				"operator": "in",
				"values": ["urban", "suburban"],
				"statuses": ["valid"],
				"sources": ["map_route"],
				"minimum_confidence": 0.8,
				"minimum_duration_ns": 1000000000,
				"camera_id": "",
				"actor_track_uid": ""
			}],
			"groups": [{
				"logic": "or",
				"predicates": [{
					"key": "event.ego.maneuver",
					"operator": "equals",
					"values": ["turn_left"],
					"statuses": [],
					"sources": [],
					"minimum_confidence": 0,
					"minimum_duration_ns": 0,
					"camera_id": "",
					"actor_track_uid": ""
				}],
				"groups": []
			}]
		},
		"sort": "confidence",
		"descending": true,
		"limit": 25,
		"offset": 0
	}`
	request := httptest.NewRequest(http.MethodPost, "/api/v1/odd/scenes/search", strings.NewReader(body))
	response := httptest.NewRecorder()

	decoded, err := decodeODDSearchRequest(response, request)

	if err != nil {
		t.Fatal(err)
	}
	if decoded.Query.Logic != "and" ||
		len(decoded.Query.Groups) != 1 ||
		decoded.Query.Groups[0].Logic != "or" ||
		decoded.Query.Predicates[0].MinimumConfidence != 0.8 ||
		decoded.Sort != "confidence" ||
		!decoded.Descending ||
		decoded.Limit != 25 {
		t.Fatalf("decoded request = %+v", decoded)
	}
}

func TestDecodeODDSearchRequestRejectsNonCanonicalBodies(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "unknown field",
			body: `{"query":{"logic":"and","predicates":[],"groups":[]},"unknown":true}`,
		},
		{
			name: "trailing object",
			body: `{"query":{"logic":"and","predicates":[],"groups":[]}} {}`,
		},
		{
			name: "oversized body",
			body: `{"query":{"logic":"and","predicates":[],"groups":[]},"sort":"` +
				strings.Repeat("a", maxODDSearchBody) + `"}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(
				http.MethodPost,
				"/api/v1/odd/scenes/search",
				strings.NewReader(test.body),
			)
			response := httptest.NewRecorder()

			if _, err := decodeODDSearchRequest(response, request); err == nil {
				t.Fatal("non-canonical ODD request was accepted")
			}
		})
	}
}

func TestWriteODDErrorMapsInvalidQueryToBadRequest(t *testing.T) {
	response := httptest.NewRecorder()

	writeODDError(
		response,
		errors.Join(service.ErrODDInvalidQuery, errors.New("bad operator")),
	)

	if response.Code != http.StatusBadRequest ||
		!strings.Contains(response.Body.String(), "invalid ODD search query") {
		t.Fatalf(
			"status = %d, body = %s",
			response.Code,
			response.Body.String(),
		)
	}
}
