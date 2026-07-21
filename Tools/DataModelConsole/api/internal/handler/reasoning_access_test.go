package handler

import (
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

func TestReasoningHandlersRejectUnknownDatasetsBeforeStorageAccess(t *testing.T) {
	s3 := &service.S3Service{}
	reasoning := NewReasoningHandler(s3)
	scenes := NewScenesHandler(s3)
	tests := []struct {
		name    string
		request *http.Request
		handle  http.HandlerFunc
	}{
		{
			name: "prompt versions",
			request: httptest.NewRequest(
				http.MethodGet,
				"/api/v1/reasoning-labels/prompt-versions?dataset=unknown",
				nil,
			),
			handle: reasoning.PromptVersions,
		},
		{
			name: "stats detail",
			request: httptest.NewRequest(
				http.MethodGet,
				"/api/v1/reasoning-labels/stats-detail?dataset=unknown&prompt_version=pv",
				nil,
			),
			handle: reasoning.StatsDetail,
		},
		{
			name: "label",
			request: requestWithDatasetRoute(
				"/api/v1/reasoning-labels/unknown/sample",
				"dataset", "unknown",
				"sample_id", "sample",
			),
			handle: reasoning.GetLabel,
		},
		{
			name: "scene search",
			request: httptest.NewRequest(
				http.MethodGet,
				"/api/v1/scenes/search?dataset=unknown&prompt_version=pv",
				nil,
			),
			handle: scenes.Search,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			test.handle(response, test.request)
			if response.Code != http.StatusNotFound {
				t.Fatalf(
					"status = %d, want %d: %s",
					response.Code,
					http.StatusNotFound,
					response.Body.String(),
				)
			}
		})
	}
}

func TestReasoningHandlersRejectInvalidDynamoKeyComponents(t *testing.T) {
	s3 := &service.S3Service{}
	reasoning := NewReasoningHandler(s3)
	scenes := NewScenesHandler(s3)
	teacher := base64.RawURLEncoding.EncodeToString(
		[]byte("provider\x00model"),
	)
	tests := []struct {
		name    string
		request *http.Request
		handle  http.HandlerFunc
	}{
		{
			name: "prompt version delimiter",
			request: httptest.NewRequest(
				http.MethodGet,
				"/api/v1/reasoning-labels/stats-detail"+
					"?dataset=kitscenes&prompt_version=%23",
				nil,
			),
			handle: reasoning.StatsDetail,
		},
		{
			name: "sample delimiter",
			request: requestWithDatasetRoute(
				"/api/v1/reasoning-labels/kitscenes/%23",
				"dataset", "kitscenes",
				"sample_id", "#",
			),
			handle: reasoning.GetLabel,
		},
		{
			name: "scene value delimiter",
			request: httptest.NewRequest(
				http.MethodGet,
				"/api/v1/scenes/search?dataset=kitscenes"+
					"&teacher="+teacher+
					"&prompt_version=prompt-v1"+
					"&field=relation_to_ego&value=%23",
				nil,
			),
			handle: scenes.Search,
		},
		{
			name: "leading whitespace",
			request: httptest.NewRequest(
				http.MethodGet,
				"/api/v1/reasoning-labels/prompt-versions"+
					"?dataset=%20kitscenes",
				nil,
			),
			handle: reasoning.PromptVersions,
		},
		{
			name: "oversized component",
			request: requestWithDatasetRoute(
				"/api/v1/reasoning-labels/kitscenes/sample",
				"dataset", "kitscenes",
				"sample_id", strings.Repeat("a", 513),
			),
			handle: reasoning.GetLabel,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			test.handle(response, test.request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf(
					"status = %d, want %d: %s",
					response.Code,
					http.StatusBadRequest,
					response.Body.String(),
				)
			}
		})
	}
}
