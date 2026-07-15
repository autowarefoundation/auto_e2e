package handler

import (
	"net/http"
	"net/http/httptest"
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
			name: "compute stats",
			request: httptest.NewRequest(
				http.MethodPost,
				"/api/v1/reasoning-labels/compute-stats?dataset=unknown&prompt_version=pv",
				nil,
			),
			handle: reasoning.ComputeStats,
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
