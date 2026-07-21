package handler

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

func TestFlyteLimitRejectsInvalidValues(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"executions":[]}`))
	}))
	defer upstream.Close()

	h := NewFlyteHandler(service.NewFlyteService(
		upstream.URL, "development", "development",
	))
	values := []string{"", "0", "-1", "abc", "1.5", "10junk", "1001", "999999999999999999999"}
	for _, value := range values {
		t.Run(value, func(t *testing.T) {
			upstreamCalls.Store(0)
			target := "/executions?limit=" + url.QueryEscape(value)
			response := httptest.NewRecorder()

			h.Executions(response, httptest.NewRequest(http.MethodGet, target, nil))

			if response.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want %d: %s", response.Code, http.StatusBadRequest, response.Body.String())
			}
			if got := upstreamCalls.Load(); got != 0 {
				t.Fatalf("upstream calls = %d, want 0", got)
			}
		})
	}
}

func TestFlyteLimitForwardsSafeValues(t *testing.T) {
	var gotLimits []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotLimits = append(gotLimits, r.URL.Query().Get("limit"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"executions":[]}`))
	}))
	defer upstream.Close()

	h := NewFlyteHandler(service.NewFlyteService(
		upstream.URL, "development", "development",
	))
	for _, target := range []string{
		"/executions",
		"/executions?limit=1000",
	} {
		response := httptest.NewRecorder()
		h.Executions(response, httptest.NewRequest(http.MethodGet, target, nil))
		if response.Code != http.StatusOK {
			t.Fatalf("%s status = %d, want %d: %s", target, response.Code, http.StatusOK, response.Body.String())
		}
	}

	want := []string{"25", "1000"}
	if len(gotLimits) != len(want) {
		t.Fatalf("upstream limits = %v, want %v", gotLimits, want)
	}
	for i := range want {
		if gotLimits[i] != want[i] {
			t.Errorf("upstream limit[%d] = %q, want %q", i, gotLimits[i], want[i])
		}
	}
}
