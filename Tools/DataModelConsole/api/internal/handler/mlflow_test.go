package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/go-chi/chi/v5"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

func TestMLflowMaxResultsRejectsInvalidValues(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/2.0/mlflow/experiments/search":
			_, _ = w.Write([]byte(`{"experiments":[]}`))
		case "/api/2.0/mlflow/runs/search":
			_, _ = w.Write([]byte(`{"runs":[]}`))
		case "/api/2.0/mlflow/registered-models/search":
			_, _ = w.Write([]byte(`{"registered_models":[]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()

	h := NewMLflowHandler(service.NewMLflowService(upstream.URL))
	router := chi.NewRouter()
	router.Get("/experiments", h.Experiments)
	router.Get("/experiments/{id}/runs", h.Runs)
	router.Get("/models", h.Models)

	endpoints := []string{"/experiments", "/experiments/1/runs", "/models"}
	values := []string{"", "0", "-1", "abc", "1.5", "10junk", "1001", "999999999999999999999"}
	for _, endpoint := range endpoints {
		for _, value := range values {
			t.Run(endpoint+"/"+value, func(t *testing.T) {
				upstreamCalls.Store(0)
				target := endpoint + "?max_results=" + url.QueryEscape(value)
				response := httptest.NewRecorder()

				router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, target, nil))

				if response.Code != http.StatusBadRequest {
					t.Fatalf("status = %d, want %d: %s", response.Code, http.StatusBadRequest, response.Body.String())
				}
				if got := upstreamCalls.Load(); got != 0 {
					t.Fatalf("upstream calls = %d, want 0", got)
				}
			})
		}
	}
}

func TestMLflowMaxResultsForwardsBoundedInteger(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/2.0/mlflow/experiments/search":
			if got := r.URL.Query().Get("max_results"); got != "1000" {
				t.Errorf("experiment max_results = %q, want 1000", got)
			}
			_, _ = w.Write([]byte(`{"experiments":[]}`))
		case "/api/2.0/mlflow/runs/search":
			var body struct {
				MaxResults int `json:"max_results"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode runs request: %v", err)
			}
			if body.MaxResults != 1000 {
				t.Errorf("run max_results = %d, want 1000", body.MaxResults)
			}
			_, _ = w.Write([]byte(`{"runs":[]}`))
		case "/api/2.0/mlflow/registered-models/search":
			if got := r.URL.Query().Get("max_results"); got != "1000" {
				t.Errorf("model max_results = %q, want 1000", got)
			}
			_, _ = w.Write([]byte(`{"registered_models":[]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()

	h := NewMLflowHandler(service.NewMLflowService(upstream.URL))
	router := chi.NewRouter()
	router.Get("/experiments", h.Experiments)
	router.Get("/experiments/{id}/runs", h.Runs)
	router.Get("/models", h.Models)

	for _, endpoint := range []string{"/experiments", "/experiments/1/runs", "/models"} {
		t.Run(endpoint, func(t *testing.T) {
			response := httptest.NewRecorder()
			target := endpoint + "?max_results=" + strconv.Itoa(1000)

			router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, target, nil))

			if response.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d: %s", response.Code, http.StatusOK, response.Body.String())
			}
		})
	}
}
