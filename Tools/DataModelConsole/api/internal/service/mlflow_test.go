package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

type repeatByteReader struct{}

func (repeatByteReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = 'x'
	}
	return len(p), nil
}

func TestRunStatsSelectsLatestFiniteADE(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/2.0/mlflow/experiments/search":
			_, _ = w.Write([]byte(`{"experiments":[{"experiment_id":"1"}]}`))
		case "/api/2.0/mlflow/runs/search":
			_, _ = w.Write([]byte(`{"runs":[
				{"info":{"run_id":"newest"},"data":{"metrics":[{"key":"eval/ade","value":"NaN"}]}},
				{"info":{"run_id":"next"},"data":{"metrics":[{"key":"eval/ade","value":"Infinity"}]}},
				{"info":{"run_id":"broken"},"data":{"metrics":[{"key":"eval/ade","value":"not-a-number"}]}},
				{"info":{"run_id":"finite"},"data":{"metrics":[{"key":"eval/ade","value":1.25}]}}
			]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()

	totalRuns, latestADE, err := NewMLflowService(upstream.URL).RunStats(t.Context())
	if err != nil {
		t.Fatalf("RunStats: %v", err)
	}
	if totalRuns != 4 {
		t.Errorf("total runs = %d, want 4", totalRuns)
	}
	if latestADE == nil || *latestADE != 1.25 {
		t.Fatalf("latest ADE = %v, want 1.25", latestADE)
	}
}

func TestProxyResponseBodyLimit(t *testing.T) {
	tests := []struct {
		name    string
		size    int64
		wantErr bool
	}{
		{name: "at limit", size: MaxProxyResponseBytes},
		{name: "one byte over limit", size: MaxProxyResponseBytes + 1, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &http.Client{
				Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusOK,
						Header:     make(http.Header),
						Body: io.NopCloser(io.LimitReader(
							repeatByteReader{},
							tt.size,
						)),
						Request: r,
					}, nil
				}),
			}

			result, err := httpGetJSON(
				t.Context(),
				client,
				"http://upstream.test",
				"/data",
				nil,
			)
			if tt.wantErr {
				if !errors.Is(err, ErrProxyResponseTooLarge) {
					t.Fatalf("error = %v, want ErrProxyResponseTooLarge", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("httpGetJSON: %v", err)
			}
			if got := int64(len(result.Body)); got != tt.size {
				t.Fatalf("body size = %d, want %d", got, tt.size)
			}
		})
	}
}

func TestProxyRequestConcurrencyIsProcessWide(t *testing.T) {
	const extraRequests = 5
	requests := maxConcurrentProxyRequests + extraRequests
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() {
		releaseOnce.Do(func() { close(release) })
	}
	defer unblock()

	started := make(chan struct{}, requests)
	var active atomic.Int32
	var peak atomic.Int32
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		current := active.Add(1)
		for {
			oldPeak := peak.Load()
			if current <= oldPeak || peak.CompareAndSwap(oldPeak, current) {
				break
			}
		}
		started <- struct{}{}
		select {
		case <-release:
		case <-r.Context().Done():
			active.Add(-1)
			return nil, r.Context().Err()
		}
		active.Add(-1)
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(`{}`)),
			Request:    r,
		}, nil
	})
	clients := []*http.Client{
		{Transport: transport},
		{Transport: transport},
	}

	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	errs := make(chan error, requests)
	var wg sync.WaitGroup
	for i := 0; i < requests; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := httpGetJSON(
				ctx,
				clients[i%len(clients)],
				"http://upstream.test",
				"/data",
				nil,
			)
			errs <- err
		}(i)
	}

	for i := 0; i < maxConcurrentProxyRequests; i++ {
		select {
		case <-started:
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for proxy requests to start")
		}
	}
	select {
	case <-started:
		t.Fatalf("more than %d proxy requests started concurrently", maxConcurrentProxyRequests)
	case <-time.After(75 * time.Millisecond):
	}

	unblock()
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Errorf("proxy request: %v", err)
		}
	}
	if got := peak.Load(); got != int32(maxConcurrentProxyRequests) {
		t.Errorf("peak concurrency = %d, want %d", got, maxConcurrentProxyRequests)
	}
}

func TestRunStatsPaginatesAndDeduplicates(t *testing.T) {
	var experimentCalls atomic.Int32
	var runCalls atomic.Int32

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/2.0/mlflow/experiments/search":
			experimentCalls.Add(1)
			switch r.URL.Query().Get("page_token") {
			case "":
				_ = json.NewEncoder(w).Encode(map[string]any{
					"experiments": []map[string]string{
						{"experiment_id": "1"},
						{"experiment_id": "1"},
					},
					"next_page_token": "experiments-2",
				})
			case "experiments-2":
				_ = json.NewEncoder(w).Encode(map[string]any{
					"experiments": []map[string]string{
						{"experiment_id": "2"},
					},
				})
			default:
				http.Error(w, "unexpected experiment token", http.StatusBadRequest)
			}
		case "/api/2.0/mlflow/runs/search":
			runCalls.Add(1)
			var payload struct {
				ExperimentIDs []string `json:"experiment_ids"`
				PageToken     string   `json:"page_token"`
			}
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if strings.Join(payload.ExperimentIDs, ",") != "1,2" {
				http.Error(w, "experiment ids were not deduplicated", http.StatusBadRequest)
				return
			}

			page := 0
			if payload.PageToken != "" {
				var err error
				page, err = strconv.Atoi(strings.TrimPrefix(payload.PageToken, "runs-"))
				if err != nil {
					http.Error(w, "unexpected run token", http.StatusBadRequest)
					return
				}
			}

			runs := []map[string]any{}
			switch page {
			case 0:
				runs = append(runs, rawRunForTest("run-0", 1.25))
			case 1:
				runs = append(
					runs,
					rawRunForTest("run-0", nil),
					rawRunForTest("run-1", nil),
				)
			case 5:
				// An empty page with a continuation token is not terminal.
			default:
				runs = append(runs, rawRunForTest(fmt.Sprintf("run-%d", page), nil))
			}

			response := map[string]any{"runs": runs}
			if page < 11 {
				response["next_page_token"] = fmt.Sprintf("runs-%d", page+1)
			}
			_ = json.NewEncoder(w).Encode(response)
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()

	totalRuns, latestADE, err := NewMLflowService(upstream.URL).RunStats(t.Context())
	if err != nil {
		t.Fatalf("RunStats: %v", err)
	}
	if totalRuns != 11 {
		t.Errorf("total runs = %d, want 11 unique runs", totalRuns)
	}
	if latestADE == nil || *latestADE != 1.25 {
		t.Fatalf("latest ADE = %v, want 1.25", latestADE)
	}
	if got := experimentCalls.Load(); got != 2 {
		t.Errorf("experiment requests = %d, want 2", got)
	}
	if got := runCalls.Load(); got != 12 {
		t.Errorf("run requests = %d, want 12", got)
	}
}

func rawRunForTest(runID string, ade any) map[string]any {
	metrics := []map[string]any{}
	if ade != nil {
		metrics = append(metrics, map[string]any{
			"key":   "eval/ade",
			"value": ade,
		})
	}
	return map[string]any{
		"info": map[string]any{"run_id": runID},
		"data": map[string]any{"metrics": metrics},
	}
}

func TestRunStatsRejectsPaginationCycles(t *testing.T) {
	tests := []struct {
		name      string
		cyclePath string
		wantError string
	}{
		{
			name:      "experiments",
			cyclePath: "/api/2.0/mlflow/experiments/search",
			wantError: "experiments pagination token cycle",
		},
		{
			name:      "runs",
			cyclePath: "/api/2.0/mlflow/runs/search",
			wantError: "runs pagination token cycle",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch r.URL.Path {
				case "/api/2.0/mlflow/experiments/search":
					response := map[string]any{
						"experiments": []map[string]string{
							{"experiment_id": "1"},
						},
					}
					if tt.cyclePath == r.URL.Path {
						response["next_page_token"] = "same"
					}
					_ = json.NewEncoder(w).Encode(response)
				case "/api/2.0/mlflow/runs/search":
					response := map[string]any{
						"runs": []map[string]any{
							rawRunForTest("run-1", nil),
						},
					}
					if tt.cyclePath == r.URL.Path {
						response["next_page_token"] = "same"
					}
					_ = json.NewEncoder(w).Encode(response)
				default:
					http.NotFound(w, r)
				}
			}))
			defer upstream.Close()

			_, _, err := NewMLflowService(upstream.URL).RunStats(t.Context())
			if err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("error = %v, want %q", err, tt.wantError)
			}
		})
	}
}

func TestRunStatsCacheAndSingleflight(t *testing.T) {
	var experimentCalls atomic.Int32
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() {
		releaseOnce.Do(func() { close(release) })
	}
	defer unblock()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/2.0/mlflow/experiments/search" {
			http.NotFound(w, r)
			return
		}
		if experimentCalls.Add(1) == 1 {
			close(entered)
		}
		select {
		case <-release:
			_, _ = w.Write([]byte(`{"experiments":[]}`))
		case <-r.Context().Done():
		}
	}))
	defer upstream.Close()

	svc := NewMLflowService(upstream.URL)
	const callers = 24
	start := make(chan struct{})
	errs := make(chan error, callers)
	var ready sync.WaitGroup
	var done sync.WaitGroup
	ready.Add(callers)
	done.Add(callers)
	for i := 0; i < callers; i++ {
		go func() {
			defer done.Done()
			ready.Done()
			<-start
			_, _, err := svc.RunStats(t.Context())
			errs <- err
		}()
	}
	ready.Wait()
	close(start)

	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for RunStats upstream request")
	}
	time.Sleep(75 * time.Millisecond)
	if got := experimentCalls.Load(); got != 1 {
		t.Fatalf("experiment requests while in flight = %d, want 1", got)
	}

	unblock()
	done.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Errorf("RunStats: %v", err)
		}
	}

	if _, _, err := svc.RunStats(t.Context()); err != nil {
		t.Fatalf("cached RunStats: %v", err)
	}
	if got := experimentCalls.Load(); got != 1 {
		t.Errorf("experiment requests after cache hit = %d, want 1", got)
	}
}

func TestRunStatsSingleflightWaiterHonorsContext(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() {
		releaseOnce.Do(func() { close(release) })
	}
	defer unblock()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/2.0/mlflow/experiments/search" {
			http.NotFound(w, r)
			return
		}
		select {
		case entered <- struct{}{}:
		default:
		}
		select {
		case <-release:
			_, _ = w.Write([]byte(`{"experiments":[]}`))
		case <-r.Context().Done():
		}
	}))
	defer upstream.Close()

	svc := NewMLflowService(upstream.URL)
	leaderErr := make(chan error, 1)
	go func() {
		_, _, err := svc.RunStats(t.Context())
		leaderErr <- err
	}()

	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for leader request")
	}

	waiterCtx, cancel := context.WithTimeout(t.Context(), 40*time.Millisecond)
	defer cancel()
	_, _, err := svc.RunStats(waiterCtx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("waiter error = %v, want context deadline exceeded", err)
	}

	unblock()
	select {
	case err := <-leaderErr:
		if err != nil {
			t.Fatalf("leader RunStats: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for leader RunStats")
	}
}

func TestRunStatsPaginationHonorsContext(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/2.0/mlflow/experiments/search" {
			http.NotFound(w, r)
			return
		}
		if r.URL.Query().Get("page_token") == "" {
			_, _ = w.Write([]byte(
				`{"experiments":[{"experiment_id":"1"}],"next_page_token":"next"}`,
			))
			return
		}
		<-r.Context().Done()
	}))
	defer upstream.Close()

	ctx, cancel := context.WithTimeout(t.Context(), 40*time.Millisecond)
	defer cancel()
	_, _, err := NewMLflowService(upstream.URL).RunStats(ctx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("RunStats error = %v, want context deadline exceeded", err)
	}
}
