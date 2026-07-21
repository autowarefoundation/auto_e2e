package handler

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

type healthPingerFunc func(context.Context) error

func (f healthPingerFunc) Ping(ctx context.Context) error {
	return f(ctx)
}

func TestReadyzDoesNotExposeS3Error(t *testing.T) {
	var calls atomic.Int32
	h := NewHealthHandler(healthPingerFunc(func(context.Context) error {
		calls.Add(1)
		return errors.New("HeadBucket https://private.example.invalid: credential secret")
	}))

	for i := 0; i < 2; i++ {
		response := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodGet, "/readyz", nil)
		h.Readyz(response, request)

		if response.Code != http.StatusServiceUnavailable {
			t.Fatalf("status = %d, want %d", response.Code, http.StatusServiceUnavailable)
		}
		if strings.Contains(response.Body.String(), "private.example.invalid") ||
			strings.Contains(response.Body.String(), "credential secret") {
			t.Fatalf("response exposed upstream error: %s", response.Body.String())
		}
		var body model.HealthResponse
		if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
			t.Fatalf("decode response: %v", err)
		}
		if body.Status != "unavailable" || body.Checks["s3"] != "unreachable" {
			t.Fatalf("response = %+v, want sanitized unavailable status", body)
		}
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("S3 checks = %d, want one cached failure", got)
	}
}

func TestReadyzCachesAndSingleflightsS3Check(t *testing.T) {
	var calls atomic.Int32
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() {
		releaseOnce.Do(func() { close(release) })
	}
	defer unblock()

	h := NewHealthHandler(healthPingerFunc(func(ctx context.Context) error {
		if calls.Add(1) == 1 {
			close(entered)
		}
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}))

	const callers = 24
	start := make(chan struct{})
	statuses := make(chan int, callers)
	var ready sync.WaitGroup
	var done sync.WaitGroup
	ready.Add(callers)
	done.Add(callers)
	for i := 0; i < callers; i++ {
		go func() {
			defer done.Done()
			ready.Done()
			<-start
			response := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodGet, "/readyz", nil).
				WithContext(t.Context())
			h.Readyz(response, request)
			statuses <- response.Code
		}()
	}
	ready.Wait()
	close(start)

	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for S3 readiness check")
	}
	time.Sleep(75 * time.Millisecond)
	if got := calls.Load(); got != 1 {
		t.Fatalf("S3 checks while in flight = %d, want 1", got)
	}

	unblock()
	done.Wait()
	close(statuses)
	for status := range statuses {
		if status != http.StatusOK {
			t.Errorf("status = %d, want %d", status, http.StatusOK)
		}
	}

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	h.Readyz(response, request)
	if response.Code != http.StatusOK {
		t.Errorf("cached status = %d, want %d", response.Code, http.StatusOK)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("S3 checks after cache hit = %d, want 1", got)
	}
}
