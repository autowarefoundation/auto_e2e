package handler

import (
	"context"
	"net/http"
	"sync"
	"time"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

const (
	readinessTimeout  = 5 * time.Second
	readinessCacheTTL = 3 * time.Second
)

type s3ReadinessChecker interface {
	Ping(context.Context) error
}

// HealthHandler serves /healthz and /readyz.
type HealthHandler struct {
	s3 s3ReadinessChecker

	readinessMu     sync.Mutex
	readinessCache  cachedReadiness
	readinessFlight *readinessCall
}

type cachedReadiness struct {
	err       error
	expiresAt time.Time
	valid     bool
}

type readinessCall struct {
	done chan struct{}
	err  error
}

// NewHealthHandler builds the health handler.
func NewHealthHandler(s3 s3ReadinessChecker) *HealthHandler {
	return &HealthHandler{s3: s3}
}

// Healthz always returns 200 (liveness).
func (h *HealthHandler) Healthz(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, model.HealthResponse{Status: "ok"})
}

// Readyz checks S3 reachability (readiness). MLflow/Flyte are proxied
// upstreams whose outage should not take the whole console out, so they are
// not gating here.
func (h *HealthHandler) Readyz(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), readinessTimeout)
	defer cancel()

	checks := map[string]string{}
	status := http.StatusOK
	if err := h.checkS3(ctx); err != nil {
		// Do not expose SDK, endpoint, account, or credential details from the
		// upstream error through an unauthenticated health endpoint.
		checks["s3"] = "unreachable"
		status = http.StatusServiceUnavailable
	} else {
		checks["s3"] = "ok"
	}

	body := model.HealthResponse{Status: "ok", Checks: checks}
	if status != http.StatusOK {
		body.Status = "unavailable"
	}
	writeJSON(w, status, body)
}

func (h *HealthHandler) checkS3(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	h.readinessMu.Lock()
	if h.readinessCache.valid &&
		time.Now().Before(h.readinessCache.expiresAt) {
		err := h.readinessCache.err
		h.readinessMu.Unlock()
		return err
	}
	if flight := h.readinessFlight; flight != nil {
		h.readinessMu.Unlock()
		select {
		case <-flight.done:
			if err := ctx.Err(); err != nil {
				return err
			}
			return flight.err
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	flight := &readinessCall{done: make(chan struct{})}
	h.readinessFlight = flight
	h.readinessMu.Unlock()

	checkErr := h.s3.Ping(ctx)

	h.readinessMu.Lock()
	flight.err = checkErr
	h.readinessCache = cachedReadiness{
		err:       checkErr,
		expiresAt: time.Now().Add(readinessCacheTTL),
		valid:     true,
	}
	h.readinessFlight = nil
	close(flight.done)
	h.readinessMu.Unlock()

	return checkErr
}
