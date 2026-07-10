package handler

import (
	"context"
	"net/http"
	"time"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

// HealthHandler serves /healthz and /readyz.
type HealthHandler struct {
	s3 *service.S3Service
}

// NewHealthHandler builds the health handler.
func NewHealthHandler(s3 *service.S3Service) *HealthHandler {
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
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	checks := map[string]string{}
	status := http.StatusOK
	if err := h.s3.Ping(ctx); err != nil {
		checks["s3"] = "unreachable: " + err.Error()
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
