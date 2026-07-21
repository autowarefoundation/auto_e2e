package handler

import (
	"log/slog"
	"net/http"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

// StatsHandler serves GET /api/v1/stats for the dashboard KPI cards.
type StatsHandler struct {
	s3     *service.S3Service
	mlflow *service.MLflowService
}

// NewStatsHandler builds the stats handler.
func NewStatsHandler(s3 *service.S3Service, mlflow *service.MLflowService) *StatsHandler {
	return &StatsHandler{s3: s3, mlflow: mlflow}
}

// Get aggregates dashboard stats. Each source degrades independently: an
// unreachable MLflow (or a failing S3 sub-query) zeroes its own fields but
// the endpoint still returns 200 so the dashboard renders partial data.
func (h *StatsHandler) Get(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	resp := model.StatsResponse{}

	if n, err := h.s3.TotalSamples(ctx); err != nil {
		slog.Warn("stats: total samples unavailable", "error", err)
	} else {
		resp.TotalSamples = n
	}

	if n, err := h.s3.CountReasoningLabels(ctx); err != nil {
		slog.Warn("stats: reasoning label count unavailable", "error", err)
	} else {
		resp.ReasoningLabels = n
	}

	if runs, ade, err := h.mlflow.RunStats(ctx); err != nil {
		slog.Warn("stats: mlflow unavailable, degrading", "error", err)
	} else {
		resp.MLflowRuns = runs
		resp.LatestADE = ade
		resp.MLflowAvailable = true
	}

	writeJSON(w, http.StatusOK, resp)
}
