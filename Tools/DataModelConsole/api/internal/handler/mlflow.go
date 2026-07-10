package handler

import (
	"log/slog"
	"net/http"

	"github.com/go-chi/chi/v5"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

// MLflowHandler exposes the read-only MLflow proxy endpoints.
type MLflowHandler struct {
	svc *service.MLflowService
}

// NewMLflowHandler builds the MLflow proxy handler.
func NewMLflowHandler(svc *service.MLflowService) *MLflowHandler {
	return &MLflowHandler{svc: svc}
}

// Experiments handles GET /api/v1/mlflow/experiments.
func (h *MLflowHandler) Experiments(w http.ResponseWriter, r *http.Request) {
	res, err := h.svc.SearchExperiments(r.Context(),
		r.URL.Query().Get("max_results"), r.URL.Query().Get("page_token"))
	h.relay(w, res, err, "mlflow experiments search")
}

// Runs handles GET /api/v1/mlflow/experiments/{id}/runs.
func (h *MLflowHandler) Runs(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	res, err := h.svc.SearchRuns(r.Context(), id,
		r.URL.Query().Get("max_results"), r.URL.Query().Get("page_token"))
	h.relay(w, res, err, "mlflow runs search")
}

// Run handles GET /api/v1/mlflow/runs/{id}.
func (h *MLflowHandler) Run(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	res, err := h.svc.GetRun(r.Context(), id)
	h.relay(w, res, err, "mlflow run get")
}

// Models handles GET /api/v1/mlflow/models.
func (h *MLflowHandler) Models(w http.ResponseWriter, r *http.Request) {
	res, err := h.svc.SearchRegisteredModels(r.Context(),
		r.URL.Query().Get("max_results"), r.URL.Query().Get("page_token"))
	h.relay(w, res, err, "mlflow registered models search")
}

// relay forwards the upstream JSON response, or a 502 on transport failure.
func (h *MLflowHandler) relay(w http.ResponseWriter, res *service.UpstreamResult, err error, op string) {
	if err != nil {
		slog.Error(op, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "mlflow unreachable")
		return
	}
	writeRawJSON(w, res.Status, res.Body)
}
