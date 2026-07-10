package handler

import (
	"log/slog"
	"net/http"

	"github.com/go-chi/chi/v5"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

// FlyteHandler exposes the read-only Flyte Admin proxy endpoints.
type FlyteHandler struct {
	svc *service.FlyteService
}

// NewFlyteHandler builds the Flyte proxy handler.
func NewFlyteHandler(svc *service.FlyteService) *FlyteHandler {
	return &FlyteHandler{svc: svc}
}

// Executions handles GET /api/v1/flyte/executions.
func (h *FlyteHandler) Executions(w http.ResponseWriter, r *http.Request) {
	limit := r.URL.Query().Get("limit")
	if limit == "" {
		limit = "25"
	}
	res, err := h.svc.ListExecutions(r.Context(), limit, r.URL.Query().Get("token"))
	h.relay(w, res, err, "flyte executions list")
}

// Execution handles GET /api/v1/flyte/executions/{id}.
func (h *FlyteHandler) Execution(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	res, err := h.svc.GetExecution(r.Context(), id)
	h.relay(w, res, err, "flyte execution get")
}

// relay forwards the upstream JSON response, or a 502 on transport failure.
func (h *FlyteHandler) relay(w http.ResponseWriter, res *service.UpstreamResult, err error, op string) {
	if err != nil {
		slog.Error(op, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "flyte admin unreachable")
		return
	}
	writeRawJSON(w, res.Status, res.Body)
}
