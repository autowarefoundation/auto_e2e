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

// Executions handles GET /api/v1/flyte/executions, normalizing the nested
// Flyte Admin {"executions":[{id,closure,spec}]} shape into the flat list the
// frontend consumes (a raw pass-through rendered blank/crashed).
func (h *FlyteHandler) Executions(w http.ResponseWriter, r *http.Request) {
	limit := r.URL.Query().Get("limit")
	if limit == "" {
		limit = "25"
	}
	res, err := h.svc.ListExecutions(r.Context(), limit, r.URL.Query().Get("token"))
	if err != nil {
		slog.Error("flyte executions list", "error", err)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "flyte admin unreachable")
		return
	}
	if res.Status != http.StatusOK {
		writeRawJSON(w, res.Status, res.Body)
		return
	}
	out, nerr := model.NormalizeFlyteExecutions(res.Body)
	if nerr != nil {
		slog.Error("normalize flyte executions", "error", nerr)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "unexpected flyte response")
		return
	}
	writeJSON(w, http.StatusOK, out)
}

// Execution handles GET /api/v1/flyte/executions/{id}.
func (h *FlyteHandler) Execution(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	// The id is interpolated into the upstream Flyte Admin path; reject
	// anything that could escape the project/domain scope (path injection).
	if !validFlyteExecutionID(id) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid execution id")
		return
	}
	res, err := h.svc.GetExecution(r.Context(), id)
	if err != nil {
		slog.Error("flyte execution get", "error", err)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "flyte admin unreachable")
		return
	}
	if res.Status != http.StatusOK {
		writeRawJSON(w, res.Status, res.Body)
		return
	}
	// The get-by-id endpoint returns an unwrapped Execution ({id,closure,spec}),
	// so normalize it to the flat shape too (the list handler already does).
	out, nerr := model.NormalizeFlyteExecution(res.Body)
	if nerr != nil {
		slog.Error("normalize flyte execution", "error", nerr)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "unexpected flyte response")
		return
	}
	writeJSON(w, http.StatusOK, out)
}

// validFlyteExecutionID accepts only Flyte-generated execution names:
// non-empty lowercase alphanumerics and hyphens. This excludes '/', '\\' and
// ".." by construction, so the id cannot traverse the upstream URL path.
func validFlyteExecutionID(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		if (c < 'a' || c > 'z') && (c < '0' || c > '9') && c != '-' {
			return false
		}
	}
	return true
}
