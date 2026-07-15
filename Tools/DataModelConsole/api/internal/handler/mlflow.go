package handler

import (
	"log/slog"
	"net/http"
	"strconv"
	"strings"

	"github.com/go-chi/chi/v5"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

const maxMLflowResults = 1000

// MLflowHandler exposes the read-only MLflow proxy endpoints.
type MLflowHandler struct {
	svc *service.MLflowService
}

// NewMLflowHandler builds the MLflow proxy handler.
func NewMLflowHandler(svc *service.MLflowService) *MLflowHandler {
	return &MLflowHandler{svc: svc}
}

// Experiments handles GET /api/v1/mlflow/experiments. The raw MLflow response
// is {"experiments":[...]}; the frontend consumes a flat array, so normalize.
func (h *MLflowHandler) Experiments(w http.ResponseWriter, r *http.Request) {
	maxResults, ok := parseMLflowMaxResults(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "max_results must be an integer between 1 and 1000")
		return
	}
	res, err := h.svc.SearchExperiments(r.Context(),
		maxResults, r.URL.Query().Get("page_token"))
	if err != nil {
		slog.Error("mlflow experiments search", "error", err)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "mlflow unreachable")
		return
	}
	if res.Status != http.StatusOK {
		writeRawJSON(w, res.Status, res.Body)
		return
	}
	out, nerr := model.NormalizeMLflowExperimentsPage(res.Body)
	if nerr != nil {
		slog.Error("normalize mlflow experiments", "error", nerr)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "unexpected mlflow response")
		return
	}
	writeJSON(w, http.StatusOK, out)
}

// Runs handles GET /api/v1/mlflow/experiments/{id}/runs, normalizing the nested
// {"runs":[{info,data}]} into the flat run list.
func (h *MLflowHandler) Runs(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if !safeUpstreamID(id) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid experiment id")
		return
	}
	maxResults, ok := parseMLflowMaxResults(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "max_results must be an integer between 1 and 1000")
		return
	}
	res, err := h.svc.SearchRuns(r.Context(), id,
		maxResults, r.URL.Query().Get("page_token"))
	if err != nil {
		slog.Error("mlflow runs search", "error", err)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "mlflow unreachable")
		return
	}
	if res.Status != http.StatusOK {
		writeRawJSON(w, res.Status, res.Body)
		return
	}
	out, nerr := model.NormalizeMLflowRunsPage(res.Body)
	if nerr != nil {
		slog.Error("normalize mlflow runs", "error", nerr)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "unexpected mlflow response")
		return
	}
	writeJSON(w, http.StatusOK, out)
}

// Run handles GET /api/v1/mlflow/runs/{id}, normalizing {"run":{info,data}}.
func (h *MLflowHandler) Run(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if !safeUpstreamID(id) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid run id")
		return
	}
	res, err := h.svc.GetRun(r.Context(), id)
	if err != nil {
		slog.Error("mlflow run get", "error", err)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "mlflow unreachable")
		return
	}
	if res.Status != http.StatusOK {
		writeRawJSON(w, res.Status, res.Body)
		return
	}
	out, nerr := model.NormalizeMLflowRun(res.Body)
	if nerr != nil {
		slog.Error("normalize mlflow run", "error", nerr)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "unexpected mlflow response")
		return
	}
	writeJSON(w, http.StatusOK, out)
}

// safeUpstreamID rejects ids that could act as path components upstream
// (defense in depth: MLflow ids currently travel as query/body values, but
// keep them from ever traversing a URL path).
func safeUpstreamID(s string) bool {
	return s != "" && !strings.ContainsAny(s, "/\\") && !strings.Contains(s, "..")
}

// Models handles GET /api/v1/mlflow/models, normalizing the
// {"registered_models":[...]} envelope into the flat model list.
func (h *MLflowHandler) Models(w http.ResponseWriter, r *http.Request) {
	maxResults, ok := parseMLflowMaxResults(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "max_results must be an integer between 1 and 1000")
		return
	}
	res, err := h.svc.SearchRegisteredModels(r.Context(),
		maxResults, r.URL.Query().Get("page_token"))
	if err != nil {
		slog.Error("mlflow registered models search", "error", err)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "mlflow unreachable")
		return
	}
	if res.Status != http.StatusOK {
		writeRawJSON(w, res.Status, res.Body)
		return
	}
	out, nerr := model.NormalizeMLflowModelsPage(res.Body)
	if nerr != nil {
		slog.Error("normalize mlflow models", "error", nerr)
		writeError(w, http.StatusBadGateway, model.CodeUpstream, "unexpected mlflow response")
		return
	}
	writeJSON(w, http.StatusOK, out)
}

func parseMLflowMaxResults(r *http.Request) (int, bool) {
	values, present := r.URL.Query()["max_results"]
	if !present {
		return 0, true
	}
	if len(values) != 1 {
		return 0, false
	}
	value, err := strconv.Atoi(values[0])
	if err != nil || value <= 0 || value > maxMLflowResults {
		return 0, false
	}
	return value, true
}
