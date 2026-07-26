package handler

import (
	"log/slog"
	"net/http"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

// ExperimentsHandler serves the joined experiment dashboard contract.
type ExperimentsHandler struct {
	svc *service.ExperimentService
}

// NewExperimentsHandler builds the cross-system experiment handler.
func NewExperimentsHandler(svc *service.ExperimentService) *ExperimentsHandler {
	return &ExperimentsHandler{svc: svc}
}

// List handles GET /api/v1/experiments.
func (h *ExperimentsHandler) List(w http.ResponseWriter, r *http.Request) {
	response, err := h.svc.List(r.Context())
	if err != nil {
		slog.Error("list joined experiments", "error", err)
		writeError(
			w,
			http.StatusBadGateway,
			model.CodeUpstream,
			"experiment sources are unavailable",
		)
		return
	}
	writeJSON(w, http.StatusOK, response)
}
