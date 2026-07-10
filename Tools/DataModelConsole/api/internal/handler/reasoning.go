package handler

import (
	"errors"
	"log/slog"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

// ReasoningHandler serves the reasoning label cache endpoints.
type ReasoningHandler struct {
	s3 *service.S3Service
}

// NewReasoningHandler builds the reasoning labels handler.
func NewReasoningHandler(s3 *service.S3Service) *ReasoningHandler {
	return &ReasoningHandler{s3: s3}
}

// Stats handles GET /api/v1/reasoning-labels/stats — counts label objects
// per dataset/teacher/prompt_version partition.
func (h *ReasoningHandler) Stats(w http.ResponseWriter, r *http.Request) {
	entries, total, err := h.s3.ReasoningStats(r.Context())
	if err != nil {
		slog.Error("reasoning stats", "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to aggregate reasoning label stats")
		return
	}
	if entries == nil {
		entries = []model.ReasoningStatsEntry{}
	}
	writeJSON(w, http.StatusOK, model.ReasoningStatsResponse{Entries: entries, Total: total})
}

// GetLabel handles GET /api/v1/reasoning-labels/{dataset}/{sample_id}.
// Optional ?teacher= and ?prompt_version= narrow the cache partition; without
// them the first matching partition is returned.
func (h *ReasoningHandler) GetLabel(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	sampleID := chi.URLParam(r, "sample_id")
	if strings.ContainsAny(dataset, "/\\") || strings.ContainsAny(sampleID, "/\\") {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid dataset or sample_id")
		return
	}

	teacher := r.URL.Query().Get("teacher")
	promptVersion := r.URL.Query().Get("prompt_version")
	// These land in the S3 key template; reject path-traversal characters the
	// same way dataset/sample_id are validated above.
	if strings.ContainsAny(teacher, "/\\") || strings.Contains(teacher, "..") ||
		strings.ContainsAny(promptVersion, "/\\") || strings.Contains(promptVersion, "..") {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid teacher or prompt_version")
		return
	}

	body, _, err := h.s3.GetReasoningLabel(r.Context(), dataset, sampleID, teacher, promptVersion)
	if err != nil {
		if errors.Is(err, service.ErrNotFound) {
			writeError(w, http.StatusNotFound, model.CodeNotFound,
				"reasoning label not found for "+dataset+"/"+sampleID)
			return
		}
		slog.Error("get reasoning label", "dataset", dataset, "sample_id", sampleID, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to fetch reasoning label")
		return
	}

	// Label files are JSON; pass through verbatim. The source S3 key is
	// intentionally NOT exposed (bucket layout disclosure).
	writeRawJSON(w, http.StatusOK, body)
}
