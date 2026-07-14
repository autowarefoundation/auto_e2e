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

// PromptVersions handles
// GET /api/v1/reasoning-labels/prompt-versions?dataset={name} — the
// teacher/prompt_version partitions of ONE dataset's label cache with counts.
func (h *ReasoningHandler) PromptVersions(w http.ResponseWriter, r *http.Request) {
	dataset := r.URL.Query().Get("dataset")
	if dataset == "" || strings.ContainsAny(dataset, "/\\") || strings.Contains(dataset, "..") {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "missing or invalid dataset")
		return
	}
	entries, err := h.s3.ReasoningPromptVersions(r.Context(), dataset)
	if err != nil {
		slog.Error("reasoning prompt versions", "dataset", dataset, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to list reasoning prompt versions")
		return
	}
	if entries == nil {
		entries = []model.ReasoningPromptVersion{}
	}
	writeJSON(w, http.StatusOK, model.ReasoningPromptVersionsResponse{Dataset: dataset, PromptVersions: entries})
}

// StatsDetail handles
// GET /api/v1/reasoning-labels/stats-detail?dataset=&version=&prompt_version=&teacher=
// — the precomputed ODD-coverage stats for one (dataset x version x
// prompt_version) reasoning-label set. Read-through DynamoDB: a hit returns the
// cached blob; a miss scans the labels from S3, aggregates, populates the
// scene-by-label index, persists, and returns.
func (h *ReasoningHandler) StatsDetail(w http.ResponseWriter, r *http.Request) {
	dataset := r.URL.Query().Get("dataset")
	promptVersion := r.URL.Query().Get("prompt_version")
	teacher := r.URL.Query().Get("teacher")
	if !validReasoningParam(dataset) || !validReasoningParam(promptVersion) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "missing or invalid dataset/prompt_version")
		return
	}
	if teacher != "" && !validReasoningParam(teacher) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid teacher")
		return
	}
	version, ok := requestedVersion(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return
	}

	resp, err := h.s3.ReasoningStatsDetail(r.Context(), dataset, version, promptVersion, teacher)
	if err != nil {
		slog.Error("reasoning stats-detail", "dataset", dataset, "prompt_version", promptVersion, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to compute reasoning stats")
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// ComputeStats handles
// GET|POST /api/v1/reasoning-labels/compute-stats?dataset=&prompt_version=&teacher=
// — force-(re)computes the stats blob AND repopulates the scene-by-label index.
// Idempotent.
func (h *ReasoningHandler) ComputeStats(w http.ResponseWriter, r *http.Request) {
	dataset := r.URL.Query().Get("dataset")
	promptVersion := r.URL.Query().Get("prompt_version")
	teacher := r.URL.Query().Get("teacher")
	if !validReasoningParam(dataset) || !validReasoningParam(promptVersion) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "missing or invalid dataset/prompt_version")
		return
	}
	if teacher != "" && !validReasoningParam(teacher) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid teacher")
		return
	}
	version, ok := requestedVersion(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return
	}

	resp, err := h.s3.ComputeReasoningStats(r.Context(), dataset, version, promptVersion, teacher)
	if err != nil {
		slog.Error("reasoning compute-stats", "dataset", dataset, "prompt_version", promptVersion, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to compute reasoning stats")
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// validReasoningParam rejects empty values and path-traversal characters for
// values that land in an S3 key template or a DynamoDB key.
func validReasoningParam(v string) bool {
	return v != "" && !strings.ContainsAny(v, "/\\") && !strings.Contains(v, "..")
}

// GetLabel handles GET /api/v1/reasoning-labels/{dataset}/{sample_id}.
// Optional version, teacher, and prompt_version pin the immutable shard member.
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
	version, ok := requestedVersion(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return
	}

	body, _, err := h.s3.GetReasoningLabelAtVersion(
		r.Context(), dataset, version, sampleID, teacher, promptVersion,
	)
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
