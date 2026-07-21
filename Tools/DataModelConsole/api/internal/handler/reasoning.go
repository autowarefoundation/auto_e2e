package handler

import (
	"errors"
	"log/slog"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/store"
)

// ReasoningHandler serves the reasoning label cache endpoints.
type ReasoningHandler struct {
	s3 *service.S3Service
}

// NewReasoningHandler builds the reasoning labels handler.
func NewReasoningHandler(s3 *service.S3Service) *ReasoningHandler {
	return &ReasoningHandler{s3: s3}
}

func writeReasoningAvailabilityError(
	w http.ResponseWriter,
	err error,
) bool {
	switch {
	case errors.Is(err, service.ErrReasoningUnavailable):
		w.Header().Set("Retry-After", "60")
		writeError(
			w,
			http.StatusServiceUnavailable,
			model.CodeUnavailable,
			"reasoning inventory is not materialized",
		)
		return true
	case errors.Is(err, service.ErrReasoningIntegrity):
		writeError(
			w,
			http.StatusBadGateway,
			model.CodeS3Error,
			"reasoning publication failed integrity validation",
		)
		return true
	default:
		return false
	}
}

// Stats handles GET /api/v1/reasoning-labels/stats — counts label objects
// per dataset/teacher/prompt_version partition.
func (h *ReasoningHandler) Stats(w http.ResponseWriter, r *http.Request) {
	entries, total, err := h.s3.ReasoningStats(r.Context())
	if err != nil {
		if writeReasoningAvailabilityError(w, err) {
			return
		}
		slog.Error("reasoning stats", "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to read reasoning label stats")
		return
	}
	if entries == nil {
		entries = []model.ReasoningStatsEntry{}
	}
	writeJSON(w, http.StatusOK, model.ReasoningStatsResponse{Entries: entries, Total: total})
}

// PromptVersions handles
// GET /api/v1/reasoning-labels/prompt-versions?dataset={name}&version={v} —
// the teacher/prompt_version partitions of one immutable dataset version.
func (h *ReasoningHandler) PromptVersions(w http.ResponseWriter, r *http.Request) {
	dataset := r.URL.Query().Get("dataset")
	if !validReasoningParam(dataset) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "missing or invalid dataset")
		return
	}
	if !h.s3.ValidDataset(dataset) {
		writeError(w, http.StatusNotFound, model.CodeNotFound, "unknown dataset: "+dataset)
		return
	}
	version, ok := requestedVersion(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return
	}
	entries, err := h.s3.ReasoningPromptVersionsAtVersion(
		r.Context(), dataset, version,
	)
	if err != nil {
		if writeReasoningAvailabilityError(w, err) {
			return
		}
		if errors.Is(err, service.ErrNotFound) {
			writeError(
				w,
				http.StatusNotFound,
				model.CodeNotFound,
				"dataset version not found",
			)
			return
		}
		slog.Error(
			"reasoning prompt versions",
			"dataset", dataset,
			"version", version,
			"error", err,
		)
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
// teacher x prompt_version) reasoning-label set. Reads are cache-only; the
// trusted materializer publishes the inventory after stats and scene rows.
func (h *ReasoningHandler) StatsDetail(w http.ResponseWriter, r *http.Request) {
	dataset := r.URL.Query().Get("dataset")
	promptVersion := r.URL.Query().Get("prompt_version")
	teacher := r.URL.Query().Get("teacher")
	if !validReasoningParam(dataset) || !validReasoningParam(promptVersion) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "missing or invalid dataset/prompt_version")
		return
	}
	if !h.s3.ValidDataset(dataset) {
		writeError(w, http.StatusNotFound, model.CodeNotFound, "unknown dataset: "+dataset)
		return
	}
	if !service.ValidReasoningTeacherID(teacher) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "missing or invalid teacher")
		return
	}
	version, ok := requestedVersion(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return
	}

	resp, err := h.s3.ReasoningStatsDetail(r.Context(), dataset, version, promptVersion, teacher)
	if err != nil {
		if writeReasoningAvailabilityError(w, err) {
			return
		}
		if errors.Is(err, service.ErrNotFound) {
			writeError(w, http.StatusNotFound, model.CodeNotFound, "reasoning label partition not found")
			return
		}
		slog.Error("reasoning stats-detail", "dataset", dataset, "prompt_version", promptVersion, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to read reasoning stats")
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// validReasoningParam applies both S3 path and DynamoDB key constraints at the
// HTTP boundary so malformed client input never becomes an upstream error.
func validReasoningParam(v string) bool {
	return store.ValidReasoningKeyComponent(v) &&
		!strings.ContainsAny(v, "/\\") &&
		!strings.Contains(v, "..")
}

// GetLabel handles GET /api/v1/reasoning-labels/{dataset}/{sample_id}.
// Optional version, teacher, and prompt_version pin the immutable shard member.
func (h *ReasoningHandler) GetLabel(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	sampleID := chi.URLParam(r, "sample_id")
	if !validReasoningParam(dataset) ||
		!validReasoningParam(sampleID) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid dataset or sample_id")
		return
	}
	if !h.s3.ValidDataset(dataset) {
		writeError(w, http.StatusNotFound, model.CodeNotFound, "unknown dataset: "+dataset)
		return
	}

	teacher := r.URL.Query().Get("teacher")
	promptVersion := r.URL.Query().Get("prompt_version")
	// These land in the S3 key template; reject path-traversal characters the
	// same way dataset/sample_id are validated above.
	if (promptVersion != "" && !validReasoningParam(promptVersion)) ||
		(teacher != "" && !service.ValidReasoningTeacherID(teacher)) ||
		(promptVersion != "" && teacher == "") {
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
		if writeReasoningAvailabilityError(w, err) {
			return
		}
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
