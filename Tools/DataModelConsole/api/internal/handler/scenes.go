package handler

import (
	"log/slog"
	"net/http"
	"strconv"
	"strings"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/store"
)

// ScenesHandler serves the scene-by-label search endpoint (backed by the
// DynamoDB scene-by-label index populated during stats computation).
type ScenesHandler struct {
	s3 *service.S3Service
}

// NewScenesHandler builds the scenes handler.
func NewScenesHandler(s3 *service.S3Service) *ScenesHandler {
	return &ScenesHandler{s3: s3}
}

// sceneSearchMaxLimit caps how many scenes one search returns.
const sceneSearchMaxLimit = 5000

// Search handles
// GET /api/v1/scenes/search?dataset=&teacher=&prompt_version=&field=&value=&limit=
// — the scenes (sample ids) carrying a specific (field,value) reasoning label
// in one exact teacher partition.
func (h *ScenesHandler) Search(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	dataset := q.Get("dataset")
	teacher := q.Get("teacher")
	promptVersion := q.Get("prompt_version")
	field := q.Get("field")
	value := q.Get("value")

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
	if !store.IsStatField(field) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "unknown field; must be a reasoning taxonomy axis")
		return
	}
	// value is a categorical label; reject traversal characters that would leak
	// into the DynamoDB key. An empty value is a client error (nothing to find).
	if value == "" || strings.ContainsAny(value, "/\\") || strings.Contains(value, "..") {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "missing or invalid value")
		return
	}

	limit := sceneSearchMaxLimit
	if v := q.Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			limit = min(n, sceneSearchMaxLimit)
		}
	}

	// Optional version scopes which published shards a scene can resolve into.
	version := ""
	if v := q.Get("version"); v != "" {
		if !service.ValidVersion(v) {
			writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
			return
		}
		version = v
	}

	// Fetch one extra row so we can report truncation truthfully instead of
	// silently capping at `limit`.
	ids, resolvedVersion, err := h.s3.SearchScenesByLabelForTeacherAtVersion(
		r.Context(),
		dataset,
		version,
		teacher,
		promptVersion,
		field,
		value,
		limit+1,
	)
	if err != nil {
		slog.Error("scene search", "dataset", dataset, "field", field, "value", value, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to search scenes by label")
		return
	}
	truncated := len(ids) > limit
	if truncated {
		ids = ids[:limit]
	}

	// Resolve each sample id to the shard that actually contains it in this
	// version; ids not packed into any published shard are marked unavailable
	// so the UI links only real samples (labels can outnumber packed frames).
	shardByID := h.s3.ResolveSampleShards(
		r.Context(), dataset, resolvedVersion, ids,
	)

	scenes := make([]model.SceneRef, 0, len(ids))
	available := 0
	for _, id := range ids {
		sh := shardByID[id]
		ok := sh != ""
		if ok {
			available++
		}
		scenes = append(scenes, model.SceneRef{SampleID: id, Shard: sh, Available: ok})
	}
	writeJSON(w, http.StatusOK, model.SceneSearchResponse{
		Dataset:       dataset,
		Teacher:       teacher,
		PromptVersion: promptVersion,
		Version:       resolvedVersion,
		Field:         field,
		Value:         value,
		Scenes:        scenes,
		Total:         len(scenes),
		Available:     available,
		Truncated:     truncated,
	})
}
