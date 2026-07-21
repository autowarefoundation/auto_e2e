package handler

import (
	"errors"
	"log/slog"
	"net/http"
	"strconv"

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
	// value is a categorical label embedded in a DynamoDB key.
	if !validReasoningParam(value) {
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
	scenes, resolvedVersion, err := h.s3.SearchScenesByLabelForTeacherAtVersion(
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
		if writeReasoningAvailabilityError(w, err) {
			return
		}
		if errors.Is(err, service.ErrNotFound) {
			writeError(
				w,
				http.StatusNotFound,
				model.CodeNotFound,
				"reasoning label partition not found",
			)
			return
		}
		slog.Error("scene search", "dataset", dataset, "field", field, "value", value, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to search scenes by label")
		return
	}
	truncated := len(scenes) > limit
	if truncated {
		scenes = scenes[:limit]
	}

	available := 0
	for _, scene := range scenes {
		if scene.Available {
			available++
		}
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
