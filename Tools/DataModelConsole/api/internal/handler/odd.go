package handler

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"

	"github.com/go-chi/chi/v5"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

// maxODDQueryBytes bounds the q parameter. The largest valid query (24
// conditions with a 128-byte key, 32 values and 8 sources of 64 bytes each)
// serialises to about 70 KiB.
const maxODDQueryBytes = 128 << 10

// ODDHandler serves the ODD scene search.
type ODDHandler struct {
	s3 *service.S3Service
}

// NewODDHandler builds the ODD handler.
func NewODDHandler(s3 *service.S3Service) *ODDHandler {
	return &ODDHandler{s3: s3}
}

// parseODDQuery decodes the q parameter: a JSON model.ODDQuery. An absent q
// selects every scene.
func parseODDQuery(raw string) (model.ODDQuery, error) {
	if raw == "" {
		return model.ODDQuery{Op: "and", Groups: []model.ODDQueryGroup{}}, nil
	}
	var query model.ODDQuery
	if len(raw) > maxODDQueryBytes {
		return query, service.ErrInvalidODDQuery{Reason: "q is too long"}
	}
	decoder := json.NewDecoder(bytes.NewReader([]byte(raw)))
	decoder.DisallowUnknownFields()
	var extra json.RawMessage
	if err := decoder.Decode(&query); err != nil || decoder.Decode(&extra) != io.EOF {
		return query, service.ErrInvalidODDQuery{Reason: "q must be one JSON query object"}
	}
	if query.Groups == nil {
		query.Groups = []model.ODDQueryGroup{}
	}
	return query, service.ValidateODDQuery(query)
}

// datasetVersion reads the required dataset and version parameters. version
// is required: newest-version discovery falls back to v1.0 on an S3 error,
// which would turn an outage into a false "no labels" 404.
func (h *ODDHandler) datasetVersion(w http.ResponseWriter, r *http.Request) (string, string, bool) {
	dataset := r.URL.Query().Get("dataset")
	if !h.s3.ValidDataset(dataset) {
		writeError(w, http.StatusNotFound, model.CodeNotFound, "unknown dataset: "+dataset)
		return "", "", false
	}
	version := r.URL.Query().Get("version")
	if !service.ValidVersion(version) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "missing or invalid version")
		return "", "", false
	}
	return dataset, version, true
}

func writeODDError(w http.ResponseWriter, err error, notFound string, logArgs ...any) {
	var invalid service.ErrInvalidODDQuery
	switch {
	case errors.As(err, &invalid):
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, invalid.Error())
	case errors.Is(err, service.ErrNotFound):
		writeError(w, http.StatusNotFound, model.CodeNotFound, notFound)
	default:
		slog.Error("odd", append(logArgs, "error", err)...)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to read ODD labels")
	}
}

// Search handles
// GET /api/v1/odd/scenes?dataset=&version=&q=<JSON ODDQuery>&limit=&offset=
// — the scenes whose valid ODD labels satisfy q.
func (h *ODDHandler) Search(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.datasetVersion(w, r)
	if !ok {
		return
	}
	query, err := parseODDQuery(r.URL.Query().Get("q"))
	if err != nil {
		writeODDError(w, err, "")
		return
	}
	limit, offset := parsePagination(r)
	response, err := h.s3.SearchODDScenes(
		r.Context(), dataset, version, query, limit, offset,
	)
	if err != nil {
		writeODDError(
			w, err, "no ODD labels for this dataset version",
			"dataset", dataset, "version", version,
		)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

// Scene handles GET /api/v1/odd/scenes/{scene_uid}?dataset=&version= — every
// valid ODD label of one scene.
func (h *ODDHandler) Scene(w http.ResponseWriter, r *http.Request) {
	sceneUID := chi.URLParam(r, "scene_uid")
	if !service.ValidODDSceneUID(sceneUID) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid scene_uid")
		return
	}
	dataset, version, ok := h.datasetVersion(w, r)
	if !ok {
		return
	}
	response, err := h.s3.ODDScene(r.Context(), dataset, version, sceneUID)
	if err != nil {
		writeODDError(
			w, err, "no ODD labels for this scene",
			"dataset", dataset, "version", version, "scene_uid", sceneUID,
		)
		return
	}
	writeJSON(w, http.StatusOK, response)
}
