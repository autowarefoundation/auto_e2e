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

// ODDHandler serves immutable scene-level ODD LabelSet read models.
type ODDHandler struct {
	s3 *service.S3Service
}

func NewODDHandler(s3 *service.S3Service) *ODDHandler {
	return &ODDHandler{s3: s3}
}

func (h *ODDHandler) coordinate(
	w http.ResponseWriter,
	r *http.Request,
) (string, string, bool) {
	dataset := r.URL.Query().Get("dataset")
	if dataset == "" {
		dataset = "kitscenes"
	}
	if !validReasoningParam(dataset) || !h.s3.ValidDataset(dataset) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid dataset")
		return "", "", false
	}
	requested, ok := requestedVersion(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return "", "", false
	}
	version, err := h.s3.ResolvedVersion(r.Context(), dataset, requested)
	if err != nil {
		writeError(w, http.StatusNotFound, model.CodeNotFound, "dataset version not found")
		return "", "", false
	}
	return dataset, version, true
}

func writeODDError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, service.ErrODDUnavailable):
		writeError(w, http.StatusServiceUnavailable, model.CodeUnavailable, "ODD LabelSet is not ready")
	case errors.Is(err, service.ErrNotFound):
		writeError(w, http.StatusNotFound, model.CodeNotFound, "ODD scene not found")
	default:
		slog.Error("read ODD LabelSet", "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "ODD LabelSet failed integrity validation")
	}
}

func setODDIdentity(w http.ResponseWriter, manifest service.ODDManifest, digest string) {
	w.Header().Set("X-ODD-LabelSet-ID", manifest.LabelSetID)
	w.Header().Set("X-ODD-Manifest-SHA256", digest)
}

func (h *ODDHandler) Ontology(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.coordinate(w, r)
	if !ok {
		return
	}
	body, manifest, digest, err := h.s3.ODDOntology(r.Context(), dataset, version)
	if err != nil {
		writeODDError(w, err)
		return
	}
	setODDIdentity(w, manifest, digest)
	writeRawJSON(w, http.StatusOK, body)
}

func (h *ODDHandler) Statistics(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.coordinate(w, r)
	if !ok {
		return
	}
	body, manifest, digest, err := h.s3.ODDStatistics(r.Context(), dataset, version)
	if err != nil {
		writeODDError(w, err)
		return
	}
	setODDIdentity(w, manifest, digest)
	writeRawJSON(w, http.StatusOK, body)
}

func (h *ODDHandler) LabelSets(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.coordinate(w, r)
	if !ok {
		return
	}
	_, manifest, digest, err := h.s3.ODDStatistics(r.Context(), dataset, version)
	if err != nil {
		writeODDError(w, err)
		return
	}
	setODDIdentity(w, manifest, digest)
	writeJSON(w, http.StatusOK, map[string]any{
		"dataset": dataset, "version": version, "labelsets": []service.ODDManifest{manifest},
	})
}

func (h *ODDHandler) Search(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.coordinate(w, r)
	if !ok {
		return
	}
	key := r.URL.Query().Get("key")
	value := r.URL.Query().Get("value")
	status := r.URL.Query().Get("status")
	source := r.URL.Query().Get("source")
	for _, value := range []string{key, value, status, source} {
		if value != "" && (!validReasoningParam(value) || strings.Contains(value, "..")) {
			writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid ODD search predicate")
			return
		}
	}
	limit, offset := parsePagination(r)
	response, err := h.s3.SearchODDScenes(
		r.Context(), dataset, version, key, value, status, source, limit, offset,
	)
	if err != nil {
		writeODDError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *ODDHandler) Scene(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.coordinate(w, r)
	if !ok {
		return
	}
	sceneUID := chi.URLParam(r, "scene_uid")
	if !validReasoningParam(sceneUID) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid scene uid")
		return
	}
	body, manifest, digest, err := h.s3.ODDScene(
		r.Context(), dataset, version, sceneUID,
	)
	if err != nil {
		writeODDError(w, err)
		return
	}
	setODDIdentity(w, manifest, digest)
	writeRawJSON(w, http.StatusOK, body)
}
