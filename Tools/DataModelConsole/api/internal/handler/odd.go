package handler

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

const maxODDSearchBody = 64 << 10

// ODDHandler serves immutable scene-level ODD LabelSet read models.
type ODDHandler struct {
	s3                     *service.S3Service
	operations             ODDOperations
	operationsRequiredRole string
}

type ODDOperations interface {
	Enabled() bool
	AllowFull() bool
	Launch(
		context.Context, string, string, string,
	) (service.ODDOperationResult, error)
	Retry(context.Context, string) (service.ODDOperationResult, error)
}

func NewODDHandler(s3 *service.S3Service) *ODDHandler {
	return &ODDHandler{s3: s3}
}

func NewODDHandlerWithOperations(
	s3 *service.S3Service,
	operations ODDOperations,
	requiredRole string,
) *ODDHandler {
	return &ODDHandler{
		s3:                     s3,
		operations:             operations,
		operationsRequiredRole: requiredRole,
	}
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
	case errors.Is(err, service.ErrODDUnavailable),
		errors.Is(err, service.ErrODDNotStarted):
		writeError(w, http.StatusServiceUnavailable, model.CodeUnavailable, "ODD LabelSet is not ready")
	case errors.Is(err, service.ErrNotFound):
		writeError(w, http.StatusNotFound, model.CodeNotFound, "ODD scene not found")
	case errors.Is(err, service.ErrODDInvalidQuery):
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, err.Error())
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

func (h *ODDHandler) ModelMetrics(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.coordinate(w, r)
	if !ok {
		return
	}
	response, manifest, digest, err := h.s3.ODDMetricProjections(
		r.Context(),
		dataset,
		version,
	)
	if err != nil {
		writeODDError(w, err)
		return
	}
	setODDIdentity(w, manifest, digest)
	writeJSON(w, http.StatusOK, response)
}

func (h *ODDHandler) LabelSets(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.coordinate(w, r)
	if !ok {
		return
	}
	_, manifest, digest, err := h.s3.ODDStatistics(r.Context(), dataset, version)
	if err != nil {
		if errors.Is(err, service.ErrODDNotStarted) {
			writeJSON(w, http.StatusOK, map[string]any{
				"dataset": dataset, "version": version,
				"state": "not_started", "labelsets": []service.ODDManifest{},
			})
			return
		}
		writeODDError(w, err)
		return
	}
	setODDIdentity(w, manifest, digest)
	writeJSON(w, http.StatusOK, map[string]any{
		"dataset": dataset, "version": version, "state": "ready",
		"labelsets": []service.ODDManifest{manifest},
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

func decodeODDSearchRequest(
	w http.ResponseWriter,
	r *http.Request,
) (service.ODDStructuredSearchRequest, error) {
	var request service.ODDStructuredSearchRequest
	body := http.MaxBytesReader(w, r.Body, maxODDSearchBody)
	defer body.Close()
	decoder := json.NewDecoder(body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		return request, err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return request, errors.New("request body must contain one JSON object")
		}
		return request, err
	}
	return request, nil
}

func (h *ODDHandler) SearchStructured(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.coordinate(w, r)
	if !ok {
		return
	}
	request, err := decodeODDSearchRequest(w, r)
	if err != nil {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid ODD search request")
		return
	}
	response, err := h.s3.SearchODDScenesStructured(
		r.Context(), dataset, version, request,
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

func (h *ODDHandler) Evidence(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.coordinate(w, r)
	if !ok {
		return
	}
	sceneUID := chi.URLParam(r, "scene_uid")
	observationUID := chi.URLParam(r, "observation_uid")
	if !validReasoningParam(sceneUID) ||
		!validReasoningParam(observationUID) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid ODD evidence coordinate")
		return
	}
	response, manifest, digest, err := h.s3.ODDEvidence(
		r.Context(), dataset, version, sceneUID, observationUID,
	)
	if err != nil {
		writeODDError(w, err)
		return
	}
	setODDIdentity(w, manifest, digest)
	writeJSON(w, http.StatusOK, response)
}
