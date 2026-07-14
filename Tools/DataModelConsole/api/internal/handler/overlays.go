package handler

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"

	"github.com/go-chi/chi/v5"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

// OverlayHandler serves immutable trajectory artifacts and geospatial products.
type OverlayHandler struct {
	s3                   *service.S3Service
	exactGeoEnabled      bool
	exactGeoRequiredRole string
}

// NewOverlayHandler builds the overlay/geo handler. Exact routes remain closed
// unless deployment configuration and a trusted edge-injected role both allow
// them.
func NewOverlayHandler(
	s3 *service.S3Service,
	exactGeoEnabled bool,
	exactGeoRequiredRole string,
) *OverlayHandler {
	return &OverlayHandler{
		s3:                   s3,
		exactGeoEnabled:      exactGeoEnabled,
		exactGeoRequiredRole: exactGeoRequiredRole,
	}
}

// Models handles GET /datasets/{name}/shards/{shard}/overlay-models.
func (h *OverlayHandler) Models(w http.ResponseWriter, r *http.Request) {
	dataset, shard, version, ok := h.shardRequest(w, r)
	if !ok {
		return
	}
	models, resolvedVersion, err := h.s3.ListOverlayModels(
		r.Context(), dataset, version, shard,
	)
	if err != nil {
		slog.Error("list overlay models", "dataset", dataset, "shard", shard, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeUnavailable, "overlay index unavailable")
		return
	}
	if models == nil {
		models = []model.OverlayModel{}
	}
	writeJSON(w, http.StatusOK, model.OverlayModelsResponse{
		Dataset: dataset,
		Version: resolvedVersion,
		Shard:   shard,
		Models:  models,
	})
}

// Body handles GET /datasets/{name}/shards/{shard}/overlays/{model_id}.
func (h *OverlayHandler) Body(w http.ResponseWriter, r *http.Request) {
	dataset, shard, version, ok := h.shardRequest(w, r)
	if !ok {
		return
	}
	modelID := chi.URLParam(r, "model_id")
	if !validArtifactID(modelID) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid model artifact id")
		return
	}
	body, _, err := h.s3.GetOverlayBody(
		r.Context(), dataset, version, shard, modelID,
	)
	if err != nil {
		if errors.Is(err, service.ErrNotFound) {
			writeError(w, http.StatusNotFound, model.CodeNotFound, "overlay not found")
			return
		}
		slog.Error("read overlay body", "dataset", dataset, "shard", shard, "model_id", modelID, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "overlay artifact failed validation")
		return
	}
	d := body.Descriptor
	w.Header().Set("Content-Type", "application/vnd.auto-e2e.overlay")
	w.Header().Set("Content-Encoding", "gzip")
	w.Header().Set("Content-Length", strconv.FormatInt(d.ByteSize, 10))
	w.Header().Set("Cache-Control", "private, max-age=31536000, immutable")
	w.Header().Set("ETag", fmt.Sprintf("%q", d.SHA256))
	w.Header().Set("X-Overlay-Schema", d.OverlaySchema)
	w.Header().Set("X-Overlay-SHA256", d.SHA256)
	w.Header().Set("X-Overlay-Sample-Count", strconv.Itoa(d.SampleCount))
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(body.Payload); err != nil {
		slog.Warn("write overlay response", "model_id", modelID, "error", err)
	}
}

// Rig handles GET /datasets/{name}/rig-projection.
func (h *OverlayHandler) Rig(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.datasetRequest(w, r)
	if !ok {
		return
	}
	body, _, err := h.s3.RigProjection(r.Context(), dataset, version)
	if err != nil {
		if errors.Is(err, service.ErrNotFound) {
			writeError(w, http.StatusNotFound, model.CodeNotFound, "rig projection not found")
			return
		}
		slog.Error("read rig projection", "dataset", dataset, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to read rig projection")
		return
	}
	if !json.Valid(body) {
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "invalid rig projection artifact")
		return
	}
	w.Header().Set("Cache-Control", "private, max-age=31536000, immutable")
	writeRawJSON(w, http.StatusOK, body)
}

// GeoStats handles GET /datasets/{name}/geo-stats.
func (h *OverlayHandler) GeoStats(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.datasetRequest(w, r)
	if !ok {
		return
	}
	stats, err := h.s3.GeoStats(r.Context(), dataset, version)
	if err != nil {
		if errors.Is(err, service.ErrNotFound) {
			writeError(w, http.StatusNotFound, model.CodeNotFound, "geo stats not found")
			return
		}
		slog.Error("read geo stats", "dataset", dataset, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to read geo stats")
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

// GeoHeatmap handles GET /datasets/{name}/geo/heatmap.
func (h *OverlayHandler) GeoHeatmap(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.datasetRequest(w, r)
	if !ok {
		return
	}
	body, _, err := h.s3.GeoHeatmap(r.Context(), dataset, version)
	if err != nil {
		if errors.Is(err, service.ErrNotFound) {
			writeError(w, http.StatusNotFound, model.CodeNotFound, "geo heatmap not found")
			return
		}
		slog.Error("read geo heatmap", "dataset", dataset, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to read geo heatmap")
		return
	}
	w.Header().Set("Content-Type", "application/geo+json")
	w.Header().Set("Cache-Control", "private, max-age=3600")
	if len(body) >= 2 && body[0] == 0x1f && body[1] == 0x8b {
		w.Header().Set("Content-Encoding", "gzip")
	}
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(body); err != nil {
		slog.Warn("write geo heatmap", "dataset", dataset, "error", err)
	}
}

// EpisodePath handles GET /datasets/{name}/geo/episodes/{episode}. Exact route
// access is disabled by default and requires a trusted edge-injected role.
func (h *OverlayHandler) EpisodePath(w http.ResponseWriter, r *http.Request) {
	dataset, version, ok := h.datasetRequest(w, r)
	if !ok {
		return
	}
	if !h.exactGeoAuthorized(r) {
		writeError(w, http.StatusForbidden, model.CodeUnavailable, "exact geo access is not authorized")
		return
	}
	episode := chi.URLParam(r, "episode")
	body, _, err := h.s3.EpisodePath(r.Context(), dataset, version, episode)
	if err != nil {
		if errors.Is(err, service.ErrNotFound) {
			writeError(w, http.StatusNotFound, model.CodeNotFound, "episode path not found")
			return
		}
		slog.Error("read exact episode path", "dataset", dataset, "episode", episode, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to read episode path")
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.Header().Set("Cache-Control", "private, no-store")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(body); err != nil {
		slog.Warn("write episode path", "dataset", dataset, "episode", episode, "error", err)
	}
}

func (h *OverlayHandler) datasetRequest(w http.ResponseWriter, r *http.Request) (string, string, bool) {
	dataset := chi.URLParam(r, "name")
	if !h.s3.ValidDataset(dataset) {
		writeError(w, http.StatusNotFound, model.CodeNotFound, "unknown dataset: "+dataset)
		return "", "", false
	}
	version, ok := requestedVersion(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return "", "", false
	}
	return dataset, version, true
}

func (h *OverlayHandler) shardRequest(w http.ResponseWriter, r *http.Request) (string, string, string, bool) {
	dataset, version, ok := h.datasetRequest(w, r)
	if !ok {
		return "", "", "", false
	}
	shard := chi.URLParam(r, "shard")
	if !validShardName(shard) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid shard name")
		return "", "", "", false
	}
	return dataset, shard, version, true
}

func (h *OverlayHandler) exactGeoAuthorized(r *http.Request) bool {
	if !h.exactGeoEnabled || h.exactGeoRequiredRole == "" {
		return false
	}
	for _, role := range strings.Split(r.Header.Get("X-Console-Roles"), ",") {
		if strings.TrimSpace(role) == h.exactGeoRequiredRole {
			return true
		}
	}
	return false
}

func validArtifactID(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, char := range value {
		if (char < '0' || char > '9') && (char < 'a' || char > 'f') {
			return false
		}
	}
	return true
}
