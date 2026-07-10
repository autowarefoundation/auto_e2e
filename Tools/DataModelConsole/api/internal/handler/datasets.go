package handler

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

// DatasetsHandler serves the S3-backed dataset browsing endpoints.
type DatasetsHandler struct {
	s3 *service.S3Service
}

// NewDatasetsHandler builds the datasets handler.
func NewDatasetsHandler(s3 *service.S3Service) *DatasetsHandler {
	return &DatasetsHandler{s3: s3}
}

// List handles GET /api/v1/datasets.
func (h *DatasetsHandler) List(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, model.DatasetListResponse{Datasets: h.s3.ListDatasets()})
}

// ListShards handles GET /api/v1/datasets/{name}/shards.
func (h *DatasetsHandler) ListShards(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")
	if !h.s3.ValidDataset(name) {
		writeError(w, http.StatusNotFound, model.CodeNotFound, "unknown dataset: "+name)
		return
	}
	limit, offset := parsePagination(r)

	shards, page, err := h.s3.ListShards(r.Context(), name, limit, offset)
	if err != nil {
		slog.Error("list shards", "dataset", name, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to list shards")
		return
	}
	if shards == nil {
		shards = []model.Shard{}
	}
	writeJSON(w, http.StatusOK, model.ShardListResponse{Dataset: name, Shards: shards, Page: page})
}

// ListSamples handles GET /api/v1/datasets/{name}/shards/{shard}/samples.
func (h *DatasetsHandler) ListSamples(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")
	shard := chi.URLParam(r, "shard")
	if !h.s3.ValidDataset(name) {
		writeError(w, http.StatusNotFound, model.CodeNotFound, "unknown dataset: "+name)
		return
	}
	if !validShardName(shard) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid shard name")
		return
	}
	limit, offset := parsePagination(r)

	samples, page, err := h.s3.ListSamples(r.Context(), name, shard, limit, offset)
	if err != nil {
		if errors.Is(err, service.ErrNotFound) {
			writeError(w, http.StatusNotFound, model.CodeNotFound, "shard not found: "+shard)
			return
		}
		slog.Error("list samples", "dataset", name, "shard", shard, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to read shard")
		return
	}
	if samples == nil {
		samples = []model.Sample{}
	}
	writeJSON(w, http.StatusOK, model.SampleListResponse{
		Dataset: name, Shard: shard, Samples: samples, Page: page,
	})
}

// GetImage handles
// GET /api/v1/datasets/{name}/shards/{shard}/samples/{key}/image/{cam}.
//
// Phase 1: streams the tar from S3, locates the member {key}.{cam}.jpg and
// pipes its bytes back with image/jpeg + Cache-Control. With
// ?presign=true it instead returns a presigned URL for the whole tar so the
// client can range-GET using the offsets from the samples listing.
func (h *DatasetsHandler) GetImage(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")
	shard := chi.URLParam(r, "shard")
	key := chi.URLParam(r, "key")
	cam := chi.URLParam(r, "cam")

	if !h.s3.ValidDataset(name) {
		writeError(w, http.StatusNotFound, model.CodeNotFound, "unknown dataset: "+name)
		return
	}
	if !validShardName(shard) || strings.ContainsAny(key, "/\\") || !validCam(cam) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid shard/key/cam")
		return
	}

	if r.URL.Query().Get("presign") == "true" {
		url, err := h.s3.PresignShard(r.Context(), name, shard)
		if err != nil {
			slog.Error("presign shard", "dataset", name, "shard", shard, "error", err)
			writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to presign shard")
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{
			"url":    url,
			"member": fmt.Sprintf("%s.%s.jpg", key, cam),
			"note":   "range-GET the tar using offset/size_bytes from the samples listing",
		})
		return
	}

	member := fmt.Sprintf("%s.%s.jpg", key, cam)
	reader, closer, size, err := h.s3.StreamTarMember(r.Context(), name, shard, member)
	if err != nil {
		if errors.Is(err, service.ErrNotFound) {
			writeError(w, http.StatusNotFound, model.CodeNotFound, "image not found: "+member)
			return
		}
		slog.Error("stream tar member", "dataset", name, "shard", shard, "member", member, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to read image from shard")
		return
	}
	defer closer.Close()

	w.Header().Set("Content-Type", "image/jpeg")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", size))
	w.Header().Set("Cache-Control", "public, max-age=3600")
	w.WriteHeader(http.StatusOK)
	if _, err := io.Copy(w, reader); err != nil {
		// Headers already sent; just log (client likely disconnected).
		slog.Warn("copy image body", "member", member, "error", err)
	}
}

// validShardName accepts plain .tar file names (no path traversal).
func validShardName(s string) bool {
	return strings.HasSuffix(s, ".tar") && !strings.ContainsAny(s, "/\\") && s != ".tar"
}

// validCam accepts cam_0 .. cam_6 style identifiers.
func validCam(s string) bool {
	if !strings.HasPrefix(s, "cam_") || len(s) < 5 {
		return false
	}
	for _, c := range s[4:] {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}
