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

// GetSample handles GET /api/v1/datasets/{name}/shards/{shard}/samples/{key}.
// One tar scan collects the member list, meta.json and the decoded ego.npy
// history/future arrays for the sample detail page.
func (h *DatasetsHandler) GetSample(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")
	shard := chi.URLParam(r, "shard")
	key := chi.URLParam(r, "key")

	if !h.s3.ValidDataset(name) {
		writeError(w, http.StatusNotFound, model.CodeNotFound, "unknown dataset: "+name)
		return
	}
	if !validShardName(shard) || key == "" || strings.ContainsAny(key, "/\\") || strings.Contains(key, "..") {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid shard/key")
		return
	}

	detail, err := h.s3.GetSampleDetail(r.Context(), name, shard, key)
	if err != nil {
		if errors.Is(err, service.ErrNotFound) {
			writeError(w, http.StatusNotFound, model.CodeNotFound, "sample not found: "+key)
			return
		}
		slog.Error("get sample detail", "dataset", name, "shard", shard, "key", key, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to read sample from shard")
		return
	}
	writeJSON(w, http.StatusOK, detail)
}

// GetShardIndex handles GET /api/v1/datasets/{name}/shards/{shard}/index.
// One tar scan produces per-member byte ranges plus a presigned tar URL so the
// ADAS player can range-GET frames directly from S3.
func (h *DatasetsHandler) GetShardIndex(w http.ResponseWriter, r *http.Request) {
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

	index, err := h.s3.BuildShardIndex(r.Context(), name, shard)
	if err != nil {
		if errors.Is(err, service.ErrNotFound) {
			writeError(w, http.StatusNotFound, model.CodeNotFound, "shard not found: "+shard)
			return
		}
		slog.Error("build shard index", "dataset", name, "shard", shard, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to index shard")
		return
	}
	writeJSON(w, http.StatusOK, index)
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

	// SECURITY note: the default path below streams exactly one tar member,
	// which is the least-privilege behavior. ?presign=true necessarily returns
	// a URL for the WHOLE shard tar (S3 cannot presign a byte range); it is
	// only acceptable because the caller already names the specific member
	// (key/cam in the path) and the /index endpoint pairs the same tar URL
	// with per-member byte ranges — tar URL + client-side range-GET is the
	// intended pattern for the player (range enforced client-side).
	if r.URL.Query().Get("presign") == "true" {
		url, err := h.s3.PresignShard(r.Context(), name, shard)
		if err != nil {
			slog.Error("presign shard", "dataset", name, "shard", shard, "error", err)
			writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to presign shard")
			return
		}
		// Never cache the presigned URL: it expires in ~15 min, and the image
		// cache behavior shares this path pattern. Without no-store CloudFront
		// could serve an expired URL (or cross-serve JPEG/JSON) for up to
		// default_ttl.
		w.Header().Set("Cache-Control", "no-store")
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
