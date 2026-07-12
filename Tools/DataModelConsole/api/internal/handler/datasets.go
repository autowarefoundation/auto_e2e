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
	writeJSON(w, http.StatusOK, model.DatasetListResponse{Datasets: h.s3.ListDatasets(r.Context())})
}

// requestedVersion reads the optional ?version= override, validating it as a
// well-formed version dir. Returns (version, ok=true) when present and valid,
// ("", true) when absent (auto-resolve), and ("", false) when present but
// malformed so the handler can 400 instead of silently auto-resolving.
func requestedVersion(r *http.Request) (version string, ok bool) {
	v := r.URL.Query().Get("version")
	if v == "" {
		return "", true // absent: auto-resolve downstream
	}
	if !service.ValidVersion(v) {
		return "", false
	}
	return v, true
}

// ListVersions handles GET /api/v1/datasets/{name}/versions — every published
// version of a dataset with its whole-training composition, newest-first.
func (h *DatasetsHandler) ListVersions(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")
	if !h.s3.ValidDataset(name) {
		writeError(w, http.StatusNotFound, model.CodeNotFound, "unknown dataset: "+name)
		return
	}
	versions, err := h.s3.ListDatasetVersions(r.Context(), name)
	if err != nil {
		slog.Error("list dataset versions", "dataset", name, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to list dataset versions")
		return
	}
	if versions == nil {
		versions = []model.DatasetVersion{}
	}
	writeJSON(w, http.StatusOK, model.DatasetVersionsResponse{Dataset: name, Versions: versions})
}

// ListShards handles GET /api/v1/datasets/{name}/shards.
func (h *DatasetsHandler) ListShards(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")
	if !h.s3.ValidDataset(name) {
		writeError(w, http.StatusNotFound, model.CodeNotFound, "unknown dataset: "+name)
		return
	}
	version, ok := requestedVersion(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return
	}
	limit, offset := parsePagination(r)

	shards, page, err := h.s3.ListShards(r.Context(), name, version, limit, offset)
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
	version, ok := requestedVersion(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return
	}
	limit, offset := parsePagination(r)

	samples, page, err := h.s3.ListSamples(r.Context(), name, version, shard, limit, offset)
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
	version, ok := requestedVersion(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return
	}

	detail, err := h.s3.GetSampleDetail(r.Context(), name, version, shard, key)
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
// One tar scan produces per-member byte ranges plus per-frame ego state/plan
// for the ADAS player.
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
	version, ok := requestedVersion(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return
	}

	index, err := h.s3.BuildShardIndex(r.Context(), name, version, shard)
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
// pipes its bytes back with image/jpeg + Cache-Control. Streaming exactly one
// tar member is the least-privilege behavior (no whole-shard URL leaks to the
// client).
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
	version, ok := requestedVersion(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return
	}

	member := fmt.Sprintf("%s.%s.jpg", key, cam)
	// Fast path: the client already has the member's tar byte range from the
	// shard index, so a bounded range GET avoids re-scanning the whole shard.
	// Fall back to the linear scan when the params are absent or unparseable.
	var reader io.Reader
	var closer io.Closer
	var size int64
	var err error
	if off, sz, rok := parseRange(r); rok {
		reader, closer, size, err = h.s3.StreamTarMemberRange(r.Context(), name, version, shard, off, sz)
	} else {
		reader, closer, size, err = h.s3.StreamTarMember(r.Context(), name, version, shard, member)
	}
	if err != nil {
		if errors.Is(err, service.ErrNotFound) {
			writeError(w, http.StatusNotFound, model.CodeNotFound, "image not found: "+member)
			return
		}
		// A crafted offset/size must not stream a whole shard as image/jpeg.
		if errors.Is(err, service.ErrRangeTooLarge) {
			writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "requested range too large")
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

// GetBlob handles GET /api/v1/datasets/{name}/shards/{shard}/blob?offset=&size=.
//
// It streams one CONTIGUOUS byte range of the shard tar — spanning several
// consecutive members (e.g. all camera JPEGs of a window of frames) — in a
// single S3 range GET. The client slices the individual JPEGs back out using
// the per-member offsets it already holds from the shard index. This collapses
// the ~6-GETs-per-frame image traffic into one request per playback window,
// which is what makes 10Hz playback fill its buffer over a high-latency link
// (each round trip is amortized across many frames instead of paid per image).
//
// The bytes are opaque here (tar headers + JPEG payloads interleaved), so the
// response is application/octet-stream; only the client, holding the index,
// knows where each member sits. maxBlobBytes caps a single span so a bad
// offset/size pair cannot ask the origin to stream an unbounded read.
func (h *DatasetsHandler) GetBlob(w http.ResponseWriter, r *http.Request) {
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
	version, ok := requestedVersion(r)
	if !ok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return
	}
	off, sz, rok := parseRange(r)
	if !rok {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "offset and size are required")
		return
	}
	// Reject an oversized span up front (avoids an S3 call); the service
	// enforces the same MaxRangeBytes cap for any caller as a backstop.
	if sz > service.MaxRangeBytes {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "requested range too large")
		return
	}

	reader, closer, size, err := h.s3.StreamTarMemberRange(r.Context(), name, version, shard, off, sz)
	if err != nil {
		if errors.Is(err, service.ErrNotFound) {
			writeError(w, http.StatusNotFound, model.CodeNotFound, "shard not found: "+shard)
			return
		}
		if errors.Is(err, service.ErrRangeTooLarge) {
			writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "requested range too large")
			return
		}
		slog.Error("stream shard blob", "dataset", name, "shard", shard, "offset", off, "size", sz, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to read shard range")
		return
	}
	defer closer.Close()

	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", size))
	// Immutable shard bytes at a fixed offset — cache aggressively at the edge.
	w.Header().Set("Cache-Control", "public, max-age=3600")
	w.WriteHeader(http.StatusOK)
	if _, err := io.Copy(w, reader); err != nil {
		slog.Warn("copy shard blob", "shard", shard, "offset", off, "error", err)
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
