// Package handler wires HTTP endpoints to the service layer.
package handler

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"strconv"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

const (
	defaultLimit = 50
	maxLimit     = 1000
)

// writeJSON marshals v and writes it with the given status.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Error("encode response", "error", err)
	}
}

// writeError emits the uniform error envelope {"error": ..., "code": ...}.
func writeError(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, model.ErrorResponse{Error: msg, Code: code})
}

// writeRawJSON passes through an upstream JSON body (proxy endpoints).
func writeRawJSON(w http.ResponseWriter, status int, body []byte) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if _, err := w.Write(body); err != nil {
		slog.Error("write proxied response", "error", err)
	}
}

// parsePagination reads ?limit=&offset= with defaults and bounds.
func parsePagination(r *http.Request) (limit, offset int) {
	limit = defaultLimit
	offset = 0
	if v := r.URL.Query().Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			limit = min(n, maxLimit)
		}
	}
	if v := r.URL.Query().Get("offset"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			offset = n
		}
	}
	return limit, offset
}

// parseRange reads the optional ?offset=&size= tar byte-range params used by
// the image endpoint's fast path. Both must be present and valid (offset>=0,
// size>0) for ok to be true; otherwise the caller falls back to a full scan.
func parseRange(r *http.Request) (offset, size int64, ok bool) {
	ov := r.URL.Query().Get("offset")
	sv := r.URL.Query().Get("size")
	if ov == "" || sv == "" {
		return 0, 0, false
	}
	off, err1 := strconv.ParseInt(ov, 10, 64)
	sz, err2 := strconv.ParseInt(sv, 10, 64)
	if err1 != nil || err2 != nil || off < 0 || sz <= 0 {
		return 0, 0, false
	}
	return off, sz, true
}
