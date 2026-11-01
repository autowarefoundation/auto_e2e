package handler

import (
	"errors"
	"log/slog"
	"net/http"
	"regexp"
	"strings"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

// maxODDConditions bounds one search; a scene carries about 50 ODD keys.
const maxODDConditions = 16

var (
	oddKeyPattern   = regexp.MustCompile(`^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$`)
	oddValuePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_]*$`)
)

// ODDHandler serves the ODD scene search.
type ODDHandler struct {
	s3 *service.S3Service
}

// NewODDHandler builds the ODD handler.
func NewODDHandler(s3 *service.S3Service) *ODDHandler {
	return &ODDHandler{s3: s3}
}

// parseODDConditions reads repeated where=<key>=<value> terms.
func parseODDConditions(raw []string) ([]model.ODDCondition, bool) {
	if len(raw) > maxODDConditions {
		return nil, false
	}
	conditions := make([]model.ODDCondition, 0, len(raw))
	for _, term := range raw {
		key, value, ok := strings.Cut(term, "=")
		if !ok || len(key) > 128 || len(value) > 64 ||
			!oddKeyPattern.MatchString(key) ||
			!oddValuePattern.MatchString(value) {
			return nil, false
		}
		conditions = append(conditions, model.ODDCondition{Key: key, Value: value})
	}
	return conditions, true
}

// Search handles
// GET /api/v1/odd/scenes?dataset=&version=&where=<key>=<value>&limit=&offset=
// — the scenes whose valid ODD labels satisfy every where term.
func (h *ODDHandler) Search(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	dataset := q.Get("dataset")
	if !h.s3.ValidDataset(dataset) {
		writeError(w, http.StatusNotFound, model.CodeNotFound, "unknown dataset: "+dataset)
		return
	}
	version := q.Get("version")
	if version != "" && !service.ValidVersion(version) {
		writeError(w, http.StatusBadRequest, model.CodeInvalidParam, "invalid version")
		return
	}
	where, ok := parseODDConditions(q["where"])
	if !ok {
		writeError(
			w, http.StatusBadRequest, model.CodeInvalidParam,
			"where must be at most 16 terms of the form <odd.key>=<value>",
		)
		return
	}
	limit, offset := parsePagination(r)

	response, err := h.s3.SearchODDScenes(
		r.Context(), dataset, version, where, limit, offset,
	)
	if errors.Is(err, service.ErrNotFound) {
		writeError(
			w, http.StatusNotFound, model.CodeNotFound,
			"no ODD labels for this dataset version",
		)
		return
	}
	if err != nil {
		slog.Error("odd scene search", "dataset", dataset, "version", version, "error", err)
		writeError(w, http.StatusBadGateway, model.CodeS3Error, "failed to read ODD labels")
		return
	}
	writeJSON(w, http.StatusOK, response)
}
