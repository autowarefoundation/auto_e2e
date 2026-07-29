package handler

import (
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"

	internalauth "github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/auth"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

const maxODDOperationBody = 4 << 10

type oddLaunchRequest struct {
	Dataset          string `json:"dataset"`
	Version          string `json:"version"`
	PublicationScope string `json:"publication_scope"`
}

type oddRetryRequest struct {
	ExecutionID string `json:"execution_id"`
}

func (h *ODDHandler) operationsPermitted(r *http.Request) bool {
	return h.operations != nil &&
		h.operations.Enabled() &&
		internalauth.HasRole(r.Context(), h.operationsRequiredRole)
}

// Operations returns only the requester's effective operation capability.
func (h *ODDHandler) Operations(w http.ResponseWriter, r *http.Request) {
	permitted := h.operationsPermitted(r)
	writeJSON(w, http.StatusOK, map[string]any{
		"enabled":    h.operations != nil && h.operations.Enabled(),
		"permitted":  permitted,
		"allow_full": permitted && h.operations.AllowFull(),
	})
}

func decodeODDOperationRequest(
	w http.ResponseWriter,
	r *http.Request,
	value any,
) error {
	body := http.MaxBytesReader(w, r.Body, maxODDOperationBody)
	defer body.Close()
	decoder := json.NewDecoder(body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("request body must contain one JSON object")
		}
		return err
	}
	return nil
}

func (h *ODDHandler) requireOperations(
	w http.ResponseWriter,
	r *http.Request,
) bool {
	if h.operationsPermitted(r) {
		return true
	}
	writeError(
		w,
		http.StatusForbidden,
		model.CodeUnavailable,
		"ODD operations are not permitted",
	)
	return false
}

func writeODDOperationError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, service.ErrODDOperationsDisabled),
		errors.Is(err, service.ErrODDFullRunsDisabled):
		writeError(
			w, http.StatusForbidden, model.CodeUnavailable,
			"ODD operation is disabled",
		)
	case errors.Is(err, service.ErrODDOperationConflict):
		writeError(
			w, http.StatusConflict, model.CodeUnavailable,
			"another ODD Dataset Labeler run is active",
		)
	case errors.Is(err, service.ErrODDInvalidOperation):
		writeError(
			w, http.StatusBadRequest, model.CodeInvalidParam,
			"invalid ODD operation",
		)
	default:
		slog.Error("execute ODD operation", "error", err)
		writeError(
			w, http.StatusBadGateway, model.CodeUpstream,
			"ODD Dataset Labeler launch failed",
		)
	}
}

// Launch starts only the standalone Dataset Labeler LaunchPlan.
func (h *ODDHandler) Launch(w http.ResponseWriter, r *http.Request) {
	if !h.requireOperations(w, r) {
		return
	}
	var request oddLaunchRequest
	if err := decodeODDOperationRequest(w, r, &request); err != nil {
		writeError(
			w, http.StatusBadRequest, model.CodeInvalidParam,
			"invalid ODD launch request",
		)
		return
	}
	if request.Dataset == "" ||
		request.Version == "" ||
		request.Version == "latest" ||
		!validReasoningParam(request.Dataset) ||
		!validReasoningParam(request.Version) ||
		!h.s3.ValidDataset(request.Dataset) ||
		(request.PublicationScope != "smoke" &&
			request.PublicationScope != "full") {
		writeError(
			w, http.StatusBadRequest, model.CodeInvalidParam,
			"invalid ODD launch coordinate",
		)
		return
	}
	resolved, err := h.s3.ResolvedVersion(
		r.Context(), request.Dataset, request.Version,
	)
	if err != nil || resolved != request.Version {
		writeError(
			w, http.StatusBadRequest, model.CodeInvalidParam,
			"dataset publication is not ready",
		)
		return
	}
	result, err := h.operations.Launch(
		r.Context(),
		request.Dataset,
		request.Version,
		request.PublicationScope,
	)
	if err != nil {
		writeODDOperationError(w, err)
		return
	}
	status := http.StatusAccepted
	if !result.Created {
		status = http.StatusOK
	}
	writeJSON(w, status, result)
}

// Retry relaunches only a terminal failed Dataset Labeler execution.
func (h *ODDHandler) Retry(w http.ResponseWriter, r *http.Request) {
	if !h.requireOperations(w, r) {
		return
	}
	var request oddRetryRequest
	if err := decodeODDOperationRequest(w, r, &request); err != nil ||
		!validFlyteExecutionID(request.ExecutionID) {
		writeError(
			w, http.StatusBadRequest, model.CodeInvalidParam,
			"invalid ODD retry request",
		)
		return
	}
	result, err := h.operations.Retry(r.Context(), request.ExecutionID)
	if err != nil {
		writeODDOperationError(w, err)
		return
	}
	status := http.StatusAccepted
	if !result.Created {
		status = http.StatusOK
	}
	writeJSON(w, status, result)
}
