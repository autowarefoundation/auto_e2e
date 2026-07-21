package handler

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

func TestWriteReasoningAvailabilityError(t *testing.T) {
	tests := []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
		wantRetry  string
	}{
		{
			name: "inventory unavailable",
			err: fmt.Errorf(
				"read inventory: %w",
				service.ErrReasoningUnavailable,
			),
			wantStatus: http.StatusServiceUnavailable,
			wantCode:   model.CodeUnavailable,
			wantRetry:  "60",
		},
		{
			name: "publication integrity",
			err: fmt.Errorf(
				"read stats: %w",
				service.ErrReasoningIntegrity,
			),
			wantStatus: http.StatusBadGateway,
			wantCode:   model.CodeS3Error,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			if !writeReasoningAvailabilityError(response, test.err) {
				t.Fatal("reasoning error was not handled")
			}
			if response.Code != test.wantStatus {
				t.Fatalf(
					"status = %d, want %d",
					response.Code,
					test.wantStatus,
				)
			}
			var body model.ErrorResponse
			if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if body.Code != test.wantCode {
				t.Fatalf("code = %q, want %q", body.Code, test.wantCode)
			}
			if got := response.Header().Get("Retry-After"); got != test.wantRetry {
				t.Fatalf(
					"Retry-After = %q, want %q",
					got,
					test.wantRetry,
				)
			}
		})
	}
}

func TestWriteReasoningAvailabilityErrorLeavesNotFoundToCaller(
	t *testing.T,
) {
	response := httptest.NewRecorder()
	if writeReasoningAvailabilityError(response, service.ErrNotFound) {
		t.Fatal("not-found error was handled as availability failure")
	}
}
