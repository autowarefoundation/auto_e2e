package handler

import (
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestWriteJSONDoesNotCommitSuccessBeforeMarshal(t *testing.T) {
	response := httptest.NewRecorder()

	writeJSON(response, http.StatusOK, map[string]float64{
		"invalid": math.NaN(),
	})

	if response.Code != http.StatusInternalServerError {
		t.Fatalf(
			"status = %d, want %d",
			response.Code,
			http.StatusInternalServerError,
		)
	}
	if response.Body.String() !=
		`{"error":"failed to encode response","code":"INTERNAL_ERROR"}` {
		t.Fatalf("body = %q", response.Body.String())
	}
}
