package handler

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	internalauth "github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/auth"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

type fakeODDOperations struct {
	enabled      bool
	allowFull    bool
	retryCalls   int
	retryResult  service.ODDOperationResult
	retryError   error
	launchCalls  int
	launchResult service.ODDOperationResult
	launchError  error
}

func (f *fakeODDOperations) Enabled() bool {
	return f.enabled
}

func (f *fakeODDOperations) AllowFull() bool {
	return f.allowFull
}

func (f *fakeODDOperations) Launch(
	context.Context,
	string,
	string,
	string,
) (service.ODDOperationResult, error) {
	f.launchCalls++
	return f.launchResult, f.launchError
}

func (f *fakeODDOperations) Retry(
	_ context.Context,
	_ string,
) (service.ODDOperationResult, error) {
	f.retryCalls++
	return f.retryResult, f.retryError
}

func oddOperatorRequest(request *http.Request) *http.Request {
	return request.WithContext(internalauth.WithPrincipal(
		request.Context(),
		internalauth.Principal{
			Subject: "operator@example.test",
			Roles:   []string{"console-odd-operator"},
		},
	))
}

func TestODDOperationCapabilityRequiresVerifiedRole(t *testing.T) {
	operations := &fakeODDOperations{enabled: true, allowFull: true}
	handler := NewODDHandlerWithOperations(
		nil, operations, "console-odd-operator",
	)
	tests := []struct {
		name      string
		request   *http.Request
		permitted bool
		allowFull bool
	}{
		{
			name:    "anonymous",
			request: httptest.NewRequest(http.MethodGet, "/operations", nil),
		},
		{
			name: "viewer supplied role header",
			request: func() *http.Request {
				request := httptest.NewRequest(
					http.MethodGet, "/operations", nil,
				)
				request.Header.Set("X-Roles", "console-odd-operator")
				return request
			}(),
		},
		{
			name: "verified wrong role",
			request: func() *http.Request {
				request := httptest.NewRequest(
					http.MethodGet, "/operations", nil,
				)
				return request.WithContext(internalauth.WithPrincipal(
					request.Context(),
					internalauth.Principal{
						Subject: "viewer@example.test",
						Roles:   []string{"viewer"},
					},
				))
			}(),
		},
		{
			name:      "verified operator",
			request:   oddOperatorRequest(httptest.NewRequest(http.MethodGet, "/operations", nil)),
			permitted: true,
			allowFull: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.Operations(response, test.request)
			var body struct {
				Enabled   bool `json:"enabled"`
				Permitted bool `json:"permitted"`
				AllowFull bool `json:"allow_full"`
			}
			if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if response.Code != http.StatusOK ||
				!body.Enabled ||
				body.Permitted != test.permitted ||
				body.AllowFull != test.allowFull {
				t.Fatalf("capability response = %d, %+v", response.Code, body)
			}
		})
	}
}

func TestODDRetryFailsClosedAndAcceptsVerifiedOperator(t *testing.T) {
	operations := &fakeODDOperations{
		enabled: true,
		retryResult: service.ODDOperationResult{
			Action:      "retry",
			ExecutionID: "odr-retried",
			Created:     true,
		},
	}
	handler := NewODDHandlerWithOperations(
		nil, operations, "console-odd-operator",
	)
	body := `{"execution_id":"odd-failed"}`

	denied := httptest.NewRecorder()
	handler.Retry(
		denied,
		httptest.NewRequest(http.MethodPost, "/retry", strings.NewReader(body)),
	)
	if denied.Code != http.StatusForbidden || operations.retryCalls != 0 {
		t.Fatalf(
			"anonymous retry = %d, calls=%d",
			denied.Code, operations.retryCalls,
		)
	}

	accepted := httptest.NewRecorder()
	request := oddOperatorRequest(
		httptest.NewRequest(http.MethodPost, "/retry", strings.NewReader(body)),
	)
	handler.Retry(accepted, request)
	if accepted.Code != http.StatusAccepted || operations.retryCalls != 1 {
		t.Fatalf(
			"operator retry = %d, calls=%d, body=%s",
			accepted.Code, operations.retryCalls, accepted.Body.String(),
		)
	}

	invalid := httptest.NewRecorder()
	request = oddOperatorRequest(httptest.NewRequest(
		http.MethodPost,
		"/retry",
		strings.NewReader(`{"execution_id":"odd-failed","extra":true}`),
	))
	handler.Retry(invalid, request)
	if invalid.Code != http.StatusBadRequest || operations.retryCalls != 1 {
		t.Fatalf(
			"invalid retry = %d, calls=%d",
			invalid.Code, operations.retryCalls,
		)
	}
}
