package handler

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	internalauth "github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/auth"
)

func TestExactGeoAuthorized(t *testing.T) {
	tests := []struct {
		name         string
		enabled      bool
		requiredRole string
		subject      string
		roles        []string
		headerRoles  string
		want         bool
	}{
		{
			name:         "forged header rejected",
			enabled:      true,
			requiredRole: "console-exact-geo",
			headerRoles:  "console-exact-geo",
		},
		{
			name:         "disabled",
			requiredRole: "console-exact-geo",
			subject:      "user-1",
			roles:        []string{"console-exact-geo"},
		},
		{
			name:    "required role unset",
			enabled: true,
			subject: "user-1",
			roles:   []string{"console-exact-geo"},
		},
		{
			name:         "subject required",
			enabled:      true,
			requiredRole: "console-exact-geo",
			roles:        []string{"console-exact-geo"},
		},
		{
			name:         "wrong role",
			enabled:      true,
			requiredRole: "console-exact-geo",
			subject:      "user-1",
			roles:        []string{"viewer"},
		},
		{
			name:         "substring rejected",
			enabled:      true,
			requiredRole: "console-exact-geo",
			subject:      "user-1",
			roles:        []string{"console-exact-geo-admin"},
		},
		{
			name:         "verified role match",
			enabled:      true,
			requiredRole: "console-exact-geo",
			subject:      "user-1",
			roles:        []string{"viewer", "console-exact-geo"},
			want:         true,
		},
		{
			name:         "role match is case sensitive",
			enabled:      true,
			requiredRole: "console-exact-geo",
			subject:      "user-1",
			roles:        []string{"Console-Exact-Geo"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := httptest.NewRequest("GET", "/api/v1/datasets/l2d/geo/episodes/e1", nil)
			if tt.headerRoles != "" {
				request.Header.Set("X-Console-Roles", tt.headerRoles)
			}
			if tt.subject != "" || len(tt.roles) > 0 {
				request = request.WithContext(internalauth.WithPrincipal(
					request.Context(),
					internalauth.Principal{
						Subject: tt.subject,
						Roles:   tt.roles,
					},
				))
			}
			got := exactGeoAuthorized(request, tt.enabled, tt.requiredRole)
			if got != tt.want {
				t.Fatalf("exactGeoAuthorized() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestValidArtifactID(t *testing.T) {
	valid := strings.Repeat("0a", 32)
	if !validArtifactID(valid) {
		t.Fatal("valid lowercase SHA-256 was rejected")
	}
	for _, value := range []string{
		"",
		strings.Repeat("a", 63),
		strings.Repeat("a", 65),
		strings.Repeat("A", 64),
		strings.Repeat("g", 64),
		strings.Repeat("/", 64),
	} {
		if validArtifactID(value) {
			t.Fatalf("invalid artifact id %q was accepted", value)
		}
	}
}

func TestParseOverlayModelsPage(t *testing.T) {
	token := strings.Repeat("a", 64)
	for _, test := range []struct {
		name      string
		query     string
		wantLimit int
		wantToken string
		wantOK    bool
	}{
		{
			name:      "defaults",
			wantLimit: 100,
			wantOK:    true,
		},
		{
			name:      "minimum limit",
			query:     "limit=1",
			wantLimit: 1,
			wantOK:    true,
		},
		{
			name:      "maximum limit and token",
			query:     "limit=100&page_token=" + token,
			wantLimit: 100,
			wantToken: token,
			wantOK:    true,
		},
		{name: "zero limit", query: "limit=0"},
		{name: "negative limit", query: "limit=-1"},
		{name: "oversized limit", query: "limit=101"},
		{name: "non-numeric limit", query: "limit=many"},
		{name: "short token", query: "page_token=abc"},
		{
			name:  "uppercase token",
			query: "page_token=" + strings.Repeat("A", 64),
		},
		{
			name:  "non-hex token",
			query: "page_token=" + strings.Repeat("g", 64),
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(
				http.MethodGet, "/overlay-models?"+test.query, nil,
			)
			limit, pageToken, ok := parseOverlayModelsPage(request)
			if ok != test.wantOK ||
				limit != test.wantLimit ||
				pageToken != test.wantToken {
				t.Fatalf(
					"parseOverlayModelsPage() = (%d, %q, %v), want (%d, %q, %v)",
					limit,
					pageToken,
					ok,
					test.wantLimit,
					test.wantToken,
					test.wantOK,
				)
			}
		})
	}
}

func TestRigCacheControlRequiresExplicitVersion(t *testing.T) {
	tests := []struct {
		name             string
		requestedVersion string
		want             string
	}{
		{name: "auto-resolved", want: "no-store"},
		{
			name:             "explicit",
			requestedVersion: "v2.1",
			want:             "private, max-age=31536000, immutable",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response := httptest.NewRecorder()

			setRigCacheControl(response, tt.requestedVersion)

			if got := response.Header().Get("Cache-Control"); got != tt.want {
				t.Fatalf("Cache-Control = %q, want %q", got, tt.want)
			}
			if response.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d", response.Code, http.StatusOK)
			}
		})
	}
}

func TestOverlayCacheControlRequiresExplicitVersion(t *testing.T) {
	tests := []struct {
		name             string
		requestedVersion string
		want             string
	}{
		{name: "auto-resolved", want: "no-store"},
		{
			name:             "explicit",
			requestedVersion: "v2.1",
			want:             "private, max-age=31536000, immutable",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response := httptest.NewRecorder()

			setOverlayCacheControl(response, tt.requestedVersion)

			if got := response.Header().Get("Cache-Control"); got != tt.want {
				t.Fatalf("Cache-Control = %q, want %q", got, tt.want)
			}
		})
	}
}
