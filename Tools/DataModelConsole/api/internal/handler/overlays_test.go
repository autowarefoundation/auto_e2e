package handler

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestExactGeoAuthorized(t *testing.T) {
	tests := []struct {
		name         string
		enabled      bool
		requiredRole string
		roles        string
		want         bool
	}{
		{"disabled", false, "console-exact-geo", "console-exact-geo", false},
		{"required role unset", true, "", "console-exact-geo", false},
		{"header absent", true, "console-exact-geo", "", false},
		{"wrong role", true, "console-exact-geo", "viewer", false},
		{"substring rejected", true, "console-exact-geo", "console-exact-geo-admin", false},
		{"role match", true, "console-exact-geo", "console-exact-geo", true},
		{"comma separated role match", true, "console-exact-geo", "viewer, console-exact-geo", true},
		{"role match is case sensitive", true, "console-exact-geo", "Console-Exact-Geo", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := httptest.NewRequest("GET", "/api/v1/datasets/l2d/geo/episodes/e1", nil)
			if tt.roles != "" {
				request.Header.Set("X-Console-Roles", tt.roles)
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
