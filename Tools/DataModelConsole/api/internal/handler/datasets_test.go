package handler

import (
	"net/http/httptest"
	"testing"
)

func TestValidShardName(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want bool
	}{
		{"plain shard name", "train-000000.tar", true},
		{"shard with underscores", "l2d_shard_000012.tar", true},
		{"traversal with slash", "../secrets.tar", false},
		{"traversal nested", "../../etc/passwd.tar", false},
		{"embedded slash", "a/b.tar", false},
		{"backslash traversal", "..\\secrets.tar", false},
		{"embedded backslash", "a\\b.tar", false},
		{"empty string", "", false},
		{"bare .tar", ".tar", false},
		{"missing .tar suffix", "train-000000", false},
		{"tar in the middle only", "train.tar.gz", false},
		{"dot-dot prefix but valid tar name", "..evil.tar", true}, // no separator: harmless as a single S3 key segment
		{"absolute unix path", "/etc/passwd.tar", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := validShardName(tt.in); got != tt.want {
				t.Errorf("validShardName(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestValidCam(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want bool
	}{
		{"cam_0", "cam_0", true},
		{"cam_6", "cam_6", true},
		{"multi-digit", "cam_12", true},
		{"prefix only", "cam_", false},
		{"non-digit suffix", "cam_x", false},
		{"mixed digit and letter", "cam_1a", false},
		{"empty", "", false},
		{"missing prefix", "0", false},
		{"wrong prefix", "camera_0", false},
		{"traversal in cam", "cam_0/../..", false},
		{"traversal replacing digits", "cam_../x", false},
		{"unicode digit rejected", "cam_٣", false}, // Arabic-Indic three, outside ASCII 0-9
		{"whitespace suffix", "cam_0 ", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := validCam(tt.in); got != tt.want {
				t.Errorf("validCam(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestParsePagination(t *testing.T) {
	tests := []struct {
		name       string
		query      string
		wantLimit  int
		wantOffset int
	}{
		{"defaults when absent", "", defaultLimit, 0},
		{"explicit values", "limit=10&offset=20", 10, 20},
		{"limit clamped to maxLimit", "limit=999999", maxLimit, 0},
		{"limit exactly maxLimit", "limit=1000", maxLimit, 0},
		{"limit just above maxLimit", "limit=1001", maxLimit, 0},
		{"zero limit falls back to default", "limit=0", defaultLimit, 0},
		{"negative limit falls back to default", "limit=-5", defaultLimit, 0},
		{"negative offset falls back to zero", "offset=-1", defaultLimit, 0},
		{"zero offset accepted", "offset=0", defaultLimit, 0},
		{"non-numeric limit ignored", "limit=abc", defaultLimit, 0},
		{"non-numeric offset ignored", "offset=abc", defaultLimit, 0},
		{"float limit ignored", "limit=1.5", defaultLimit, 0},
		{"huge offset passes through", "offset=123456789", defaultLimit, 123456789},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := httptest.NewRequest("GET", "/api/v1/datasets?"+tt.query, nil)
			limit, offset := parsePagination(r)
			if limit != tt.wantLimit {
				t.Errorf("parsePagination(%q) limit = %d, want %d", tt.query, limit, tt.wantLimit)
			}
			if offset != tt.wantOffset {
				t.Errorf("parsePagination(%q) offset = %d, want %d", tt.query, offset, tt.wantOffset)
			}
		})
	}
}
