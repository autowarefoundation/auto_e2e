package handler

import (
	"net/http/httptest"
	"testing"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
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

func TestIndexWithoutExactGeoDoesNotMutateCachedIndex(t *testing.T) {
	pose := &model.GeoPose{
		LatitudeDeg:           35.6812,
		LongitudeDeg:          139.7671,
		HeadingDegCWFromNorth: 90,
		TimestampNS:           42,
	}
	cached := &model.ShardIndex{
		Fps:     10,
		Version: "v2.1",
		Shard:   "train-000000.tar",
		Samples: []model.IndexSample{{
			Key:         "l2d-v1-e000001-f000042",
			SampleUID:   "l2d-v1-e000001-f000042",
			PoseCurrent: pose,
			Members: map[string]model.MemberRange{
				"cam_0.jpg": {Offset: 512, Size: 128},
				"pose.npy":  {Offset: 1024, Size: 36},
				"gps.npy":   {Offset: 1536, Size: 1040},
			},
		}},
	}

	redacted := indexWithoutExactGeo(cached)
	if redacted == cached {
		t.Fatal("redaction returned the cached index pointer")
	}
	if redacted.Samples[0].PoseCurrent != nil {
		t.Fatal("redacted index still contains exact pose")
	}
	if cached.Samples[0].PoseCurrent != pose {
		t.Fatal("redaction mutated the cached index")
	}
	if redacted.Samples[0].SampleUID != cached.Samples[0].SampleUID {
		t.Fatal("redaction changed non-sensitive sample fields")
	}
	if _, ok := redacted.Samples[0].Members["pose.npy"]; ok {
		t.Fatal("redacted index still contains pose member offset")
	}
	if _, ok := redacted.Samples[0].Members["gps.npy"]; ok {
		t.Fatal("redacted index still contains GPS member offset")
	}
	if _, ok := redacted.Samples[0].Members["cam_0.jpg"]; !ok {
		t.Fatal("redaction removed camera member")
	}
	if len(cached.Samples[0].Members) != 3 {
		t.Fatal("redaction mutated cached member map")
	}
}

func TestCameraMemberRangeRequiresMatchingSampleAndCamera(t *testing.T) {
	index := &model.ShardIndex{Samples: []model.IndexSample{{
		Key: "sample-a",
		Members: map[string]model.MemberRange{
			"cam_0.jpg": {Offset: 512, Size: 128},
		},
	}}}
	got, ok := cameraMemberRange(index, "sample-a", "cam_0.jpg")
	if !ok || got.Offset != 512 || got.Size != 128 {
		t.Fatalf("camera range = %+v, %v", got, ok)
	}
	if _, ok := cameraMemberRange(index, "sample-b", "cam_0.jpg"); ok {
		t.Fatal("matched a range from another sample")
	}
	if _, ok := cameraMemberRange(index, "sample-a", "cam_1.jpg"); ok {
		t.Fatal("matched a range from another camera")
	}
}

func TestRangeOverlapsExactGeo(t *testing.T) {
	index := &model.ShardIndex{Samples: []model.IndexSample{{
		Members: map[string]model.MemberRange{
			"cam_0.jpg": {Offset: 512, Size: 128},
			"pose.npy":  {Offset: 1024, Size: 36},
			"gps.npy":   {Offset: 1536, Size: 1040},
		},
	}}}
	tests := []struct {
		name   string
		offset int64
		size   int64
		want   bool
	}{
		{"camera only", 512, 128, false},
		{"ends at pose", 896, 128, false},
		{"starts at pose", 1024, 1, true},
		{"spans pose", 900, 200, true},
		{"starts at GPS", 1536, 1, true},
		{"after GPS", 2576, 10, false},
		{"invalid size", 0, 0, true},
		{"overflow", int64(^uint64(0) >> 1), 2, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := rangeOverlapsExactGeo(index, tt.offset, tt.size); got != tt.want {
				t.Fatalf("rangeOverlapsExactGeo(%d, %d) = %v, want %v", tt.offset, tt.size, got, tt.want)
			}
		})
	}
}
