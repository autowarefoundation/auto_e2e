package service

import (
	"archive/tar"
	"bytes"
	"io"
	"math"
	"strings"
	"testing"
)

func TestSampleKeyOf(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "normal webdataset member",
			in:   "ep0_000064.cam_0.jpg",
			want: "ep0_000064",
		},
		{
			name: "single extension",
			in:   "ep0_000064.json",
			want: "ep0_000064",
		},
		{
			name: "no dot returns whole base",
			in:   "README",
			want: "README",
		},
		{
			name: "hidden file (leading dot) is not truncated to empty key",
			in:   ".hidden",
			want: ".hidden",
		},
		{
			name: "hidden file with extension keeps full base (i==0 guard)",
			in:   ".hidden.json",
			want: ".hidden.json",
		},
		{
			name: "nested path uses base name",
			in:   "a/b/ep1_000001.cam_2.jpg",
			want: "ep1_000001",
		},
		{
			name: "trailing dot",
			in:   "name.",
			want: "name",
		},
		{
			name: "empty name degrades to path.Base dot",
			in:   "",
			want: ".",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := sampleKeyOf(tt.in); got != tt.want {
				t.Errorf("sampleKeyOf(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestCacheDataset(t *testing.T) {
	tests := []struct{ in, want string }{
		{"nvidia_av", "nvidia_PhysicalAI-Autonomous-Vehicles"},
		{"l2d", "yaak-ai_L2D"},
		{"nvidia_PhysicalAI-Autonomous-Vehicles", "nvidia_PhysicalAI-Autonomous-Vehicles"}, // already a partition name
		{"unknown", "unknown"},
	}
	for _, tt := range tests {
		if got := cacheDataset(tt.in); got != tt.want {
			t.Errorf("cacheDataset(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestCacheSampleID(t *testing.T) {
	tests := []struct {
		in     string
		want   string
		wantOK bool
	}{
		{"25cd4769_000000", "s00000000", true}, // nvidia hash-prefixed -> flat index
		{"25cd4769_000012", "s00000012", true},
		{"ep0_000064", "s00000064", true}, // L2D episode-prefixed
		{"s00000000", "s00000000", true},  // already flat: passthrough
		{"s00000123", "s00000123", true},
		{"totally_garbage", "", false}, // underscore, non-numeric suffix -> not a frame key
		{"hello", "", false},           // no underscore, not s<digits> -> not a frame key
	}
	for _, tt := range tests {
		got, ok := cacheSampleID(tt.in)
		if got != tt.want || ok != tt.wantOK {
			t.Errorf("cacheSampleID(%q) = (%q, %v), want (%q, %v)", tt.in, got, ok, tt.want, tt.wantOK)
		}
	}
}

func TestParseReasoningKey(t *testing.T) {
	tests := []struct {
		name        string
		key         string
		wantDataset string
		wantTeacher string
		wantPV      string
		wantOK      bool
	}{
		{
			name:        "valid key",
			key:         "reasoning_labels_cache/dataset=l2d/teacher=mock/prompt_version=v3/ep0_000064.json",
			wantDataset: "l2d",
			wantTeacher: "mock",
			wantPV:      "v3",
			wantOK:      true,
		},
		{
			name:   "missing teacher and prompt_version partitions",
			key:    "reasoning_labels_cache/dataset=l2d/ep0_000064.json",
			wantOK: false,
		},
		{
			name:   "missing prompt_version partition",
			key:    "reasoning_labels_cache/dataset=l2d/teacher=mock/ep0_000064.json",
			wantOK: false,
		},
		{
			name:   "malformed prefix (different root)",
			key:    "other_prefix/dataset=l2d/teacher=mock/prompt_version=v3/ep0_000064.json",
			wantOK: false,
		},
		{
			name:   "partitions out of order",
			key:    "reasoning_labels_cache/teacher=mock/dataset=l2d/prompt_version=v3/ep0_000064.json",
			wantOK: false,
		},
		{
			name:   "plain object without partition markers",
			key:    "reasoning_labels_cache/a/b/c/d.json",
			wantOK: false,
		},
		{
			name:   "empty key",
			key:    "",
			wantOK: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds, teacher, pv, ok := parseReasoningKey(tt.key)
			if ok != tt.wantOK {
				t.Fatalf("parseReasoningKey(%q) ok = %v, want %v", tt.key, ok, tt.wantOK)
			}
			if !tt.wantOK {
				if ds != "" || teacher != "" || pv != "" {
					t.Errorf("parseReasoningKey(%q) non-empty values on !ok: (%q,%q,%q)", tt.key, ds, teacher, pv)
				}
				return
			}
			if ds != tt.wantDataset || teacher != tt.wantTeacher || pv != tt.wantPV {
				t.Errorf("parseReasoningKey(%q) = (%q,%q,%q), want (%q,%q,%q)",
					tt.key, ds, teacher, pv, tt.wantDataset, tt.wantTeacher, tt.wantPV)
			}
		})
	}
}

func TestPaginate(t *testing.T) {
	items := func(n int) []int {
		out := make([]int, n)
		for i := range out {
			out[i] = i
		}
		return out
	}

	tests := []struct {
		name       string
		total      int
		limit      int
		offset     int
		wantItems  []int
		wantLimit  int
		wantOffset int
		wantMore   bool
	}{
		{
			name:       "offset beyond total returns empty page",
			total:      5,
			limit:      10,
			offset:     100,
			wantItems:  []int{},
			wantLimit:  10,
			wantOffset: 5, // clamped to total
			wantMore:   false,
		},
		{
			name:       "zero limit falls back to default 50",
			total:      5,
			limit:      0,
			offset:     0,
			wantItems:  []int{0, 1, 2, 3, 4},
			wantLimit:  50,
			wantOffset: 0,
			wantMore:   false,
		},
		{
			name:       "negative limit falls back to default 50",
			total:      3,
			limit:      -1,
			offset:     0,
			wantItems:  []int{0, 1, 2},
			wantLimit:  50,
			wantOffset: 0,
			wantMore:   false,
		},
		{
			name:       "exact boundary: last full page has More=false",
			total:      10,
			limit:      5,
			offset:     5,
			wantItems:  []int{5, 6, 7, 8, 9},
			wantLimit:  5,
			wantOffset: 5,
			wantMore:   false,
		},
		{
			name:       "first page of two has More=true",
			total:      10,
			limit:      5,
			offset:     0,
			wantItems:  []int{0, 1, 2, 3, 4},
			wantLimit:  5,
			wantOffset: 0,
			wantMore:   true,
		},
		{
			name:       "partial trailing page",
			total:      7,
			limit:      5,
			offset:     5,
			wantItems:  []int{5, 6},
			wantLimit:  5,
			wantOffset: 5,
			wantMore:   false,
		},
		{
			name:       "negative offset clamped to zero",
			total:      4,
			limit:      2,
			offset:     -3,
			wantItems:  []int{0, 1},
			wantLimit:  2,
			wantOffset: 0,
			wantMore:   true,
		},
		{
			name:       "offset equal to total returns empty, More=false",
			total:      4,
			limit:      2,
			offset:     4,
			wantItems:  []int{},
			wantLimit:  2,
			wantOffset: 4,
			wantMore:   false,
		},
		{
			name:       "empty input",
			total:      0,
			limit:      10,
			offset:     0,
			wantItems:  []int{},
			wantLimit:  10,
			wantOffset: 0,
			wantMore:   false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := items(tt.total)
			got, page := paginate(in, tt.limit, tt.offset, tt.total)

			if len(got) != len(tt.wantItems) {
				t.Fatalf("paginate items len = %d, want %d (got %v)", len(got), len(tt.wantItems), got)
			}
			for i := range got {
				if got[i] != tt.wantItems[i] {
					t.Fatalf("paginate items = %v, want %v", got, tt.wantItems)
				}
			}
			if page.Limit != tt.wantLimit {
				t.Errorf("page.Limit = %d, want %d", page.Limit, tt.wantLimit)
			}
			if page.Offset != tt.wantOffset {
				t.Errorf("page.Offset = %d, want %d", page.Offset, tt.wantOffset)
			}
			if page.Total != tt.total {
				t.Errorf("page.Total = %d, want %d", page.Total, tt.total)
			}
			if page.More != tt.wantMore {
				t.Errorf("page.More = %v, want %v", page.More, tt.wantMore)
			}
		})
	}
}

// TestPaginate_HugeOffsetDoesNotPanic is a regression test for an integer
// overflow: paginate computes end := offset + limit BEFORE clamping offset to
// total, so an offset near MaxInt overflows end to a negative value and
// items[offset:end] panics with "slice bounds out of range". The offset is
// remotely attacker-controlled: handler.parsePagination accepts any
// non-negative int (?offset=9223372036854775807 parses fine on 64-bit), so
// GET /api/v1/datasets/l2d/shards?offset=9223372036854775807 panics per
// request (recovered by chi middleware.Recoverer into a 500, with a stack
// trace logged). Correct behavior: return an empty page with More=false.
// This test FAILS until the production fix lands (clamp offset before
// computing end, and guard end < offset for overflow).
func TestPaginate_HugeOffsetDoesNotPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("paginate(items, 50, math.MaxInt, 5) panicked: %v (int overflow in end := offset + limit)", r)
		}
	}()
	items := []int{0, 1, 2, 3, 4}
	got, page := paginate(items, 50, math.MaxInt, len(items))
	if len(got) != 0 {
		t.Errorf("expected empty page for huge offset, got %v", got)
	}
	if page.More {
		t.Errorf("page.More = true, want false for offset beyond total")
	}
}

func TestCountingReader_AccumulatesAcrossReads(t *testing.T) {
	src := strings.Repeat("x", 1000)
	cr := &countingReader{r: strings.NewReader(src)}

	buf := make([]byte, 137) // deliberately not a divisor of 1000
	var total int64
	for {
		n, err := cr.Read(buf)
		total += int64(n)
		if cr.n != total {
			t.Fatalf("countingReader.n = %d after reading %d bytes total", cr.n, total)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("unexpected read error: %v", err)
		}
	}
	if total != int64(len(src)) {
		t.Fatalf("read %d bytes, want %d", total, len(src))
	}
	if cr.n != int64(len(src)) {
		t.Fatalf("countingReader.n = %d, want %d", cr.n, len(src))
	}

	// EOF reads after exhaustion must not change the count.
	if n, err := cr.Read(buf); n != 0 || err != io.EOF {
		t.Fatalf("post-EOF read = (%d, %v), want (0, EOF)", n, err)
	}
	if cr.n != int64(len(src)) {
		t.Fatalf("countingReader.n changed after EOF read: %d", cr.n)
	}
}

// TestCountingReader_TarDataOffsets validates the invariant ListSamples relies
// on: after tar.Reader.Next() the counting reader's n is exactly the byte
// offset of the member's data within the tar stream, so raw[n:n+size]
// reproduces the member content (the basis for Phase 2 range-GET extraction).
func TestCountingReader_TarDataOffsets(t *testing.T) {
	files := []struct {
		name string
		body string
	}{
		{"ep0_000000.cam_0.jpg", strings.Repeat("A", 700)}, // spans >1 block, forces padding
		{"ep0_000000.json", `{"speed": 1.0}`},
		{"ep0_000001.cam_0.jpg", strings.Repeat("B", 512)}, // exact block size
	}

	var raw bytes.Buffer
	tw := tar.NewWriter(&raw)
	for _, f := range files {
		if err := tw.WriteHeader(&tar.Header{
			Name:     f.name,
			Mode:     0o644,
			Size:     int64(len(f.body)),
			Typeflag: tar.TypeReg,
		}); err != nil {
			t.Fatalf("write header %s: %v", f.name, err)
		}
		if _, err := tw.Write([]byte(f.body)); err != nil {
			t.Fatalf("write body %s: %v", f.name, err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("close tar writer: %v", err)
	}
	rawBytes := raw.Bytes()

	cr := &countingReader{r: bytes.NewReader(rawBytes)}
	tr := tar.NewReader(cr)
	i := 0
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("tar next: %v", err)
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		if i >= len(files) {
			t.Fatalf("unexpected extra member %q", hdr.Name)
		}
		offset := cr.n // same accounting as ListSamples
		want := files[i].body
		if hdr.Size != int64(len(want)) {
			t.Errorf("member %s size = %d, want %d", hdr.Name, hdr.Size, len(want))
		}
		got := string(rawBytes[offset : offset+hdr.Size])
		if got != want {
			t.Errorf("member %s: raw[%d:%d] does not match member content", hdr.Name, offset, offset+hdr.Size)
		}
		i++
	}
	if i != len(files) {
		t.Fatalf("iterated %d members, want %d", i, len(files))
	}
}
