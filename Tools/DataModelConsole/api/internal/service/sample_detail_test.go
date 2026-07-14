package service

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestParseSampleKey(t *testing.T) {
	tests := []struct {
		name    string
		key     string
		wantEp  string
		wantIdx int
		wantOK  bool
	}{
		{"l2d ep prefix stripped", "ep0_000064", "0", 64, true},
		{"l2d multi-digit episode", "ep12_000100", "12", 100, true},
		{"nvidia hex hash kept verbatim", "25cd4769_000064", "25cd4769", 64, true},
		{"pipeline flat s-index parses frame", "s00000064", "", 64, true},
		{"pipeline flat s-index zero", "s00000000", "", 0, true},
		{"non-numeric frame suffix is not a frame key", "ep0_abc", "", 0, false},
		{"ep prefix with non-digit rest kept whole", "epX_000001", "epX", 1, true},
		{"bare non-s non-underscore key", "garbage", "", 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ep, idx, ok := parseSampleKey(tt.key)
			if ep != tt.wantEp || idx != tt.wantIdx || ok != tt.wantOK {
				t.Errorf("parseSampleKey(%q) = (%q, %d, %v), want (%q, %d, %v)",
					tt.key, ep, idx, ok, tt.wantEp, tt.wantIdx, tt.wantOK)
			}
		})
	}
}

func TestDecodeFloat32LE(t *testing.T) {
	want := []float32{1.5, -2.25, 0, 3.14159}
	buf := make([]byte, len(want)*4)
	for i, f := range want {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	got := decodeFloat32LE(buf)
	if len(got) != len(want) {
		t.Fatalf("decoded %d floats, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("float[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	// Trailing partial float is ignored, not panicked on.
	if got := decodeFloat32LE(buf[:6]); len(got) != 1 {
		t.Errorf("partial buffer decoded %d floats, want 1", len(got))
	}
	if got := decodeFloat32LE(nil); len(got) != 0 {
		t.Errorf("nil buffer decoded %d floats, want 0", len(got))
	}
}

// TestTripFrameFromMeta validates that BuildShardIndex reads the trip-global
// frame index from meta.json (its frame_idx), which differs from the
// intra-shard playback ordinal derived from the key suffix. A key like
// "x_000000" gives FrameIdx 0 while its meta.json frame_idx may be 64.
func TestTripFrameFromMeta(t *testing.T) {
	tests := []struct {
		name   string
		body   string
		want   int
		wantOK bool
	}{
		{"trip frame present", `{"frame_idx": 64}`, 64, true},
		{"trip frame zero present", `{"frame_idx": 0}`, 0, true},
		{"other fields ignored", `{"episode": "abc", "frame_idx": 12, "t": 1.5}`, 12, true},
		{"no frame_idx field", `{"episode": "abc"}`, 0, false},
		{"malformed json", `not json`, 0, false},
		{"empty body", ``, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := tripFrameFromMeta([]byte(tt.body))
			if got != tt.want || ok != tt.wantOK {
				t.Errorf("tripFrameFromMeta(%q) = (%d, %v), want (%d, %v)",
					tt.body, got, ok, tt.want, tt.wantOK)
			}
		})
	}
}

func TestMemberSuffixOf(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"ep0_000064.cam_0.jpg", "cam_0.jpg"},
		{"ep0_000064.ego.npy", "ego.npy"},
		{"ep0_000064.meta.json", "meta.json"},
		{"a/b/ep1_000001.cam_2.jpg", "cam_2.jpg"},
		{"README", ""},
		{".hidden", ""},
	}
	for _, tt := range tests {
		if got := memberSuffixOf(tt.in); got != tt.want {
			t.Errorf("memberSuffixOf(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
