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
	}{
		{"l2d ep prefix stripped", "ep0_000064", "0", 64},
		{"l2d multi-digit episode", "ep12_000100", "12", 100},
		{"nvidia hex hash kept verbatim", "25cd4769_000064", "25cd4769", 64},
		{"pipeline flat s-index parses frame", "s00000064", "", 64},
		{"pipeline flat s-index zero", "s00000000", "", 0},
		{"non-numeric frame suffix", "ep0_abc", "0", 0},
		{"ep prefix with non-digit rest kept whole", "epX_000001", "epX", 1},
		{"bare non-s non-underscore key", "garbage", "", 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ep, idx := parseSampleKey(tt.key)
			if ep != tt.wantEp || idx != tt.wantIdx {
				t.Errorf("parseSampleKey(%q) = (%q, %d), want (%q, %d)",
					tt.key, ep, idx, tt.wantEp, tt.wantIdx)
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
