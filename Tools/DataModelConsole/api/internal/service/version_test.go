package service

import "testing"

func TestIsVersionDir(t *testing.T) {
	cases := map[string]bool{
		"v1":       true,
		"v1.0":     true,
		"v2.0":     true,
		"v10":      true,
		"v1.10":    true,
		"v1.2.3":   true,
		"":         false,
		"v":        false,
		"1.0":      false,
		"vx":       false,
		"v1.x":     false,
		"shards":   false,
		"v1.0-rc1": false,
	}
	for in, want := range cases {
		if got := isVersionDir(in); got != want {
			t.Errorf("isVersionDir(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestVersionLess(t *testing.T) {
	// versionLess(a, b) == a is older than b.
	cases := []struct {
		a, b string
		want bool
	}{
		{"v1.0", "v2.0", true},
		{"v2.0", "v1.0", false},
		{"v9", "v10", true},  // numeric, not lexical
		{"v10", "v9", false}, // v10 is newer
		{"v1.2", "v1.10", true},
		{"v1.10", "v1.2", false},
		{"v1.0", "v1.0", false},
		{"v1", "v1.0", true}, // fewer components sorts older
	}
	for _, c := range cases {
		if got := versionLess(c.a, c.b); got != c.want {
			t.Errorf("versionLess(%q, %q) = %v, want %v", c.a, c.b, got, c.want)
		}
	}
}

// TestNewestSelection mirrors discoverNewestVersion's sort: the greatest
// version must come first after sorting newest-first.
func TestNewestSelection(t *testing.T) {
	versions := []string{"v1.0", "v2.0", "v1.10", "v1.2"}
	// newest-first (same comparator as discoverNewestVersion)
	for i := 0; i < len(versions); i++ {
		for j := i + 1; j < len(versions); j++ {
			if versionLess(versions[i], versions[j]) {
				versions[i], versions[j] = versions[j], versions[i]
			}
		}
	}
	if versions[0] != "v2.0" {
		t.Errorf("newest = %q, want v2.0 (order: %v)", versions[0], versions)
	}
}
