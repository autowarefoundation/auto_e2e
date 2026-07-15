package service

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestValidDatasetIncludesPublishedKITScenesAlias(t *testing.T) {
	service := &S3Service{}
	for _, dataset := range []string{"kitscenes", "l2d", "nvidia_av"} {
		if !service.ValidDataset(dataset) {
			t.Fatalf("ValidDataset(%q) = false", dataset)
		}
	}
	if service.ValidDataset("KITScenes") {
		t.Fatal("ValidDataset accepted a non-canonical alias")
	}
}

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

// TestSortVersionsNewestFirst pins the ordering used by ListDatasetVersions:
// numeric-per-component, newest first (so v10 precedes v9 and v2.0 precedes
// v1.10). This is the same comparator the endpoint returns to the UI, which
// defaults to versions[0].
func TestSortVersionsNewestFirst(t *testing.T) {
	got := []string{"v1.0", "v10", "v2.0", "v1.10", "v9", "v1.2"}
	sortVersionsNewestFirst(got)
	want := []string{"v10", "v9", "v2.0", "v1.10", "v1.2", "v1.0"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("sortVersionsNewestFirst = %v, want %v", got, want)
	}
}

// TestValidVersion guards the ?version= override validation: only well-formed
// version dirs are accepted; garbage must be rejected so the handler 400s
// instead of building an S3 key like "<dataset>/../secrets/shards/".
func TestValidVersion(t *testing.T) {
	cases := map[string]bool{
		"v1.0":       true,
		"v2.0":       true,
		"v10":        true,
		"v1.10":      true,
		"":           false,
		"latest":     false,
		"v1.0/../..": false,
		"../v1.0":    false,
		"v1.0-rc1":   false,
	}
	for in, want := range cases {
		if got := ValidVersion(in); got != want {
			t.Errorf("ValidVersion(%q) = %v, want %v", in, got, want)
		}
	}
}

// TestShardsPrefix pins the S3 key layout the version override targets: a
// pinned version must produce exactly "<dataset>/<version>/shards/".
func TestShardsPrefix(t *testing.T) {
	cases := []struct {
		dataset, version, want string
	}{
		{"l2d", "v1.0", "l2d/v1.0/shards/"},
		{"l2d", "v2.0", "l2d/v2.0/shards/"},
		{"nvidia_av", "v2.0", "nvidia_av/v2.0/shards/"},
	}
	for _, c := range cases {
		if got := shardsPrefix(c.dataset, c.version); got != c.want {
			t.Errorf("shardsPrefix(%q,%q) = %q, want %q", c.dataset, c.version, got, c.want)
		}
	}
}

func TestCanonicalVersionsRequirePublicationManifest(t *testing.T) {
	cases := map[string]bool{
		"v1.0":  false,
		"v2.0":  false,
		"v2.1":  true,
		"v2.2":  true,
		"v3.0":  true,
		"v10.0": true,
	}
	for version, want := range cases {
		if got := requiresPublicationManifest(version); got != want {
			t.Errorf(
				"requiresPublicationManifest(%q) = %v, want %v",
				version, got, want,
			)
		}
	}
}

// TestShardManifestUnmarshal pins the manifest decode ListDatasetVersions
// depends on: the pipeline-written shards/manifest.json shape must map onto
// DatasetVersion's composition fields, and a manifest missing a field (the
// historical minimal form) must decode its zero value rather than erroring.
func TestShardManifestUnmarshal(t *testing.T) {
	full := `{"total_samples": 436, "shards": 1, "hz": 10, "image_size": 256,
		          "dataset": "yaak-ai/L2D", "episodes": 5, "num_views": 6,
		          "has_map": true, "has_world_model": false, "has_gps": true,
		          "geometry_type": "pseudo"}`
	var m shardManifest
	if err := json.Unmarshal([]byte(full), &m); err != nil {
		t.Fatalf("unmarshal full manifest: %v", err)
	}
	want := shardManifest{TotalSamples: 436, Shards: 1, Episodes: 5, NumViews: 6, HasMap: true, HasWorldModel: false, HasGPS: true}
	if !reflect.DeepEqual(m, want) {
		t.Errorf("full manifest decoded to %+v, want %+v", m, want)
	}

	// A manifest carrying only total_samples (no composition fields) must decode
	// cleanly with the rest zeroed.
	var partial shardManifest
	if err := json.Unmarshal([]byte(`{"total_samples": 10}`), &partial); err != nil {
		t.Fatalf("unmarshal partial manifest: %v", err)
	}
	if partial.TotalSamples != 10 || partial.Episodes != 0 || partial.NumViews != 0 || partial.HasMap {
		t.Errorf("partial manifest = %+v, want only TotalSamples=10", partial)
	}
}
