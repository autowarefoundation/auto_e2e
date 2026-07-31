package service

import "testing"

func artifactPublication(
	version string,
	totalSamples int,
	shards ...string,
) *publicationManifest {
	entries := make([]publicationShardEntry, len(shards))
	for index, name := range shards {
		entries[index] = publicationShardEntry{
			Name:            name,
			ContentIdentity: version + "-pack-content",
		}
	}
	return &publicationManifest{
		Dataset:      "kitscenes",
		Version:      version,
		TotalSamples: totalSamples,
		ShardCount:   len(entries),
		Episodes:     len(entries),
		ShardEntries: entries,
	}
}

func TestSameSceneInventoryIgnoresPackContentIdentity(t *testing.T) {
	current := artifactPublication(
		"v3.1",
		42_667,
		"scene-b.tar",
		"scene-a.tar",
	)
	previous := artifactPublication(
		"v2.2",
		42_667,
		"scene-a.tar",
		"scene-b.tar",
	)

	if !sameSceneInventory(current, previous) {
		t.Fatal("identical KITScenes inventory was rejected after repacking")
	}
}

func TestSameSceneInventoryRejectsChangedCoordinateSet(t *testing.T) {
	reference := artifactPublication(
		"v3.1",
		42_667,
		"scene-a.tar",
		"scene-b.tar",
	)
	tests := []struct {
		name      string
		candidate *publicationManifest
	}{
		{
			name: "sample count",
			candidate: artifactPublication(
				"v3.0",
				42_666,
				"scene-a.tar",
				"scene-b.tar",
			),
		},
		{
			name: "scene count",
			candidate: artifactPublication(
				"v3.0",
				42_667,
				"scene-a.tar",
			),
		},
		{
			name: "scene identity",
			candidate: artifactPublication(
				"v3.0",
				42_667,
				"scene-a.tar",
				"scene-c.tar",
			),
		},
		{
			name: "dataset",
			candidate: func() *publicationManifest {
				manifest := artifactPublication(
					"v3.0",
					42_667,
					"scene-a.tar",
					"scene-b.tar",
				)
				manifest.Dataset = "other"
				return manifest
			}(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if sameSceneInventory(reference, test.candidate) {
				t.Fatal("changed artifact coordinate set was accepted")
			}
		})
	}
}
