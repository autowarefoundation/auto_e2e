// Package store implements the DynamoDB-backed cache for the console API:
// shard playback indexes (read-through, replacing the OOM-prone in-memory
// map), precomputed reasoning-label statistics, and a scene-by-label search
// index. It follows a single-table design on table `auto-e2e-console`
// (pk HASH, sk RANGE); every partition/sort key is constructed by the pure
// functions in this file so the layout is testable without AWS.
package store

import "fmt"

// metaSK is the sort key shared by the singleton items (one shard index / one
// stats blob per partition key). The partition key already fully identifies the
// item; the sort key is a fixed sentinel.
const metaSK = "META"

// ShardIndexPK is the partition key of a cached shard playback index:
// IDX#{dataset}#{version}#{shard}. sk is metaSK.
func ShardIndexPK(dataset, version, shard string) string {
	return fmt.Sprintf("IDX#%s#%s#%s", dataset, version, shard)
}

// StatsPK is the partition key of a precomputed reasoning-stats blob:
// STATS#{dataset}#{version}#{promptVersion}. sk is metaSK.
func StatsPK(dataset, version, promptVersion string) string {
	return fmt.Sprintf("STATS#%s#%s#%s", dataset, version, promptVersion)
}

// SceneLabelPK is the partition key that groups every scene carrying one
// (field,value) reasoning label: LBL#{dataset}#{promptVersion}#{field}#{value}.
// Querying this pk returns all scenes with that label (via SceneLabelSK sorts).
//
// Note the scene index is keyed by (dataset, promptVersion) only — NOT by
// dataset version — because reasoning labels are not partitioned by shard
// version in S3 (they are keyed by the flat s%08d sample id).
func SceneLabelPK(dataset, promptVersion, field, value string) string {
	return fmt.Sprintf("LBL#%s#%s#%s#%s", dataset, promptVersion, field, value)
}

// SceneLabelVersionPK isolates the sample_uid-based index for one immutable
// packed dataset version from legacy s%08d rows.
func SceneLabelVersionPK(dataset, version, promptVersion, field, value string) string {
	return fmt.Sprintf(
		"LBLV2#%s#%s#%s#%s#%s",
		dataset, version, promptVersion, field, value,
	)
}

// SceneLabelSK is the sort key of one scene under a SceneLabelPK:
// SCENE#{sampleID}. The SCENE# prefix keeps scene rows distinct from any future
// metadata row that might share the partition.
func SceneLabelSK(sampleID string) string {
	return fmt.Sprintf("SCENE#%s", sampleID)
}

// ShardModelPK groups the canonical model overlays available for one immutable
// dataset shard. Querying this base-table partition powers the model picker.
func ShardModelPK(dataset, version, shard string) string {
	return fmt.Sprintf("SHARD#%s#%s#%s", dataset, version, shard)
}

// ModelSK identifies one content-addressed checkpoint within ShardModelPK.
func ModelSK(modelArtifactID string) string {
	return fmt.Sprintf("MODEL#%s", modelArtifactID)
}

// OverlaySetPK identifies the write-then-publish gate for all shard overlays
// produced from one model and immutable dataset version.
func OverlaySetPK(modelArtifactID, dataset, version string) string {
	return fmt.Sprintf("OVLSET#%s#%s#%s", modelArtifactID, dataset, version)
}

// GeoPK identifies the privacy-filtered geospatial summary for a dataset
// version. Exact episode paths remain in access-controlled S3 objects.
func GeoPK(dataset, version string) string {
	return fmt.Sprintf("GEO#%s#%s", dataset, version)
}
