// Package store implements the DynamoDB-backed cache for the console API:
// shard playback indexes (read-through, replacing the OOM-prone in-memory
// map), precomputed reasoning-label statistics, and a scene-by-label search
// index. It follows a single-table design on table `auto-e2e-console`
// (pk HASH, sk RANGE, GSI gsi1); every partition/sort key is constructed by
// the pure functions in this file so the layout is testable without AWS.
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

// SceneLabelSK is the sort key of one scene under a SceneLabelPK:
// SCENE#{sampleID}. The SCENE# prefix keeps scene rows distinct from any future
// metadata row that might share the partition.
func SceneLabelSK(sampleID string) string {
	return fmt.Sprintf("SCENE#%s", sampleID)
}
