// Package model defines the JSON types exchanged by the console API.
package model

import (
	"encoding/json"
	"time"
)

// ErrorResponse is the uniform error envelope: {"error": "...", "code": "..."}.
type ErrorResponse struct {
	Error string `json:"error"`
	Code  string `json:"code"`
}

// Well-known error codes.
const (
	CodeNotFound     = "NOT_FOUND"
	CodeBadRequest   = "BAD_REQUEST"
	CodeUpstream     = "UPSTREAM_ERROR"
	CodeInternal     = "INTERNAL_ERROR"
	CodeS3Error      = "S3_ERROR"
	CodeUnavailable  = "SERVICE_UNAVAILABLE"
	CodeInvalidParam = "INVALID_PARAMETER"
)

// Page carries pagination metadata for list responses.
type Page struct {
	Limit  int  `json:"limit"`
	Offset int  `json:"offset"`
	Total  int  `json:"total"`
	More   bool `json:"more"`
}

// Dataset is a top-level dataset entry (l2d, nvidia_av, ...).
type Dataset struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Prefix  string `json:"prefix"` // S3 prefix of the shards
}

// DatasetListResponse wraps GET /api/v1/datasets.
type DatasetListResponse struct {
	Datasets []Dataset `json:"datasets"`
}

// Shard is one WebDataset .tar object.
type Shard struct {
	Name         string    `json:"name"` // e.g. train-000000.tar
	Key          string    `json:"key"`  // full S3 key
	SizeBytes    int64     `json:"size_bytes"`
	LastModified time.Time `json:"last_modified"`
}

// ShardListResponse wraps GET /api/v1/datasets/{name}/shards.
type ShardListResponse struct {
	Dataset string  `json:"dataset"`
	Shards  []Shard `json:"shards"`
	Page    Page    `json:"page"`
}

// TarMember is one file inside a shard, e.g. ep0_000064.cam_0.jpg.
type TarMember struct {
	Name      string `json:"name"`
	SizeBytes int64  `json:"size_bytes"`
	Offset    int64  `json:"offset"` // byte offset of the member data within the tar
}

// Sample groups tar members that share a sample key (WebDataset convention:
// key is the member name up to the first dot).
type Sample struct {
	Key     string      `json:"key"` // e.g. ep0_000064
	Members []TarMember `json:"members"`
}

// SampleListResponse wraps GET .../shards/{shard}/samples.
type SampleListResponse struct {
	Dataset string   `json:"dataset"`
	Shard   string   `json:"shard"`
	Samples []Sample `json:"samples"`
	Page    Page     `json:"page"`
}

// SampleDetail wraps GET .../shards/{shard}/samples/{key}: the parsed
// identity, raw meta.json, camera list and decoded ego signal arrays of one
// WebDataset sample.
type SampleDetail struct {
	Key        string          `json:"key"`
	EpisodeID  string          `json:"episode_id"`  // parsed from key: "ep0_000064" -> "0"; nvidia "25cd4769_000064" -> "25cd4769"
	FrameIdx   int             `json:"frame_idx"`   // parsed from key suffix -> 64
	Meta       json.RawMessage `json:"meta"`        // raw meta.json bytes
	Cameras    []string        `json:"cameras"`     // ["cam_0",...,"cam_6"] present for this sample
	EgoHistory []float32       `json:"ego_history"` // 256 floats (64 steps x 4 signals)
	EgoFuture  []float32       `json:"ego_future"`  // 128 floats (64 steps x 2 signals)
}

// ShardIndex wraps GET .../shards/{shard}/index: everything the ADAS player
// needs to play a shard (per-member byte ranges + per-frame ego state/plan),
// built from a single tar scan. Frames are fetched member-by-member through
// the image endpoint, so no whole-shard presigned URL is emitted.
type ShardIndex struct {
	Fps     int           `json:"fps"` // 10
	Samples []IndexSample `json:"samples"`
}

// IndexSample is one playback frame in a ShardIndex.
type IndexSample struct {
	Key          string                 `json:"key"`
	EpisodeID    string                 `json:"episode_id"`    // parsed from key: episode-global identity
	FrameIdx     int                    `json:"frame_idx"`     // intra-shard playback ordinal (key suffix)
	TripFrame    int                    `json:"trip_frame"`    // trip-global frame index from meta.json (-1 if absent)
	Members      map[string]MemberRange `json:"members"`       // "cam_0.jpg" -> {offset,size}
	EgoNow       []float32              `json:"ego_now"`       // last history row: [speed, accel, yaw_rate, curvature] (4 floats)
	EgoHistory   []float32              `json:"ego_history"`   // 256 floats = 64 steps x [speed, accel, yaw_rate, curvature] (past)
	EgoFuture    []float32              `json:"ego_future"`    // 128 floats = 64 steps x [accel, curvature] (the future plan)
	HasReasoning bool                   `json:"has_reasoning"` // whether an offline reasoning label exists for this sample
}

// MemberRange is the byte range of a tar member's data within the shard tar.
type MemberRange struct {
	Offset int64 `json:"offset"`
	Size   int64 `json:"size"`
}

// ReasoningStatsEntry is one dataset/teacher/prompt_version bucket with its
// label object count.
type ReasoningStatsEntry struct {
	Dataset       string `json:"dataset"`
	Teacher       string `json:"teacher"`
	PromptVersion string `json:"prompt_version"`
	Count         int    `json:"count"`
}

// ReasoningStatsResponse wraps GET /api/v1/reasoning-labels/stats.
type ReasoningStatsResponse struct {
	Entries []ReasoningStatsEntry `json:"entries"`
	Total   int                   `json:"total"`
}

// StatsResponse wraps GET /api/v1/stats (dashboard KPI cards). MLflow-derived
// fields degrade to zero/null when the tracking server is unreachable.
type StatsResponse struct {
	TotalSamples    int      `json:"total_samples"`
	ReasoningLabels int      `json:"reasoning_labels"`
	MLflowRuns      int      `json:"mlflow_runs"`
	LatestADE       *float64 `json:"latest_ade"`
	// MLflowAvailable reports whether the MLflow-derived fields (MLflowRuns,
	// LatestADE) were actually fetched. False (the default) means the tracking
	// server was unreachable, so a zero run count is "unknown", not "no runs".
	MLflowAvailable bool `json:"mlflow_available"`
}

// HealthResponse is returned by /healthz and /readyz.
type HealthResponse struct {
	Status string            `json:"status"`
	Checks map[string]string `json:"checks,omitempty"`
}
