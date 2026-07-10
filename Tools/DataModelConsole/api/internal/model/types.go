// Package model defines the JSON types exchanged by the console API.
package model

import "time"

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

// HealthResponse is returned by /healthz and /readyz.
type HealthResponse struct {
	Status string            `json:"status"`
	Checks map[string]string `json:"checks,omitempty"`
}
