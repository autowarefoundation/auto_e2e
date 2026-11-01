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

// DatasetVersion is one packed shard-set version of a dataset (e.g. l2d/v2.0),
// summarising the whole training composition at that version. Fields sourced
// from the version's shards/manifest.json; SizeBytes/Shards are computed from a
// ListObjects sum so a manifest-less (historical v1.0) version still reports.
type DatasetVersion struct {
	Version       string `json:"version"`         // e.g. "v2.0"
	TotalSamples  int    `json:"total_samples"`   // manifest total_samples (0 if no manifest)
	Shards        int    `json:"shards"`          // count of .tar objects under shards/
	Episodes      int    `json:"episodes"`        // manifest episodes
	NumViews      int    `json:"num_views"`       // manifest num_views (cameras per sample)
	HasMap        bool   `json:"has_map"`         // manifest has_map
	HasWorldModel bool   `json:"has_world_model"` // manifest has_world_model
	HasGPS        bool   `json:"has_gps"`         // v2.1 pose.npy/gps.npy members
	SizeBytes     int64  `json:"size_bytes"`      // sum of shard .tar object sizes
	HasManifest   bool   `json:"has_manifest"`    // whether shards/manifest.json was present
}

// DatasetVersionsResponse wraps GET /api/v1/datasets/{name}/versions
// (newest-first).
type DatasetVersionsResponse struct {
	Dataset  string           `json:"dataset"`
	Versions []DatasetVersion `json:"versions"`
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
	Fps               int           `json:"fps"` // 10
	Version           string        `json:"version"`
	Shard             string        `json:"shard"`
	BlobRangesAllowed bool          `json:"blob_ranges_allowed"`
	Samples           []IndexSample `json:"samples"`
}

// IndexSample is one playback frame in a ShardIndex.
type IndexSample struct {
	Key           string                 `json:"key"`
	SampleUID     string                 `json:"sample_uid"`
	SplitGroupUID string                 `json:"split_group_uid"`
	SplitBucket   int                    `json:"split_bucket"`
	EpisodeID     string                 `json:"episode_id"`  // parsed from key: episode-global identity
	FrameIdx      int                    `json:"frame_idx"`   // intra-shard playback ordinal (key suffix)
	TripFrame     int                    `json:"trip_frame"`  // trip-global frame index from meta.json (-1 if absent)
	Members       map[string]MemberRange `json:"members"`     // "cam_0.jpg" -> {offset,size}
	EgoNow        []float32              `json:"ego_now"`     // last history row: [speed, accel, yaw_rate, curvature] (4 floats)
	EgoHistory    []float32              `json:"ego_history"` // 256 floats = 64 steps x [speed, accel, yaw_rate, curvature] (past)
	EgoFuture     []float32              `json:"ego_future"`  // 128 floats = 64 steps x [accel, curvature] (the future plan)
	PoseCurrent   *GeoPose               `json:"pose_current,omitempty"`
	HasReasoning  bool                   `json:"has_reasoning"` // whether an offline reasoning label exists for this sample
}

// MemberRange is the byte range of a tar member's data within the shard tar.
type MemberRange struct {
	Offset int64 `json:"offset"`
	Size   int64 `json:"size"`
}

// GeoPose is the absolute pose packed in pose.npy. Heading is a compass
// bearing in degrees clockwise from north.
type GeoPose struct {
	LatitudeDeg           float64  `json:"latitude_deg"`
	LongitudeDeg          float64  `json:"longitude_deg"`
	HeadingDegCWFromNorth float64  `json:"heading_deg_cw_from_north"`
	TimestampNS           int64    `json:"timestamp_ns,string"`
	GPSAccuracyM          *float32 `json:"gps_accuracy_m"`
}

// OverlayModel is one ready canonical overlay available for a shard.
type OverlayModel struct {
	ModelArtifactID     string  `json:"model_artifact_id"`
	RegisteredModelName string  `json:"registered_model_name"`
	ModelVersion        int     `json:"model_version"`
	RunID               string  `json:"run_id"`
	ModelName           string  `json:"model_name"`
	EvalADE             float64 `json:"eval_ade"`
	EvalFDE             float64 `json:"eval_fde"`
	ValFraction         float64 `json:"val_fraction"`
	OverlaySchema       string  `json:"overlay_schema"`
	SampleCount         int     `json:"sample_count"`
}

// OverlayModelsResponse lists only models whose whole overlay set is ready.
type OverlayModelsResponse struct {
	Dataset       string         `json:"dataset"`
	Version       string         `json:"version"`
	Shard         string         `json:"shard"`
	Models        []OverlayModel `json:"models"`
	NextPageToken string         `json:"next_page_token,omitempty"`
}

// OverlayDescriptor accompanies the binary response through HTTP headers and
// is also useful to clients that need to inspect the selected artifact.
type OverlayDescriptor struct {
	ModelArtifactID string `json:"model_artifact_id"`
	OverlaySchema   string `json:"overlay_schema"`
	SHA256          string `json:"sha256"`
	ByteSize        int64  `json:"byte_size"`
	SampleCount     int    `json:"sample_count"`
}

// SemanticOccupancyDescriptor identifies one immutable 2D BEV semantic body.
type SemanticOccupancyDescriptor struct {
	ModelArtifactID string `json:"model_artifact_id"`
	Schema          string `json:"schema"`
	GeometryID      string `json:"geometry_id"`
	TaxonomyVersion string `json:"taxonomy_version"`
	HeadVersion     string `json:"head_version"`
	SHA256          string `json:"sha256"`
	ByteSize        int64  `json:"byte_size"`
}

// SemanticOccupancyModelSource records the immutable source of one published
// occupancy model. Weight and generated occupancy identities are independent.
type SemanticOccupancyModelSource struct {
	CodeLicenseSPDX         string `json:"code_license_spdx"`
	Config                  string `json:"config"`
	LicenseSPDX             string `json:"license_spdx"`
	Repository              string `json:"repository"`
	RepositoryRevision      string `json:"repository_revision"`
	TrainingDataLicenseSPDX string `json:"training_data_license_spdx"`
	WeightSHA256            string `json:"weight_sha256"`
	WeightSourceURL         string `json:"weight_source_url"`
}

// SemanticOccupancyModel is one independently published occupancy model that
// contains a body for the requested dataset shard.
type SemanticOccupancyModel struct {
	ModelArtifactID       string                       `json:"model_artifact_id"`
	DisplayName           string                       `json:"display_name"`
	ModelFamily           string                       `json:"model_family"`
	ArtifactKind          string                       `json:"artifact_kind"`
	ArtifactSchema        string                       `json:"artifact_schema"`
	CreatedAt             string                       `json:"created_at"`
	DatasetManifestSHA256 string                       `json:"dataset_manifest_sha256"`
	GeometryID            string                       `json:"geometry_id"`
	TaxonomyVersion       string                       `json:"taxonomy_version"`
	HeadVersion           string                       `json:"head_version"`
	InputContract         string                       `json:"input_contract"`
	SupportedClasses      []string                     `json:"supported_classes"`
	TeacherAvailable      bool                         `json:"teacher_available"`
	Limitations           []string                     `json:"limitations"`
	ModelSource           SemanticOccupancyModelSource `json:"model_source"`
	ProducerConfig        map[string]any               `json:"producer_config"`
	SampleCount           int                          `json:"sample_count"`
	ShardCount            int                          `json:"shard_count"`
	ShardSampleCount      int                          `json:"shard_sample_count"`
}

// SemanticOccupancyModelsResponse lists occupancy publications independently
// from trajectory overlay model records.
type SemanticOccupancyModelsResponse struct {
	Dataset string                   `json:"dataset"`
	Version string                   `json:"version"`
	Shard   string                   `json:"shard"`
	Models  []SemanticOccupancyModel `json:"models"`
}

// GeoStatsResponse is the privacy-filtered dataset-level ODD geography.
// Summary is kept as JSON because its per-region dimensions may evolve without
// changing the serving envelope.
type GeoStatsResponse struct {
	Dataset    string          `json:"dataset"`
	Version    string          `json:"version"`
	Summary    json.RawMessage `json:"summary"`
	HeatmapURL string          `json:"heatmap_url"`
	NSamples   int             `json:"n_samples"`
	ComputedAt string          `json:"computed_at"`
}

// ReasoningStatsEntry is one dataset/teacher/prompt_version bucket with its
// label object count. Teacher is an opaque URL-safe identity; the provider and
// model fields are the human-readable provenance.
type ReasoningStatsEntry struct {
	Dataset         string `json:"dataset"`
	Teacher         string `json:"teacher"`
	TeacherProvider string `json:"teacher_provider"`
	TeacherModel    string `json:"teacher_model"`
	PromptVersion   string `json:"prompt_version"`
	Count           int    `json:"count"`
}

// ReasoningStatsResponse wraps GET /api/v1/reasoning-labels/stats.
type ReasoningStatsResponse struct {
	Entries []ReasoningStatsEntry `json:"entries"`
	Total   int                   `json:"total"`
}

// ReasoningPromptVersion is one teacher/prompt_version partition of a single
// dataset's reasoning-label cache with its label object count.
type ReasoningPromptVersion struct {
	Teacher         string `json:"teacher"`
	TeacherProvider string `json:"teacher_provider"`
	TeacherModel    string `json:"teacher_model"`
	PromptVersion   string `json:"prompt_version"`
	Count           int    `json:"count"`
}

// ReasoningPromptVersionsResponse wraps
// GET /api/v1/reasoning-labels/prompt-versions?dataset={name}.
type ReasoningPromptVersionsResponse struct {
	Dataset        string                   `json:"dataset"`
	PromptVersions []ReasoningPromptVersion `json:"prompt_versions"`
}

// ReasoningInventory is the atomic publication pointer for one immutable
// dataset version. Generation selects a complete namespace whose stats, scene
// rows, and sample lookups were all persisted before this item was replaced.
type ReasoningInventory struct {
	Generation            string                   `json:"generation"`
	DatasetManifestSHA256 string                   `json:"dataset_manifest_sha256"`
	PromptVersions        []ReasoningPromptVersion `json:"prompt_versions"`
	Total                 int                      `json:"total"`
	SceneRows             int                      `json:"scene_rows"`
}

// ReasoningSampleLookup is the generation-scoped DynamoDB pointer to one
// embedded reasoning.json tar member. It is internal serving metadata and is
// never serialized directly to the public API.
type ReasoningSampleLookup struct {
	SampleID string
	Shard    string
	Offset   int64
	Size     int64
}

// ReasoningMaterializationResponse summarizes one trusted, all-partition
// materialization run.
type ReasoningMaterializationResponse struct {
	Dataset               string `json:"dataset"`
	Version               string `json:"version"`
	Generation            string `json:"generation"`
	DatasetManifestSHA256 string `json:"dataset_manifest_sha256"`
	ComputedAt            string `json:"computed_at"`
	Partitions            int    `json:"partitions"`
	TotalRecords          int    `json:"total_records"`
	SceneRows             int    `json:"scene_rows"`
	Reused                bool   `json:"reused,omitempty"`
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

// Reasoning-label statistics, precomputed and cached in DynamoDB, describe the
// ODD a (dataset x version x prompt_version) label set covers: the categorical
// distribution of each taxonomy axis across all horizons of all labels, plus a
// confidence histogram. The v2 context axes (weather, geo, road topology, ...)
// are NULL in the data, so they are not represented here.

// HistogramBucket is one bucket of a value-count histogram (confidence, speed).
type HistogramBucket struct {
	Bucket string `json:"bucket"`
	Count  int    `json:"count"`
}

// ReasoningStatsBlob is the precomputed statistics for one
// (dataset x version x prompt_version) reasoning-label set. ByField maps each
// taxonomy axis name (relation_to_ego, hazard_event, cause, longitudinal_response,
// lateral_response, tactical_response, rule_response) to its value->count map,
// aggregated across every horizon of every label in the set.
type ReasoningStatsBlob struct {
	NRecords            int                       `json:"n_records"`     // all valid records, including explicit abstentions
	NLabels             int                       `json:"n_labels"`      // successful records contributing coverage
	NAbstained          int                       `json:"n_abstained"`   // explicit teacher-error records
	HorizonCount        int                       `json:"horizon_count"` // total rows from successful records
	ByField             map[string]map[string]int `json:"by_field"`
	ConfidenceHistogram []HistogramBucket         `json:"confidence_histogram"`
	// SpeedHistogram is populated only when ego speed is cheaply joinable;
	// omitted otherwise (the reasoning cache carries no ego signal).
	SpeedHistogram []HistogramBucket `json:"speed_histogram,omitempty"`
}

// ReasoningStatsDetailResponse wraps GET /api/v1/reasoning-labels/stats-detail.
// Stats is the precomputed blob; ComputedAt is when it was materialised (RFC3339,
// empty when just computed inline and not yet persisted).
type ReasoningStatsDetailResponse struct {
	Dataset         string             `json:"dataset"`
	Version         string             `json:"version"`
	PromptVersion   string             `json:"prompt_version"`
	Teacher         string             `json:"teacher"`
	TeacherProvider string             `json:"teacher_provider,omitempty"`
	TeacherModel    string             `json:"teacher_model,omitempty"`
	ComputedAt      string             `json:"computed_at,omitempty"`
	Cached          bool               `json:"cached"` // true when served from a DynamoDB hit
	Stats           ReasoningStatsBlob `json:"stats"`
}

// SceneRef identifies one scene carrying a searched reasoning label. Shard is
// persisted with the generation-scoped label row during materialization, so
// search does not scan shard indexes. Available remains part of the public
// contract and is true for complete materialized rows.
type SceneRef struct {
	SampleID  string `json:"sample_id"`
	Shard     string `json:"shard,omitempty"`
	Available bool   `json:"available"`
}

// SceneSearchResponse wraps GET /api/v1/scenes/search: the scenes carrying a
// given (field,value) reasoning label for a (dataset, prompt_version). Total is
// the number of matched sample ids returned; Available is how many of them are
// present in the requested version's published shards (linkable). Truncated is
// true when the label index held more matches than the requested limit.
type SceneSearchResponse struct {
	Dataset       string     `json:"dataset"`
	PromptVersion string     `json:"prompt_version"`
	Teacher       string     `json:"teacher"`
	Version       string     `json:"version,omitempty"`
	Field         string     `json:"field"`
	Value         string     `json:"value"`
	Scenes        []SceneRef `json:"scenes"`
	Total         int        `json:"total"`
	Available     int        `json:"available"`
	Truncated     bool       `json:"truncated"`
}

// ODDQuery selects scenes by their ODD labels: the top-level Op ("and" or
// "or") joins Groups, and each group's Op joins its Conditions.
type ODDQuery struct {
	Op     string          `json:"op"`
	Groups []ODDQueryGroup `json:"groups"`
}

// ODDQueryGroup is one parenthesised group of conditions.
type ODDQueryGroup struct {
	Op         string              `json:"op"`
	Conditions []ODDQueryCondition `json:"conditions"`
}

// ODDQueryCondition holds when the scene has a valid labelled interval of Key
// whose values include one of Values and that lasts at least MinDurationS,
// has at least MinConfidence and comes from one of Sources (any source when
// empty). Not inverts it: no such interval exists.
type ODDQueryCondition struct {
	Key           string   `json:"key"`
	Values        []string `json:"values"`
	Not           bool     `json:"not,omitempty"`
	MinDurationS  float64  `json:"min_duration_s,omitempty"`
	MinConfidence float64  `json:"min_confidence,omitempty"`
	Sources       []string `json:"sources,omitempty"`
}

// ODDQuality is the review state the ODD LabelSet manifest declares.
type ODDQuality struct {
	StructuralStatus    string `json:"structural_status"`
	AuditStatus         string `json:"audit_status"`
	CertificationStatus string `json:"certification_status"`
}

// ODDFacetValue counts the scenes carrying one value of a key.
type ODDFacetValue struct {
	Value  string `json:"value"`
	Scenes int    `json:"scenes"`
}

// ODDFacet lists the values of one ODD key over a set of scenes, and the
// sources that labelled it.
type ODDFacet struct {
	Key     string          `json:"key"`
	Values  []ODDFacetValue `json:"values"`
	Sources []string        `json:"sources"`
}

// ODDInterval is one labelled interval. StartNS and EndNS are offsets from
// the scene start.
type ODDInterval struct {
	StartNS    int64    `json:"start_ns"`
	EndNS      int64    `json:"end_ns"`
	Values     []string `json:"values"`
	Source     string   `json:"source"`
	Confidence float64  `json:"confidence"`
	CameraID   string   `json:"camera_id,omitempty"`
}

// ODDLane is the intervals of one key in a scene, in start order.
type ODDLane struct {
	Key       string        `json:"key"`
	Intervals []ODDInterval `json:"intervals"`
}

// ODDScene is one scene. Shard is the shard name in the requested version,
// which the player opens. In a search result Lanes holds the intervals that
// satisfied the query; in a scene response it holds every valid label.
type ODDScene struct {
	SceneUID   string    `json:"scene_uid"`
	Shard      string    `json:"shard"`
	DurationNS int64     `json:"duration_ns"`
	DistanceM  float64   `json:"distance_m"`
	Lanes      []ODDLane `json:"lanes"`
}

// ODDLabelSetInfo identifies the LabelSet behind a response. LabelVersion is
// the dataset version it was made for; it differs from Version when the
// labels are carried forward to a later pack of the same scenes.
type ODDLabelSetInfo struct {
	Dataset        string     `json:"dataset"`
	Version        string     `json:"version"`
	LabelVersion   string     `json:"label_version"`
	LabelSetID     string     `json:"labelset_id"`
	ManifestSHA256 string     `json:"manifest_sha256"`
	Quality        ODDQuality `json:"quality"`
	LabeledScenes  int        `json:"labeled_scenes"`
}

// ODDSearchResponse wraps GET /api/v1/odd/scenes. Vocabulary covers every
// labelled scene (the choices for building a query); Facets covers the
// matched scenes.
type ODDSearchResponse struct {
	ODDLabelSetInfo
	Query      ODDQuery   `json:"query"`
	Vocabulary []ODDFacet `json:"vocabulary"`
	Facets     []ODDFacet `json:"facets"`
	Scenes     []ODDScene `json:"scenes"`
	Page       Page       `json:"page"`
}

// ODDSceneResponse wraps GET /api/v1/odd/scenes/{scene_uid}.
type ODDSceneResponse struct {
	ODDLabelSetInfo
	Scene ODDScene `json:"scene"`
}
