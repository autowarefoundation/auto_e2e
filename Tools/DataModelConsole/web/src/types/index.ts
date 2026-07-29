// Domain types for DataModelConsole.
// Mirrors the Go API JSON shapes (api/internal/model/types.go).

// ---------------------------------------------------------------------------
// Pagination
// ---------------------------------------------------------------------------

export interface Page {
  limit: number;
  offset: number;
  total: number;
  more: boolean;
}

export interface TokenPage<T> {
  items: T[];
  next_page_token?: string;
}

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

export interface Dataset {
  name: string; // "l2d" | "nvidia_av" | ...
  version: string; // e.g. "v1.0"
  prefix: string; // S3 prefix of the shards
}

export interface DatasetListResponse {
  datasets: Dataset[];
}

// DatasetVersion summarises one packed shard-set version's WHOLE training
// composition (GET /api/v1/datasets/{name}/versions). Manifest-derived counts
// are zero when has_manifest is false (historical v1.0 without a manifest);
// shards/size_bytes are always the real ListObjects tally.
export interface DatasetVersion {
  version: string; // e.g. "v2.0"
  total_samples: number;
  shards: number;
  episodes: number;
  num_views: number;
  has_map: boolean;
  has_world_model: boolean;
  has_gps: boolean;
  size_bytes: number;
  has_manifest: boolean;
}

export interface DatasetVersionsResponse {
  dataset: string;
  versions: DatasetVersion[]; // newest-first
}

export interface Shard {
  name: string; // e.g. "train-000000.tar"
  key: string; // full S3 key
  size_bytes: number;
  last_modified: string; // RFC3339
}

export interface ShardListResponse {
  dataset: string;
  shards: Shard[];
  page: Page;
}

// TarMember is one file inside a shard, e.g. "ep0_000064.cam_0.jpg".
export interface TarMember {
  name: string;
  size_bytes: number;
  offset: number; // byte offset of the member data within the tar
}

// Sample groups tar members sharing a WebDataset key (name up to first dot).
export interface Sample {
  key: string; // e.g. "ep0_000064"
  members: TarMember[];
}

export interface SampleListResponse {
  dataset: string;
  shard: string;
  samples: Sample[];
  page: Page;
}

// SampleDetail is GET .../shards/{shard}/samples/{key}.
// ego_history: 256 floats = 64 steps x [speed, accel, yaw_rate, curvature].
// ego_future: 128 floats = 64 steps x [accel, curvature].
export interface SampleDetail {
  key: string;
  episode_id: string;
  frame_idx: number;
  meta: Record<string, unknown>;
  cameras: string[]; // e.g. ["cam_0", ..., "cam_6"]
  ego_history: number[];
  ego_future: number[];
}

// ---------------------------------------------------------------------------
// Scene-level ODD LabelSets
// ---------------------------------------------------------------------------

export type ODDStatus =
  | "valid"
  | "unavailable"
  | "not_observable"
  | "ambiguous";

export type ODDSupportState =
  | "supported_certified"
  | "supported_experimental"
  | "unsupported_missing_source"
  | "disabled_pending_audit";

export interface ODDDatasetSupport {
  support_state: ODDSupportState;
  quality_tier: string;
  valid_scene_count: number;
  eligible_scene_count: number;
  observable_scene_coverage: number;
  attempted_count: number;
  successful_count: number;
}

export interface ODDLabelDefinition {
  key: string;
  namespace: "odd" | "event" | "perception";
  display_name: string;
  description: string;
  cardinality: "single" | "multi";
  values: { value: string }[];
  primary_sources: string[];
  backends: string[];
  subject: string;
  temporal_scope: string;
  quality_tier: string;
  none_semantics?: string | null;
  dataset_support: ODDDatasetSupport;
}

export interface ODDOntology {
  schema_version: string;
  ontology_version: string;
  ontology_sha256: string;
  dataset_name: string;
  dataset_version: string;
  labelset_id: string;
  statuses: string[];
  sources: string[];
  labels: ODDLabelDefinition[];
}

export interface ODDRatioInterval {
  lower: number;
  upper: number;
  method: string;
  replicates?: number;
}

export interface ODDConfidenceBin {
  lower: number;
  upper: number;
  observation_count: number;
  duration_ns: number;
  distance_m: number;
}

export interface ODDConfidenceSummary {
  observation_count: number;
  duration_weighted_mean: number;
  p10: number;
  p50: number;
  p90: number;
  bins: ODDConfidenceBin[];
}

export interface ODDValueStatistic {
  value: string;
  scene_count: number;
  scene_ratio: number;
  scene_ratio_ci95: ODDRatioInterval;
  duration_ns: number;
  duration_ratio: number;
  duration_ratio_ci95: ODDRatioInterval;
  distance_m: number;
  distance_ratio: number;
  distance_ratio_ci95: ODDRatioInterval;
  valid_interval_count: number;
  event_instance_count: number;
  confidence: ODDConfidenceSummary;
}

export interface ODDKeyStatistic {
  key: string;
  namespace: string;
  quality_tier: string;
  valid_scene_count: number;
  eligible_scene_count: number;
  observable_scene_coverage: number;
  eligible_duration_ns: number;
  valid_duration_ns: number;
  observable_duration_coverage: number;
  eligible_distance_m: number;
  valid_distance_m: number;
  observable_distance_coverage: number;
  valid_interval_count: number;
  attempted_count: number;
  successful_count: number;
  conflict_count: number;
  status_scene_counts: Record<string, number>;
  status_duration_ns: Record<string, number>;
  status_distance_m: Record<string, number>;
  source_scene_counts: Record<string, number>;
  source_duration_ns: Record<string, number>;
  source_distance_m: Record<string, number>;
  confidence: ODDConfidenceSummary;
  values: ODDValueStatistic[];
}

export interface ODDCooccurrenceStatistics {
  minimum_overlap_ns: number;
  odd_pairs: Array<{
    left_key: string;
    left_value: string;
    right_key: string;
    right_value: string;
    scene_count: number;
    overlap_duration_ns: number;
    overlap_distance_m: number;
  }>;
  odd_event: Array<{
    odd_key: string;
    odd_value: string;
    event_key: string;
    event_value: string;
    scene_count: number;
    event_instance_count: number;
    overlap_duration_ns: number;
    overlap_distance_m: number;
  }>;
}

export interface ODDStatistics {
  schema_version: string;
  labelset_id: string;
  scene_count: number;
  scene_duration_ns: number;
  scene_distance_m: number;
  distance_weighting: {
    method_scene_counts: Record<string, number>;
    normalization: string;
  };
  keys: ODDKeyStatistic[];
  cooccurrences: ODDCooccurrenceStatistics;
}

export interface ODDArtifact {
  key: string;
  sha256: string;
  byte_size: number;
  content_type?: string;
  format?: string;
  row_count?: number;
  schema_version?: string;
  authoritative?: boolean;
}

export interface ODDQualityState {
  schema_version: string;
  structural_status: string;
  audit_status: string;
  certification_status: string;
}

export interface ODDLabelSet {
  schema_version: string;
  status: string;
  labelset_id: string;
  dataset_name: string;
  dataset_version: string;
  dataset_manifest_uri: string;
  dataset_manifest_sha256: string;
  ontology_version: string;
  ontology_sha256: string;
  labeler_version: string;
  labeler_image_digest: string;
  labeler_source_revision: string;
  publication_scope: string;
  expected_scene_count: number;
  scene_count: number;
  openai_compatible: Record<string, string>;
  quality: ODDQualityState;
  artifacts: Record<string, ODDArtifact>;
}

export interface ODDLabelSetsResponse {
  dataset: string;
  version: string;
  labelsets: ODDLabelSet[];
}

export interface ODDObservation {
  schema_version?: string;
  observation_uid: string;
  scene_uid: string;
  key: string;
  status: ODDStatus;
  values: string[];
  confidence: number;
  source: string;
  start_timestamp_ns: number;
  end_timestamp_ns: number;
  evidence_uids: string[];
  conflicting_evidence_uids: string[];
  measurements: Record<string, string | number | boolean>;
  provenance: Record<string, unknown>;
  camera_id?: string | null;
  actor_track_uid?: string | null;
  event_uid?: string | null;
}

export interface ODDEventPhase {
  phase: "onset" | "active" | "resolution";
  start_timestamp_ns: number;
  end_timestamp_ns: number;
}

export interface ODDEvent {
  schema_version?: string;
  event_uid: string;
  scene_uid: string;
  start_timestamp_ns: number;
  end_timestamp_ns: number;
  primary_event_key: string;
  actor_track_uids: string[];
  observation_uids: string[];
  phases: ODDEventPhase[];
  confidence: number;
  status: ODDStatus;
  supporting_evidence_uids: string[];
  provenance: Record<string, unknown>;
}

export interface ODDEvidenceScope {
  dataset_name: string;
  dataset_version: string;
  scene_uid: string;
  start_timestamp_ns: number;
  end_timestamp_ns: number;
  subject_type: string;
  subject_id?: string | null;
  anchor_timestamp_ns?: number | null;
  camera_ids: string[];
  coordinate_frame?: string | null;
  spatial_roi?: Record<string, unknown> | null;
}

export interface ODDLabelEvidence {
  schema_version: string;
  evidence_uid: string;
  label_key: string;
  cardinality: "single" | "multi";
  values: string[];
  candidate_values: Array<{
    value: string;
    score: number;
    evidence_ref?: string | null;
  }>;
  status: ODDStatus;
  confidence: number;
  source: string;
  scope: ODDEvidenceScope;
  measurements: Array<{
    name: string;
    value: string | number | boolean;
    unit: string;
    quality: string;
    aggregation: string;
  }>;
  evidence_refs: Array<{
    artifact_uri: string;
    artifact_sha256: string;
    timestamp_ns?: number | null;
    camera_id?: string | null;
  }>;
  provenance: {
    labeler_name: string;
    labeler_version: string;
    code_commit: string;
    container_image_digest: string;
    config_sha256: string;
    ontology_sha256: string;
    input_artifact_sha256s: string[];
    model_provider?: string | null;
    model_name?: string | null;
    model_revision?: string | null;
    prompt_sha256?: string | null;
    decoding_config_sha256?: string | null;
    lookback_ns: number;
    lookahead_ns: number;
    details: Record<string, unknown>;
  };
}

export interface ODDSceneRecord {
  scene_uid: string;
  dataset_name: string;
  dataset_version: string;
  start_timestamp_ns: number;
  end_timestamp_ns: number;
  distance_m: number;
  observations: ODDObservation[];
  evidence: ODDLabelEvidence[];
  events: ODDEvent[];
  provenance: Record<string, unknown>;
}

export interface ODDSceneObservationSummary {
  key: string;
  status: ODDStatus;
  values: string[];
  source: string;
  confidence: number;
  duration_ns: number;
  first_timestamp_ns: number;
  interval_count?: number;
  camera_id?: string;
  actor_track_uid?: string;
  event_uid?: string;
}

export interface ODDSceneEventSummary {
  event_uid: string;
  primary_event_key: string;
  primary_values: string[];
  start_timestamp_ns: number;
  end_timestamp_ns: number;
  status: ODDStatus;
  confidence: number;
  actor_track_uids: string[];
  outcome: string;
}

export interface ODDSceneSummary {
  scene_uid: string;
  shard_name: string;
  start_timestamp_ns: number;
  end_timestamp_ns: number;
  distance_m: number;
  observations: ODDSceneObservationSummary[];
  events: ODDSceneEventSummary[];
  matched?: ODDSceneObservationSummary[];
  matched_duration_ns?: number;
  match_confidence?: number;
  first_matched_timestamp_ns?: number;
}

export type ODDSearchOperator =
  | "exists"
  | "contains"
  | "equals"
  | "in"
  | "not_equals";

export interface ODDSearchPredicate {
  key: string;
  operator: ODDSearchOperator;
  values: string[];
  statuses: ODDStatus[];
  sources: string[];
  minimum_confidence: number;
  minimum_duration_ns: number;
  camera_id: string;
  actor_track_uid: string;
}

export interface ODDSearchGroup {
  logic: "and" | "or";
  predicates: ODDSearchPredicate[];
  groups: ODDSearchGroup[];
}

export interface ODDStructuredSearchRequest {
  query: ODDSearchGroup;
  sort:
    | "scene_uid"
    | "confidence"
    | "matched_duration"
    | "scene_duration"
    | "recording_time";
  descending: boolean;
  limit: number;
  offset: number;
}

export interface ODDSearchResponse {
  dataset: string;
  version: string;
  labelset_id: string;
  scenes: ODDSceneSummary[];
  total: number;
  limit: number;
  offset: number;
  more: boolean;
  manifest_sha256: string;
}

export interface ODDEvidenceResponse {
  dataset: string;
  version: string;
  labelset_id: string;
  scene_uid: string;
  observation: ODDObservation;
  supporting_evidence: ODDLabelEvidence[];
  conflicting_evidence: ODDLabelEvidence[];
  related_events: ODDEvent[];
  scene_provenance: Record<string, unknown>;
  manifest_sha256: string;
}

// ---------------------------------------------------------------------------
// Shard index (ADAS player)
// ---------------------------------------------------------------------------

// MemberRange locates one tar member's raw bytes for HTTP Range requests.
export interface MemberRange {
  offset: number;
  size: number;
}

// IndexSample is one frame entry of the shard index.
// ego_now: [speed, accel, yaw_rate, curvature] at this frame.
// ego_future: 128 floats = 64 steps x [accel, curvature] — the future plan.
export interface IndexSample {
  key: string;
  sample_uid: string;
  split_group_uid: string;
  split_bucket: number;
  episode_id: string;
  frame_idx: number; // intra-shard playback ordinal (key suffix)
  trip_frame: number; // trip-global frame index from meta.json (-1 if absent)
  members: Record<string, MemberRange>; // "cam_0.jpg" -> range
  ego_now: number[];
  ego_history: number[]; // 256 floats = 64 steps x [speed, accel, yaw_rate, curvature]
  ego_future: number[];
  pose_current?: GeoPose;
  has_reasoning: boolean;
}

export interface GeoPose {
  latitude_deg: number;
  longitude_deg: number;
  heading_deg_cw_from_north: number;
  timestamp_ns: string;
  gps_accuracy_m: number | null;
}

// ShardIndex is GET .../shards/{shard}/index — everything the client needs
// to play a shard as a 10Hz video (frames fetched per-member via the image
// endpoint).
export interface ShardIndex {
  fps: number; // 10
  version: string;
  shard: string;
  blob_ranges_allowed: boolean;
  samples: IndexSample[];
}

// ---------------------------------------------------------------------------
// Model trajectory overlays and geographic products
// ---------------------------------------------------------------------------

export interface OverlayModel {
  model_artifact_id: string;
  registered_model_name: string;
  model_version: number;
  run_id: string;
  model_name: string;
  eval_ade: number;
  eval_fde: number;
  val_fraction: number;
  overlay_schema: string;
  sample_count: number;
}

export interface OverlayModelsResponse {
  dataset: string;
  version: string;
  shard: string;
  models: OverlayModel[];
  next_page_token?: string;
}

export interface RigProjectionDocument {
  schema_version: string;
  dataset: string;
  geometry_type: "pinhole" | "rectified_pinhole" | "ftheta" | "pseudo";
  image_size?: number | [number, number];
  projection: Record<string, unknown> | null;
}

export interface GeoSummary {
  schema_version?: string;
  bbox: [number, number, number, number] | null;
  episode_count: number;
  path_point_count: number;
  sample_pose_count: number;
  privacy?: {
    k_anonymity: number;
    endpoint_exclusion_frames: number;
    heatmap_grid_degrees: number;
  };
}

export interface GeoStats {
  dataset: string;
  version: string;
  summary: GeoSummary;
  heatmap_url?: string;
  n_samples: number;
  computed_at?: string;
}

export interface GeoJSONPointFeature {
  type: "Feature";
  geometry: {
    type: "Point";
    coordinates: [number, number];
  };
  properties: {
    sample_count: number;
    episode_count: number;
  };
}

export interface GeoJSONFeatureCollection {
  type: "FeatureCollection";
  features: GeoJSONPointFeature[];
}

// ---------------------------------------------------------------------------
// Reasoning Labels
// ---------------------------------------------------------------------------

// ReasoningHorizon is one of 5 horizon entries in a label record
// (compositional action-relevant ontology).
export interface ReasoningHorizon {
  horizon_sec: number; // e.g. 0.5, 1.0, 2.0, 3.0, 4.0
  relation_to_ego?: string;
  hazard_event?: string[];
  cause?: string[];
  longitudinal_response?: string;
  lateral_response?: string;
  tactical_response?: string;
  rule_response?: string;
  confidence?: number;
  evidence?: string;
}

export interface ReasoningLabelRecord {
  schema_version?: string;
  sample_id: string;
  // v1 kept short `dataset`/`teacher`; the v2 producer writes the fuller
  // `dataset_name` / `teacher_model` / `teacher_provider` plus abstention.
  dataset?: string;
  dataset_name?: string;
  teacher?: string;
  teacher_model?: string;
  teacher_provider?: string;
  prompt_version?: string;
  abstained?: boolean;
  teacher_error?: string | null;
  horizons: ReasoningHorizon[];
  created_at?: string;
}

// ReasoningStatsEntry is one dataset/teacher/prompt_version bucket.
export interface ReasoningStatsEntry {
  dataset: string;
  teacher: string;
  teacher_provider: string;
  teacher_model: string;
  prompt_version: string;
  count: number;
}

export interface ReasoningLabelStats {
  entries: ReasoningStatsEntry[];
  total: number;
}

// ReasoningPromptVersion is one teacher/prompt_version partition of ONE
// dataset's label cache (GET /api/v1/reasoning-labels/prompt-versions).
export interface ReasoningPromptVersion {
  teacher: string;
  teacher_provider: string;
  teacher_model: string;
  prompt_version: string;
  count: number;
}

export interface ReasoningPromptVersionsResponse {
  dataset: string;
  prompt_versions: ReasoningPromptVersion[];
}

// ---------------------------------------------------------------------------
// Reasoning stats-detail (ODD / label composition)
// ---------------------------------------------------------------------------

// ConfidenceBucket is one bar of the teacher-confidence histogram.
export interface ConfidenceBucket {
  bucket: string; // e.g. "0.9-1.0"
  count: number;
}

// ReasoningStatsBlob is the aggregated composition over every horizon of every
// label in a (dataset, version, prompt_version) partition. by_field maps an
// ODD axis (relation_to_ego, hazard_event, cause, longitudinal_response,
// lateral_response, tactical_response, rule_response) to a value->count map.
export interface ReasoningStatsBlob {
  n_labels: number;
  horizon_count: number;
  by_field: Record<string, Record<string, number>>;
  confidence_histogram: ConfidenceBucket[];
}

// ReasoningStatsDetail is GET /api/v1/reasoning-labels/stats-detail. The first
// call for an uncomputed partition triggers a cold S3 scan (~50s); cached is
// true once the result is memoized server-side.
export interface ReasoningStatsDetail {
  dataset: string;
  version: string;
  prompt_version: string;
  teacher: string;
  teacher_provider?: string;
  teacher_model?: string;
  computed_at: string; // RFC3339
  cached: boolean;
  stats: ReasoningStatsBlob;
}

// SceneHit is one sample carrying a given (field=value) label. shard is the
// published shard that actually holds the sample in the requested version (the
// server resolves it from the shard indexes); available is false when the
// label exists but no published shard in this version contains the frame, so
// the UI links only real samples instead of a guessed shard that 404s.
export interface SceneHit {
  sample_id: string;
  shard?: string;
  available: boolean;
  dataset?: string;
  teacher?: string;
  prompt_version?: string;
}

// SceneSearchResult is GET /api/v1/scenes/search. total = returned hits;
// available = how many are present in this version's shards (linkable);
// truncated = the label index held more than the requested limit.
export interface SceneSearchResult {
  dataset: string;
  teacher: string;
  prompt_version: string;
  version?: string;
  field: string;
  value: string;
  scenes: SceneHit[];
  total: number;
  available: number;
  truncated: boolean;
}

// ---------------------------------------------------------------------------
// MLflow (proxy)
// ---------------------------------------------------------------------------

export interface MLflowExperiment {
  experiment_id: string;
  name: string;
  artifact_location: string;
  lifecycle_stage: string;
  run_count: number;
  last_update_time: number; // epoch millis
}

export interface MLflowMetric {
  key: string;
  value: number;
  timestamp: number; // epoch millis
  step: number;
}

export interface MLflowRun {
  run_id: string;
  run_name: string;
  experiment_id: string;
  status: "RUNNING" | "SCHEDULED" | "FINISHED" | "FAILED" | "KILLED";
  start_time: number; // epoch millis
  end_time: number; // epoch millis, 0 if running
  params: Record<string, string>;
  metrics: Record<string, number>; // latest value per key
  metric_history?: MLflowMetric[];
}

export interface MLflowRegisteredModel {
  name: string;
  latest_versions: {
    version: string;
    stage: string;
    run_id: string;
    status: string;
  }[];
}

// ---------------------------------------------------------------------------
// Cross-system experiments
// ---------------------------------------------------------------------------

export interface ExperimentDisplacementMetrics {
  ade?: number;
  fde?: number;
  gate_pass?: boolean;
}

export interface ExperimentExecution {
  execution_id: string;
  workflow_name: string;
  phase: FlytePhase | "";
  started_at: string;
  duration_s: number;
  url?: string;
}

export interface ExperimentModelVersion {
  name: string;
  version: string;
  role?: string;
  status: string;
  url?: string;
}

export interface ExperimentRecord {
  run_id: string;
  run_name: string;
  experiment_id: string;
  experiment_name: string;
  mlflow_status: MLflowRun["status"];
  start_time: number;
  end_time: number;
  dataset: string;
  dataset_version: string;
  data_fingerprint?: string;
  validation_scope?: string;
  validation_split_id?: string;
  backbone?: string;
  fusion_mode?: string;
  route_conditioning?: boolean;
  seed?: string;
  epochs?: string;
  epochs_completed?: string;
  lineage_status: "complete" | "partial" | "missing";
  primary_execution_id?: string;
  primary_execution_url?: string;
  mlflow_url?: string;
  evaluation?: ExperimentDisplacementMetrics;
  validation?: ExperimentDisplacementMetrics;
  train_execution?: ExperimentExecution;
  eval_execution?: ExperimentExecution;
  model_versions: ExperimentModelVersion[];
  params: Record<string, string>;
  tags: Record<string, string>;
  metrics: Record<string, number>;
}

export interface ExperimentsResponse {
  generated_at: string;
  summary: {
    total: number;
    running: number;
    failed: number;
    evaluated: number;
    registered: number;
    unlinked: number;
  };
  experiments: ExperimentRecord[];
}

// ---------------------------------------------------------------------------
// Flyte (proxy)
// ---------------------------------------------------------------------------

export type FlytePhase =
  | "UNDEFINED"
  | "QUEUED"
  | "RUNNING"
  | "SUCCEEDED"
  | "SUCCEEDING"
  | "FAILED"
  | "FAILING"
  | "ABORTED"
  | "ABORTING"
  | "TIMED_OUT";

export interface FlyteNode {
  node_id: string;
  display_name: string;
  phase: FlytePhase;
  started_at?: string; // RFC3339
  duration_s?: number;
  inputs?: Record<string, unknown>;
  outputs?: Record<string, unknown>;
}

export interface FlyteExecution {
  execution_id: string;
  workflow_name: string; // e.g. "wf_train_il"
  phase: FlytePhase;
  started_at: string; // RFC3339
  duration_s: number;
  nodes?: FlyteNode[];
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

export interface DashboardStats {
  total_samples: number;
  reasoning_labels: number;
  mlflow_runs: number;
  latest_ade: number | null;
  // False means MLflow was unreachable, so mlflow_runs/latest_ade are unknown
  // (not genuinely zero/null).
  mlflow_available: boolean;
}
