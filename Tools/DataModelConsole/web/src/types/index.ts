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
// Shard index (ADAS player)
// ---------------------------------------------------------------------------

// MemberRange locates one tar member's raw bytes for HTTP Range requests.
export interface MemberRange {
  offset: number;
  size: number;
}

// IndexSample is one frame entry of the shard index.
// ego_now: [speed, accel, yaw_rate, curvature] at this frame.
export interface IndexSample {
  key: string;
  frame_idx: number;
  members: Record<string, MemberRange>; // "cam_0.jpg" -> range
  ego_now: number[];
  has_reasoning: boolean;
}

// ShardIndex is GET .../shards/{shard}/index — everything the client needs
// to play a shard as a 10Hz video via ranged GETs on the presigned tar.
export interface ShardIndex {
  presigned_tar_url: string;
  expires_at: string; // RFC3339
  fps: number; // 10
  samples: IndexSample[];
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
  dataset?: string;
  teacher?: string;
  prompt_version?: string;
  horizons: ReasoningHorizon[];
  created_at?: string;
}

// ReasoningStatsEntry is one dataset/teacher/prompt_version bucket.
export interface ReasoningStatsEntry {
  dataset: string;
  teacher: string;
  prompt_version: string;
  count: number;
}

export interface ReasoningLabelStats {
  entries: ReasoningStatsEntry[];
  total: number;
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
}
