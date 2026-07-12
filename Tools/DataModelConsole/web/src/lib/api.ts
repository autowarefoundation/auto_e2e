// Typed API client for the DataModelConsole Go API.
// All calls are client-side fetches against /api/v1/* (docs/DESIGN.md section 6).

import type {
  DashboardStats,
  Dataset,
  DatasetListResponse,
  DatasetVersionsResponse,
  FlyteExecution,
  MLflowExperiment,
  MLflowRegisteredModel,
  MLflowRun,
  ReasoningLabelRecord,
  ReasoningLabelStats,
  ReasoningPromptVersionsResponse,
  ReasoningStatsDetail,
  SampleDetail,
  SampleListResponse,
  SceneSearchResult,
  ShardIndex,
  ShardListResponse,
} from "@/types";

// Same-origin by default (ALB routes /api -> Go API). Local dev overrides via
// NEXT_PUBLIC_API_URL=http://localhost:8080 in .env.local.
const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

export class ApiError extends Error {
  readonly status: number;
  readonly url: string;

  constructor(status: number, url: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.url = url;
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${path}`;
  let res: Response;
  try {
    res = await fetch(url, {
      ...init,
      headers: { Accept: "application/json", ...init?.headers },
    });
  } catch (err) {
    throw new ApiError(
      0,
      url,
      `Network error: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.text();
      if (body) detail = body.slice(0, 500);
    } catch {
      // keep statusText
    }
    throw new ApiError(res.status, url, `API ${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

export function getDashboardStats(): Promise<DashboardStats> {
  return apiFetch<DashboardStats>("/api/v1/stats");
}

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

export async function listDatasets(): Promise<Dataset[]> {
  const res = await apiFetch<DatasetListResponse>("/api/v1/datasets");
  return res.datasets ?? [];
}

// listDatasetVersions returns every published version of a dataset (newest
// first) with its whole-training composition, powering the version selector.
export async function listDatasetVersions(
  dataset: string,
): Promise<DatasetVersionsResponse["versions"]> {
  const res = await apiFetch<DatasetVersionsResponse>(
    `/api/v1/datasets/${encodeURIComponent(dataset)}/versions`,
  );
  return res.versions ?? [];
}

// versionParam builds the "&version=" / "?version=" suffix for an optional
// pinned dataset version. Empty/undefined means "let the API auto-resolve the
// newest version" (the historical behavior), so nothing is appended.
function versionParam(version: string | undefined, sep: "?" | "&"): string {
  return version ? `${sep}version=${encodeURIComponent(version)}` : "";
}

export function listShards(
  dataset: string,
  offset = 0,
  limit = 50,
  version?: string,
): Promise<ShardListResponse> {
  const q = new URLSearchParams({
    offset: String(offset),
    limit: String(limit),
  });
  if (version) q.set("version", version);
  return apiFetch<ShardListResponse>(
    `/api/v1/datasets/${encodeURIComponent(dataset)}/shards?${q.toString()}`,
  );
}

// listShardsForEpisode returns a dataset's shards name-sorted (playback order).
// Groundwork for same-trip continuity (auto-advance to the next shard at
// end-of-shard); today each dataset has a single shard, so live FrameStore
// stitching is deferred until a second shard lands.
export async function listShardsForEpisode(
  dataset: string,
  version?: string,
): Promise<ShardListResponse["shards"]> {
  const res = await listShards(dataset, 0, 200, version);
  return [...(res.shards ?? [])].sort((a, b) => a.name.localeCompare(b.name));
}

export function listSamples(
  dataset: string,
  shard: string,
  version?: string,
  offset = 0,
  limit = 0,
): Promise<SampleListResponse> {
  const q = new URLSearchParams();
  if (version) q.set("version", version);
  if (offset > 0) q.set("offset", String(offset));
  if (limit > 0) q.set("limit", String(limit));
  const qs = q.toString();
  return apiFetch<SampleListResponse>(
    `/api/v1/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard)}/samples${qs ? `?${qs}` : ""}`,
  );
}

export function getSample(
  dataset: string,
  shard: string,
  key: string,
  version?: string,
): Promise<SampleDetail> {
  return apiFetch<SampleDetail>(
    `/api/v1/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard)}/samples/${encodeURIComponent(key)}${versionParam(version, "?")}`,
  );
}

// getSampleImageUrl builds the raw JPEG endpoint URL for an <img src>.
// cam is passed as the "cam_${n}" identifier the API requires. When the tar
// byte range is known (from the shard index) it is passed as ?offset=&size=
// so the API serves the member with a bounded S3 range GET instead of a
// full-shard tar scan.
export function getSampleImageUrl(
  dataset: string,
  shard: string,
  key: string,
  cam: number,
  range?: { offset: number; size: number },
  version?: string,
): string {
  const base = `${BASE_URL}/api/v1/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard)}/samples/${encodeURIComponent(key)}/image/cam_${cam}`;
  const q = new URLSearchParams();
  if (range && range.size > 0) {
    q.set("offset", String(range.offset));
    q.set("size", String(range.size));
  }
  if (version) q.set("version", version);
  const qs = q.toString();
  return qs ? `${base}?${qs}` : base;
}

// getShardBlobUrl builds the contiguous-range endpoint URL. The player fetches
// one span covering a whole window of frames' camera members in a single GET
// and slices the JPEGs out client-side (using the per-member offsets from the
// shard index), amortizing one network round trip across many frames — the
// difference between playback that fills its buffer at 10Hz and one that
// starves on per-image latency.
export function getShardBlobUrl(
  dataset: string,
  shard: string,
  offset: number,
  size: number,
  version?: string,
): string {
  const base = `${BASE_URL}/api/v1/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard)}/blob`;
  const q = new URLSearchParams({ offset: String(offset), size: String(size) });
  if (version) q.set("version", version);
  return `${base}?${q.toString()}`;
}

// getShardIndex fetches the playback index: per-frame member byte ranges +
// ego_now / ego_future signals (ADAS player data source).
export function getShardIndex(
  dataset: string,
  shard: string,
  version?: string,
): Promise<ShardIndex> {
  return apiFetch<ShardIndex>(
    `/api/v1/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard)}/index${versionParam(version, "?")}`,
  );
}

// ---------------------------------------------------------------------------
// Reasoning labels
// ---------------------------------------------------------------------------

export function getReasoningLabelStats(): Promise<ReasoningLabelStats> {
  return apiFetch<ReasoningLabelStats>("/api/v1/reasoning-labels/stats");
}

// getReasoningPromptVersions lists ONE dataset's reasoning-label
// teacher/prompt_version partitions with per-partition counts (the label
// version axis shown on the dataset detail page).
export async function getReasoningPromptVersions(
  dataset: string,
): Promise<ReasoningPromptVersionsResponse["prompt_versions"]> {
  const res = await apiFetch<ReasoningPromptVersionsResponse>(
    `/api/v1/reasoning-labels/prompt-versions?dataset=${encodeURIComponent(dataset)}`,
  );
  return res.prompt_versions ?? [];
}

export function getReasoningLabel(
  dataset: string,
  sampleId: string,
  promptVersion?: string,
): Promise<ReasoningLabelRecord> {
  // Pin the prompt_version so the player shows labels from ONE run, not an
  // arbitrary partition when several prompt_versions exist for a sample.
  const qs = promptVersion
    ? `?prompt_version=${encodeURIComponent(promptVersion)}`
    : "";
  return apiFetch<ReasoningLabelRecord>(
    `/api/v1/reasoning-labels/${encodeURIComponent(dataset)}/${encodeURIComponent(sampleId)}${qs}`,
  );
}

// getReasoningStatsDetail fetches the aggregated ODD / label composition for
// one (dataset, version, prompt_version) partition: per-field value counts +
// a confidence histogram over every horizon. The FIRST call for an uncomputed
// partition triggers a cold S3 scan the API caches afterward — that can take
// ~50s, so a generous 120s client timeout is used to avoid a false error.
export function getReasoningStatsDetail(
  dataset: string,
  version: string,
  promptVersion: string,
): Promise<ReasoningStatsDetail> {
  const q = new URLSearchParams({
    dataset,
    version,
    prompt_version: promptVersion,
  });
  return apiFetch<ReasoningStatsDetail>(
    `/api/v1/reasoning-labels/stats-detail?${q.toString()}`,
    { signal: AbortSignal.timeout(120_000) },
  );
}

// searchScenesByLabel lists sample_ids in a (dataset, prompt_version) partition
// whose label carries field=value on any horizon (the drill-down behind a
// clicked ODD bar).
export function searchScenesByLabel(
  dataset: string,
  promptVersion: string,
  field: string,
  value: string,
  limit = 50,
  version?: string,
): Promise<SceneSearchResult> {
  const q = new URLSearchParams({
    dataset,
    prompt_version: promptVersion,
    field,
    value,
    limit: String(limit),
  });
  // version scopes which published shards a scene can resolve into, so the
  // drawer links to the shard that actually holds each sample.
  if (version) q.set("version", version);
  return apiFetch<SceneSearchResult>(`/api/v1/scenes/search?${q.toString()}`);
}

// ---------------------------------------------------------------------------
// MLflow proxy
// ---------------------------------------------------------------------------

export function listExperiments(): Promise<MLflowExperiment[]> {
  return apiFetch<MLflowExperiment[]>("/api/v1/mlflow/experiments");
}

export function listRuns(experimentId: string): Promise<MLflowRun[]> {
  return apiFetch<MLflowRun[]>(
    `/api/v1/mlflow/experiments/${encodeURIComponent(experimentId)}/runs`,
  );
}

export function getRun(runId: string): Promise<MLflowRun> {
  return apiFetch<MLflowRun>(
    `/api/v1/mlflow/runs/${encodeURIComponent(runId)}`,
  );
}

export function listRegisteredModels(): Promise<MLflowRegisteredModel[]> {
  return apiFetch<MLflowRegisteredModel[]>("/api/v1/mlflow/models");
}

// ---------------------------------------------------------------------------
// Flyte proxy
// ---------------------------------------------------------------------------

export function listExecutions(limit = 50): Promise<FlyteExecution[]> {
  return apiFetch<FlyteExecution[]>(`/api/v1/flyte/executions?limit=${limit}`);
}

export function getExecution(executionId: string): Promise<FlyteExecution> {
  return apiFetch<FlyteExecution>(
    `/api/v1/flyte/executions/${encodeURIComponent(executionId)}`,
  );
}
