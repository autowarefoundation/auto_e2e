// Typed API client for the DataModelConsole Go API.
// All calls are client-side fetches against /api/v1/* (docs/DESIGN.md section 6).

import type {
  DashboardStats,
  Dataset,
  DatasetListResponse,
  FlyteExecution,
  MLflowExperiment,
  MLflowRegisteredModel,
  MLflowRun,
  ReasoningLabelRecord,
  ReasoningLabelStats,
  SampleDetail,
  SampleListResponse,
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

export function listShards(
  dataset: string,
  offset = 0,
  limit = 50,
): Promise<ShardListResponse> {
  const q = new URLSearchParams({
    offset: String(offset),
    limit: String(limit),
  });
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
): Promise<ShardListResponse["shards"]> {
  const res = await listShards(dataset, 0, 200);
  return [...(res.shards ?? [])].sort((a, b) => a.name.localeCompare(b.name));
}

export function listSamples(
  dataset: string,
  shard: string,
): Promise<SampleListResponse> {
  return apiFetch<SampleListResponse>(
    `/api/v1/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard)}/samples`,
  );
}

export function getSample(
  dataset: string,
  shard: string,
  key: string,
): Promise<SampleDetail> {
  return apiFetch<SampleDetail>(
    `/api/v1/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard)}/samples/${encodeURIComponent(key)}`,
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
): string {
  const base = `${BASE_URL}/api/v1/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard)}/samples/${encodeURIComponent(key)}/image/cam_${cam}`;
  if (range && range.size > 0) {
    const q = new URLSearchParams({
      offset: String(range.offset),
      size: String(range.size),
    });
    return `${base}?${q.toString()}`;
  }
  return base;
}

// getShardIndex fetches the playback index: per-frame member byte ranges +
// ego_now / ego_future signals (ADAS player data source).
export function getShardIndex(
  dataset: string,
  shard: string,
): Promise<ShardIndex> {
  return apiFetch<ShardIndex>(
    `/api/v1/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard)}/index`,
  );
}

// ---------------------------------------------------------------------------
// Reasoning labels
// ---------------------------------------------------------------------------

export function getReasoningLabelStats(): Promise<ReasoningLabelStats> {
  return apiFetch<ReasoningLabelStats>("/api/v1/reasoning-labels/stats");
}

export function getReasoningLabel(
  dataset: string,
  sampleId: string,
): Promise<ReasoningLabelRecord> {
  return apiFetch<ReasoningLabelRecord>(
    `/api/v1/reasoning-labels/${encodeURIComponent(dataset)}/${encodeURIComponent(sampleId)}`,
  );
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
