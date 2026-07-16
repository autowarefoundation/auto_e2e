// Typed API client for the DataModelConsole Go API.
// All calls are client-side fetches against /api/v1/* (docs/DESIGN.md section 6).

import type {
  DashboardStats,
  Dataset,
  DatasetListResponse,
  DatasetVersionsResponse,
  FlyteExecution,
  GeoJSONFeatureCollection,
  GeoStats,
  MLflowExperiment,
  MLflowRegisteredModel,
  MLflowRun,
  OverlayModel,
  OverlayModelsResponse,
  ReasoningLabelRecord,
  ReasoningLabelStats,
  ReasoningPromptVersionsResponse,
  ReasoningStatsDetail,
  TokenPage,
  SampleDetail,
  SampleListResponse,
  SceneSearchResult,
  ShardIndex,
  ShardListResponse,
  RigProjectionDocument,
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

async function apiFetchResponse(
  path: string,
  accept: string,
): Promise<Response> {
  const url = `${BASE_URL}${path}`;
  let res: Response;
  try {
    res = await fetch(url, { headers: { Accept: accept } });
  } catch (err) {
    throw new ApiError(
      0,
      url,
      `Network error: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  if (!res.ok) {
    const detail = (await res.text().catch(() => res.statusText)).slice(0, 500);
    throw new ApiError(res.status, url, `API ${res.status}: ${detail}`);
  }
  return res;
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

// listShardsForEpisode returns every dataset shard name-sorted (playback
// order). Publications can exceed one API page (KITScenes has 533 shards), so
// follow the offset continuation until the server reports no more results.
export async function listShardsForEpisode(
  dataset: string,
  version?: string,
): Promise<ShardListResponse["shards"]> {
  const shards: ShardListResponse["shards"] = [];
  let offset = 0;
  while (true) {
    const res = await listShards(dataset, offset, 1000, version);
    const page = res.shards ?? [];
    shards.push(...page);
    if (!res.page.more) break;
    if (page.length === 0) {
      throw new Error("shard pagination made no progress");
    }
    offset = res.page.offset + page.length;
  }
  return shards.sort((a, b) => a.name.localeCompare(b.name));
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
// Model trajectory overlays and geographic products
// ---------------------------------------------------------------------------

const OVERLAY_MODELS_PAGE_LIMIT = 100;
const MAX_OVERLAY_MODEL_PAGES = 20;

export async function listShardOverlayModels(
  dataset: string,
  shard: string,
  version?: string,
): Promise<OverlayModelsResponse> {
  const modelsByID = new Map<string, OverlayModel>();
  const seenPageTokens = new Set<string>();
  let pageToken = "";
  let coordinates:
    | Pick<OverlayModelsResponse, "dataset" | "version" | "shard">
    | undefined;

  // DynamoDB may return a LastEvaluatedKey when a page exactly reaches Limit,
  // even if no item follows it. Permit one empty terminal probe after the
  // bounded data pages, but never accept a 21st page containing models.
  for (let page = 0; page <= MAX_OVERLAY_MODEL_PAGES; page++) {
    if (pageToken) {
      if (seenPageTokens.has(pageToken)) {
        throw new Error("overlay model pagination token entered a cycle");
      }
      seenPageTokens.add(pageToken);
    }

    const query = new URLSearchParams({
      limit: String(OVERLAY_MODELS_PAGE_LIMIT),
    });
    const requestedVersion = version ?? coordinates?.version;
    if (requestedVersion) query.set("version", requestedVersion);
    if (pageToken) query.set("page_token", pageToken);

    const response = await apiFetch<OverlayModelsResponse>(
      `/api/v1/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard)}/overlay-models?${query.toString()}`,
    );
    if (
      response.dataset !== dataset ||
      response.shard !== shard ||
      !response.version ||
      (requestedVersion && response.version !== requestedVersion)
    ) {
      throw new Error("overlay model pagination returned invalid coordinates");
    }
    if (!coordinates) {
      coordinates = {
        dataset: response.dataset,
        version: response.version,
        shard: response.shard,
      };
    } else if (
      response.dataset !== coordinates.dataset ||
      response.version !== coordinates.version ||
      response.shard !== coordinates.shard
    ) {
      throw new Error("overlay model pagination changed coordinates");
    }

    if (page === MAX_OVERLAY_MODEL_PAGES) {
      if (
        (response.models?.length ?? 0) !== 0 ||
        response.next_page_token
      ) {
        throw new Error(
          `overlay model pagination exceeded ${MAX_OVERLAY_MODEL_PAGES} pages`,
        );
      }
      return {
        ...coordinates,
        models: [...modelsByID.values()].sort((a, b) => {
          const versionOrder = b.model_version - a.model_version;
          if (versionOrder !== 0) return versionOrder;
          if (a.model_artifact_id < b.model_artifact_id) return -1;
          if (a.model_artifact_id > b.model_artifact_id) return 1;
          return 0;
        }),
      };
    }

    for (const model of response.models ?? []) {
      if (!modelsByID.has(model.model_artifact_id)) {
        modelsByID.set(model.model_artifact_id, model);
      }
    }

    pageToken = response.next_page_token ?? "";
    if (!pageToken) {
      return {
        ...coordinates,
        models: [...modelsByID.values()].sort((a, b) => {
          const versionOrder = b.model_version - a.model_version;
          if (versionOrder !== 0) return versionOrder;
          if (a.model_artifact_id < b.model_artifact_id) return -1;
          if (a.model_artifact_id > b.model_artifact_id) return 1;
          return 0;
        }),
      };
    }
  }

  throw new Error(
    `overlay model pagination exceeded ${MAX_OVERLAY_MODEL_PAGES} pages`,
  );
}

export async function getShardOverlay(
  dataset: string,
  shard: string,
  modelArtifactId: string,
  version?: string,
): Promise<ArrayBuffer> {
  const response = await apiFetchResponse(
    `/api/v1/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard)}/overlays/${encodeURIComponent(modelArtifactId)}${versionParam(version, "?")}`,
    "application/vnd.auto-e2e.overlay",
  );
  return response.arrayBuffer();
}

export function getRigProjection(
  dataset: string,
  version?: string,
): Promise<RigProjectionDocument> {
  return apiFetch<RigProjectionDocument>(
    `/api/v1/datasets/${encodeURIComponent(dataset)}/rig-projection${versionParam(version, "?")}`,
  );
}

export function getShardRigProjection(
  dataset: string,
  shard: string,
  version?: string,
): Promise<RigProjectionDocument> {
  return apiFetch<RigProjectionDocument>(
    `/api/v1/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard)}/rig-projection${versionParam(version, "?")}`,
  );
}

export function getGeoStats(
  dataset: string,
  version?: string,
): Promise<GeoStats> {
  return apiFetch<GeoStats>(
    `/api/v1/datasets/${encodeURIComponent(dataset)}/geo-stats${versionParam(version, "?")}`,
  );
}

export function getGeoHeatmap(
  heatmapURL: string,
): Promise<GeoJSONFeatureCollection> {
  return apiFetch<GeoJSONFeatureCollection>(heatmapURL);
}

export async function getEpisodeGPSPath(
  dataset: string,
  episode: string,
  version?: string,
): Promise<ArrayBuffer> {
  const response = await apiFetchResponse(
    `/api/v1/datasets/${encodeURIComponent(dataset)}/geo/episodes/${encodeURIComponent(episode)}${versionParam(version, "?")}`,
    "application/octet-stream",
  );
  return response.arrayBuffer();
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
  version?: string,
): Promise<ReasoningPromptVersionsResponse["prompt_versions"]> {
  const query = new URLSearchParams({ dataset });
  if (version) query.set("version", version);
  const res = await apiFetch<ReasoningPromptVersionsResponse>(
    `/api/v1/reasoning-labels/prompt-versions?${query.toString()}`,
  );
  return res.prompt_versions ?? [];
}

export function getReasoningLabel(
  dataset: string,
  sampleId: string,
  promptVersion?: string,
  version?: string,
  teacher?: string,
): Promise<ReasoningLabelRecord> {
  const query = new URLSearchParams();
  if (teacher) query.set("teacher", teacher);
  if (promptVersion) query.set("prompt_version", promptVersion);
  if (version) query.set("version", version);
  const qs = query.toString();
  return apiFetch<ReasoningLabelRecord>(
    `/api/v1/reasoning-labels/${encodeURIComponent(dataset)}/${encodeURIComponent(sampleId)}${qs ? `?${qs}` : ""}`,
  );
}

// getReasoningStatsDetail fetches the aggregated ODD / label composition for
// one (dataset, version, prompt_version) partition: per-field value counts +
// a confidence histogram over every horizon. Reads are served from the
// materialized DynamoDB index and never trigger a browser-driven S3 scan.
export function getReasoningStatsDetail(
  dataset: string,
  version: string,
  promptVersion: string,
  teacher?: string,
): Promise<ReasoningStatsDetail> {
  const q = new URLSearchParams({ dataset, version, prompt_version: promptVersion });
  if (teacher) q.set("teacher", teacher);
  return apiFetch<ReasoningStatsDetail>(
    `/api/v1/reasoning-labels/stats-detail?${q.toString()}`,
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
  teacher?: string,
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
  if (teacher) q.set("teacher", teacher);
  return apiFetch<SceneSearchResult>(`/api/v1/scenes/search?${q.toString()}`);
}

// ---------------------------------------------------------------------------
// MLflow proxy
// ---------------------------------------------------------------------------

export function listExperimentsPage(
  pageToken = "",
  maxResults = 100,
): Promise<TokenPage<MLflowExperiment>> {
  const query = new URLSearchParams({ max_results: String(maxResults) });
  if (pageToken) query.set("page_token", pageToken);
  return apiFetch<TokenPage<MLflowExperiment> | MLflowExperiment[]>(
    `/api/v1/mlflow/experiments?${query.toString()}`,
  ).then(normalizeTokenPage);
}

export function listRunsPage(
  experimentId: string,
  pageToken = "",
  maxResults = 100,
): Promise<TokenPage<MLflowRun>> {
  const query = new URLSearchParams({ max_results: String(maxResults) });
  if (pageToken) query.set("page_token", pageToken);
  return apiFetch<TokenPage<MLflowRun> | MLflowRun[]>(
    `/api/v1/mlflow/experiments/${encodeURIComponent(experimentId)}/runs?${query.toString()}`,
  ).then(normalizeTokenPage);
}

export function getRun(runId: string): Promise<MLflowRun> {
  return apiFetch<MLflowRun>(
    `/api/v1/mlflow/runs/${encodeURIComponent(runId)}`,
  );
}

export function listRegisteredModelsPage(
  pageToken = "",
  maxResults = 100,
): Promise<TokenPage<MLflowRegisteredModel>> {
  const query = new URLSearchParams({ max_results: String(maxResults) });
  if (pageToken) query.set("page_token", pageToken);
  return apiFetch<
    TokenPage<MLflowRegisteredModel> | MLflowRegisteredModel[]
  >(
    `/api/v1/mlflow/models?${query.toString()}`,
  ).then(normalizeTokenPage);
}

// ---------------------------------------------------------------------------
// Flyte proxy
// ---------------------------------------------------------------------------

export function listExecutionsPage(
  limit = 50,
  pageToken = "",
): Promise<TokenPage<FlyteExecution>> {
  const query = new URLSearchParams({ limit: String(limit) });
  if (pageToken) query.set("token", pageToken);
  return apiFetch<TokenPage<FlyteExecution> | FlyteExecution[]>(
    `/api/v1/flyte/executions?${query.toString()}`,
  ).then(normalizeTokenPage);
}

function normalizeTokenPage<T>(response: TokenPage<T> | T[]): TokenPage<T> {
  if (Array.isArray(response)) {
    return { items: response };
  }
  if (!response || !Array.isArray(response.items)) {
    throw new Error("Invalid paginated API response.");
  }
  return response;
}

export function getExecution(executionId: string): Promise<FlyteExecution> {
  return apiFetch<FlyteExecution>(
    `/api/v1/flyte/executions/${encodeURIComponent(executionId)}`,
  );
}
