"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { Loader2, Search, X } from "lucide-react";

import { ErrorState } from "@/components/error-state";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/hooks/use-api";
import {
  getReasoningPromptVersions,
  getReasoningStatsDetail,
  listDatasetVersions,
  listDatasets,
  searchScenesByLabel,
} from "@/lib/api";
import { formatNumber, formatTimestamp, friendlyDataset } from "@/lib/format";
import type {
  ReasoningStatsDetail,
  SceneSearchResult,
} from "@/types";

// ---------------------------------------------------------------------------
// Chart palette (dataviz skill — references/palette.md, dark column). Each ODD
// axis is a distinct ENTITY, so it gets one fixed categorical hue and every bar
// in that axis shares it (bars compare magnitude within one series; color never
// tracks rank). Validated as a set on the slate surface: all 8 clear the L band,
// chroma floor, and 3:1 contrast (worst adjacent CVD ΔE 10.3 — the floor band,
// legal here because bars carry visible value labels as the secondary channel).
const HUE = {
  blue: "#3987e5",
  aqua: "#199e70",
  yellow: "#c98500",
  green: "#008300",
  violet: "#9085e9",
  red: "#e66767",
  magenta: "#d55181",
  orange: "#d95926",
} as const;

// ODD axis render order + friendly title + assigned hue. Fixed order = fixed
// hue assignment (never cycled). Fields absent from a partition are skipped.
const AXES: { field: string; title: string; hue: string }[] = [
  { field: "relation_to_ego", title: "Relation to ego", hue: HUE.blue },
  { field: "hazard_event", title: "Hazard event", hue: HUE.aqua },
  { field: "cause", title: "Cause", hue: HUE.yellow },
  { field: "longitudinal_response", title: "Longitudinal response", hue: HUE.green },
  { field: "lateral_response", title: "Lateral response", hue: HUE.violet },
  { field: "tactical_response", title: "Tactical response", hue: HUE.red },
  { field: "rule_response", title: "Rule response", hue: HUE.magenta },
];

const DATASETS_FALLBACK = ["l2d", "nvidia_av"];

function pctLabel(count: number, total: number): string {
  if (total <= 0) return "0%";
  const p = (count / total) * 100;
  return `${p < 10 ? p.toFixed(1) : p.toFixed(0)}%`;
}

// ---------------------------------------------------------------------------
// One ODD-axis horizontal bar chart. Bars are max-normalized (widest = the
// modal value), sorted desc, and each is a clickable button that drills into
// the matching scenes. Bar height 16px (< 24px cap), square at the baseline
// (left), 4px rounded data-end (right); labels/values wear text tokens only.
// ---------------------------------------------------------------------------
function AxisChart({
  title,
  hue,
  counts,
  active,
  onSelect,
}: {
  title: string;
  hue: string;
  counts: Record<string, number>;
  active: string | null;
  onSelect: (value: string) => void;
}) {
  const rows = useMemo(
    () => Object.entries(counts).sort((a, b) => b[1] - a[1]),
    [counts],
  );
  const total = useMemo(() => rows.reduce((s, [, c]) => s + c, 0), [rows]);
  const max = rows.length ? rows[0][1] : 1;

  return (
    <Card className="min-w-0 border-slate-800 bg-slate-950/50">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <span
            aria-hidden
            className="inline-block size-2.5 shrink-0 rounded-[3px]"
            style={{ background: hue }}
          />
          {title}
          {/* `total` is the sum of this axis's per-value counts (occurrences),
              NOT the horizon count: scalar axes skip blank horizons and the two
              multi-label axes count several members per horizon, so it differs
              from stats.horizon_count. Label it occurrences to match the per-bar
              percentages (which use `total` as the denominator). */}
          <span className="text-[10px] font-normal text-slate-500">
            {rows.length} values · {formatNumber(total)} occurrences
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1.5">
        {rows.map(([value, count]) => {
          const isActive = active === value;
          return (
            <button
              key={value}
              type="button"
              onClick={() => onSelect(value)}
              title={`${value}: ${formatNumber(count)} (${pctLabel(count, total)}) — click to list scenes`}
              className={`group flex w-full items-center gap-3 rounded-md px-1.5 py-1 text-left transition-colors hover:bg-slate-900/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 ${
                isActive ? "bg-slate-900/70 ring-1 ring-slate-600" : ""
              }`}
            >
              <span
                className="w-36 shrink-0 truncate text-right font-mono text-[11px] text-slate-300"
                title={value}
              >
                {value}
              </span>
              <span className="relative h-4 min-w-0 flex-1 overflow-hidden rounded-[4px] bg-slate-800/40">
                <span
                  className="absolute inset-y-0 left-0 rounded-r-[4px]"
                  style={{
                    width: `${Math.max((count / max) * 100, 1.5)}%`,
                    background: hue,
                  }}
                />
              </span>
              <span className="w-[5.5rem] shrink-0 text-right font-mono text-[11px] tabular-nums text-slate-400">
                {formatNumber(count)}
                <span className="ml-1 text-slate-600">
                  {pctLabel(count, total)}
                </span>
              </span>
            </button>
          );
        })}
        {rows.length === 0 && (
          <p className="py-2 text-center text-xs text-slate-500">No values</p>
        )}
      </CardContent>
    </Card>
  );
}

// Confidence histogram — a single-series magnitude chart (fixed 0.0..1.0
// buckets, so order is preserved, not sorted). Columns, blue series hue.
function ConfidenceHistogram({
  buckets,
}: {
  buckets: { bucket: string; count: number }[];
}) {
  const max = Math.max(1, ...buckets.map((b) => b.count));
  return (
    <Card className="border-slate-800 bg-slate-950/50">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <span
            aria-hidden
            className="inline-block size-2.5 shrink-0 rounded-[3px]"
            style={{ background: HUE.blue }}
          />
          Teacher confidence
          <span className="text-[10px] font-normal text-slate-500">
            per horizon
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex h-40 items-end gap-1.5">
          {buckets.map((b) => (
            <div
              key={b.bucket}
              className="flex min-w-0 flex-1 flex-col items-center justify-end gap-1"
              title={`${b.bucket}: ${formatNumber(b.count)}`}
            >
              <span className="font-mono text-[9px] tabular-nums text-slate-400">
                {b.count > 0 ? formatNumber(b.count) : ""}
              </span>
              <div
                className="w-full rounded-t-[4px]"
                style={{
                  height: `${(b.count / max) * 100}%`,
                  minHeight: b.count > 0 ? 2 : 0,
                  background: HUE.blue,
                }}
              />
              <span className="w-full truncate text-center font-mono text-[9px] text-slate-500">
                {b.bucket}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Scene drill-down drawer: lists sample_ids carrying the clicked field=value,
// each linking to its sample-detail page (shard derived from the id).
// ---------------------------------------------------------------------------
function SceneDrawer({
  dataset,
  version,
  promptVersion,
  field,
  value,
  onClose,
}: {
  dataset: string;
  version: string;
  promptVersion: string;
  field: string;
  value: string;
  onClose: () => void;
}) {
  const { data, error, loading, reload } = useApi<SceneSearchResult>(
    () => searchScenesByLabel(dataset, promptVersion, field, value, 200, version),
    [dataset, version, promptVersion, field, value],
  );

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Carry both version and prompt_version downstream so the linked sample/player
  // shows the SAME reasoning run the user was browsing (not an arbitrary one).
  const linkQuery = (() => {
    const q = new URLSearchParams();
    if (version) q.set("version", version);
    if (promptVersion) q.set("prompt_version", promptVersion);
    const s = q.toString();
    return s ? `?${s}` : "";
  })();
  const scenes = data?.scenes ?? [];

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div
        className="absolute inset-0 bg-black/60"
        onClick={onClose}
        aria-hidden
      />
      <div
        role="dialog"
        aria-label={`Scenes with ${field} = ${value}`}
        className="relative z-10 flex h-full w-full max-w-md flex-col border-l border-slate-800 bg-slate-950 shadow-xl"
      >
        <div className="flex items-start justify-between gap-3 border-b border-slate-800 p-4">
          <div className="min-w-0">
            <p className="text-[10px] uppercase tracking-wider text-slate-500">
              Scenes · {field}
            </p>
            <p className="truncate font-mono text-sm text-slate-200" title={value}>
              {value}
            </p>
            <p className="mt-0.5 text-xs text-slate-500">
              {friendlyDataset(dataset)} {version} ·{" "}
              {loading
                ? "…"
                : `${formatNumber(data?.available ?? 0)} of ${formatNumber(data?.total ?? 0)} in this version${data?.truncated ? " (first 200 shown)" : ""}`}
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onClose}
            aria-label="Close"
          >
            <X className="size-4" />
          </Button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {error ? (
            <ErrorState error={error} onRetry={reload} />
          ) : loading ? (
            <div className="space-y-2">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-8 w-full" />
              ))}
            </div>
          ) : scenes.length === 0 ? (
            <p className="text-sm text-slate-500">No matching scenes.</p>
          ) : (
            <ul className="space-y-1">
              {scenes.map((s) => {
                // Link only samples the server resolved to a real shard in this
                // version; an unavailable sample (label exists but frame not
                // packed here) renders as a non-link with a hint, never a 404.
                if (!s.available || !s.shard) {
                  return (
                    <li key={s.sample_id}>
                      <div
                        className="flex items-center justify-between gap-3 rounded-md border border-slate-800/60 bg-slate-900/20 px-3 py-1.5"
                        title="This label exists but the frame is not packed into the selected dataset version"
                      >
                        <span className="font-mono text-xs text-slate-500">
                          {s.sample_id}
                        </span>
                        <span className="font-mono text-[10px] text-slate-600">
                          not in {version}
                        </span>
                      </div>
                    </li>
                  );
                }
                const href = `/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(s.shard)}/samples/${encodeURIComponent(s.sample_id)}${linkQuery}`;
                return (
                  <li key={s.sample_id}>
                    <Link
                      href={href}
                      className="flex items-center justify-between gap-3 rounded-md border border-slate-800 bg-slate-900/40 px-3 py-1.5 transition-colors hover:border-slate-600 hover:bg-slate-900"
                    >
                      <span className="font-mono text-xs text-blue-400">
                        {s.sample_id}
                      </span>
                      <span className="font-mono text-[10px] text-slate-500">
                        {s.shard}
                      </span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
function ReasoningLabelsInner() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const urlDataset = searchParams.get("dataset") ?? "";
  const urlVersion = searchParams.get("version") ?? "";
  const urlPromptVersion = searchParams.get("prompt_version") ?? "";

  // Dataset options come from the API (l2d / nvidia_av), with a static fallback.
  const datasetsApi = useApi(listDatasets);
  const datasetNames = useMemo(() => {
    const names = (datasetsApi.data ?? []).map((d) => d.name);
    return names.length ? names : DATASETS_FALLBACK;
  }, [datasetsApi.data]);

  const dataset = useMemo(
    () => (datasetNames.includes(urlDataset) ? urlDataset : datasetNames[0]),
    [datasetNames, urlDataset],
  );

  // Dataset versions + reasoning prompt_versions for the chosen dataset.
  const versionsApi = useApi(
    () => listDatasetVersions(dataset),
    [dataset],
  );
  const promptVersionsApi = useApi(
    () => getReasoningPromptVersions(dataset),
    [dataset],
  );

  // Only offer versions that actually packed samples: reasoning-label stats are
  // keyed by prompt_version (not dataset version), so an empty version must not
  // be selectable — it would attribute the full label set to a 0-sample version
  // and scope scene availability to shards that do not exist. Fall back to the
  // raw list if the tally is unavailable so the selector is never empty.
  const versionList = useMemo(() => {
    const all = versionsApi.data ?? [];
    const nonEmpty = all.filter((v) => v.total_samples > 0);
    return nonEmpty.length ? nonEmpty : all;
  }, [versionsApi.data]);
  const version = useMemo(() => {
    if (versionList.length === 0) return "";
    return versionList.find((v) => v.version === urlVersion)?.version ??
      versionList[0].version;
  }, [versionList, urlVersion]);

  // Prompt versions, sorted by count desc so the default is the richest label
  // set (the one worth inspecting first).
  const promptVersions = useMemo(
    () => [...(promptVersionsApi.data ?? [])].sort((a, b) => b.count - a.count),
    [promptVersionsApi.data],
  );
  const promptVersion = useMemo(() => {
    if (promptVersions.length === 0) return "";
    return promptVersions.find((p) => p.prompt_version === urlPromptVersion)
      ?.prompt_version ?? promptVersions[0].prompt_version;
  }, [promptVersions, urlPromptVersion]);

  // Canonicalize the URL to the resolved (dataset, version, prompt_version)
  // once all three are known, so selections persist and deep links are stable.
  useEffect(() => {
    if (!dataset || !version || !promptVersion) return;
    if (
      urlDataset !== dataset ||
      urlVersion !== version ||
      urlPromptVersion !== promptVersion
    ) {
      const q = new URLSearchParams({
        dataset,
        version,
        prompt_version: promptVersion,
      });
      router.replace(`${pathname}?${q.toString()}`, { scroll: false });
    }
  }, [
    dataset,
    version,
    promptVersion,
    urlDataset,
    urlVersion,
    urlPromptVersion,
    pathname,
    router,
  ]);

  const setParam = useCallback(
    (patch: Record<string, string>) => {
      const q = new URLSearchParams(searchParams.toString());
      for (const [k, v] of Object.entries(patch)) q.set(k, v);
      // Changing dataset invalidates version/prompt_version; let them re-resolve.
      if ("dataset" in patch) {
        q.delete("version");
        q.delete("prompt_version");
      }
      router.replace(`${pathname}?${q.toString()}`, { scroll: false });
    },
    [pathname, router, searchParams],
  );

  // Stats-detail: a manual fetch (not useApi) so we can gate on all three
  // selectors being resolved and keep the "computing…" state honest through a
  // cold ~50s S3 scan.
  const [stats, setStats] = useState<ReasoningStatsDetail | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState<Error | null>(null);
  const [reloadGen, setReloadGen] = useState(0);

  useEffect(() => {
    if (!dataset || !version || !promptVersion) return;
    let cancelled = false;
    setStatsLoading(true);
    setStatsError(null);
    setStats(null);
    getReasoningStatsDetail(dataset, version, promptVersion)
      .then((d) => {
        if (!cancelled) {
          setStats(d);
          setStatsLoading(false);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setStatsError(e instanceof Error ? e : new Error(String(e)));
          setStatsLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [dataset, version, promptVersion, reloadGen]);

  const [drawer, setDrawer] = useState<{ field: string; value: string } | null>(
    null,
  );
  // Close the drawer whenever the partition under it changes.
  useEffect(() => {
    setDrawer(null);
  }, [dataset, version, promptVersion]);

  const byField = useMemo(
    () => stats?.stats.by_field ?? {},
    [stats],
  );
  const extraAxes = useMemo(
    () =>
      Object.keys(byField).filter(
        (f) => !AXES.some((a) => a.field === f),
      ),
    [byField],
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Reasoning Labels — ODD composition</h2>
        <p className="text-sm text-slate-400">
          Per-axis distribution of teacher reasoning labels across all 5 horizons
          — the operational-design-domain each dataset version + prompt covers.
          Click any bar to list the scenes carrying that label.
        </p>
      </div>

      {/* Selectors */}
      <Card className="border-slate-800 bg-slate-950/50">
        <CardContent className="flex flex-wrap items-end gap-4">
          <div className="flex flex-col gap-1">
            <label
              htmlFor="rl-dataset"
              className="text-[10px] uppercase tracking-wider text-slate-500"
            >
              Dataset
            </label>
            <select
              id="rl-dataset"
              value={dataset}
              onChange={(e) => setParam({ dataset: e.target.value })}
              className="h-9 rounded-md border border-slate-700 bg-slate-900 px-3 font-mono text-sm"
            >
              {datasetNames.map((d) => (
                <option key={d} value={d}>
                  {friendlyDataset(d)}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1">
            <label
              htmlFor="rl-version"
              className="text-[10px] uppercase tracking-wider text-slate-500"
            >
              Dataset version
            </label>
            <select
              id="rl-version"
              value={version}
              onChange={(e) => setParam({ version: e.target.value })}
              disabled={versionsApi.loading || versionList.length === 0}
              className="h-9 rounded-md border border-slate-700 bg-slate-900 px-3 font-mono text-sm disabled:opacity-50"
            >
              {versionList.length === 0 && <option value="">—</option>}
              {versionList.map((v) => (
                <option key={v.version} value={v.version}>
                  {v.version}
                  {v.has_manifest
                    ? ` · ${formatNumber(v.total_samples)} samples`
                    : ""}
                </option>
              ))}
            </select>
          </div>

          <div className="flex min-w-0 flex-col gap-1">
            <label
              htmlFor="rl-prompt"
              className="text-[10px] uppercase tracking-wider text-slate-500"
            >
              Prompt version (reasoning labels)
            </label>
            <select
              id="rl-prompt"
              value={promptVersion}
              onChange={(e) => setParam({ prompt_version: e.target.value })}
              disabled={
                promptVersionsApi.loading || promptVersions.length === 0
              }
              className="h-9 max-w-[26rem] rounded-md border border-slate-700 bg-slate-900 px-3 font-mono text-sm disabled:opacity-50"
            >
              {promptVersions.length === 0 && (
                <option value="">no reasoning labels</option>
              )}
              {promptVersions.map((p) => (
                <option key={p.prompt_version} value={p.prompt_version}>
                  {p.prompt_version} · {formatNumber(p.count)} labels
                </option>
              ))}
            </select>
          </div>
        </CardContent>
      </Card>

      {/* Header / provenance */}
      {versionsApi.error ? (
        <ErrorState
          error={versionsApi.error}
          onRetry={versionsApi.reload}
        />
      ) : promptVersionsApi.error ? (
        <ErrorState
          error={promptVersionsApi.error}
          onRetry={promptVersionsApi.reload}
        />
      ) : promptVersions.length === 0 && !promptVersionsApi.loading ? (
        <p className="text-sm text-slate-500">
          No reasoning labels for {friendlyDataset(dataset)}.
        </p>
      ) : statsError ? (
        <ErrorState
          error={statsError}
          onRetry={() => setReloadGen((g) => g + 1)}
        />
      ) : statsLoading ? (
        <Card className="border-slate-800 bg-slate-950/50">
          <CardContent className="flex items-center gap-3 py-8 text-sm text-slate-300">
            <Loader2 className="size-5 animate-spin text-blue-400" />
            <div>
              <p className="font-medium">Computing statistics…</p>
              <p className="text-xs text-slate-500">
                First run scans every label in this partition (~30–60s), then it
                is cached.
              </p>
            </div>
          </CardContent>
        </Card>
      ) : stats ? (
        <>
          <Card className="border-slate-800 bg-slate-950/50">
            <CardContent className="flex flex-wrap items-center gap-x-8 gap-y-3">
              <div>
                <p className="text-[10px] uppercase tracking-wider text-slate-500">
                  Labeled samples
                </p>
                <p className="font-mono text-2xl font-semibold text-slate-100">
                  {formatNumber(stats.stats.n_labels)}
                </p>
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-wider text-slate-500">
                  Horizons
                </p>
                <p className="font-mono text-2xl font-semibold text-slate-100">
                  {formatNumber(stats.stats.horizon_count)}
                </p>
              </div>
              <div className="min-w-0">
                <p className="text-[10px] uppercase tracking-wider text-slate-500">
                  Composition
                </p>
                <p className="truncate text-sm text-slate-300">
                  {friendlyDataset(stats.dataset)} {stats.version} ·{" "}
                  <span className="font-mono">{stats.prompt_version}</span>
                </p>
              </div>
              <div className="ml-auto flex flex-col items-end gap-1">
                <Badge
                  variant={stats.cached ? "secondary" : "outline"}
                  className="text-[10px]"
                >
                  {stats.cached ? "cached" : "freshly computed"}
                </Badge>
                <span className="text-[10px] text-slate-500">
                  computed {formatTimestamp(stats.computed_at)}
                </span>
              </div>
            </CardContent>
          </Card>

          {/* ODD axis charts */}
          <div className="grid gap-4 lg:grid-cols-2">
            {AXES.filter((a) => byField[a.field]).map((a) => (
              <AxisChart
                key={a.field}
                title={a.title}
                hue={a.hue}
                counts={byField[a.field]}
                active={drawer?.field === a.field ? drawer.value : null}
                onSelect={(value) => setDrawer({ field: a.field, value })}
              />
            ))}
            {extraAxes.map((f) => (
              <AxisChart
                key={f}
                title={f}
                hue={HUE.orange}
                counts={byField[f]}
                active={drawer?.field === f ? drawer.value : null}
                onSelect={(value) => setDrawer({ field: f, value })}
              />
            ))}
          </div>

          <ConfidenceHistogram
            buckets={stats.stats.confidence_histogram ?? []}
          />
        </>
      ) : null}

      {/* Fallback inspector cue */}
      <p className="flex items-center gap-2 text-xs text-slate-600">
        <Search className="size-3.5" />
        Tip: click a bar to open the matching scenes; each links to its
        sample-detail page.
      </p>

      {drawer && stats && (
        <SceneDrawer
          dataset={dataset}
          version={version}
          promptVersion={promptVersion}
          field={drawer.field}
          value={drawer.value}
          onClose={() => setDrawer(null)}
        />
      )}
    </div>
  );
}

export default function ReasoningLabelsPage() {
  return (
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <ReasoningLabelsInner />
    </Suspense>
  );
}
