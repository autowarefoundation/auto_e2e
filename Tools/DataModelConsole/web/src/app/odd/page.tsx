"use client";

// ODD scene search. ODD labels are scene-level catalog metadata (road,
// environment, traffic, image condition, ego events); each label is an
// interval inside its scene. The page builds a query from groups of
// conditions, lists the matching scenes with the intervals that matched on a
// time axis, and can expand any scene to every label it carries. The applied
// query lives in the URL so a search can be shared.

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import {
  ChartNoAxesColumn,
  ChevronLeft,
  ChevronRight,
  Layers,
  Loader2,
  Play,
  ShieldAlert,
} from "lucide-react";

import { QueryBuilder, emptyCondition } from "@/components/odd/query-builder";
import { SceneTimeline } from "@/components/odd/scene-timeline";
import { ErrorState } from "@/components/error-state";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/hooks/use-api";
import { ApiError, getODDScene, listDatasets, searchODDScenes } from "@/lib/api";
import { friendlyDataset } from "@/lib/format";
import {
  MAX_CONDITIONS,
  MAX_GROUPS,
  cleanQuery,
  conditionCount,
  describeQuery,
  keyLabel,
  namespaceLabel,
  namespaceOf,
  parseQueryParam,
  seconds,
  valueColor,
  valueLabel,
  withIDs,
} from "@/lib/odd";
import { cn } from "@/lib/utils";
import type { ODDFacet, ODDQuery, ODDScene, ODDSearchResponse } from "@/types";

const PAGE_SIZE = 20;
const APPLY_DELAY_MS = 350;

function groupByNamespace<T extends { key: string }>(items: T[]): [string, T[]][] {
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const namespace = namespaceOf(item.key);
    groups.set(namespace, [...(groups.get(namespace) ?? []), item]);
  }
  return [...groups.entries()];
}

function StatTile({
  label,
  value,
  detail,
  busy,
}: {
  label: string;
  value: string;
  detail?: React.ReactNode;
  busy?: boolean;
}) {
  return (
    <div className="min-w-0 rounded-xl border border-slate-800 bg-gradient-to-b from-slate-900/80 to-slate-950/60 p-4">
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-wider text-slate-500">
        {label}
        {busy && <Loader2 className="size-3 animate-spin text-slate-400" aria-label="Updating" />}
      </div>
      <div className="mt-1 text-2xl font-semibold tabular-nums text-slate-50">{value}</div>
      {detail && <div className="mt-1 text-xs text-slate-400">{detail}</div>}
    </div>
  );
}

function SceneCard({
  scene,
  rank,
  response,
  vocabulary,
  hasConditions,
}: {
  scene: ODDScene;
  rank: number;
  response: ODDSearchResponse;
  vocabulary: Map<string, ODDFacet>;
  hasConditions: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const all = useApi(
    () => getODDScene(response.dataset, response.version, scene.scene_uid),
    // The manifest digest changes when the LabelSet behind the version does.
    [response.dataset, response.version, response.manifest_sha256, scene.scene_uid],
    expanded,
  );
  const playerHref = `/scenes/${encodeURIComponent(response.dataset)}/${encodeURIComponent(scene.shard)}/0?version=${encodeURIComponent(response.version)}`;
  const shortShard = scene.shard.replace(/-train-\d+\.tar$/, "");

  return (
    <article
      aria-label={`Scene ${scene.shard}`}
      className="rounded-xl border border-slate-800 bg-slate-950/50 p-4 transition-colors hover:border-slate-700"
    >
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <span className="grid size-7 place-items-center rounded-md bg-slate-800 text-xs font-semibold tabular-nums text-slate-300">
          {rank}
        </span>
        <div className="min-w-0">
          <div className="truncate font-mono text-sm text-slate-100" title={scene.shard}>
            {shortShard}
          </div>
          <div className="text-[11px] tabular-nums text-slate-500">
            {seconds(scene.duration_ns)} · {scene.distance_m.toFixed(0)} m ·{" "}
            <span className="font-mono">{scene.scene_uid.slice(0, 8)}</span>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          <Button
            type="button"
            variant={expanded ? "secondary" : "ghost"}
            size="sm"
            aria-expanded={expanded}
            onClick={() => setExpanded(!expanded)}
          >
            <Layers className="size-3.5" />
            {expanded ? "Matched labels" : "All labels"}
          </Button>
          <Link
            href={playerHref}
            aria-label={`Open ${scene.shard} in the player`}
            className="inline-flex h-8 items-center gap-1.5 rounded-md bg-sky-500/15 px-3 text-xs font-medium text-sky-200 ring-1 ring-sky-500/40 transition-colors hover:bg-sky-500/25"
          >
            <Play className="size-3.5" /> Play
          </Link>
        </div>
      </header>

      <div className="mt-4">
        {!expanded ? (
          scene.lanes.length > 0 ? (
            <SceneTimeline lanes={scene.lanes} durationNs={scene.duration_ns} vocabulary={vocabulary} />
          ) : (
            <p className="text-xs text-slate-500">
              {hasConditions
                ? "This scene matched through “is not” conditions only, so there are no matched intervals to draw."
                : "The query has no conditions, so there are no matched intervals to draw."}{" "}
              Open all labels to see what this scene carries.
            </p>
          )
        ) : all.error ? (
          <ErrorState error={all.error} onRetry={all.reload} service="S3" />
        ) : all.loading || !all.data ? (
          <Skeleton className="h-40 w-full" />
        ) : (
          <div className="space-y-5">
            {groupByNamespace(all.data.scene.lanes).map(([namespace, lanes]) => (
              <section key={namespace} aria-label={namespaceLabel(namespace)}>
                <h4 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  {namespaceLabel(namespace)} · {lanes.length}
                </h4>
                <SceneTimeline lanes={lanes} durationNs={scene.duration_ns} vocabulary={vocabulary} />
              </section>
            ))}
          </div>
        )}
      </div>
    </article>
  );
}

function LabelExplorer({
  facets,
  matched,
  vocabulary,
  onPick,
  canAdd,
}: {
  facets: ODDFacet[];
  matched: number;
  vocabulary: Map<string, ODDFacet>;
  onPick: (key: string, value: string) => void;
  canAdd: boolean;
}) {
  const [filter, setFilter] = useState("");
  const shown = facets.filter(
    (f) => !filter || f.key.includes(filter.toLowerCase()) || keyLabel(f.key).toLowerCase().includes(filter.toLowerCase()),
  );
  return (
    <Card className="border-slate-800 bg-slate-950/50">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <ChartNoAxesColumn className="size-4 text-sky-400" /> Label explorer
        </CardTitle>
        <p className="text-xs text-slate-500">
          Share of the {matched} matched scenes carrying each value.{" "}
          {canAdd
            ? "Click a value to add it as a new group."
            : `The query is at its limit of ${MAX_GROUPS} groups or ${MAX_CONDITIONS} conditions.`}
        </p>
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter labels…"
          aria-label="Filter labels"
          className="mt-2 h-8 w-full rounded-md border border-slate-700 bg-slate-900 px-2 text-xs placeholder:text-slate-600"
        />
      </CardHeader>
      <CardContent className="max-h-[70vh] space-y-4 overflow-y-auto">
        {groupByNamespace(shown).map(([namespace, items]) => (
          <section key={namespace}>
            <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              {namespaceLabel(namespace)}
            </h4>
            <div className="space-y-3">
              {items.map((facet) => (
                <div key={facet.key}>
                  <div className="mb-1 truncate text-xs text-slate-300" title={facet.key}>
                    {keyLabel(facet.key)}
                  </div>
                  <div className="space-y-0.5">
                    {facet.values.map(({ value, scenes }) => (
                      <button
                        key={value}
                        type="button"
                        onClick={() => onPick(facet.key, value)}
                        disabled={!canAdd}
                        aria-label={`Add ${keyLabel(facet.key)} is ${valueLabel(value)}, ${scenes} scenes`}
                        className="group grid w-full grid-cols-[6.5rem_1fr_2.5rem] items-center gap-2 rounded px-1 py-0.5 text-left hover:bg-slate-900 disabled:cursor-not-allowed disabled:hover:bg-transparent"
                      >
                        <span className="truncate text-[11px] text-slate-400 group-hover:text-slate-200">
                          {valueLabel(value)}
                        </span>
                        <span className="relative h-2.5 overflow-hidden rounded-full bg-slate-800/70">
                          <span
                            className="absolute inset-y-0 left-0 rounded-full"
                            style={{
                              width: `${matched ? (scenes / matched) * 100 : 0}%`,
                              background: valueColor(vocabulary, facet.key, value),
                            }}
                          />
                        </span>
                        <span className="text-right text-[11px] tabular-nums text-slate-500">{scenes}</span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>
        ))}
      </CardContent>
    </Card>
  );
}

function ODDPageInner() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const datasets = useApi(listDatasets);

  const dataset = useMemo(() => {
    const items = datasets.data ?? [];
    const requested = searchParams.get("dataset");
    return (
      items.find((item) => item.name === requested)?.name ??
      items.find((item) => item.name === "kitscenes")?.name ??
      items[0]?.name ??
      ""
    );
  }, [datasets.data, searchParams]);
  const version =
    searchParams.get("version") ||
    datasets.data?.find((item) => item.name === dataset)?.version ||
    "";
  const qParam = searchParams.get("q");
  const applied = useMemo(() => parseQueryParam(qParam), [qParam]);
  const appliedKey = JSON.stringify(applied);
  const offset = Math.max(0, parseInt(searchParams.get("offset") ?? "", 10) || 0);

  const [draft, setDraft] = useState<ODDQuery>(() => withIDs(applied));
  // The query this page last wrote to the URL. While that write is in
  // flight the user may keep editing, so its arrival must not reset the
  // draft; any other new URL query (back/forward, a shared link) does.
  const selfApplied = useRef(appliedKey);
  useEffect(() => {
    if (appliedKey === selfApplied.current) return;
    selfApplied.current = appliedKey;
    setDraft((current) => (JSON.stringify(cleanQuery(current)) === appliedKey ? current : withIDs(applied)));
  }, [applied, appliedKey]);

  function navigate(next: { dataset?: string; query?: ODDQuery; offset?: number }, push = true) {
    const query = new URLSearchParams();
    query.set("dataset", next.dataset ?? dataset);
    if (next.dataset === undefined && version) query.set("version", version);
    const q = next.query ?? applied;
    if (q.groups.length > 0) query.set("q", JSON.stringify(q));
    if ((next.offset ?? 0) > 0) query.set("offset", String(next.offset));
    const url = `${pathname}?${query.toString()}`;
    if (push) router.push(url, { scroll: false });
    else router.replace(url, { scroll: false });
  }
  const navigateRef = useRef(navigate);
  navigateRef.current = navigate;

  // The dataset a switch is navigating to, until the URL shows it. A draft
  // write still pending from the old dataset must not land on the new one.
  const switchingTo = useRef<string | null>(null);
  useEffect(() => {
    if (switchingTo.current === dataset) switchingTo.current = null;
  }, [dataset]);

  // Apply the draft shortly after the last edit.
  useEffect(() => {
    const cleaned = cleanQuery(draft);
    if (JSON.stringify(cleaned) === appliedKey) return;
    const timer = setTimeout(() => {
      if (switchingTo.current !== null) return;
      selfApplied.current = JSON.stringify(cleaned);
      navigateRef.current({ query: cleaned }, false);
    }, APPLY_DELAY_MS);
    return () => clearTimeout(timer);
  }, [draft, appliedKey]);

  const search = useApi(
    () => searchODDScenes(dataset, version, applied, offset, PAGE_SIZE),
    [dataset, version, appliedKey, offset],
    Boolean(dataset && version),
  );
  const noLabels = search.error instanceof ApiError && search.error.status === 404;
  const response = search.data;
  const vocabulary = useMemo(
    () => new Map((response?.vocabulary ?? []).map((f) => [f.key, f])),
    [response?.vocabulary],
  );

  const canAdd = draft.groups.length < MAX_GROUPS && conditionCount(draft) < MAX_CONDITIONS;
  function addCondition(key: string, value: string) {
    if (!canAdd) return;
    setDraft((current) => ({
      ...current,
      groups: [...current.groups, { op: "and", conditions: [{ ...emptyCondition(key), values: [value] }] }],
    }));
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold">ODD Search</h2>
          <p className="max-w-2xl text-sm text-slate-400">
            Find scenes by their operational design domain labels. Every label is an interval inside
            its scene, so results show where in the scene each condition holds.
          </p>
        </div>
        {datasets.error ? null : (
          <div className="flex items-center gap-2">
            <select
              value={dataset}
              onChange={(event) => {
                switchingTo.current = event.target.value;
                setDraft({ op: "and", groups: [] });
                navigate({ dataset: event.target.value, query: { op: "and", groups: [] } });
              }}
              disabled={datasets.loading || !datasets.data?.length}
              className="h-9 rounded-md border border-slate-700 bg-slate-900 px-3 text-sm"
              aria-label="Dataset"
            >
              {!datasets.data?.length && <option value="">—</option>}
              {(datasets.data ?? []).map((item) => (
                <option key={item.name} value={item.name}>
                  {friendlyDataset(item.name)}
                </option>
              ))}
            </select>
            <span className="rounded-md bg-slate-900 px-2 py-1.5 font-mono text-xs text-slate-300 ring-1 ring-slate-800">
              {version || "—"}
            </span>
          </div>
        )}
      </div>

      {datasets.error ? (
        <ErrorState error={datasets.error} onRetry={datasets.reload} />
      ) : noLabels ? (
        <Card className="border-slate-800 bg-slate-950/50">
          <CardContent className="flex items-center gap-3 py-8 text-sm text-slate-400">
            <ShieldAlert className="size-5 text-slate-500" />
            No ODD labels for {friendlyDataset(dataset)} {version}.
          </CardContent>
        </Card>
      ) : search.error && (!response || response.dataset !== dataset || response.version !== version) ? (
        <ErrorState error={search.error} onRetry={search.reload} service="S3" />
      ) : !response || response.dataset !== dataset || response.version !== version ? (
        <div className="space-y-4">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            <StatTile
              label="Matched scenes"
              value={`${response.page.total}`}
              detail={`of ${response.labeled_scenes} scenes match.`}
              busy={search.loading}
            />
            <StatTile
              label="Share"
              value={`${response.labeled_scenes ? Math.round((response.page.total / response.labeled_scenes) * 100) : 0}%`}
              detail={<span className="block truncate" title={describeQuery(applied)}>{describeQuery(applied)}</span>}
            />
            <StatTile
              label="Labels from"
              value={`${response.label_version}`}
              detail={
                <span className="flex flex-wrap items-center gap-1.5">
                  <span className="font-mono">{response.dataset} {response.label_version}</span>
                  <span className="truncate font-mono text-slate-600" title={response.labelset_id}>
                    {response.labelset_id.slice(0, 14)}…
                  </span>
                  <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-200 ring-1 ring-amber-500/30">
                    {response.quality.certification_status}
                  </span>
                  <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300">
                    {response.quality.audit_status}
                  </span>
                </span>
              }
            />
          </div>

          <Card className="border-slate-800 bg-slate-950/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Query</CardTitle>
            </CardHeader>
            <CardContent>
              <QueryBuilder draft={draft} onChange={setDraft} vocabulary={response.vocabulary} />
            </CardContent>
          </Card>

          {search.error && (
            // The results below are from the previous query.
            <ErrorState error={search.error} onRetry={search.reload} service="S3" />
          )}
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
            <section aria-label="Matched scenes" className={cn("space-y-3", search.loading && "opacity-70")}>
              {response.scenes.length === 0 && response.page.total > 0 ? (
                // A shared or stale URL can point past the last page.
                <Card className="border-slate-800 bg-slate-950/50">
                  <CardContent className="flex flex-col items-center gap-3 py-10 text-sm text-slate-500">
                    This page is past the {response.page.total} matched scenes.
                    <Button variant="secondary" size="sm" onClick={() => navigate({ offset: 0 })}>
                      Back to the first page
                    </Button>
                  </CardContent>
                </Card>
              ) : response.scenes.length === 0 ? (
                <Card className="border-slate-800 bg-slate-950/50">
                  <CardContent className="py-10 text-center text-sm text-slate-500">
                    No scene matches this query. Loosen a condition or switch a group to ANY.
                  </CardContent>
                </Card>
              ) : (
                response.scenes.map((scene, i) => (
                  <SceneCard
                    key={scene.scene_uid}
                    scene={scene}
                    rank={response.page.offset + i + 1}
                    response={response}
                    vocabulary={vocabulary}
                    hasConditions={applied.groups.length > 0}
                  />
                ))
              )}
              {response.page.total > PAGE_SIZE && response.scenes.length > 0 && (
                <nav aria-label="Pages" className="flex items-center justify-between pt-1 text-xs text-slate-400">
                  <span className="tabular-nums">
                    {response.page.offset + 1}–{response.page.offset + response.scenes.length} of {response.page.total}
                  </span>
                  <span className="flex gap-2">
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={search.loading || response.page.offset === 0}
                      onClick={() => navigate({ offset: Math.max(0, response.page.offset - PAGE_SIZE) })}
                    >
                      <ChevronLeft className="size-3.5" /> Previous
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={search.loading || !response.page.more}
                      onClick={() => navigate({ offset: response.page.offset + PAGE_SIZE })}
                    >
                      Next <ChevronRight className="size-3.5" />
                    </Button>
                  </span>
                </nav>
              )}
            </section>
            <aside>
              <LabelExplorer
                facets={response.facets}
                matched={response.page.total}
                vocabulary={vocabulary}
                onPick={addCondition}
                canAdd={canAdd}
              />
            </aside>
          </div>
        </>
      )}
    </div>
  );
}

export default function ODDPage() {
  return (
    <Suspense fallback={<Skeleton className="h-64 w-full" />}>
      <ODDPageInner />
    </Suspense>
  );
}
