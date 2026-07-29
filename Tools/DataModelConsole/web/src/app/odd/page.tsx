"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  BookOpen,
  ChartNoAxesColumn,
  Database,
  ExternalLink,
  LoaderCircle,
  Plus,
  Play,
  RotateCcw,
  Search,
  Trash2,
} from "lucide-react";

import { ErrorState } from "@/components/error-state";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/hooks/use-api";
import {
  getODDLabelSets,
  getODDOntology,
  getODDOperations,
  getODDStatistics,
  launchODDDatasetLabeler,
  listExecutionsPage,
  retryODDDatasetLabeler,
  searchODDScenesStructured,
} from "@/lib/api";
import type {
  FlyteExecution,
  ODDKeyStatistic,
  ODDOperationalState,
  ODDOperationsCapability,
  ODDRatioInterval,
  ODDSearchPredicate,
  ODDStatus,
  ODDStructuredSearchRequest,
  ODDValueStatistic,
} from "@/types";

const DEFAULT_DATASET = "kitscenes";
const DEFAULT_VERSION = "v3.0";
const TABS = [
  { id: "overview", label: "Overview", icon: ChartNoAxesColumn },
  { id: "search", label: "Search", icon: Search },
  { id: "ontology", label: "Ontology", icon: BookOpen },
  { id: "labelsets", label: "LabelSets", icon: Database },
] as const;
const NAMESPACES = ["odd", "event", "perception"] as const;
const WEIGHTINGS = ["scene", "duration", "distance"] as const;
const STATUS_STYLE: Record<string, string> = {
  valid: "border-emerald-800 text-emerald-300",
  unavailable: "border-slate-700 text-slate-400",
  not_observable: "border-amber-800 text-amber-300",
  ambiguous: "border-rose-800 text-rose-300",
};

type Tab = (typeof TABS)[number]["id"];
type Namespace = (typeof NAMESPACES)[number];
type Weighting = (typeof WEIGHTINGS)[number];

function isTab(value: string | null): value is Tab {
  return TABS.some((tab) => tab.id === value);
}

function isODDDatasetLabelerExecution(execution: FlyteExecution): boolean {
  const name = execution.workflow_name.toLowerCase();
  return (
    name === "odd-dataset-labeler" ||
    name.endsWith("wf_generate_odd_labelset") ||
    name.endsWith("wf-generate-odd-labelset")
  );
}

function isActiveFlytePhase(phase: FlyteExecution["phase"]): boolean {
  return ["QUEUED", "RUNNING", "SUCCEEDING"].includes(phase);
}

function isFailedFlytePhase(phase: FlyteExecution["phase"]): boolean {
  return ["FAILED", "FAILING", "ABORTED", "ABORTING", "TIMED_OUT"].includes(
    phase,
  );
}

function oddOperationalState(
  catalogState: ODDOperationalState,
  executions: FlyteExecution[],
): ODDOperationalState {
  const latest = executions[0];
  if (!latest) return catalogState;
  if (isActiveFlytePhase(latest.phase)) return "running";
  if (isFailedFlytePhase(latest.phase)) return "failed";
  if (latest.phase === "SUCCEEDED") {
    return catalogState === "ready" ? "ready" : "failed";
  }
  return catalogState;
}

function oddRunState(
  execution: FlyteExecution,
  index: number,
  hasReadyLabelSet: boolean,
): ODDOperationalState {
  if (isActiveFlytePhase(execution.phase)) return "running";
  if (isFailedFlytePhase(execution.phase)) return "failed";
  if (execution.phase === "SUCCEEDED") {
    return index === 0 && hasReadyLabelSet ? "ready" : "superseded";
  }
  return "failed";
}

function oddStateTone(
  state: ODDOperationalState,
): "success" | "warning" | "error" | "info" | "neutral" {
  switch (state) {
    case "ready":
      return "success";
    case "running":
      return "warning";
    case "failed":
      return "error";
    case "superseded":
      return "info";
    default:
      return "neutral";
  }
}

function parseSearchRequest(value: string | null): ODDStructuredSearchRequest {
  if (!value) return createRequest([createPredicate()]);
  try {
    const parsed = JSON.parse(value) as ODDStructuredSearchRequest;
    if (
      !parsed ||
      !parsed.query ||
      !["and", "or"].includes(parsed.query.logic) ||
      !Array.isArray(parsed.query.predicates) ||
      !Array.isArray(parsed.query.groups) ||
      !Number.isInteger(parsed.limit) ||
      !Number.isInteger(parsed.offset)
    ) {
      return createRequest([createPredicate()]);
    }
    return parsed;
  } catch {
    return createRequest([createPredicate()]);
  }
}

function percent(value: number): string {
  if (!Number.isFinite(value)) return "0%";
  return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
}

function formatDurationNS(value: number): string {
  const seconds = Math.max(0, value) / 1e9;
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)} h`;
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)} min`;
  return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
}

function formatDistance(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(1)} km`;
  return `${value.toFixed(0)} m`;
}

function createPredicate(key = "odd.road.context"): ODDSearchPredicate {
  return {
    key,
    operator: "exists",
    values: [],
    statuses: ["valid"],
    sources: [],
    minimum_confidence: 0,
    minimum_duration_ns: 0,
    camera_id: "",
    actor_track_uid: "",
  };
}

function createRequest(
  predicates: ODDSearchPredicate[],
  logic: "and" | "or" = "and",
): ODDStructuredSearchRequest {
  return {
    query: { logic, predicates, groups: [] },
    sort: "confidence",
    descending: true,
    limit: 50,
    offset: 0,
  };
}

function valueRatio(value: ODDValueStatistic, weighting: Weighting): number {
  if (weighting === "duration") return value.duration_ratio ?? 0;
  if (weighting === "distance") return value.distance_ratio ?? 0;
  return value.scene_ratio;
}

function valueInterval(
  value: ODDValueStatistic,
  weighting: Weighting,
): ODDRatioInterval {
  if (weighting === "duration") return value.duration_ratio_ci95;
  if (weighting === "distance") return value.distance_ratio_ci95;
  return value.scene_ratio_ci95;
}

function valueAmount(
  value: ODDValueStatistic,
  weighting: Weighting,
): string {
  if (weighting === "duration") return formatDurationNS(value.duration_ns ?? 0);
  if (weighting === "distance") return formatDistance(value.distance_m ?? 0);
  return `${value.scene_count.toLocaleString()} scenes`;
}

function valueDenominator(
  key: ODDKeyStatistic,
  weighting: Weighting,
): string {
  if (weighting === "duration") return formatDurationNS(key.valid_duration_ns);
  if (weighting === "distance") return formatDistance(key.valid_distance_m);
  return `${key.valid_scene_count.toLocaleString()} scenes`;
}

function keyCoverage(key: ODDKeyStatistic, weighting: Weighting): number {
  if (weighting === "duration") return key.observable_duration_coverage ?? 0;
  if (weighting === "distance") return key.observable_distance_coverage ?? 0;
  return key.observable_scene_coverage;
}

function keyCoverageAmounts(
  key: ODDKeyStatistic,
  weighting: Weighting,
): [string, string] {
  if (weighting === "duration") {
    return [
      formatDurationNS(key.valid_duration_ns),
      formatDurationNS(key.eligible_duration_ns),
    ];
  }
  if (weighting === "distance") {
    return [
      formatDistance(key.valid_distance_m),
      formatDistance(key.eligible_distance_m),
    ];
  }
  return [
    `${key.valid_scene_count.toLocaleString()} scenes`,
    `${key.eligible_scene_count.toLocaleString()} scenes`,
  ];
}

function weightedRecordAmount(
  sceneCount: number,
  durationNS: number,
  distanceM: number,
  weighting: Weighting,
): string {
  if (weighting === "duration") return formatDurationNS(durationNS);
  if (weighting === "distance") return formatDistance(distanceM);
  return `${sceneCount.toLocaleString()} scenes`;
}

function EventTimeline({
  scene,
}: {
  scene: {
    start_timestamp_ns: number;
    end_timestamp_ns: number;
    events?: Array<{
      event_uid: string;
      primary_event_key: string;
      primary_values: string[];
      start_timestamp_ns: number;
      end_timestamp_ns: number;
      outcome: string;
    }>;
  };
}) {
  const events = scene.events ?? [];
  if (events.length === 0) return null;
  const duration = Math.max(
    1,
    scene.end_timestamp_ns - scene.start_timestamp_ns,
  );
  return (
    <div className="mt-3 space-y-2" aria-label="Event timeline">
      {events.map((event) => {
        const left =
          ((event.start_timestamp_ns - scene.start_timestamp_ns) / duration) *
          100;
        const width =
          ((event.end_timestamp_ns - event.start_timestamp_ns) / duration) *
          100;
        return (
          <div
            key={event.event_uid}
            className="grid grid-cols-[minmax(0,1fr)_5rem] items-center gap-3"
          >
            <div className="min-w-0">
              <div className="mb-1 flex min-w-0 items-center justify-between gap-2 text-[10px]">
                <span className="truncate font-mono text-slate-400">
                  {event.primary_event_key} ·{" "}
                  {event.primary_values.join(", ") || "present"}
                </span>
                <span className="shrink-0 text-slate-600">{event.outcome}</span>
              </div>
              <div className="relative h-1.5 bg-slate-900">
                <span
                  className="absolute top-0 h-full bg-amber-500"
                  style={{
                    left: `${Math.max(0, left)}%`,
                    width: `${Math.max(1, Math.min(100 - left, width))}%`,
                  }}
                />
              </div>
            </div>
            <span className="text-right font-mono text-[10px] text-slate-600">
              {formatDurationNS(
                event.end_timestamp_ns - event.start_timestamp_ns,
              )}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function DatasetLabelerRuns({
  executions,
  loading,
  state,
  hasReadyLabelSet,
  capability,
  pending,
  error,
  onLaunch,
  onRetry,
}: {
  executions: FlyteExecution[];
  loading: boolean;
  state: ODDOperationalState;
  hasReadyLabelSet: boolean;
  capability: ODDOperationsCapability | null;
  pending: string;
  error: string;
  onLaunch: (scope: "smoke" | "full") => void;
  onRetry: (executionID: string) => void;
}) {
  return (
    <section className="border-t border-slate-800 pt-5">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[10px] uppercase text-slate-600">
            Dataset Labeler runs
          </p>
          <div className="mt-1">
            <StatusBadge label={state} tone={oddStateTone(state)} />
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          {capability?.permitted && (
            <>
              <Button
                size="xs"
                variant="outline"
                disabled={pending !== ""}
                onClick={() => onLaunch("smoke")}
              >
                {pending === "launch:smoke" ? (
                  <LoaderCircle className="animate-spin" />
                ) : (
                  <Play />
                )}
                Run smoke
              </Button>
              {capability.allow_full && (
                <Button
                  size="xs"
                  variant="outline"
                  disabled={pending !== ""}
                  onClick={() => onLaunch("full")}
                >
                  {pending === "launch:full" ? (
                    <LoaderCircle className="animate-spin" />
                  ) : (
                    <Play />
                  )}
                  Run full
                </Button>
              )}
            </>
          )}
          <Link
            href="/runs"
            className="flex items-center gap-1 text-xs text-slate-500 hover:text-cyan-300"
          >
            All runs
            <ExternalLink className="size-3" />
          </Link>
        </div>
      </div>
      {error && (
        <p role="alert" className="mt-3 text-xs text-rose-400">
          {error}
        </p>
      )}
      {loading ? (
        <Skeleton className="mt-3 h-16 w-full" />
      ) : executions.length === 0 ? (
        <p className="mt-3 text-xs text-slate-600">
          No recent Dataset Labeler execution was returned by Flyte.
        </p>
      ) : (
        <div className="mt-3 divide-y divide-slate-900">
          {executions.slice(0, 5).map((execution, index) => {
            const runState = oddRunState(execution, index, hasReadyLabelSet);
            return (
              <div
                key={execution.execution_id}
                className="grid gap-2 py-2 sm:grid-cols-[minmax(0,1fr)_auto_auto_auto]"
              >
                <Link
                  href={`/runs/${encodeURIComponent(execution.execution_id)}`}
                  className="break-all font-mono text-xs text-slate-300 hover:text-cyan-300"
                >
                  {execution.execution_id}
                </Link>
                <span className="font-mono text-[10px] text-slate-600">
                  {execution.phase} ·{" "}
                  {execution.duration_s > 0
                    ? `${Math.round(execution.duration_s).toLocaleString()} s`
                    : "in progress"}
                </span>
                <StatusBadge
                  label={runState}
                  tone={oddStateTone(runState)}
                />
                {capability?.permitted &&
                ["FAILED", "ABORTED", "TIMED_OUT"].includes(
                  execution.phase,
                ) ? (
                  <Button
                    size="icon-xs"
                    variant="ghost"
                    title="Retry Dataset Labeler run"
                    aria-label={`Retry ${execution.execution_id}`}
                    disabled={pending !== ""}
                    onClick={() => onRetry(execution.execution_id)}
                  >
                    {pending === execution.execution_id ? (
                      <LoaderCircle className="animate-spin" />
                    ) : (
                      <RotateCcw />
                    )}
                  </Button>
                ) : (
                  <span className="hidden size-6 sm:block" />
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

function ODDPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const dataset = searchParams.get("dataset")?.trim() || DEFAULT_DATASET;
  const version = searchParams.get("version")?.trim() || DEFAULT_VERSION;
  const selectedOntologyKey = searchParams.get("key");
  const initialRequest = parseSearchRequest(searchParams.get("query"));
  const [tab, setTab] = useState<Tab>(() => {
    const requested = searchParams.get("tab");
    return isTab(requested) ? requested : "overview";
  });
  const [namespace, setNamespace] = useState<Namespace>("odd");
  const [weighting, setWeighting] = useState<Weighting>("scene");
  const labelsets = useApi(
    () => getODDLabelSets(dataset, version),
    [dataset, version],
  );
  const readyLabelSet = labelsets.data?.labelsets[0];
  const ontology = useApi(
    () => getODDOntology(dataset, version),
    [dataset, version],
    Boolean(readyLabelSet),
  );
  const statistics = useApi(
    () => getODDStatistics(dataset, version),
    [dataset, version],
    Boolean(readyLabelSet),
  );
  const executions = useApi(
    () => listExecutionsPage(100),
    [],
  );
  const operations = useApi(getODDOperations, []);
  const [operationPending, setOperationPending] = useState("");
  const [operationError, setOperationError] = useState("");
  const [logic, setLogic] = useState<"and" | "or">(
    initialRequest.query.logic,
  );
  const [predicates, setPredicates] = useState<ODDSearchPredicate[]>(
    initialRequest.query.predicates.length > 0
      ? initialRequest.query.predicates
      : [createPredicate()],
  );
  const [appliedRequest, setAppliedRequest] =
    useState<ODDStructuredSearchRequest>(initialRequest);
  const searchIdentity = JSON.stringify(appliedRequest);
  const search = useApi(
    () =>
      searchODDScenesStructured(dataset, version, appliedRequest),
    [dataset, version, searchIdentity],
    Boolean(readyLabelSet),
  );
  const definitionByKey = useMemo(
    () =>
      new Map(
        ontology.data?.labels.map((definition) => [
          definition.key,
          definition,
        ]) ?? [],
      ),
    [ontology.data],
  );
  const counts = useMemo(
    () =>
      new Map(
        statistics.data?.keys.map((item) => [item.key, item]) ?? [],
      ),
    [statistics.data],
  );
  const oddExecutions =
    executions.data?.items.filter((execution) =>
      isODDDatasetLabelerExecution(execution),
    ) ?? [];
  const catalogState =
    labelsets.data?.state ?? (readyLabelSet ? "ready" : "not_started");
  const operationalState = oddOperationalState(catalogState, oddExecutions);

  useEffect(() => {
    const requested = searchParams.get("tab");
    if (isTab(requested)) setTab(requested);
  }, [searchParams]);

  useEffect(() => {
    if (tab !== "ontology" || !selectedOntologyKey || !ontology.data) return;
    document
      .getElementById(`ontology-${selectedOntologyKey}`)
      ?.scrollIntoView({ block: "start" });
  }, [ontology.data, selectedOntologyKey, tab]);

  function replaceURL(
    nextTab: Tab,
    request: ODDStructuredSearchRequest | null = null,
    coordinate: { dataset?: string; version?: string } = {},
  ) {
    const params = new URLSearchParams(searchParams.toString());
    params.set("dataset", coordinate.dataset ?? dataset);
    params.set("version", coordinate.version ?? version);
    params.set("tab", nextTab);
    if (request) {
      params.set("query", JSON.stringify(request));
    } else if (nextTab !== "search") {
      params.delete("query");
    }
    router.replace(`/odd?${params.toString()}`, { scroll: false });
  }

  function selectTab(nextTab: Tab) {
    setTab(nextTab);
    replaceURL(nextTab, nextTab === "search" ? appliedRequest : null);
  }

  function selectCoordinate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const nextDataset = String(form.get("dataset") ?? "").trim();
    const nextVersion = String(form.get("version") ?? "").trim();
    if (!nextDataset || !nextVersion) return;
    setTab("overview");
    replaceURL("overview", null, {
      dataset: nextDataset,
      version: nextVersion,
    });
  }

  function updatePredicate(
    index: number,
    patch: Partial<ODDSearchPredicate>,
  ) {
    setPredicates((current) =>
      current.map((predicate, itemIndex) =>
        itemIndex === index ? { ...predicate, ...patch } : predicate,
      ),
    );
  }

  function applySearch(
    nextPredicates = predicates,
    nextLogic = logic,
    offset = 0,
  ) {
    const request = {
      ...appliedRequest,
      query: { logic: nextLogic, predicates: nextPredicates, groups: [] },
      offset,
    };
    setAppliedRequest(request);
    replaceURL("search", request);
  }

  function patchAppliedRequest(
    patch: Partial<ODDStructuredSearchRequest>,
  ) {
    const request = { ...appliedRequest, ...patch };
    setAppliedRequest(request);
    replaceURL("search", request);
  }

  function searchValue(key: string, value: string) {
    const predicate = {
      ...createPredicate(key),
      operator: "contains" as const,
      values: [value],
    };
    setPredicates([predicate]);
    setLogic("and");
    const request = createRequest([predicate]);
    setAppliedRequest(request);
    setTab("search");
    replaceURL("search", request);
  }

  function searchPair(
    leftKey: string,
    leftValue: string,
    rightKey: string,
    rightValue: string,
  ) {
    const nextPredicates = [
      { ...createPredicate(leftKey), operator: "contains" as const, values: [leftValue] },
      { ...createPredicate(rightKey), operator: "contains" as const, values: [rightValue] },
    ];
    setPredicates(nextPredicates);
    setLogic("and");
    const request = createRequest(nextPredicates);
    setAppliedRequest(request);
    setTab("search");
    replaceURL("search", request);
  }

  function refreshOperations() {
    labelsets.reload();
    executions.reload();
    window.setTimeout(executions.reload, 1_000);
  }

  async function launchLabeler(scope: "smoke" | "full") {
    if (
      scope === "full" &&
      !window.confirm(`Run full ODD labeling for ${dataset} / ${version}?`)
    ) {
      return;
    }
    setOperationPending(`launch:${scope}`);
    setOperationError("");
    try {
      await launchODDDatasetLabeler(dataset, version, scope);
      refreshOperations();
    } catch (error) {
      setOperationError(
        error instanceof Error ? error.message : "ODD launch failed.",
      );
    } finally {
      setOperationPending("");
    }
  }

  async function retryLabeler(executionID: string) {
    setOperationPending(executionID);
    setOperationError("");
    try {
      await retryODDDatasetLabeler(executionID);
      refreshOperations();
    } catch (error) {
      setOperationError(
        error instanceof Error ? error.message : "ODD retry failed.",
      );
    } finally {
      setOperationPending("");
    }
  }

  if (labelsets.loading) {
    return <Skeleton className="h-[34rem] w-full" />;
  }
  if (labelsets.error) {
    return (
      <ErrorState
        error={labelsets.error}
        onRetry={() => {
          ontology.reload();
          statistics.reload();
          labelsets.reload();
        }}
      />
    );
  }
  if (!readyLabelSet) {
    return (
      <div className="space-y-6">
        <h1 className="text-xl font-semibold">ODD Dashboard</h1>
        <p className="text-sm text-slate-500">
          No ready ODD LabelSet is published for {dataset} / {version}.
        </p>
        <DatasetLabelerRuns
          executions={oddExecutions}
          loading={executions.loading}
          state={operationalState}
          hasReadyLabelSet={false}
          capability={operations.data}
          pending={operationPending}
          error={operationError}
          onLaunch={launchLabeler}
          onRetry={retryLabeler}
        />
      </div>
    );
  }
  if (ontology.loading || statistics.loading) {
    return <Skeleton className="h-[34rem] w-full" />;
  }
  if (ontology.error || statistics.error) {
    return (
      <ErrorState
        error={ontology.error ?? statistics.error!}
        onRetry={() => {
          ontology.reload();
          statistics.reload();
          labelsets.reload();
        }}
      />
    );
  }
  if (!ontology.data || !statistics.data) return null;

  const ontologyData = ontology.data;
  const statisticsData = statistics.data;
  const visibleKeys = statisticsData.keys.filter(
    (item) => item.namespace === namespace,
  );
  const commonOddPairs = [...statisticsData.cooccurrences.odd_pairs]
    .filter((item) => item.scene_count > 0)
    .sort(
      (left, right) =>
        right.scene_count - left.scene_count ||
        right.overlap_duration_ns - left.overlap_duration_ns,
    )
    .slice(0, 6);
  const rareOddPairs = [...statisticsData.cooccurrences.odd_pairs]
    .filter((item) => item.scene_count > 0)
    .sort(
      (left, right) =>
        left.scene_count - right.scene_count ||
        left.overlap_duration_ns - right.overlap_duration_ns,
    )
    .slice(0, 6);
  const commonOddEvents = [...statisticsData.cooccurrences.odd_event]
    .filter((item) => item.scene_count > 0)
    .sort(
      (left, right) =>
        right.event_instance_count - left.event_instance_count ||
        right.overlap_duration_ns - left.overlap_duration_ns,
    )
    .slice(0, 6);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-5 border-b border-slate-800 pb-4">
        <div className="space-y-3">
          <h1 className="mt-1 text-xl font-semibold">ODD Dashboard</h1>
          <form
            onSubmit={selectCoordinate}
            className="flex flex-wrap items-end gap-2"
          >
            <label className="space-y-1 text-[9px] uppercase text-slate-600">
              Dataset
              <input
                key={`dataset-${dataset}`}
                name="dataset"
                defaultValue={dataset}
                className="block h-8 w-32 border border-slate-800 bg-slate-950 px-2 font-mono text-xs normal-case text-slate-300"
              />
            </label>
            <label className="space-y-1 text-[9px] uppercase text-slate-600">
              Version
              <input
                key={`version-${version}`}
                name="version"
                defaultValue={version}
                className="block h-8 w-24 border border-slate-800 bg-slate-950 px-2 font-mono text-xs normal-case text-slate-300"
              />
            </label>
            <button
              type="submit"
              className="h-8 border border-slate-700 px-3 text-xs text-slate-400 hover:border-cyan-700 hover:text-cyan-300"
            >
              Open
            </button>
          </form>
        </div>
        <div className="grid grid-cols-3 gap-x-6 text-right">
          <div>
            <p className="font-mono text-sm text-slate-200">
              {statisticsData.scene_count.toLocaleString()}
            </p>
            <p className="text-[10px] uppercase text-slate-600">Scenes</p>
          </div>
          <div>
            <p className="font-mono text-sm text-slate-200">
              {formatDurationNS(statisticsData.scene_duration_ns)}
            </p>
            <p className="text-[10px] uppercase text-slate-600">Duration</p>
          </div>
          <div>
            <p className="font-mono text-sm text-slate-200">
              {formatDistance(statisticsData.scene_distance_m ?? 0)}
            </p>
            <p className="text-[10px] uppercase text-slate-600">Distance</p>
          </div>
        </div>
      </header>

      <nav
        aria-label="ODD views"
        className="flex max-w-full gap-1 overflow-x-auto border-b border-slate-800"
      >
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            onClick={() => selectTab(id)}
            className={`flex h-10 shrink-0 items-center gap-2 border-b-2 px-3 text-sm ${
              tab === id
                ? "border-cyan-500 text-slate-100"
                : "border-transparent text-slate-500 hover:text-slate-300"
            }`}
          >
            <Icon className="size-4" />
            {label}
          </button>
        ))}
      </nav>

      {tab === "overview" && (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex border border-slate-800">
              {NAMESPACES.map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => setNamespace(item)}
                  className={`h-8 px-3 text-xs capitalize ${
                    namespace === item
                      ? "bg-slate-800 text-slate-100"
                      : "text-slate-500 hover:text-slate-300"
                  }`}
                >
                  {item}
                </button>
              ))}
            </div>
            <div className="flex border border-slate-800">
              {WEIGHTINGS.map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => setWeighting(item)}
                  className={`h-8 px-3 text-xs capitalize ${
                    weighting === item
                      ? "bg-cyan-950 text-cyan-300"
                      : "text-slate-500 hover:text-slate-300"
                  }`}
                >
                  {item === "scene"
                    ? "Scene presence"
                    : item === "duration"
                      ? "Duration share"
                      : "Distance share"}
                </button>
              ))}
            </div>
          </div>
          <p className="text-[10px] text-slate-600">
            Scene presence can exceed 100% across values because one scene may
            contain multiple conditions over time.
          </p>

          <div className="divide-y divide-slate-900">
            {visibleKeys.map((item) => (
              <section
                key={item.key}
                className="grid gap-4 py-5 lg:grid-cols-[18rem_minmax(0,1fr)]"
              >
                <div className="min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <p className="break-words font-mono text-xs text-slate-300">
                      {item.key}
                    </p>
                    {item.conflict_count > 0 && (
                      <span className="shrink-0 border border-rose-900 px-1.5 py-0.5 font-mono text-[10px] text-rose-400">
                        {item.conflict_count} conflicts
                      </span>
                    )}
                  </div>
                  {(() => {
                    const [numerator, denominator] = keyCoverageAmounts(
                      item,
                      weighting,
                    );
                    return (
                      <p className="mt-2 text-xs text-slate-500">
                        Observable {percent(keyCoverage(item, weighting))} ·{" "}
                        {numerator} / {denominator}
                      </p>
                    );
                  })()}
                  <div className="mt-2 flex flex-wrap gap-1">
                    {ontologyData.statuses.map((status) => (
                      <span
                        key={status}
                        className={`border px-1.5 py-0.5 font-mono text-[10px] ${
                          STATUS_STYLE[status] ?? STATUS_STYLE.unavailable
                        }`}
                      >
                        {status}{" "}
                        {weightedRecordAmount(
                          item.status_scene_counts[status] ?? 0,
                          item.status_duration_ns[status] ?? 0,
                          item.status_distance_m[status] ?? 0,
                          weighting,
                        )}
                      </span>
                    ))}
                  </div>
                  <details className="mt-3">
                    <summary className="w-fit cursor-pointer text-[10px] uppercase text-slate-600">
                      Source coverage
                    </summary>
                    <div className="mt-2 space-y-1">
                      {Object.entries(item.source_scene_counts).map(
                        ([source, sceneCount]) => (
                          <p
                            key={source}
                            className="flex justify-between gap-3 font-mono text-[10px] text-slate-600"
                          >
                            <span>{source}</span>
                            <span>
                              {weightedRecordAmount(
                                sceneCount,
                                item.source_duration_ns[source] ?? 0,
                                item.source_distance_m[source] ?? 0,
                                weighting,
                              )}
                            </span>
                          </p>
                        ),
                      )}
                    </div>
                  </details>
                </div>
                <div className="space-y-2.5">
                  {item.values.map((entry) => {
                    const ratio = valueRatio(entry, weighting);
                    const interval = valueInterval(entry, weighting);
                    return (
                      <button
                        key={entry.value}
                        type="button"
                        aria-label={`${entry.value} ${percent(ratio)}`}
                        onClick={() => searchValue(item.key, entry.value)}
                        className="grid w-full grid-cols-[minmax(6rem,12rem)_minmax(5rem,1fr)_minmax(10rem,auto)] items-center gap-2 text-left text-xs"
                      >
                        <span className="truncate font-mono text-slate-400">
                          {entry.value}
                        </span>
                        <span className="h-2 overflow-hidden bg-slate-900">
                          <span
                            className="block h-full bg-cyan-600"
                            style={{ width: `${Math.min(100, ratio * 100)}%` }}
                          />
                        </span>
                        <span className="text-right font-mono text-[10px] text-slate-500">
                          <span className="block">
                            {valueAmount(entry, weighting)} /{" "}
                            {valueDenominator(item, weighting)} ·{" "}
                            {percent(ratio)}
                          </span>
                          <span className="mt-0.5 block text-[9px] text-slate-700">
                            95% CI {percent(interval.lower)}–
                            {percent(interval.upper)}
                          </span>
                        </span>
                      </button>
                    );
                  })}
                </div>
              </section>
            ))}
          </div>
          {namespace === "odd" && (
            <section className="grid gap-6 border-t border-slate-800 pt-5 lg:grid-cols-3">
              {[
                {
                  title: "Common ODD combinations",
                  rows: commonOddPairs,
                },
                {
                  title: "Rare ODD combinations",
                  rows: rareOddPairs,
                },
              ].map((group) => (
                <div key={group.title}>
                  <p className="text-[10px] uppercase text-slate-600">
                    {group.title}
                  </p>
                  <div className="mt-2 divide-y divide-slate-900">
                    {group.rows.map((row) => (
                      <button
                        key={`${row.left_key}:${row.left_value}:${row.right_key}:${row.right_value}`}
                        type="button"
                        onClick={() =>
                          searchPair(
                            row.left_key,
                            row.left_value,
                            row.right_key,
                            row.right_value,
                          )
                        }
                        className="grid w-full gap-1 py-2 text-left"
                      >
                        <span className="break-words font-mono text-[10px] text-slate-400">
                          {row.left_value} + {row.right_value}
                        </span>
                        <span className="text-[10px] text-slate-600">
                          {row.scene_count.toLocaleString()} scenes ·{" "}
                          {formatDurationNS(row.overlap_duration_ns)} overlap
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
              <div>
                <p className="text-[10px] uppercase text-slate-600">
                  ODD and event overlap
                </p>
                <div className="mt-2 divide-y divide-slate-900">
                  {commonOddEvents.map((row) => (
                    <button
                      key={`${row.odd_key}:${row.odd_value}:${row.event_key}:${row.event_value}`}
                      type="button"
                      onClick={() =>
                        searchPair(
                          row.odd_key,
                          row.odd_value,
                          row.event_key,
                          row.event_value,
                        )
                      }
                      className="grid w-full gap-1 py-2 text-left"
                    >
                      <span className="break-words font-mono text-[10px] text-slate-400">
                        {row.odd_value} + {row.event_value}
                      </span>
                      <span className="text-[10px] text-slate-600">
                        {row.scene_count.toLocaleString()} scenes ·{" "}
                        {row.event_instance_count.toLocaleString()} events
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </section>
          )}
        </div>
      )}

      {tab === "search" && (
        <div className="space-y-6">
          <section className="space-y-4 border-b border-slate-800 pb-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex border border-slate-800">
                {(["and", "or"] as const).map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => setLogic(item)}
                    className={`h-8 px-3 font-mono text-xs uppercase ${
                      logic === item
                        ? "bg-slate-800 text-slate-100"
                        : "text-slate-500"
                    }`}
                  >
                    {item}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2">
                <select
                  aria-label="Search sort"
                  value={appliedRequest.sort}
                  onChange={(event) =>
                    patchAppliedRequest({
                      sort: event.target
                        .value as ODDStructuredSearchRequest["sort"],
                      offset: 0,
                    })
                  }
                  className="h-8 border border-slate-700 bg-slate-950 px-2 text-xs text-slate-300"
                >
                  <option value="confidence">Confidence</option>
                  <option value="matched_duration">Matched duration</option>
                  <option value="scene_duration">Scene duration</option>
                  <option value="recording_time">Recording time</option>
                  <option value="scene_uid">Scene UID</option>
                </select>
                <button
                  type="button"
                  title={
                    appliedRequest.descending
                      ? "Descending order"
                      : "Ascending order"
                  }
                  aria-label={
                    appliedRequest.descending
                      ? "Descending order"
                      : "Ascending order"
                  }
                  onClick={() =>
                    patchAppliedRequest({
                      descending: !appliedRequest.descending,
                      offset: 0,
                    })
                  }
                  className="grid size-8 place-items-center border border-slate-700 text-slate-400 hover:text-slate-100"
                >
                  {appliedRequest.descending ? (
                    <ArrowDown className="size-4" />
                  ) : (
                    <ArrowUp className="size-4" />
                  )}
                </button>
              </div>
            </div>

            <div className="space-y-3">
              {predicates.map((predicate, index) => {
                const definition = definitionByKey.get(predicate.key);
                return (
                  <div
                    key={`${index}-${predicate.key}`}
                    className="grid gap-3 border-l-2 border-slate-800 pl-3 xl:grid-cols-[minmax(14rem,1.5fr)_8rem_minmax(9rem,1fr)_9rem_9rem_7rem_2rem]"
                  >
                    <label className="space-y-1 text-[10px] uppercase text-slate-600">
                      Label
                      <select
                        value={predicate.key}
                        onChange={(event) =>
                          updatePredicate(index, {
                            key: event.target.value,
                            values: [],
                          })
                        }
                        className="h-9 w-full border border-slate-700 bg-slate-950 px-2 font-mono text-xs normal-case text-slate-200"
                      >
                        {ontologyData.labels.map((item) => (
                          <option key={item.key} value={item.key}>
                            {item.key}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="space-y-1 text-[10px] uppercase text-slate-600">
                      Operator
                      <select
                        value={predicate.operator}
                        onChange={(event) =>
                          updatePredicate(index, {
                            operator: event.target
                              .value as ODDSearchPredicate["operator"],
                          })
                        }
                        className="h-9 w-full border border-slate-700 bg-slate-950 px-2 text-xs normal-case text-slate-200"
                      >
                        <option value="exists">exists</option>
                        <option value="contains">contains</option>
                        <option value="equals">equals</option>
                        <option value="in">in</option>
                        <option value="not_equals">not equals</option>
                      </select>
                    </label>
                    <label className="space-y-1 text-[10px] uppercase text-slate-600">
                      Value
                      <select
                        multiple={predicate.operator === "in"}
                        value={
                          predicate.operator === "in"
                            ? predicate.values
                            : predicate.values[0] ?? ""
                        }
                        disabled={predicate.operator === "exists"}
                        onChange={(event) =>
                          updatePredicate(index, {
                            values:
                              predicate.operator === "in"
                                ? Array.from(
                                    event.target.selectedOptions,
                                    (option) => option.value,
                                  ).filter(Boolean)
                                : event.target.value
                                  ? [event.target.value]
                                  : [],
                          })
                        }
                        className={`w-full border border-slate-700 bg-slate-950 px-2 font-mono text-xs normal-case text-slate-200 disabled:text-slate-700 ${
                          predicate.operator === "in" ? "h-20 py-1" : "h-9"
                        }`}
                      >
                        <option value="">Any value</option>
                        {definition?.values.map((candidate) => (
                          <option key={candidate.value} value={candidate.value}>
                            {candidate.value}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="space-y-1 text-[10px] uppercase text-slate-600">
                      Status
                      <select
                        value={predicate.statuses[0] ?? ""}
                        onChange={(event) =>
                          updatePredicate(index, {
                            statuses: event.target.value
                              ? [event.target.value as ODDStatus]
                              : [],
                          })
                        }
                        className="h-9 w-full border border-slate-700 bg-slate-950 px-2 text-xs normal-case text-slate-200"
                      >
                        <option value="">Any status</option>
                        {ontologyData.statuses.map((item) => (
                          <option key={item}>{item}</option>
                        ))}
                      </select>
                    </label>
                    <label className="space-y-1 text-[10px] uppercase text-slate-600">
                      Source
                      <select
                        value={predicate.sources[0] ?? ""}
                        onChange={(event) =>
                          updatePredicate(index, {
                            sources: event.target.value
                              ? [event.target.value]
                              : [],
                          })
                        }
                        className="h-9 w-full border border-slate-700 bg-slate-950 px-2 text-xs normal-case text-slate-200"
                      >
                        <option value="">Any source</option>
                        {ontologyData.sources.map((item) => (
                          <option key={item}>{item}</option>
                        ))}
                      </select>
                    </label>
                    <label className="space-y-1 text-[10px] uppercase text-slate-600">
                      Min conf.
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        value={predicate.minimum_confidence}
                        onChange={(event) =>
                          updatePredicate(index, {
                            minimum_confidence: Number(event.target.value),
                          })
                        }
                        className="h-9 w-full border border-slate-700 bg-slate-950 px-2 font-mono text-xs normal-case text-slate-200"
                      />
                    </label>
                    <button
                      type="button"
                      title="Remove predicate"
                      aria-label="Remove predicate"
                      disabled={predicates.length === 1}
                      onClick={() =>
                        setPredicates((current) =>
                          current.filter(
                            (_, itemIndex) => itemIndex !== index,
                          ),
                        )
                      }
                      className="mt-4 grid size-8 place-items-center text-slate-600 hover:text-rose-400 disabled:text-slate-800"
                    >
                      <Trash2 className="size-4" />
                    </button>
                    <details className="xl:col-span-7">
                      <summary className="w-fit cursor-pointer text-[10px] uppercase text-slate-600">
                        Interval and scope
                      </summary>
                      <div className="mt-2 grid gap-3 sm:grid-cols-3">
                        <label className="space-y-1 text-[10px] uppercase text-slate-600">
                          Min duration (s)
                          <input
                            type="number"
                            min="0"
                            step="0.1"
                            value={predicate.minimum_duration_ns / 1e9}
                            onChange={(event) =>
                              updatePredicate(index, {
                                minimum_duration_ns:
                                  Number(event.target.value) * 1e9,
                              })
                            }
                            className="h-9 w-full border border-slate-700 bg-slate-950 px-2 font-mono text-xs normal-case text-slate-200"
                          />
                        </label>
                        <label className="space-y-1 text-[10px] uppercase text-slate-600">
                          Camera ID
                          <input
                            value={predicate.camera_id}
                            onChange={(event) =>
                              updatePredicate(index, {
                                camera_id: event.target.value,
                              })
                            }
                            className="h-9 w-full border border-slate-700 bg-slate-950 px-2 font-mono text-xs normal-case text-slate-200"
                          />
                        </label>
                        <label className="space-y-1 text-[10px] uppercase text-slate-600">
                          Actor track UID
                          <input
                            value={predicate.actor_track_uid}
                            onChange={(event) =>
                              updatePredicate(index, {
                                actor_track_uid: event.target.value,
                              })
                            }
                            className="h-9 w-full border border-slate-700 bg-slate-950 px-2 font-mono text-xs normal-case text-slate-200"
                          />
                        </label>
                      </div>
                    </details>
                  </div>
                );
              })}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3">
              <button
                type="button"
                onClick={() =>
                  setPredicates((current) => [
                    ...current,
                    createPredicate(ontologyData.labels[0]?.key),
                  ])
                }
                disabled={predicates.length >= 8}
                className="flex h-8 items-center gap-2 text-xs text-slate-500 hover:text-slate-200 disabled:text-slate-800"
              >
                <Plus className="size-4" />
                Add predicate
              </button>
              <button
                type="button"
                onClick={() => applySearch()}
                className="flex h-9 items-center gap-2 bg-cyan-700 px-4 text-xs font-medium text-white hover:bg-cyan-600"
              >
                <Search className="size-4" />
                Search scenes
              </button>
            </div>
          </section>

          {search.loading ? (
            <Skeleton className="h-48 w-full" />
          ) : search.error ? (
            <ErrorState error={search.error} onRetry={search.reload} />
          ) : (
            <section>
              <div className="mb-2 flex items-center justify-between gap-3">
                <p className="text-xs text-slate-500">
                  {(search.data?.total ?? 0).toLocaleString()} matching scenes
                </p>
                <p className="font-mono text-[10px] text-slate-600">
                  {search.data?.offset ?? 0}–
                  {Math.min(
                    search.data?.total ?? 0,
                    (search.data?.offset ?? 0) +
                      (search.data?.scenes.length ?? 0),
                  )}
                </p>
              </div>
              <div className="divide-y divide-slate-900">
                {search.data?.scenes.map((scene) => {
                  const firstMatch =
                    scene.first_matched_timestamp_ns ??
                    scene.start_timestamp_ns;
                  const frame = Math.max(
                    0,
                    Math.floor(
                      (firstMatch - scene.start_timestamp_ns) / 100_000_000,
                    ),
                  );
                  return (
                    <div key={scene.scene_uid} className="py-4">
                      <div className="grid gap-2 sm:grid-cols-[1fr_auto]">
                        <div className="min-w-0">
                          <Link
                            href={`/scenes/${dataset}/${encodeURIComponent(scene.shard_name)}/${frame}?version=${version}`}
                            className="break-all font-mono text-xs text-slate-200 hover:text-cyan-300"
                          >
                            {scene.scene_uid}
                          </Link>
                          <div className="mt-2 flex flex-wrap gap-1">
                            {(scene.matched ?? []).map((match) => (
                              <span
                                key={`${match.key}-${match.first_timestamp_ns}-${match.values.join(",")}`}
                                className={`border px-1.5 py-0.5 font-mono text-[10px] ${
                                  STATUS_STYLE[match.status] ??
                                  STATUS_STYLE.unavailable
                                }`}
                              >
                                {match.key} ·{" "}
                                {match.values.join(", ") || match.status}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div className="grid grid-cols-3 gap-x-4 text-right font-mono text-[10px] text-slate-500">
                          <span>
                            {(scene.match_confidence ?? 0).toFixed(2)}
                            <small className="block font-sans text-[9px] uppercase text-slate-700">
                              confidence
                            </small>
                          </span>
                          <span>
                            {formatDurationNS(
                              scene.matched_duration_ns ?? 0,
                            )}
                            <small className="block font-sans text-[9px] uppercase text-slate-700">
                              matched
                            </small>
                          </span>
                          <span>
                            {formatDistance(scene.distance_m)}
                            <small className="block font-sans text-[9px] uppercase text-slate-700">
                              distance
                            </small>
                          </span>
                        </div>
                      </div>
                      <EventTimeline scene={scene} />
                    </div>
                  );
                })}
              </div>
              {(search.data?.offset ?? 0) > 0 || search.data?.more ? (
                <div className="mt-4 flex justify-end gap-2">
                  <button
                    type="button"
                    title="Previous page"
                    aria-label="Previous page"
                    disabled={(search.data?.offset ?? 0) === 0}
                    onClick={() =>
                      patchAppliedRequest({
                        offset: Math.max(
                          0,
                          appliedRequest.offset - appliedRequest.limit,
                        ),
                      })
                    }
                    className="grid size-8 place-items-center border border-slate-800 text-slate-500 hover:text-slate-100 disabled:text-slate-800"
                  >
                    <ArrowLeft className="size-4" />
                  </button>
                  <button
                    type="button"
                    title="Next page"
                    aria-label="Next page"
                    disabled={!search.data?.more}
                    onClick={() =>
                      patchAppliedRequest({
                        offset:
                          appliedRequest.offset + appliedRequest.limit,
                      })
                    }
                    className="grid size-8 place-items-center border border-slate-800 text-slate-500 hover:text-slate-100 disabled:text-slate-800"
                  >
                    <ArrowRight className="size-4" />
                  </button>
                </div>
              ) : null}
            </section>
          )}
        </div>
      )}

      {tab === "ontology" && (
        <div className="divide-y divide-slate-900">
          {ontologyData.labels.map((item) => (
            <details
              key={item.key}
              id={`ontology-${item.key}`}
              open={selectedOntologyKey === item.key || undefined}
              className="scroll-mt-20 py-4"
            >
              <summary className="cursor-pointer list-none">
                <div className="grid gap-1 sm:grid-cols-[1fr_auto]">
                  <span className="break-words font-mono text-xs text-slate-200">
                    {item.key}
                  </span>
                  <span className="text-xs text-slate-500">
                    {item.cardinality} · {item.subject} · {item.temporal_scope}{" "}
                    · {item.dataset_support.support_state}
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate-500">
                  {item.description}
                </p>
                <p className="mt-1 font-mono text-[10px] text-slate-600">
                  {item.primary_sources.join(", ")} ·{" "}
                  {item.backends.join(", ")} · {item.quality_tier}
                </p>
              </summary>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {item.values.map((candidate) => {
                  const statistic = counts
                    .get(item.key)
                    ?.values.find((row) => row.value === candidate.value);
                  return (
                    <button
                      key={candidate.value}
                      type="button"
                      onClick={() => searchValue(item.key, candidate.value)}
                      className="border border-slate-800 px-2 py-1 font-mono text-[11px] text-slate-400 hover:border-cyan-800 hover:text-cyan-300"
                    >
                      {candidate.value} · {statistic?.scene_count ?? 0}
                    </button>
                  );
                })}
              </div>
              <dl className="mt-4 grid gap-x-5 gap-y-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
                <div>
                  <dt className="text-[9px] uppercase text-slate-700">
                    Dataset support
                  </dt>
                  <dd className="mt-1 font-mono text-[10px] text-slate-400">
                    {dataset} / {version} ·{" "}
                    {item.dataset_support.support_state}
                  </dd>
                  <dd className="mt-1 font-mono text-[9px] text-slate-600">
                    {item.dataset_support.valid_scene_count.toLocaleString()} /{" "}
                    {item.dataset_support.eligible_scene_count.toLocaleString()}{" "}
                    scenes observable ·{" "}
                    {percent(
                      item.dataset_support.observable_scene_coverage,
                    )}
                  </dd>
                </div>
                <div>
                  <dt className="text-[9px] uppercase text-slate-700">
                    Sources
                  </dt>
                  <dd className="mt-1 font-mono text-[10px] text-slate-400">
                    {item.primary_sources.join(", ")}
                  </dd>
                </div>
                <div>
                  <dt className="text-[9px] uppercase text-slate-700">
                    Inference
                  </dt>
                  <dd className="mt-1 font-mono text-[10px] text-slate-400">
                    {item.backends.join(", ")}
                  </dd>
                </div>
                <div>
                  <dt className="text-[9px] uppercase text-slate-700">
                    Status semantics
                  </dt>
                  <dd className="mt-1 font-mono text-[10px] text-slate-400">
                    {ontologyData.statuses.join(", ")}
                  </dd>
                </div>
              </dl>
              {item.none_semantics && (
                <p className="mt-3 text-xs text-amber-400">
                  none: {item.none_semantics}
                </p>
              )}
            </details>
          ))}
        </div>
      )}

      {tab === "labelsets" &&
        (labelsets.loading ? (
          <Skeleton className="h-48 w-full" />
        ) : labelsets.error ? (
          <ErrorState error={labelsets.error} onRetry={labelsets.reload} />
        ) : readyLabelSet ? (
          <div className="space-y-6">
            <section className="grid gap-5 border-b border-slate-800 pb-5 sm:grid-cols-2 lg:grid-cols-4">
              <div>
                <p className="text-[10px] uppercase text-slate-600">
                  Publication
                </p>
                <p className="mt-1 font-mono text-xs text-emerald-300">
                  {operationalState} · {readyLabelSet.publication_scope}
                </p>
              </div>
              <div>
                <p className="text-[10px] uppercase text-slate-600">
                  Structural validation
                </p>
                <p className="mt-1 font-mono text-xs text-slate-200">
                  {readyLabelSet.quality.structural_status}
                </p>
              </div>
              <div>
                <p className="text-[10px] uppercase text-slate-600">
                  Audit
                </p>
                <p className="mt-1 font-mono text-xs text-amber-300">
                  {readyLabelSet.quality.audit_status}
                </p>
              </div>
              <div>
                <p className="text-[10px] uppercase text-slate-600">
                  Certification
                </p>
                <p className="mt-1 font-mono text-xs text-slate-200">
                  {readyLabelSet.quality.certification_status}
                </p>
              </div>
            </section>
            <section className="grid gap-x-8 gap-y-5 md:grid-cols-2">
              <div>
                <p className="text-[10px] uppercase text-slate-600">
                  LabelSet ID
                </p>
                <p className="mt-1 break-all font-mono text-xs text-slate-300">
                  {readyLabelSet.labelset_id}
                </p>
              </div>
              <div>
                <p className="text-[10px] uppercase text-slate-600">
                  Dataset manifest
                </p>
                <p className="mt-1 break-all font-mono text-xs text-slate-300">
                  {readyLabelSet.dataset_manifest_sha256}
                </p>
              </div>
              <div>
                <p className="text-[10px] uppercase text-slate-600">
                  Ontology
                </p>
                <p className="mt-1 font-mono text-xs text-slate-300">
                  {readyLabelSet.ontology_version}
                </p>
              </div>
              <div>
                <p className="text-[10px] uppercase text-slate-600">
                  Labeler
                </p>
                <p className="mt-1 font-mono text-xs text-slate-300">
                  {readyLabelSet.labeler_version}
                </p>
              </div>
              <div>
                <p className="text-[10px] uppercase text-slate-600">
                  Scene coverage
                </p>
                <p className="mt-1 font-mono text-xs text-slate-300">
                  {readyLabelSet.scene_count.toLocaleString()} /{" "}
                  {readyLabelSet.expected_scene_count.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-[10px] uppercase text-slate-600">
                  Artifacts
                </p>
                <p className="mt-1 font-mono text-xs text-slate-300">
                  {Object.keys(readyLabelSet.artifacts).length} immutable ·{" "}
                  {
                    Object.values(readyLabelSet.artifacts).filter(
                      (artifact) => artifact.authoritative,
                    ).length
                  }{" "}
                  authoritative
                </p>
              </div>
            </section>
            <DatasetLabelerRuns
              executions={oddExecutions}
              loading={executions.loading}
              state={operationalState}
              hasReadyLabelSet
              capability={operations.data}
              pending={operationPending}
              error={operationError}
              onLaunch={launchLabeler}
              onRetry={retryLabeler}
            />
          </div>
        ) : (
          <p className="text-sm text-slate-500">
            No ready ODD LabelSet is published.
          </p>
        ))}
    </div>
  );
}

export default function ODDPage() {
  return (
    <Suspense fallback={<Skeleton className="h-[34rem] w-full" />}>
      <ODDPageContent />
    </Suspense>
  );
}
