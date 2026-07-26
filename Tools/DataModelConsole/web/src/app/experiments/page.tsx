"use client";

import { Dialog } from "@base-ui/react/dialog";
import {
  ArrowDown,
  BarChart3,
  ExternalLink,
  GitCompareArrows,
  RotateCcw,
  Search,
  X,
} from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useMemo, useState } from "react";

import { ErrorState } from "@/components/error-state";
import {
  StatusBadge,
  flytePhaseTone,
  mlflowStatusTone,
} from "@/components/status-badge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useApi } from "@/hooks/use-api";
import { listJoinedExperiments } from "@/lib/api";
import {
  formatEpochMillis,
  formatMeters,
} from "@/lib/format";
import { cn } from "@/lib/utils";
import type {
  ExperimentDisplacementMetrics,
  ExperimentRecord,
  FlytePhase,
} from "@/types";

type MetricKey = "ade" | "fde";
type MetricSource = "evaluation" | "validation";

const SELECT_CLASS =
  "h-9 min-w-32 rounded-md border border-slate-700 bg-slate-950 px-2.5 text-xs text-slate-200 outline-none focus:border-cyan-500";

function shortDataset(dataset: string): string {
  if (dataset.includes("KITScenes")) return "KITScenes";
  if (dataset.includes("PhysicalAI")) return "PhysicalAI AV";
  if (dataset.includes("L2D")) return "L2D";
  return dataset || "Unknown dataset";
}

function effectiveStatus(record: ExperimentRecord): string {
  if (record.eval_execution?.phase) return record.eval_execution.phase;
  if (!record.evaluation && record.validation && record.mlflow_status === "FINISHED") {
    return "EVAL PENDING";
  }
  if (record.train_execution?.phase) return record.train_execution.phase;
  return record.mlflow_status;
}

function statusCategory(record: ExperimentRecord): string {
  const status = effectiveStatus(record);
  if (["RUNNING", "SUCCEEDING", "QUEUED", "SCHEDULED", "EVAL PENDING"].includes(status)) {
    return "running";
  }
  if (["FAILED", "FAILING", "ABORTED", "ABORTING", "TIMED_OUT", "KILLED"].includes(status)) {
    return "failed";
  }
  if (status === "SUCCEEDED" || status === "FINISHED") return "succeeded";
  return "other";
}

function ExperimentStatus({ record }: { record: ExperimentRecord }) {
  const status = effectiveStatus(record);
  const flyteStatuses = new Set([
    "UNDEFINED",
    "QUEUED",
    "RUNNING",
    "SUCCEEDED",
    "SUCCEEDING",
    "FAILED",
    "FAILING",
    "ABORTED",
    "ABORTING",
    "TIMED_OUT",
  ]);
  const tone = flyteStatuses.has(status)
    ? flytePhaseTone(status as FlytePhase)
    : mlflowStatusTone(status === "EVAL PENDING" ? "SCHEDULED" : status);
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <StatusBadge label={status} tone={tone} />
      {record.lineage_status !== "complete" && (
        <Badge
          variant="outline"
          className="border-amber-500/40 bg-amber-500/10 font-mono text-[10px] text-amber-400"
        >
          {record.lineage_status === "missing" ? "UNLINKED" : "PARTIAL"}
        </Badge>
      )}
    </div>
  );
}

function metricSource(
  record: ExperimentRecord,
): { label: "Eval" | "Val"; metrics: ExperimentDisplacementMetrics } | null {
  if (record.evaluation) return { label: "Eval", metrics: record.evaluation };
  if (record.validation) return { label: "Val", metrics: record.validation };
  return null;
}

function Results({ record, compact = false }: { record: ExperimentRecord; compact?: boolean }) {
  const source = metricSource(record);
  if (!source) {
    return <span className="text-xs text-slate-500">Metrics pending</span>;
  }
  return (
    <div className={cn("grid grid-cols-[auto_1fr_1fr] items-center gap-x-2", compact && "w-full")}>
      <span
        className={cn(
          "rounded-sm border px-1 py-0.5 font-mono text-[9px]",
          source.label === "Eval"
            ? "border-emerald-500/40 text-emerald-400"
            : "border-amber-500/40 text-amber-400",
        )}
      >
        {source.label}
      </span>
      <span className="text-right font-mono text-xs">
        <span className="mr-1 text-[9px] text-slate-500">ADE</span>
        {formatMeters(source.metrics.ade)}
      </span>
      <span className="text-right font-mono text-xs">
        <span className="mr-1 text-[9px] text-slate-500">FDE</span>
        {formatMeters(source.metrics.fde)}
      </span>
    </div>
  );
}

function RunTitle({ record }: { record: ExperimentRecord }) {
  const scope = record.validation_scope
    ? record.validation_scope === "subset"
      ? "Smoke"
      : record.validation_scope[0].toUpperCase() + record.validation_scope.slice(1)
    : "Scope unknown";
  return (
    <div className="min-w-0">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-medium text-slate-100">
          {shortDataset(record.dataset)}
          {record.dataset_version ? ` ${record.dataset_version}` : ""}
        </span>
        <Badge variant="outline" className="border-slate-700 text-[10px] text-slate-400">
          {scope}
        </Badge>
        {record.route_conditioning !== undefined && (
          <Badge
            variant="outline"
            className={cn(
              "text-[10px]",
              record.route_conditioning
                ? "border-emerald-500/40 text-emerald-400"
                : "border-amber-500/40 text-amber-400",
            )}
          >
            Route {record.route_conditioning ? "ON" : "OFF"}
          </Badge>
        )}
      </div>
      <p className="mt-1 truncate text-[11px] text-slate-400">
        {[record.backbone, record.epochs && `${record.epochs} epochs`, record.seed && `seed ${record.seed}`]
          .filter(Boolean)
          .join(" · ") || record.run_name}
      </p>
      <p className="mt-0.5 truncate font-mono text-[10px] text-slate-600">
        {record.primary_execution_id || record.run_id}
      </p>
    </div>
  );
}

function stopPropagation(event: React.SyntheticEvent) {
  event.stopPropagation();
}

function SourceLinks({ record }: { record: ExperimentRecord }) {
  const registry = record.model_versions[0];
  return (
    <div className="flex flex-wrap gap-1" onClick={stopPropagation}>
      {record.primary_execution_url && (
        <a
          href={record.primary_execution_url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex h-7 items-center gap-1 rounded-md border border-slate-700 px-2 text-[10px] text-slate-300 hover:border-cyan-500 hover:text-cyan-300"
        >
          Flyte <ExternalLink className="size-3" />
        </a>
      )}
      {record.mlflow_url && (
        <a
          href={record.mlflow_url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex h-7 items-center gap-1 rounded-md border border-slate-700 px-2 text-[10px] text-slate-300 hover:border-cyan-500 hover:text-cyan-300"
        >
          MLflow <ExternalLink className="size-3" />
        </a>
      )}
      {registry?.url && (
        <a
          href={registry.url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex h-7 items-center gap-1 rounded-md border border-slate-700 px-2 font-mono text-[10px] text-slate-300 hover:border-cyan-500 hover:text-cyan-300"
        >
          v{registry.version} <ExternalLink className="size-3" />
        </a>
      )}
    </div>
  );
}

function SummaryStrip({ records }: { records: ExperimentRecord[] }) {
  const values = [
    ["Runs", records.length],
    ["Active", records.filter((record) => statusCategory(record) === "running").length],
    ["Failed", records.filter((record) => statusCategory(record) === "failed").length],
    ["Evaluated", records.filter((record) => record.evaluation).length],
    ["Model versions", records.reduce((sum, record) => sum + record.model_versions.length, 0)],
  ] as const;
  return (
    <section className="grid grid-cols-2 border-y border-slate-800 sm:grid-cols-5">
      {values.map(([label, value], index) => (
        <div
          key={label}
          className={cn(
            "px-3 py-3",
            index > 0 && "sm:border-l sm:border-slate-800",
            index % 2 === 1 && "border-l border-slate-800 sm:border-l",
          )}
        >
          <p className="text-[10px] uppercase text-slate-500">{label}</p>
          <p className="mt-1 font-mono text-xl text-slate-100">{value}</p>
        </div>
      ))}
    </section>
  );
}

function MetricTrend({
  records,
  metric,
  onMetricChange,
}: {
  records: ExperimentRecord[];
  metric: MetricKey;
  onMetricChange: (metric: MetricKey) => void;
}) {
  const trend = useMemo(() => {
    const anchor = records.find((record) => metricSource(record)?.metrics[metric] !== undefined);
    if (!anchor) return null;
    const source: MetricSource = anchor.evaluation ? "evaluation" : "validation";
    const cohort = records
      .filter(
        (record) =>
          record.dataset === anchor.dataset &&
          record.dataset_version === anchor.dataset_version &&
          record.validation_scope === anchor.validation_scope &&
          record.validation_split_id === anchor.validation_split_id &&
          record[source]?.[metric] !== undefined,
      )
      .sort((a, b) => a.start_time - b.start_time)
      .slice(-12);
    const values = cohort.map((record) => record[source]?.[metric] as number);
    if (values.length === 0) return null;
    return {
      anchor,
      cohort,
      source,
      values,
      min: Math.min(...values),
      max: Math.max(...values),
    };
  }, [metric, records]);

  if (!trend) return null;
  const width = 720;
  const height = 150;
  const left = 38;
  const right = 18;
  const top = 18;
  const bottom = 26;
  const range = Math.max(trend.max - trend.min, 0.001);
  const point = (value: number, index: number) => ({
    x:
      trend.values.length === 1
        ? width / 2
        : left + (index * (width - left - right)) / (trend.values.length - 1),
    y: top + ((trend.max - value) * (height - top - bottom)) / range,
  });
  const points = trend.values.map((value, index) => point(value, index));
  const best = Math.min(...trend.values);
  const latest = trend.values.at(-1);

  return (
    <section className="border-b border-slate-800 pb-5">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-sm font-medium">
            <BarChart3 className="size-4 text-cyan-400" />
            {shortDataset(trend.anchor.dataset)} {trend.anchor.dataset_version || ""}
            <span className="text-slate-500">·</span>
            <span className="capitalize text-slate-400">
              {trend.anchor.validation_scope || "unknown"} {trend.source}
            </span>
          </div>
          <p className="mt-1 font-mono text-[10px] text-slate-500">
            {trend.anchor.validation_split_id || "split not recorded"}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-[10px] text-slate-500">
            best {formatMeters(best)} · latest {formatMeters(latest)}
          </span>
          <div className="flex rounded-md border border-slate-700 p-0.5">
            {(["ade", "fde"] as const).map((key) => (
              <button
                key={key}
                type="button"
                aria-pressed={metric === key}
                onClick={() => onMetricChange(key)}
                className={cn(
                  "h-6 rounded px-2 font-mono text-[10px] uppercase",
                  metric === key
                    ? "bg-slate-700 text-slate-50"
                    : "text-slate-500 hover:text-slate-300",
                )}
              >
                {key}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="h-40 w-full overflow-hidden">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-full w-full"
          role="img"
          aria-label={`${trend.source} ${metric.toUpperCase()} trend`}
          preserveAspectRatio="none"
        >
          <line x1={left} y1={top} x2={left} y2={height - bottom} stroke="#334155" />
          <line
            x1={left}
            y1={height - bottom}
            x2={width - right}
            y2={height - bottom}
            stroke="#334155"
          />
          <text x="0" y={top + 4} fill="#64748b" fontSize="9">
            {trend.max.toFixed(2)}
          </text>
          <text x="0" y={height - bottom + 4} fill="#64748b" fontSize="9">
            {trend.min.toFixed(2)}
          </text>
          {points.length > 1 && (
            <polyline
              points={points.map(({ x, y }) => `${x},${y}`).join(" ")}
              fill="none"
              stroke="#22d3ee"
              strokeWidth="2"
              vectorEffect="non-scaling-stroke"
            />
          )}
          {points.map(({ x, y }, index) => {
            const record = trend.cohort[index];
            const color =
              record.route_conditioning === true
                ? "#34d399"
                : record.route_conditioning === false
                  ? "#fbbf24"
                  : "#22d3ee";
            return (
              <circle key={record.run_id} cx={x} cy={y} r="4" fill={color}>
                <title>
                  {record.run_name}: {trend.values[index].toFixed(3)} m
                </title>
              </circle>
            );
          })}
        </svg>
      </div>
      <div className="flex justify-end gap-4 text-[10px] text-slate-500">
        <span className="flex items-center gap-1">
          <span className="size-2 bg-emerald-400" /> Route on
        </span>
        <span className="flex items-center gap-1">
          <span className="size-2 bg-amber-400" /> Route off
        </span>
      </div>
    </section>
  );
}

function DetailPanel({
  record,
  onClose,
}: {
  record: ExperimentRecord | null;
  onClose: () => void;
}) {
  const navigationMetrics = record
    ? Object.entries(record.metrics)
        .filter(([key]) => key.startsWith("eval/navigation/"))
        .sort(([left], [right]) => left.localeCompare(right))
    : [];
  const configRows = record
    ? [
        ["Dataset", `${record.dataset}${record.dataset_version ? ` / ${record.dataset_version}` : ""}`],
        ["Validation", [record.validation_scope, record.validation_split_id].filter(Boolean).join(" · ")],
        ["Backbone", record.backbone],
        ["Fusion", record.fusion_mode],
        ["Route conditioning", record.route_conditioning === undefined ? "" : record.route_conditioning ? "On" : "Off"],
        ["Epochs", record.epochs_completed || record.epochs],
        ["Seed", record.seed],
      ].filter(([, value]) => value)
    : [];

  return (
    <Dialog.Root open={record !== null} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-50 bg-slate-950/70" />
        <Dialog.Popup className="fixed inset-y-0 right-0 z-50 w-full overflow-y-auto border-l border-slate-800 bg-slate-950 p-5 outline-none sm:w-[540px]">
          {record && (
            <>
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <Dialog.Title className="text-base font-semibold">
                    {shortDataset(record.dataset)} {record.dataset_version}
                  </Dialog.Title>
                  <Dialog.Description className="mt-1 truncate font-mono text-[11px] text-slate-500">
                    {record.run_id}
                  </Dialog.Description>
                </div>
                <Dialog.Close
                  className="rounded-md p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-100"
                  aria-label="Close experiment details"
                >
                  <X className="size-4" />
                </Dialog.Close>
              </div>

              <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-y border-slate-800 py-3">
                <ExperimentStatus record={record} />
                <SourceLinks record={record} />
              </div>

              <section className="py-5">
                <h3 className="text-xs font-medium uppercase text-slate-500">Results</h3>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  {record.evaluation && (
                    <div className="border-l-2 border-emerald-500 pl-3">
                      <p className="text-[10px] uppercase text-emerald-400">Evaluation</p>
                      <p className="mt-1 font-mono text-sm">
                        ADE {formatMeters(record.evaluation.ade)}
                      </p>
                      <p className="font-mono text-sm">
                        FDE {formatMeters(record.evaluation.fde)}
                      </p>
                    </div>
                  )}
                  {record.validation && (
                    <div className="border-l-2 border-amber-500 pl-3">
                      <p className="text-[10px] uppercase text-amber-400">Validation</p>
                      <p className="mt-1 font-mono text-sm">
                        ADE {formatMeters(record.validation.ade)}
                      </p>
                      <p className="font-mono text-sm">
                        FDE {formatMeters(record.validation.fde)}
                      </p>
                    </div>
                  )}
                  {!record.evaluation && !record.validation && (
                    <p className="col-span-2 text-sm text-slate-500">Metrics pending</p>
                  )}
                </div>
              </section>

              <section className="border-t border-slate-800 py-5">
                <h3 className="text-xs font-medium uppercase text-slate-500">Configuration</h3>
                <dl className="mt-3 divide-y divide-slate-800">
                  {configRows.map(([label, value]) => (
                    <div key={label} className="grid grid-cols-[9rem_1fr] gap-3 py-2 text-xs">
                      <dt className="text-slate-500">{label}</dt>
                      <dd className="break-words font-mono text-slate-200">{value}</dd>
                    </div>
                  ))}
                </dl>
              </section>

              {navigationMetrics.length > 0 && (
                <section className="border-t border-slate-800 py-5">
                  <h3 className="text-xs font-medium uppercase text-slate-500">
                    Navigation evaluation
                  </h3>
                  <dl className="mt-3 divide-y divide-slate-800">
                    {navigationMetrics.map(([key, value]) => (
                      <div key={key} className="grid grid-cols-[1fr_auto] gap-3 py-2 text-xs">
                        <dt className="break-all font-mono text-slate-500">
                          {key.replace("eval/navigation/", "")}
                        </dt>
                        <dd className="font-mono text-slate-200">{value.toFixed(3)}</dd>
                      </div>
                    ))}
                  </dl>
                </section>
              )}

              <section className="border-t border-slate-800 py-5">
                <h3 className="text-xs font-medium uppercase text-slate-500">Provenance</h3>
                <dl className="mt-3 space-y-3 text-xs">
                  <div>
                    <dt className="text-slate-500">Flyte execution</dt>
                    <dd className="mt-1 break-all font-mono text-slate-300">
                      {record.primary_execution_id || "Not recorded"}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-500">MLflow run</dt>
                    <dd className="mt-1 break-all font-mono text-slate-300">{record.run_id}</dd>
                  </div>
                  {record.data_fingerprint && (
                    <div>
                      <dt className="text-slate-500">Data fingerprint</dt>
                      <dd className="mt-1 break-all font-mono text-slate-300">
                        {record.data_fingerprint}
                      </dd>
                    </div>
                  )}
                  {record.model_versions.length > 0 && (
                    <div>
                      <dt className="text-slate-500">Registry versions</dt>
                      <dd className="mt-2 flex flex-wrap gap-1.5">
                        {record.model_versions.map((version) => (
                          <Badge
                            key={`${version.name}-${version.version}`}
                            variant="outline"
                            className="border-cyan-500/30 font-mono text-cyan-300"
                          >
                            v{version.version}
                            {version.role ? ` · ${version.role}` : ""}
                          </Badge>
                        ))}
                      </dd>
                    </div>
                  )}
                </dl>
              </section>
            </>
          )}
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function ComparePanel({
  records,
  open,
  onClose,
}: {
  records: ExperimentRecord[];
  open: boolean;
  onClose: () => void;
}) {
  const compatibilityKeys = new Set(
    records.map((record) =>
      [
        record.dataset,
        record.dataset_version,
        record.data_fingerprint,
        record.validation_split_id,
      ].join("|"),
    ),
  );
  const comparable = compatibilityKeys.size <= 1;
  const rows = [
    ["Dataset", (record: ExperimentRecord) => shortDataset(record.dataset)],
    ["Version", (record: ExperimentRecord) => record.dataset_version || "-"],
    ["Scope", (record: ExperimentRecord) => record.validation_scope || "-"],
    ["Route", (record: ExperimentRecord) => record.route_conditioning === undefined ? "-" : record.route_conditioning ? "On" : "Off"],
    ["Backbone", (record: ExperimentRecord) => record.backbone || "-"],
    ["Epochs", (record: ExperimentRecord) => record.epochs_completed || record.epochs || "-"],
    ["Seed", (record: ExperimentRecord) => record.seed || "-"],
    ["Eval ADE", (record: ExperimentRecord) => formatMeters(record.evaluation?.ade)],
    ["Eval FDE", (record: ExperimentRecord) => formatMeters(record.evaluation?.fde)],
    ["Val ADE", (record: ExperimentRecord) => formatMeters(record.validation?.ade)],
    ["Val FDE", (record: ExperimentRecord) => formatMeters(record.validation?.fde)],
  ] as const;

  return (
    <Dialog.Root open={open} onOpenChange={(next) => !next && onClose()}>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-50 bg-slate-950/70" />
        <Dialog.Popup className="fixed inset-x-3 top-10 z-50 max-h-[calc(100vh-5rem)] overflow-auto rounded-md border border-slate-700 bg-slate-950 p-5 outline-none md:inset-x-auto md:left-1/2 md:w-[min(920px,calc(100vw-4rem))] md:-translate-x-1/2">
          <div className="flex items-start justify-between">
            <div>
              <Dialog.Title className="text-base font-semibold">Compare experiments</Dialog.Title>
              <Dialog.Description className="mt-1 text-xs text-slate-500">
                {records.length} selected runs
              </Dialog.Description>
            </div>
            <Dialog.Close
              className="rounded-md p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-100"
              aria-label="Close comparison"
            >
              <X className="size-4" />
            </Dialog.Close>
          </div>
          {!comparable && (
            <div className="mt-4 border-l-2 border-amber-500 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
              Dataset fingerprint or validation split differs. Scores are not directly comparable.
            </div>
          )}
          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-[640px] border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="w-28 p-2 text-left font-medium text-slate-500">Field</th>
                  {records.map((record) => (
                    <th key={record.run_id} className="p-2 text-left font-medium">
                      {record.run_name}
                      <span className="mt-1 block font-mono text-[9px] text-slate-600">
                        {record.run_id.slice(0, 12)}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map(([label, read]) => {
                  const values = records.map(read);
                  const differs = new Set(values).size > 1;
                  return (
                    <tr key={label} className="border-b border-slate-800">
                      <th className="p-2 text-left font-normal text-slate-500">{label}</th>
                      {values.map((value, index) => (
                        <td
                          key={records[index].run_id}
                          className={cn(
                            "p-2 font-mono text-slate-300",
                            differs && "bg-cyan-500/5 text-cyan-200",
                          )}
                        >
                          {value}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function ExperimentsPageInner() {
  const { data, error, loading, reload } = useApi(listJoinedExperiments);
  const [search, setSearch] = useState("");
  const [dataset, setDataset] = useState("all");
  const [version, setVersion] = useState("all");
  const [scope, setScope] = useState("all");
  const [status, setStatus] = useState("all");
  const [period, setPeriod] = useState("all");
  const [trendMetric, setTrendMetric] = useState<MetricKey>("ade");
  const [compareIDs, setCompareIDs] = useState<string[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const records = useMemo(() => data?.experiments ?? [], [data]);

  const datasets = useMemo(
    () => [...new Set(records.map((record) => record.dataset).filter(Boolean))].sort(),
    [records],
  );
  const versions = useMemo(
    () =>
      [
        ...new Set(
          records
            .filter((record) => dataset === "all" || record.dataset === dataset)
            .map((record) => record.dataset_version)
            .filter(Boolean),
        ),
      ].sort(),
    [dataset, records],
  );
  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return records.filter((record) => {
      if (dataset !== "all" && record.dataset !== dataset) return false;
      if (version !== "all" && record.dataset_version !== version) return false;
      if (scope !== "all" && (record.validation_scope || "unknown") !== scope) return false;
      if (status === "unlinked" && record.lineage_status === "complete") return false;
      if (status !== "all" && status !== "unlinked" && statusCategory(record) !== status) return false;
      if (period !== "all") {
        const days = Number(period);
        const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;
        if (record.start_time < cutoff) return false;
      }
      if (!needle) return true;
      return [
        record.run_name,
        record.run_id,
        record.primary_execution_id,
        record.dataset,
        record.dataset_version,
        record.backbone,
        record.validation_split_id,
      ]
        .filter(Boolean)
        .some((value) => value?.toLowerCase().includes(needle));
    });
  }, [dataset, period, records, scope, search, status, version]);

  const selectedRecord =
    records.find((record) => record.run_id === searchParams.get("run")) ?? null;
  const compareRecords = compareIDs
    .map((id) => records.find((record) => record.run_id === id))
    .filter((record): record is ExperimentRecord => Boolean(record));

  const openDetails = (runID: string) => {
    const next = new URLSearchParams(searchParams.toString());
    next.set("run", runID);
    router.push(`${pathname}?${next.toString()}`, { scroll: false });
  };
  const closeDetails = () => {
    const next = new URLSearchParams(searchParams.toString());
    next.delete("run");
    const query = next.toString();
    router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
  };
  const toggleCompare = (runID: string) => {
    setCompareIDs((current) =>
      current.includes(runID)
        ? current.filter((id) => id !== runID)
        : current.length < 3
          ? [...current, runID]
          : current,
    );
  };
  const resetFilters = () => {
    setSearch("");
    setDataset("all");
    setVersion("all");
    setScope("all");
    setStatus("all");
    setPeriod("all");
  };

  if (error) {
    return <ErrorState error={error} onRetry={reload} service="experiment sources" />;
  }
  if (loading) {
    return (
      <div className="space-y-5">
        <Skeleton className="h-14 w-64" />
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-44 w-full" />
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold">Experiments</h2>
          <p className="mt-1 text-sm text-slate-400">
            Flyte execution lineage, MLflow metrics, and registered checkpoints.
          </p>
        </div>
        {compareIDs.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-slate-500">{compareIDs.length}/3 selected</span>
            <Button
              size="sm"
              variant="outline"
              disabled={compareIDs.length < 2}
              onClick={() => setCompareOpen(true)}
            >
              <GitCompareArrows className="size-3.5" />
              Compare
            </Button>
            <Button
              size="icon-sm"
              variant="ghost"
              aria-label="Clear comparison"
              title="Clear comparison"
              onClick={() => setCompareIDs([])}
            >
              <X className="size-3.5" />
            </Button>
          </div>
        )}
      </div>

      <SummaryStrip records={filtered} />

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-52 flex-1 sm:max-w-80">
          <Search className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-slate-500" />
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search run or execution ID"
            aria-label="Search experiments"
            className="h-9 w-full rounded-md border border-slate-700 bg-slate-950 pl-9 pr-3 text-xs text-slate-200 outline-none placeholder:text-slate-600 focus:border-cyan-500"
          />
        </div>
        <select
          value={dataset}
          onChange={(event) => {
            setDataset(event.target.value);
            setVersion("all");
          }}
          aria-label="Filter by dataset"
          className={SELECT_CLASS}
        >
          <option value="all">All datasets</option>
          {datasets.map((value) => (
            <option key={value} value={value}>
              {shortDataset(value)}
            </option>
          ))}
        </select>
        <select
          value={version}
          onChange={(event) => setVersion(event.target.value)}
          aria-label="Filter by dataset version"
          className={SELECT_CLASS}
        >
          <option value="all">All versions</option>
          {versions.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
        <select
          value={scope}
          onChange={(event) => setScope(event.target.value)}
          aria-label="Filter by validation scope"
          className={SELECT_CLASS}
        >
          <option value="all">All scopes</option>
          <option value="full">Full</option>
          <option value="subset">Smoke</option>
          <option value="unknown">Unknown scope</option>
        </select>
        <select
          value={status}
          onChange={(event) => setStatus(event.target.value)}
          aria-label="Filter by status"
          className={SELECT_CLASS}
        >
          <option value="all">All statuses</option>
          <option value="running">Active</option>
          <option value="succeeded">Succeeded</option>
          <option value="failed">Failed</option>
          <option value="unlinked">Unlinked</option>
        </select>
        <select
          value={period}
          onChange={(event) => setPeriod(event.target.value)}
          aria-label="Filter by period"
          className={SELECT_CLASS}
        >
          <option value="all">All time</option>
          <option value="7">Last 7 days</option>
          <option value="30">Last 30 days</option>
          <option value="90">Last 90 days</option>
        </select>
        <Button
          type="button"
          size="icon"
          variant="ghost"
          onClick={resetFilters}
          aria-label="Reset filters"
          title="Reset filters"
        >
          <RotateCcw className="size-4" />
        </Button>
      </div>

      <MetricTrend records={filtered} metric={trendMetric} onMetricChange={setTrendMetric} />

      <section>
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-medium">Experiment runs</h3>
          <span className="font-mono text-[10px] text-slate-500">{filtered.length} results</span>
        </div>

        <div className="hidden border-y border-slate-800 md:block">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="w-8">
                  <span className="sr-only">Compare</span>
                </TableHead>
                <TableHead>Started</TableHead>
                <TableHead>Experiment</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="min-w-60 text-right">Results</TableHead>
                <TableHead>Registry</TableHead>
                <TableHead>Sources</TableHead>
                <TableHead className="w-8">
                  <span className="sr-only">Details</span>
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((record) => {
                const checked = compareIDs.includes(record.run_id);
                return (
                  <TableRow
                    key={record.run_id}
                    className="cursor-pointer"
                    data-state={checked ? "selected" : undefined}
                    onClick={() => openDetails(record.run_id)}
                  >
                    <TableCell onClick={stopPropagation}>
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={!checked && compareIDs.length >= 3}
                        onChange={() => toggleCompare(record.run_id)}
                        aria-label={`Compare ${record.run_name}`}
                        className="size-3.5 accent-cyan-500"
                      />
                    </TableCell>
                    <TableCell className="font-mono text-[10px] text-slate-400">
                      {formatEpochMillis(record.start_time)}
                    </TableCell>
                    <TableCell className="max-w-72 whitespace-normal">
                      <RunTitle record={record} />
                    </TableCell>
                    <TableCell>
                      <ExperimentStatus record={record} />
                    </TableCell>
                    <TableCell>
                      <Results record={record} />
                    </TableCell>
                    <TableCell>
                      <div className="flex max-w-44 flex-wrap gap-1">
                        {record.model_versions.slice(0, 3).map((model) => (
                          <Badge
                            key={`${model.name}-${model.version}`}
                            variant="outline"
                            className="border-cyan-500/30 font-mono text-[10px] text-cyan-300"
                          >
                            v{model.version}
                            {model.role ? ` · ${model.role}` : ""}
                          </Badge>
                        ))}
                        {record.model_versions.length === 0 && (
                          <span className="text-xs text-slate-600">-</span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <SourceLinks record={record} />
                    </TableCell>
                    <TableCell>
                      <ArrowDown className="-rotate-90 size-4 text-slate-600" />
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>

        <div className="border-y border-slate-800 md:hidden">
          {filtered.map((record) => {
            const checked = compareIDs.includes(record.run_id);
            return (
              <article
                key={record.run_id}
                className="border-b border-slate-800 px-1 py-4 last:border-b-0"
                onClick={() => openDetails(record.run_id)}
              >
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={!checked && compareIDs.length >= 3}
                    onChange={() => toggleCompare(record.run_id)}
                    onClick={stopPropagation}
                    aria-label={`Compare ${record.run_name}`}
                    className="mt-1 size-4 accent-cyan-500"
                  />
                  <div className="min-w-0 flex-1">
                    <RunTitle record={record} />
                    <div className="mt-3">
                      <ExperimentStatus record={record} />
                    </div>
                    <div className="mt-3 border-y border-slate-800 py-2">
                      <Results record={record} compact />
                    </div>
                    <div className="mt-3 flex items-center justify-between gap-2">
                      <span className="font-mono text-[10px] text-slate-500">
                        {formatEpochMillis(record.start_time)}
                      </span>
                      <SourceLinks record={record} />
                    </div>
                  </div>
                </div>
              </article>
            );
          })}
        </div>

        {filtered.length === 0 && (
          <div className="border-y border-slate-800 py-16 text-center text-sm text-slate-500">
            No experiment runs match the current filters.
          </div>
        )}
      </section>

      <DetailPanel record={selectedRecord} onClose={closeDetails} />
      <ComparePanel
        records={compareRecords}
        open={compareOpen && compareRecords.length >= 2}
        onClose={() => setCompareOpen(false)}
      />
    </div>
  );
}

export default function ExperimentsPage() {
  return (
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <ExperimentsPageInner />
    </Suspense>
  );
}
