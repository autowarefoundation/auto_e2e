"use client";

import Link from "next/link";
import { ExternalLink, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { ErrorState } from "@/components/error-state";
import { Skeleton } from "@/components/ui/skeleton";
import type {
  ODDMetricName,
  ODDMetricProjection,
  ODDMetricProjectionsResponse,
  ODDMetricSlice,
  ODDStatus,
} from "@/types";

const DELTA_METRICS: Array<{ id: ODDMetricName; label: string }> = [
  { id: "ade_3s_m", label: "ADE 3s" },
  { id: "ade_horizon_m", label: "ADE horizon" },
  { id: "fde_horizon_m", label: "FDE horizon" },
];

const OVERALL_METRICS: Array<{ id: ODDMetricName; label: string }> = [
  { id: "ade_1s_m", label: "ADE 1s" },
  { id: "ade_3s_m", label: "ADE 3s" },
  { id: "ade_horizon_m", label: "ADE horizon" },
  { id: "fde_horizon_m", label: "FDE horizon" },
];

const STATUS_OPTIONS: Array<ODDStatus | "all"> = [
  "all",
  "valid",
  "unavailable",
  "not_observable",
  "ambiguous",
];

const STATUS_STYLE: Record<ODDStatus, string> = {
  valid: "border-emerald-900 text-emerald-300",
  unavailable: "border-slate-700 text-slate-400",
  not_observable: "border-amber-900 text-amber-300",
  ambiguous: "border-rose-900 text-rose-300",
};

interface ODDModelMetricsProps {
  data: ODDMetricProjectionsResponse | null;
  loading: boolean;
  error: Error | null;
  onRetry: () => void;
}

function shortDigest(value: string): string {
  return value.length > 12 ? value.slice(0, 12) : value;
}

function metricValue(value: number): string {
  return `${value.toFixed(2)} m`;
}

function deltaValue(value: number): string {
  if (Math.abs(value) < 0.005) return "0.00 m";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)} m`;
}

function deltaTone(value: number): string {
  if (value > 0.005) return "text-rose-300";
  if (value < -0.005) return "text-emerald-300";
  return "text-slate-500";
}

function sliceValue(slice: ODDMetricSlice): string {
  return slice.value ?? slice.status;
}

function projectionLabel(projection: ODDMetricProjection): string {
  return `v${projection.model.model_version} · ${projection.model.registered_model_name}`;
}

export function ODDModelMetrics({
  data,
  loading,
  error,
  onRetry,
}: ODDModelMetricsProps) {
  const [projectionID, setProjectionID] = useState("");
  const [deltaMetric, setDeltaMetric] =
    useState<ODDMetricName>("fde_horizon_m");
  const [status, setStatus] = useState<ODDStatus | "all">("all");
  const [query, setQuery] = useState("");

  useEffect(() => {
    const projections = data?.projections ?? [];
    if (
      projections.length > 0 &&
      !projections.some((item) => item.projection_id === projectionID)
    ) {
      setProjectionID(projections[0].projection_id);
    }
  }, [data, projectionID]);

  const projection = useMemo(
    () =>
      data?.projections.find(
        (item) => item.projection_id === projectionID,
      ) ?? data?.projections[0],
    [data, projectionID],
  );

  const slices = useMemo(() => {
    if (!projection) return [];
    const normalizedQuery = query.trim().toLowerCase();
    const overall = projection.overall.metrics[deltaMetric];
    return projection.slices
      .filter((slice) => status === "all" || slice.status === status)
      .filter((slice) => {
        if (!normalizedQuery) return true;
        return [
          slice.kind,
          slice.key,
          slice.value ?? "",
          slice.status,
        ]
          .join(" ")
          .toLowerCase()
          .includes(normalizedQuery);
      })
      .sort(
        (left, right) =>
          right.metrics[deltaMetric] -
            overall -
            (left.metrics[deltaMetric] - overall) ||
          right.sample_count - left.sample_count ||
          left.key.localeCompare(right.key),
      );
  }, [deltaMetric, projection, query, status]);

  if (loading) return <Skeleton className="h-[28rem] w-full" />;
  if (error) return <ErrorState error={error} onRetry={onRetry} />;
  if (!data || data.projections.length === 0 || !projection) {
    return (
      <div className="border-y border-slate-800 py-12 text-center">
        <p className="text-sm text-slate-400">
          No validation metric projection is published for this LabelSet.
        </p>
      </div>
    );
  }

  const overall = projection.overall.metrics;
  const baseline = overall[deltaMetric];

  return (
    <div className="space-y-6">
      <section className="grid gap-4 border-b border-slate-800 pb-5 lg:grid-cols-[minmax(15rem,1fr)_auto] lg:items-end">
        <label className="space-y-1 text-[10px] uppercase text-slate-600">
          Model projection
          <select
            aria-label="Model projection"
            value={projection.projection_id}
            onChange={(event) => setProjectionID(event.target.value)}
            className="block h-9 w-full max-w-xl border border-slate-700 bg-slate-950 px-2 font-mono text-xs normal-case text-slate-200"
          >
            {data.projections.map((item) => (
              <option key={item.projection_id} value={item.projection_id}>
                {projectionLabel(item)}
              </option>
            ))}
          </select>
        </label>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="border border-cyan-900 px-2 py-1 font-mono text-cyan-300">
            validation only
          </span>
          <Link
            href={`/runs/${encodeURIComponent(projection.model.run_id)}`}
            className="flex items-center gap-1.5 border border-slate-700 px-2 py-1 font-mono text-slate-300 hover:border-cyan-700 hover:text-cyan-300"
          >
            Run {shortDigest(projection.model.run_id)}
            <ExternalLink className="size-3" />
          </Link>
        </div>
      </section>

      <section className="grid gap-x-6 gap-y-4 border-b border-slate-800 pb-5 sm:grid-cols-2 xl:grid-cols-4">
        {OVERALL_METRICS.map((metric) => (
          <div key={metric.id} className="border-l border-slate-800 pl-3">
            <p className="text-[10px] uppercase text-slate-600">
              {metric.label}
            </p>
            <p className="mt-1 font-mono text-xl text-slate-100">
              {metricValue(overall[metric.id])}
            </p>
            <p className="mt-1 font-mono text-[10px] text-slate-600">
              {projection.overall.sample_count.toLocaleString()} samples ·{" "}
              {projection.overall.scene_count.toLocaleString()} scenes
            </p>
          </div>
        ))}
      </section>

      <section className="grid gap-x-8 gap-y-3 border-b border-slate-800 pb-5 text-xs sm:grid-cols-2 xl:grid-cols-4">
        <div>
          <p className="text-[9px] uppercase text-slate-700">
            Evaluation dataset
          </p>
          <p className="mt-1 font-mono text-slate-300">
            {projection.evaluation_dataset.dataset} /{" "}
            {projection.evaluation_dataset.version}
          </p>
        </div>
        <div>
          <p className="text-[9px] uppercase text-slate-700">ODD LabelSet</p>
          <p className="mt-1 font-mono text-slate-300">
            {projection.labelset.dataset} / {projection.labelset.version}
          </p>
        </div>
        <div>
          <p className="text-[9px] uppercase text-slate-700">
            Frozen validation
          </p>
          <p className="mt-1 font-mono text-slate-300">
            {projection.validation.sample_count.toLocaleString()} samples ·{" "}
            {projection.validation.group_count.toLocaleString()} groups
          </p>
        </div>
        <div>
          <p className="text-[9px] uppercase text-slate-700">Projection</p>
          <p
            className="mt-1 truncate font-mono text-slate-300"
            title={projection.projection_id}
          >
            {shortDigest(projection.projection_id)}
          </p>
        </div>
      </section>

      <section className="space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="flex max-w-full overflow-x-auto border border-slate-800">
            {DELTA_METRICS.map((metric) => (
              <button
                key={metric.id}
                type="button"
                onClick={() => setDeltaMetric(metric.id)}
                className={`h-8 shrink-0 px-3 text-xs ${
                  deltaMetric === metric.id
                    ? "bg-slate-800 text-slate-100"
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                {metric.label} delta
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-end gap-2">
            <label className="space-y-1 text-[9px] uppercase text-slate-700">
              Status
              <select
                value={status}
                onChange={(event) =>
                  setStatus(event.target.value as ODDStatus | "all")
                }
                className="block h-8 border border-slate-700 bg-slate-950 px-2 text-xs normal-case text-slate-300"
              >
                {STATUS_OPTIONS.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>
            </label>
            <label className="relative block space-y-1 text-[9px] uppercase text-slate-700">
              Label or value
              <Search className="absolute bottom-2 left-2 size-3 text-slate-600" />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                className="block h-8 w-48 border border-slate-700 bg-slate-950 pl-7 pr-2 font-mono text-xs normal-case text-slate-300"
              />
            </label>
          </div>
        </div>

        <div className="overflow-x-auto border-y border-slate-800">
          <table className="w-full min-w-[62rem] table-fixed text-left">
            <thead className="text-[9px] uppercase text-slate-600">
              <tr className="border-b border-slate-800">
                <th className="w-[27%] px-3 py-2 font-medium">Label / value</th>
                <th className="w-[12%] px-3 py-2 font-medium">Status</th>
                <th className="w-[9%] px-3 py-2 text-right font-medium">
                  Samples
                </th>
                <th className="w-[8%] px-3 py-2 text-right font-medium">
                  Scenes
                </th>
                <th className="w-[10%] px-3 py-2 text-right font-medium">
                  ADE 3s
                </th>
                <th className="w-[11%] px-3 py-2 text-right font-medium">
                  ADE horizon
                </th>
                <th className="w-[10%] px-3 py-2 text-right font-medium">
                  FDE horizon
                </th>
                <th className="w-[13%] px-3 py-2 text-right font-medium">
                  Delta
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-900">
              {slices.map((slice) => {
                const delta = slice.metrics[deltaMetric] - baseline;
                return (
                  <tr key={`${slice.kind}:${slice.key}:${slice.value}:${slice.status}`}>
                    <td className="px-3 py-3 align-top">
                      <p className="break-words font-mono text-xs text-slate-300">
                        {slice.key}
                      </p>
                      <p className="mt-1 break-words font-mono text-[10px] text-cyan-400">
                        {sliceValue(slice)}
                        <span className="ml-2 text-slate-700">
                          {slice.kind}
                        </span>
                      </p>
                    </td>
                    <td className="px-3 py-3 align-top">
                      <span
                        className={`inline-block border px-1.5 py-0.5 font-mono text-[9px] ${STATUS_STYLE[slice.status]}`}
                      >
                        {slice.status}
                      </span>
                    </td>
                    <td className="px-3 py-3 text-right align-top font-mono text-xs text-slate-400">
                      {slice.sample_count.toLocaleString()}
                    </td>
                    <td className="px-3 py-3 text-right align-top font-mono text-xs text-slate-400">
                      {slice.scene_count.toLocaleString()}
                    </td>
                    <td className="px-3 py-3 text-right align-top font-mono text-xs text-slate-300">
                      {metricValue(slice.metrics.ade_3s_m)}
                    </td>
                    <td className="px-3 py-3 text-right align-top font-mono text-xs text-slate-300">
                      {metricValue(slice.metrics.ade_horizon_m)}
                    </td>
                    <td className="px-3 py-3 text-right align-top font-mono text-xs text-slate-300">
                      {metricValue(slice.metrics.fde_horizon_m)}
                    </td>
                    <td
                      className={`px-3 py-3 text-right align-top font-mono text-xs ${deltaTone(delta)}`}
                    >
                      {deltaValue(delta)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between gap-3 text-[10px] text-slate-600">
          <span>{slices.length.toLocaleString()} slices</span>
          <span className="font-mono">
            baseline {metricValue(baseline)}
          </span>
        </div>
      </section>
    </div>
  );
}
