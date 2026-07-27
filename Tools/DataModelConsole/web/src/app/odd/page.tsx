"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { BookOpen, ChartNoAxesColumn, Database, Search } from "lucide-react";

import { ErrorState } from "@/components/error-state";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/hooks/use-api";
import {
  getODDOntology,
  getODDStatistics,
  searchODDScenes,
} from "@/lib/api";

const DATASET = "kitscenes";
const VERSION = "v3.0";
const TABS = [
  { id: "overview", label: "Overview", icon: ChartNoAxesColumn },
  { id: "search", label: "Search", icon: Search },
  { id: "ontology", label: "Ontology", icon: BookOpen },
  { id: "labelsets", label: "LabelSets", icon: Database },
] as const;

type Tab = (typeof TABS)[number]["id"];

function percent(value: number): string {
  return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
}

export default function ODDPage() {
  const [tab, setTab] = useState<Tab>("overview");
  const ontology = useApi(
    () => getODDOntology(DATASET, VERSION),
    [DATASET, VERSION],
  );
  const statistics = useApi(
    () => getODDStatistics(DATASET, VERSION),
    [DATASET, VERSION],
  );
  const [key, setKey] = useState("");
  const [value, setValue] = useState("");
  const [status, setStatus] = useState("valid");
  const [source, setSource] = useState("");
  const search = useApi(
    () =>
      searchODDScenes(DATASET, VERSION, {
        key,
        value,
        status,
        source,
        limit: 100,
      }),
    [DATASET, VERSION, key, value, status, source],
  );
  const definition = useMemo(
    () => ontology.data?.labels.find((item) => item.key === key),
    [ontology.data, key],
  );
  const counts = useMemo(
    () =>
      new Map(
        statistics.data?.keys.map((item) => [item.key, item]) ?? [],
      ),
    [statistics.data],
  );

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
        }}
      />
    );
  }
  if (!ontology.data || !statistics.data) return null;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <p className="font-mono text-xs text-slate-500">
            {DATASET} / {VERSION}
          </p>
          <h1 className="mt-1 text-xl font-semibold">ODD Dashboard</h1>
        </div>
        <div className="text-right font-mono text-xs text-slate-400">
          <p>{statistics.data.scene_count.toLocaleString()} scenes</p>
          <p className="max-w-72 truncate text-slate-600">
            {statistics.data.labelset_id}
          </p>
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
            onClick={() => setTab(id)}
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
        <div className="space-y-8">
          <section>
            <h2 className="text-sm font-semibold">Scene composition</h2>
            <p className="mt-1 text-xs text-slate-500">
              Ratio denominator is scenes with a valid observation for each key.
              Missingness is shown separately as observable coverage.
            </p>
          </section>
          {statistics.data.keys
            .filter((item) => item.namespace === "odd")
            .map((item) => (
              <section
                key={item.key}
                className="grid gap-3 border-b border-slate-900 pb-5 lg:grid-cols-[18rem_1fr]"
              >
                <div className="min-w-0">
                  <p className="break-words font-mono text-xs text-slate-300">
                    {item.key}
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    Observable {percent(item.observable_scene_coverage)} ·{" "}
                    {item.valid_scene_count}/{item.eligible_scene_count} scenes
                  </p>
                </div>
                <div className="space-y-2">
                  {item.values.map((entry) => (
                    <button
                      key={entry.value}
                      type="button"
                      onClick={() => {
                        setKey(item.key);
                        setValue(entry.value);
                        setStatus("valid");
                        setTab("search");
                      }}
                      className="grid w-full grid-cols-[8rem_1fr_4.5rem] items-center gap-2 text-left text-xs sm:grid-cols-[12rem_1fr_5rem]"
                    >
                      <span className="truncate font-mono text-slate-400">
                        {entry.value}
                      </span>
                      <span className="h-2 overflow-hidden bg-slate-900">
                        <span
                          className="block h-full bg-cyan-600"
                          style={{ width: `${entry.scene_ratio * 100}%` }}
                        />
                      </span>
                      <span className="text-right font-mono text-slate-500">
                        {entry.scene_count} · {percent(entry.scene_ratio)}
                      </span>
                    </button>
                  ))}
                </div>
              </section>
            ))}
        </div>
      )}

      {tab === "search" && (
        <div className="space-y-5">
          <section className="grid gap-3 border-b border-slate-800 pb-5 md:grid-cols-4">
            <label className="space-y-1 text-xs text-slate-500">
              Key
              <select
                value={key}
                onChange={(event) => {
                  setKey(event.target.value);
                  setValue("");
                }}
                className="h-9 w-full border border-slate-700 bg-slate-950 px-2 text-xs text-slate-200"
              >
                <option value="">Any key</option>
                {ontology.data.labels.map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.key}
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-1 text-xs text-slate-500">
              Value
              <select
                value={value}
                onChange={(event) => setValue(event.target.value)}
                className="h-9 w-full border border-slate-700 bg-slate-950 px-2 text-xs text-slate-200"
              >
                <option value="">Any value</option>
                {definition?.values.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.value} ({counts.get(key)?.values.find((row) => row.value === item.value)?.scene_count ?? 0})
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-1 text-xs text-slate-500">
              Status
              <select
                value={status}
                onChange={(event) => setStatus(event.target.value)}
                className="h-9 w-full border border-slate-700 bg-slate-950 px-2 text-xs text-slate-200"
              >
                <option value="">Any status</option>
                {ontology.data.statuses.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>
            </label>
            <label className="space-y-1 text-xs text-slate-500">
              Source
              <select
                value={source}
                onChange={(event) => setSource(event.target.value)}
                className="h-9 w-full border border-slate-700 bg-slate-950 px-2 text-xs text-slate-200"
              >
                <option value="">Any source</option>
                {ontology.data.sources.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>
            </label>
          </section>
          {search.loading ? (
            <Skeleton className="h-48 w-full" />
          ) : search.error ? (
            <ErrorState error={search.error} onRetry={search.reload} />
          ) : (
            <section>
              <p className="mb-3 text-xs text-slate-500">
                {search.data?.total ?? 0} matching scenes
              </p>
              <div className="divide-y divide-slate-900">
                {search.data?.scenes.map((scene) => (
                  <Link
                    key={scene.scene_uid}
                    href={`/scenes/${DATASET}/${encodeURIComponent(scene.shard_name)}/0?version=${VERSION}`}
                    className="grid gap-1 py-3 hover:bg-slate-950 sm:grid-cols-[1fr_auto]"
                  >
                    <span className="break-all font-mono text-xs text-slate-300">
                      {scene.scene_uid}
                    </span>
                    <span className="text-xs text-slate-500">
                      {scene.observations.length} summaries ·{" "}
                      {scene.distance_m.toFixed(0)} m
                    </span>
                  </Link>
                ))}
              </div>
            </section>
          )}
        </div>
      )}

      {tab === "ontology" && (
        <div className="divide-y divide-slate-900">
          {ontology.data.labels.map((item) => (
            <details key={item.key} className="py-4">
              <summary className="cursor-pointer list-none">
                <div className="grid gap-1 sm:grid-cols-[1fr_auto]">
                  <span className="break-words font-mono text-xs text-slate-200">
                    {item.key}
                  </span>
                  <span className="text-xs text-slate-500">
                    {item.cardinality} · {item.primary_sources.join(", ")}
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate-500">{item.description}</p>
              </summary>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {item.values.map((candidate) => (
                  <span
                    key={candidate.value}
                    className="border border-slate-800 px-2 py-1 font-mono text-[11px] text-slate-400"
                  >
                    {candidate.value} ·{" "}
                    {counts.get(item.key)?.values.find((row) => row.value === candidate.value)?.scene_count ?? 0}
                  </span>
                ))}
              </div>
            </details>
          ))}
        </div>
      )}

      {tab === "labelsets" && (
        <section className="grid gap-4 md:grid-cols-3">
          <div>
            <p className="text-xs text-slate-500">LabelSet</p>
            <p className="mt-1 break-all font-mono text-xs text-slate-200">
              {statistics.data.labelset_id}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Ontology</p>
            <p className="mt-1 font-mono text-xs text-slate-200">
              {ontology.data.ontology_version}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Coverage</p>
            <p className="mt-1 font-mono text-xs text-slate-200">
              {statistics.data.scene_count} ready scenes
            </p>
          </div>
        </section>
      )}
    </div>
  );
}
