"use client";

import { Suspense, useCallback, useEffect } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Boxes, ChevronDown, Loader2 } from "lucide-react";

import { ErrorState } from "@/components/error-state";
import { StatusBadge, mlflowStatusTone } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { useTokenPages } from "@/hooks/use-token-pages";
import { listExperimentsPage, listRunsPage } from "@/lib/api";
import {
  formatDuration,
  formatEpochMillis,
  formatMeters,
  formatMetric,
} from "@/lib/format";

const METRIC_COLUMNS = ["train/loss", "eval/ade", "eval/fde"] as const;

// Human-readable expansions + meter-formatting for displacement-error metrics.
const METRIC_META: Record<
  (typeof METRIC_COLUMNS)[number],
  { title: string; meters: boolean }
> = {
  "train/loss": { title: "Training loss", meters: false },
  "eval/ade": { title: "Average Displacement Error (meters)", meters: true },
  "eval/fde": { title: "Final Displacement Error (meters)", meters: true },
};

function RunsTable({ experimentId }: { experimentId: string }) {
  const {
    items: runs,
    error,
    loading,
    loadingMore,
    hasMore,
    loadMore,
    reload,
  } = useTokenPages(
    (token) => listRunsPage(experimentId, token),
    [experimentId],
    (run) => run.run_id,
  );

  if (error && runs.length === 0) {
    return <ErrorState error={error} onRetry={reload} service="MLflow" />;
  }
  if (loading) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-9 w-full" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Run</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Started</TableHead>
            <TableHead className="text-right">Duration</TableHead>
            {METRIC_COLUMNS.map((m) => (
              <TableHead
                key={m}
                className="text-right font-mono text-[11px]"
                title={METRIC_META[m].title}
                aria-label={METRIC_META[m].title}
              >
                {m}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((run) => (
            <TableRow key={run.run_id}>
              <TableCell>
                <div className="flex flex-col">
                  <span className="text-xs">{run.run_name || run.run_id}</span>
                  <span className="font-mono text-[10px] text-slate-500">
                    {run.run_id}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <StatusBadge
                  label={run.status}
                  tone={mlflowStatusTone(run.status)}
                />
              </TableCell>
              <TableCell className="text-xs text-slate-400">
                {formatEpochMillis(run.start_time)}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">
                {run.end_time > 0
                  ? formatDuration((run.end_time - run.start_time) / 1000)
                  : "-"}
              </TableCell>
              {METRIC_COLUMNS.map((m) => (
                <TableCell key={m} className="text-right font-mono text-xs">
                  {METRIC_META[m].meters
                    ? formatMeters(run.metrics?.[m])
                    : formatMetric(run.metrics?.[m])}
                </TableCell>
              ))}
            </TableRow>
          ))}
          {runs.length === 0 && (
            <TableRow>
              <TableCell
                colSpan={4 + METRIC_COLUMNS.length}
                className="text-center text-sm text-slate-500"
              >
                No runs found
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      {error && (
        <ErrorState
          error={error}
          onRetry={hasMore ? loadMore : reload}
          service="MLflow"
        />
      )}
      {hasMore && (
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={loadMore}
          disabled={loadingMore}
        >
          {loadingMore ? (
            <Loader2 className="size-3.5 animate-spin" />
          ) : (
            <ChevronDown className="size-3.5" />
          )}
          Load more runs
        </Button>
      )}
    </div>
  );
}

function ModelsPageInner() {
  const {
    items: data,
    error,
    loading,
    loadingMore,
    hasMore,
    loadMore,
    reload,
  } = useTokenPages(
    (token) => listExperimentsPage(token),
    [],
    (experiment) => experiment.experiment_id,
  );
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const urlExperiment = searchParams.get("experiment") ?? "";
  const selected =
    data.find((exp) => exp.experiment_id === urlExperiment)
      ?.experiment_id ?? null;
  const defaultExperiment = data[0]?.experiment_id ?? null;

  // Resolve shared links to experiments beyond the first page before deciding
  // that the URL selection is invalid.
  useEffect(() => {
    if (loading || error || selected || !defaultExperiment) return;

    if (urlExperiment && hasMore) {
      if (!loadingMore) loadMore();
      return;
    }

    const q = new URLSearchParams(searchParams.toString());
    q.set("experiment", defaultExperiment);
    router.replace(`${pathname}?${q.toString()}`, { scroll: false });
  }, [
    defaultExperiment,
    error,
    hasMore,
    loadMore,
    loading,
    loadingMore,
    pathname,
    router,
    searchParams,
    selected,
    urlExperiment,
  ]);

  // User choices are navigation: push each one so Back and Forward restore the
  // URL-derived selection and trigger the matching runs request.
  const selectExperiment = useCallback(
    (id: string) => {
      if (id === selected) return;

      const q = new URLSearchParams(searchParams.toString());
      q.set("experiment", id);
      router.push(`${pathname}?${q.toString()}`, { scroll: false });
    },
    [pathname, router, searchParams, selected],
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Models</h2>
        <p className="text-sm text-slate-400">
          MLflow experiments and runs (proxied by the console API).
        </p>
      </div>

      {error && data.length === 0 ? (
        <ErrorState error={error} onRetry={reload} service="MLflow" />
      ) : loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 2 }).map((_, i) => (
            <Skeleton key={i} className="h-28 w-full" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data.map((exp) => (
            <button
              key={exp.experiment_id}
              type="button"
              onClick={() => selectExperiment(exp.experiment_id)}
              aria-pressed={selected === exp.experiment_id}
              className="text-left"
            >
              <Card
                className={cn(
                  "border-slate-800 bg-slate-950/50 transition-colors hover:border-slate-600",
                  selected === exp.experiment_id && "border-blue-500/60",
                )}
              >
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 font-mono text-sm">
                    <Boxes className="size-4 text-blue-500" />
                    {exp.name}
                  </CardTitle>
                </CardHeader>
                {/* No run count: MLflow's experiments/search does not return
                    one, and a per-experiment runs/search per card would be N
                    extra calls on load. Showing "0 runs" was simply wrong, so
                    the card shows the id/stage only. */}
                <CardContent className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-mono text-[10px]">
                    id {exp.experiment_id}
                  </span>
                </CardContent>
              </Card>
            </button>
          ))}
          {data.length === 0 && (
            <p className="text-sm text-slate-500">No experiments found.</p>
          )}
        </div>
      )}
      {error && data.length > 0 && (
        <ErrorState
          error={error}
          onRetry={hasMore ? loadMore : reload}
          service="MLflow"
        />
      )}
      {hasMore && (
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={loadMore}
          disabled={loadingMore}
        >
          {loadingMore ? (
            <Loader2 className="size-3.5 animate-spin" />
          ) : (
            <ChevronDown className="size-3.5" />
          )}
          Load more experiments
        </Button>
      )}

      {selected && (
        <Card className="border-slate-800 bg-slate-950/50">
          <CardHeader>
            <CardTitle className="text-sm">
              Runs —{" "}
              <span className="font-mono">
                {data.find((e) => e.experiment_id === selected)?.name ??
                  selected}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <RunsTable key={selected} experimentId={selected} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default function ModelsPage() {
  // useSearchParams (in ModelsPageInner) must be under a Suspense boundary.
  return (
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <ModelsPageInner />
    </Suspense>
  );
}
