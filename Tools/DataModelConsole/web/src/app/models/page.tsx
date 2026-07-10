"use client";

import { useState } from "react";
import { Boxes } from "lucide-react";

import { ErrorState } from "@/components/error-state";
import { StatusBadge, mlflowStatusTone } from "@/components/status-badge";
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
import { useApi } from "@/hooks/use-api";
import { listExperiments, listRuns } from "@/lib/api";
import {
  formatDuration,
  formatEpochMillis,
  formatMetric,
  formatNumber,
} from "@/lib/format";

const METRIC_COLUMNS = ["train/loss", "eval/ade", "eval/fde"] as const;

function RunsTable({ experimentId }: { experimentId: string }) {
  const { data, error, loading, reload } = useApi(
    () => listRuns(experimentId),
    [experimentId],
  );

  if (error) return <ErrorState error={error} onRetry={reload} />;
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
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Run</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Started</TableHead>
          <TableHead className="text-right">Duration</TableHead>
          {METRIC_COLUMNS.map((m) => (
            <TableHead key={m} className="text-right font-mono text-[11px]">
              {m}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {(data ?? []).map((run) => (
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
                {formatMetric(run.metrics?.[m])}
              </TableCell>
            ))}
          </TableRow>
        ))}
        {(data ?? []).length === 0 && (
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
  );
}

export default function ModelsPage() {
  const { data, error, loading, reload } = useApi(listExperiments);
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Models</h2>
        <p className="text-sm text-slate-400">
          MLflow experiments and runs (proxied by the console API).
        </p>
      </div>

      {error ? (
        <ErrorState error={error} onRetry={reload} />
      ) : loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 2 }).map((_, i) => (
            <Skeleton key={i} className="h-28 w-full" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {(data ?? []).map((exp) => (
            <button
              key={exp.experiment_id}
              type="button"
              onClick={() => setSelected(exp.experiment_id)}
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
                <CardContent className="flex items-center justify-between text-xs text-slate-400">
                  <span>
                    {formatNumber(exp.run_count)} run
                    {exp.run_count === 1 ? "" : "s"}
                  </span>
                  <span className="font-mono text-[10px]">
                    id {exp.experiment_id}
                  </span>
                </CardContent>
              </Card>
            </button>
          ))}
          {(data ?? []).length === 0 && (
            <p className="text-sm text-slate-500">No experiments found.</p>
          )}
        </div>
      )}

      {selected && (
        <Card className="border-slate-800 bg-slate-950/50">
          <CardHeader>
            <CardTitle className="text-sm">
              Runs —{" "}
              <span className="font-mono">
                {data?.find((e) => e.experiment_id === selected)?.name ??
                  selected}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <RunsTable experimentId={selected} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
