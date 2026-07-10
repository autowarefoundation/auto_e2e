"use client";

import Link from "next/link";

import { ErrorState } from "@/components/error-state";
import { StatusBadge, flytePhaseTone } from "@/components/status-badge";
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
import { useApi } from "@/hooks/use-api";
import { getDashboardStats, listExecutions } from "@/lib/api";
import {
  formatDuration,
  formatMetric,
  formatNumber,
  formatTimestamp,
} from "@/lib/format";

function KpiCard({
  title,
  value,
  loading,
}: {
  title: string;
  value: string;
  loading: boolean;
}) {
  return (
    <Card className="border-slate-800 bg-slate-950/50">
      <CardHeader className="pb-2">
        <CardTitle className="text-xs font-medium uppercase tracking-wider text-slate-400">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-8 w-24" />
        ) : (
          <p className="font-mono text-2xl font-semibold">{value}</p>
        )}
      </CardContent>
    </Card>
  );
}

export default function HomePage() {
  const stats = useApi(getDashboardStats);
  const executions = useApi(() => listExecutions(10));

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Overview</h2>
        <p className="text-sm text-slate-400">
          Datasets, reasoning labels, training runs and pipeline executions at
          a glance.
        </p>
      </div>

      {stats.error ? (
        <ErrorState error={stats.error} onRetry={stats.reload} />
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <KpiCard
            title="Total Samples"
            value={formatNumber(stats.data?.total_samples)}
            loading={stats.loading}
          />
          <KpiCard
            title="Reasoning Labels"
            value={formatNumber(stats.data?.reasoning_labels)}
            loading={stats.loading}
          />
          <KpiCard
            title="MLflow Runs"
            value={formatNumber(stats.data?.mlflow_runs)}
            loading={stats.loading}
          />
          <KpiCard
            title="Latest ADE"
            value={formatMetric(stats.data?.latest_ade)}
            loading={stats.loading}
          />
        </div>
      )}

      <Card className="border-slate-800 bg-slate-950/50">
        <CardHeader>
          <CardTitle className="text-sm">Recent Flyte Executions</CardTitle>
        </CardHeader>
        <CardContent>
          {executions.error ? (
            <ErrorState error={executions.error} onRetry={executions.reload} />
          ) : executions.loading ? (
            <div className="space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-9 w-full" />
              ))}
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Execution</TableHead>
                  <TableHead>Workflow</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Started</TableHead>
                  <TableHead className="text-right">Duration</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {(executions.data ?? []).slice(0, 10).map((e) => (
                  <TableRow key={e.execution_id}>
                    <TableCell>
                      <Link
                        href={`/runs/${encodeURIComponent(e.execution_id)}`}
                        className="font-mono text-xs text-blue-500 hover:underline"
                      >
                        {e.execution_id}
                      </Link>
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {e.workflow_name}
                    </TableCell>
                    <TableCell>
                      <StatusBadge
                        label={e.phase}
                        tone={flytePhaseTone(e.phase)}
                      />
                    </TableCell>
                    <TableCell className="text-xs text-slate-400">
                      {formatTimestamp(e.started_at)}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {formatDuration(e.duration_s)}
                    </TableCell>
                  </TableRow>
                ))}
                {(executions.data ?? []).length === 0 && (
                  <TableRow>
                    <TableCell
                      colSpan={5}
                      className="text-center text-sm text-slate-500"
                    >
                      No executions found
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
