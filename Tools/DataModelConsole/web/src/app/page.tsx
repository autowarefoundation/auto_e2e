"use client";

import Link from "next/link";
import { ChevronRight } from "lucide-react";

import { ErrorState, humanizeError } from "@/components/error-state";
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
import { getDashboardStats, listExecutionsPage } from "@/lib/api";
import {
  formatDuration,
  formatMeters,
  formatNumber,
  formatTimestamp,
} from "@/lib/format";

function KpiCard({
  title,
  value,
  loading,
  href,
  subtitle,
}: {
  title: string;
  value: string;
  loading: boolean;
  href?: string;
  subtitle?: string;
}) {
  const card = (
    <Card
      className={
        "border-slate-800 bg-slate-950/50" +
        (href ? " h-full cursor-pointer transition-colors hover:border-slate-500" : "")
      }
    >
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-xs font-medium uppercase tracking-wider text-slate-400">
          {title}
          {href && <ChevronRight className="size-3.5 text-slate-600" />}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-8 w-24" />
        ) : (
          <p className="font-mono text-2xl font-semibold">{value}</p>
        )}
        {subtitle && (
          <p className="mt-1 text-[10px] text-slate-500">{subtitle}</p>
        )}
      </CardContent>
    </Card>
  );
  return href ? (
    <Link href={href} className="block">
      {card}
    </Link>
  ) : (
    card
  );
}

export default function HomePage() {
  const stats = useApi(getDashboardStats);
  const executions = useApi(async () => {
    const page = await listExecutionsPage(10);
    return page.items;
  });

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
        (() => {
          // When MLflow is unreachable the API returns mlflow_available=false;
          // surface that explicitly instead of a fabricated 0 runs / — ADE.
          const mlflowDown = !!stats.data && !stats.data.mlflow_available;
          return (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <KpiCard
            title="Total Samples"
            value={formatNumber(stats.data?.total_samples)}
            loading={stats.loading}
            href="/datasets"
            subtitle="from pipeline manifest"
          />
          <KpiCard
            title="Reasoning Labels"
            value={formatNumber(stats.data?.reasoning_labels)}
            loading={stats.loading}
            href="/reasoning-labels"
            subtitle="label objects across all teacher/prompt versions (a sample may be labelled more than once)"
          />
          <KpiCard
            title="MLflow Runs"
            value={mlflowDown ? "—" : formatNumber(stats.data?.mlflow_runs)}
            loading={stats.loading}
            href="/models"
            subtitle={mlflowDown ? "Unavailable — upstream unreachable" : undefined}
          />
          <KpiCard
            title="Latest ADE"
            value={mlflowDown ? "—" : formatMeters(stats.data?.latest_ade)}
            loading={stats.loading}
            href="/models"
            subtitle={
              mlflowDown
                ? "Unavailable — upstream unreachable"
                : "Average Displacement Error"
            }
          />
        </div>
          );
        })()
      )}

      <Card className="border-slate-800 bg-slate-950/50">
        <CardHeader>
          <CardTitle className="text-sm">Recent Flyte Executions</CardTitle>
        </CardHeader>
        <CardContent>
          {executions.error ? (
            <Table>
              <TableBody>
                <TableRow>
                  <TableCell className="text-center text-sm text-slate-500">
                    {humanizeError(executions.error, "Flyte")}{" "}
                    <button
                      onClick={executions.reload}
                      className="text-blue-500 hover:underline"
                    >
                      Retry
                    </button>
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
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
