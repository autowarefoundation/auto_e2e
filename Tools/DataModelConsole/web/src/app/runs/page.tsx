"use client";

import Link from "next/link";
import { ChevronDown, Loader2 } from "lucide-react";

import { ErrorState } from "@/components/error-state";
import { StatusBadge, flytePhaseTone } from "@/components/status-badge";
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
import { useTokenPages } from "@/hooks/use-token-pages";
import { listExecutionsPage } from "@/lib/api";
import { formatDuration, formatTimestamp } from "@/lib/format";

export default function RunsPage() {
  const {
    items: executions,
    error,
    loading,
    loadingMore,
    hasMore,
    loadMore,
    reload,
  } = useTokenPages(
    (token) => listExecutionsPage(50, token),
    [],
    (execution) => execution.execution_id,
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Runs</h2>
        <p className="text-sm text-slate-400">
          Flyte workflow executions (project auto-e2e, domain development).
        </p>
      </div>

      <Card className="border-slate-800 bg-slate-950/50">
        <CardHeader>
          <CardTitle className="text-sm">Executions</CardTitle>
        </CardHeader>
        <CardContent>
          {error && executions.length === 0 ? (
            <ErrorState error={error} onRetry={reload} service="Flyte" />
          ) : loading ? (
            <div className="space-y-2">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-9 w-full" />
              ))}
            </div>
          ) : (
            <div className="space-y-3">
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
                  {executions.map((execution) => (
                    <TableRow key={execution.execution_id}>
                      <TableCell>
                        <Link
                          href={`/runs/${encodeURIComponent(execution.execution_id)}`}
                          className="font-mono text-xs text-blue-500 hover:underline"
                        >
                          {execution.execution_id}
                        </Link>
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        {execution.workflow_name}
                      </TableCell>
                      <TableCell>
                        <StatusBadge
                          label={execution.phase}
                          tone={flytePhaseTone(execution.phase)}
                        />
                      </TableCell>
                      <TableCell className="text-xs text-slate-400">
                        {formatTimestamp(execution.started_at)}
                      </TableCell>
                      <TableCell className="text-right font-mono text-xs">
                        {formatDuration(execution.duration_s)}
                      </TableCell>
                    </TableRow>
                  ))}
                  {executions.length === 0 && (
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
              {error && (
                <ErrorState
                  error={error}
                  onRetry={hasMore ? loadMore : reload}
                  service="Flyte"
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
                  Load more executions
                </Button>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
