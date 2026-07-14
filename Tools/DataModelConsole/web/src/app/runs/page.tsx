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
import { listExecutions } from "@/lib/api";
import { formatDuration, formatTimestamp } from "@/lib/format";

export default function RunsPage() {
  const { data, error, loading, reload } = useApi(() => listExecutions(50));

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
          {error ? (
            <ErrorState error={error} onRetry={reload} />
          ) : loading ? (
            <div className="space-y-2">
              {Array.from({ length: 8 }).map((_, i) => (
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
                {(data ?? []).map((e) => (
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
                {(data ?? []).length === 0 && (
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
