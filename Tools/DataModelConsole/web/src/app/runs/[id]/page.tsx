"use client";

import Link from "next/link";
import { use } from "react";

import { ErrorState } from "@/components/error-state";
import { StatusBadge, flytePhaseTone } from "@/components/status-badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/hooks/use-api";
import { getExecution } from "@/lib/api";
import { formatDuration, formatTimestamp } from "@/lib/format";

// Phase 1: node DAG rendered as a simplified ordered list.
export default function ExecutionDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const executionId = decodeURIComponent(id);
  const { data, error, loading, reload } = useApi(
    () => getExecution(executionId),
    [executionId],
  );

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs text-slate-500">
          <Link href="/runs" className="hover:text-slate-300">
            Runs
          </Link>{" "}
          / <span className="font-mono">{executionId}</span>
        </p>
        <h2 className="mt-1 font-mono text-lg font-semibold">{executionId}</h2>
      </div>

      {error ? (
        <ErrorState error={error} onRetry={reload} />
      ) : loading ? (
        <div className="space-y-4">
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : data ? (
        <>
          <Card className="border-slate-800 bg-slate-950/50">
            <CardContent className="flex flex-wrap items-center gap-4 text-sm">
              <span className="font-mono text-xs">{data.workflow_name}</span>
              <Separator orientation="vertical" className="h-4" />
              <StatusBadge label={data.phase} tone={flytePhaseTone(data.phase)} />
              <Separator orientation="vertical" className="h-4" />
              <span className="text-xs text-slate-400">
                started {formatTimestamp(data.started_at)}
              </span>
              <Separator orientation="vertical" className="h-4" />
              <span className="font-mono text-xs">
                {formatDuration(data.duration_s)}
              </span>
            </CardContent>
          </Card>

          <Card className="border-slate-800 bg-slate-950/50">
            <CardHeader>
              <CardTitle className="text-sm">Nodes</CardTitle>
            </CardHeader>
            <CardContent>
              {(data.nodes ?? []).length === 0 ? (
                <p className="text-sm text-slate-500">
                  No node details available.
                </p>
              ) : (
                <ol className="space-y-2">
                  {(data.nodes ?? []).map((node, i) => (
                    <li
                      key={node.node_id}
                      className="rounded-md border border-slate-800 bg-slate-900/40 p-3"
                    >
                      <div className="flex flex-wrap items-center gap-3">
                        <span className="font-mono text-xs text-slate-500">
                          {i + 1}.
                        </span>
                        <span className="font-mono text-xs">
                          {node.display_name || node.node_id}
                        </span>
                        <StatusBadge
                          label={node.phase}
                          tone={flytePhaseTone(node.phase)}
                        />
                        <span className="ml-auto font-mono text-xs text-slate-400">
                          {formatDuration(node.duration_s)}
                        </span>
                      </div>
                      {(node.inputs || node.outputs) && (
                        <details className="mt-2 text-xs">
                          <summary className="cursor-pointer text-slate-500 hover:text-slate-300">
                            Inputs / Outputs
                          </summary>
                          <div className="mt-2 grid gap-2 lg:grid-cols-2">
                            <pre className="max-h-40 overflow-auto rounded border border-slate-800 bg-slate-950 p-2 font-mono text-[10px] text-slate-400">
                              {JSON.stringify(node.inputs ?? {}, null, 2)}
                            </pre>
                            <pre className="max-h-40 overflow-auto rounded border border-slate-800 bg-slate-950 p-2 font-mono text-[10px] text-slate-400">
                              {JSON.stringify(node.outputs ?? {}, null, 2)}
                            </pre>
                          </div>
                        </details>
                      )}
                    </li>
                  ))}
                </ol>
              )}
            </CardContent>
          </Card>
        </>
      ) : null}
    </div>
  );
}
