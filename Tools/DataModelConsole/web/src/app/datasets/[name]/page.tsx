"use client";

import Link from "next/link";
import { use } from "react";

import { ErrorState } from "@/components/error-state";
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
import { listShards } from "@/lib/api";
import { formatBytes, formatTimestamp } from "@/lib/format";

export default function DatasetDetailPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const { name } = use(params);
  const dataset = decodeURIComponent(name);
  const { data, error, loading, reload } = useApi(
    () => listShards(dataset),
    [dataset],
  );

  const shards = data?.shards ?? [];

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs text-slate-500">
          <Link href="/datasets" className="hover:text-slate-300">
            Datasets
          </Link>{" "}
          / <span className="font-mono">{dataset}</span>
        </p>
        <h2 className="mt-1 font-mono text-lg font-semibold">{dataset}</h2>
        <p className="text-sm text-slate-400">
          Shards{data?.page ? ` (${data.page.total} total)` : ""}
        </p>
      </div>

      <Card className="border-slate-800 bg-slate-950/50">
        <CardHeader>
          <CardTitle className="text-sm">Shards</CardTitle>
        </CardHeader>
        <CardContent>
          {error ? (
            <ErrorState error={error} onRetry={reload} />
          ) : loading ? (
            <div className="space-y-2">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-9 w-full" />
              ))}
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Shard</TableHead>
                  <TableHead className="text-right">Size</TableHead>
                  <TableHead>Last Modified</TableHead>
                  <TableHead className="text-right">Player</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {shards.map((shard) => (
                  <TableRow key={shard.name}>
                    <TableCell>
                      <Link
                        href={`/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard.name)}`}
                        className="font-mono text-xs text-blue-500 hover:underline"
                      >
                        {shard.name}
                      </Link>
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {formatBytes(shard.size_bytes)}
                    </TableCell>
                    <TableCell className="text-xs text-slate-400">
                      {formatTimestamp(shard.last_modified)}
                    </TableCell>
                    <TableCell className="text-right">
                      <Link
                        href={`/scenes/${encodeURIComponent(dataset)}/${encodeURIComponent(shard.name)}/0`}
                        className="font-mono text-xs text-blue-500 hover:underline"
                      >
                        Play
                      </Link>
                    </TableCell>
                  </TableRow>
                ))}
                {shards.length === 0 && (
                  <TableRow>
                    <TableCell
                      colSpan={4}
                      className="text-center text-sm text-slate-500"
                    >
                      No shards found
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
