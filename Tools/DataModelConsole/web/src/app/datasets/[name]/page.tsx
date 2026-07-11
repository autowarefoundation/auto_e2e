"use client";

import Link from "next/link";
import { use, useCallback, useEffect, useState } from "react";

import { ErrorState } from "@/components/error-state";
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
import { useApi } from "@/hooks/use-api";
import { listShards } from "@/lib/api";
import { formatBytes, formatTimestamp } from "@/lib/format";
import type { Shard } from "@/types";

const PAGE_SIZE = 50;

export default function DatasetDetailPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const { name } = use(params);
  const dataset = decodeURIComponent(name);
  const { data, error, loading, reload } = useApi(
    () => listShards(dataset, 0, PAGE_SIZE),
    [dataset],
  );

  // Additional pages appended by "Load more" (the first page comes from useApi).
  const [extra, setExtra] = useState<Shard[]>([]);
  const [more, setMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [moreError, setMoreError] = useState<Error | null>(null);

  // Reset appended state whenever the first page (dataset) reloads.
  useEffect(() => {
    setExtra([]);
    setMore(data?.page?.more ?? false);
    setMoreError(null);
  }, [data]);

  const shards = [...(data?.shards ?? []), ...extra];

  const loadMore = useCallback(async () => {
    setLoadingMore(true);
    setMoreError(null);
    try {
      const res = await listShards(dataset, shards.length, PAGE_SIZE);
      setExtra((prev) => [...prev, ...res.shards]);
      setMore(res.page.more);
    } catch (err) {
      setMoreError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoadingMore(false);
    }
  }, [dataset, shards.length]);

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
          {moreError && (
            <div className="mt-3">
              <ErrorState error={moreError} onRetry={loadMore} />
            </div>
          )}
          {more && !loading && (
            <div className="mt-3 flex justify-center">
              <Button
                variant="outline"
                size="sm"
                onClick={loadMore}
                disabled={loadingMore}
              >
                {loadingMore ? "Loading…" : "Load more"}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
