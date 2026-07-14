"use client";

import Link from "next/link";
import { Database } from "lucide-react";

import { ErrorState } from "@/components/error-state";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/hooks/use-api";
import { listDatasets } from "@/lib/api";

export default function DatasetsPage() {
  const { data, error, loading, reload } = useApi(listDatasets);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Datasets</h2>
        <p className="text-sm text-slate-400">
          WebDataset shards stored in the platform datasets bucket.
        </p>
      </div>

      {error ? (
        <ErrorState error={error} onRetry={reload} />
      ) : loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 2 }).map((_, i) => (
            <Skeleton key={i} className="h-40 w-full" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {(data ?? []).map((ds) => (
            <Link key={ds.name} href={`/datasets/${encodeURIComponent(ds.name)}`}>
              <Card className="border-slate-800 bg-slate-950/50 transition-colors hover:border-slate-600">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 font-mono text-base">
                    <Database className="size-4 text-blue-500" />
                    {ds.name}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex flex-wrap gap-1">
                    <Badge variant="secondary" className="text-[10px]">
                      {ds.version}
                    </Badge>
                  </div>
                  <p className="truncate font-mono text-xs text-slate-500">
                    {ds.prefix}
                  </p>
                </CardContent>
              </Card>
            </Link>
          ))}
          {(data ?? []).length === 0 && (
            <p className="text-sm text-slate-500">No datasets found.</p>
          )}
        </div>
      )}
    </div>
  );
}
