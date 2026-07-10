"use client";

import Link from "next/link";
import { use } from "react";
import { Play } from "lucide-react";

import { CameraImage } from "@/components/camera-image";
import { ErrorState } from "@/components/error-state";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/hooks/use-api";
import { listSamples } from "@/lib/api";
import { formatBytes } from "@/lib/format";

export default function ShardSamplesPage({
  params,
}: {
  params: Promise<{ name: string; shard: string }>;
}) {
  const { name, shard } = use(params);
  const dataset = decodeURIComponent(name);
  const shardName = decodeURIComponent(shard);
  const { data, error, loading, reload } = useApi(
    () => listSamples(dataset, shardName),
    [dataset, shardName],
  );

  const samples = data?.samples ?? [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs text-slate-500">
            <Link href="/datasets" className="hover:text-slate-300">
              Datasets
            </Link>{" "}
            /{" "}
            <Link
              href={`/datasets/${encodeURIComponent(dataset)}`}
              className="font-mono hover:text-slate-300"
            >
              {dataset}
            </Link>{" "}
            / <span className="font-mono">{shardName}</span>
          </p>
          <h2 className="mt-1 font-mono text-lg font-semibold">{shardName}</h2>
          <p className="text-sm text-slate-400">
            Samples with front camera (cam_0) thumbnail.
          </p>
        </div>
        <Link
          href={`/scenes/${encodeURIComponent(dataset)}/${encodeURIComponent(shardName)}/0`}
          className="inline-flex items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-200 transition-colors hover:border-slate-500"
        >
          <Play className="size-3.5" />
          Open in player
        </Link>
      </div>

      {error ? (
        <ErrorState error={error} onRetry={reload} />
      ) : loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-44 w-full" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {samples.map((sample) => {
            const totalBytes = sample.members.reduce(
              (acc, m) => acc + m.size_bytes,
              0,
            );
            return (
              <Link
                key={sample.key}
                href={`/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shardName)}/samples/${encodeURIComponent(sample.key)}`}
              >
                <Card className="gap-0 overflow-hidden border-slate-800 bg-slate-950/50 py-0 transition-colors hover:border-slate-600">
                  <CameraImage
                    dataset={dataset}
                    shard={shardName}
                    sampleKey={sample.key}
                    cam={0}
                    className="aspect-video w-full"
                  />
                  <CardContent className="space-y-1.5 p-3">
                    <p className="font-mono text-xs">{sample.key}</p>
                    <div className="flex items-center gap-2 text-[10px] text-slate-500">
                      <span>{sample.members.length} members</span>
                      <span>{formatBytes(totalBytes)}</span>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
          {samples.length === 0 && (
            <p className="text-sm text-slate-500">No samples found.</p>
          )}
        </div>
      )}
    </div>
  );
}
