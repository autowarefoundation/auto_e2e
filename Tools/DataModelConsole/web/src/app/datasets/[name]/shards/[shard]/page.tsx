"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  Suspense,
  use,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { Loader2, Play } from "lucide-react";

import { CameraImage } from "@/components/camera-image";
import { ErrorState } from "@/components/error-state";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/hooks/use-api";
import { listSamples } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import type { Sample } from "@/types";

const PAGE_SIZE = 60;

function ShardSamplesInner({
  dataset,
  shardName,
}: {
  dataset: string;
  shardName: string;
}) {
  const searchParams = useSearchParams();
  const version = searchParams.get("version") ?? "";
  const versionQuery = version ? `?version=${encodeURIComponent(version)}` : "";
  const { data, error, loading, reload } = useApi(
    () => listSamples(dataset, shardName, version || undefined, 0, PAGE_SIZE),
    [dataset, shardName, version],
  );

  // First page from useApi; extra pages append via "Load more" so a shard with
  // hundreds of samples is fully browsable (the API caps one page, previously
  // stranding everything past the first 50).
  const [extra, setExtra] = useState<Sample[]>([]);
  const [more, setMore] = useState(false);
  const [total, setTotal] = useState(0);
  const [loadingMore, setLoadingMore] = useState(false);
  const [moreError, setMoreError] = useState<Error | null>(null);
  const pageGeneration = useRef(0);

  useEffect(() => {
    pageGeneration.current++;
    setExtra([]);
    setMore(data?.page?.more ?? false);
    setTotal(data?.page?.total ?? 0);
    setLoadingMore(false);
    setMoreError(null);
  }, [dataset, shardName, version, data]);

  const samples = [...(data?.samples ?? []), ...extra];

  const loadMore = useCallback(async () => {
    const generation = ++pageGeneration.current;
    setLoadingMore(true);
    setMoreError(null);
    try {
      const res = await listSamples(
        dataset,
        shardName,
        version || undefined,
        samples.length,
        PAGE_SIZE,
      );
      if (generation !== pageGeneration.current) return;
      setExtra((prev) => [...prev, ...res.samples]);
      setMore(res.page.more);
    } catch (err) {
      if (generation !== pageGeneration.current) return;
      setMoreError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      if (generation === pageGeneration.current) setLoadingMore(false);
    }
  }, [dataset, shardName, version, samples.length]);

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
              href={`/datasets/${encodeURIComponent(dataset)}${versionQuery}`}
              className="font-mono hover:text-slate-300"
            >
              {dataset}
            </Link>{" "}
            / <span className="font-mono">{shardName}</span>
          </p>
          <h2 className="mt-1 font-mono text-lg font-semibold">{shardName}</h2>
          <p className="text-sm text-slate-400">
            Samples with front camera (cam_0) thumbnail.
            {total > 0 && (
              <span className="text-slate-500">
                {" "}
                Showing {samples.length} of {total}.
              </span>
            )}
          </p>
        </div>
        <Link
          href={`/scenes/${encodeURIComponent(dataset)}/${encodeURIComponent(shardName)}/0${versionQuery}`}
          className="inline-flex items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-200 transition-colors hover:border-slate-500"
        >
          <Play className="size-3.5" />
          Open in player
        </Link>
      </div>

      {error ? (
        <ErrorState error={error} onRetry={reload} />
      ) : loading ? (
        <div className="space-y-4">
          <p className="flex items-center gap-2 text-xs text-slate-500">
            <Loader2 className="size-3.5 animate-spin" />
            Scanning shard index — first open takes ~10s, then it is cached.
          </p>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-44 w-full" />
            ))}
          </div>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {samples.map((sample) => {
            const totalBytes = sample.members.reduce(
              (acc, m) => acc + m.size_bytes,
              0,
            );
            const thumbnail = sample.members.find((member) =>
              member.name.endsWith(".cam_0.jpg"),
            );
            return (
              <Link
                key={sample.key}
                href={`/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shardName)}/samples/${encodeURIComponent(sample.key)}${versionQuery}`}
              >
                <Card className="gap-0 overflow-hidden border-slate-800 bg-slate-950/50 py-0 transition-colors hover:border-slate-600">
                  <CameraImage
                    dataset={dataset}
                    shard={shardName}
                    sampleKey={sample.key}
                    cam={0}
                    className="aspect-video w-full"
                    range={
                      thumbnail
                        ? {
                            offset: thumbnail.offset,
                            size: thumbnail.size_bytes,
                          }
                        : undefined
                    }
                    version={version || undefined}
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

      {!loading && !error && more && (
        <div className="flex flex-col items-center gap-2">
          {moreError && <ErrorState error={moreError} onRetry={loadMore} />}
          <Button
            variant="outline"
            size="sm"
            onClick={loadMore}
            disabled={loadingMore}
          >
            {loadingMore ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              `Load more (${samples.length} of ${total})`
            )}
          </Button>
        </div>
      )}
    </div>
  );
}

export default function ShardSamplesPage({
  params,
}: {
  params: Promise<{ name: string; shard: string }>;
}) {
  const { name, shard } = use(params);
  const dataset = decodeURIComponent(name);
  const shardName = decodeURIComponent(shard);

  return (
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <ShardSamplesInner dataset={dataset} shardName={shardName} />
    </Suspense>
  );
}
