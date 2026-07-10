"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { use, useEffect, useMemo } from "react";
import { ChevronLeft, ChevronRight, Play } from "lucide-react";

import { CameraImage } from "@/components/camera-image";
import { EgoSignal } from "@/components/ego-signal";
import { ErrorState } from "@/components/error-state";
import { ReasoningTimeline } from "@/components/reasoning-timeline";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/hooks/use-api";
import { ApiError, getReasoningLabel, getSample } from "@/lib/api";

const CAMERA_COUNT = 7; // cam_0..cam_6; last cell of the 2x4 mosaic = metadata

// Sample keys look like "ep0_000064": episode id + "_" + frame index
// zero-padded to 6 digits. Sibling keys are frame +/- 1.
function siblingKey(key: string, delta: number): string | null {
  const m = key.match(/^(.*)_(\d{6})$/);
  if (!m) return null;
  const frame = parseInt(m[2], 10) + delta;
  if (frame < 0) return null;
  return `${m[1]}_${String(frame).padStart(6, "0")}`;
}

export default function SampleDetailPage({
  params,
}: {
  params: Promise<{ name: string; shard: string; key: string }>;
}) {
  const { name, shard, key } = use(params);
  const dataset = decodeURIComponent(name);
  const shardName = decodeURIComponent(shard);
  const sampleKey = decodeURIComponent(key);
  const router = useRouter();

  const { data, error, loading, reload } = useApi(
    () => getSample(dataset, shardName, sampleKey),
    [dataset, shardName, sampleKey],
  );

  // Reasoning label is a separate endpoint; 404 simply means "no label".
  const reasoning = useApi(
    () =>
      getReasoningLabel(dataset, sampleKey).catch((err: unknown) => {
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }),
    [dataset, sampleKey],
  );

  const prevKey = useMemo(() => siblingKey(sampleKey, -1), [sampleKey]);
  const nextKey = useMemo(() => siblingKey(sampleKey, +1), [sampleKey]);

  const sampleUrl = (k: string) =>
    `/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shardName)}/samples/${encodeURIComponent(k)}`;

  // Keyboard navigation: ArrowLeft/p = prev frame, ArrowRight/n = next frame.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
      if ((e.key === "ArrowLeft" || e.key === "p") && prevKey) {
        e.preventDefault();
        router.push(sampleUrl(prevKey));
      } else if ((e.key === "ArrowRight" || e.key === "n") && nextKey) {
        e.preventDefault();
        router.push(sampleUrl(nextKey));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prevKey, nextKey, dataset, shardName]);

  const frameIdx = data?.frame_idx;

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
            /{" "}
            <Link
              href={`/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shardName)}`}
              className="font-mono hover:text-slate-300"
            >
              {shardName}
            </Link>{" "}
            / <span className="font-mono">{sampleKey}</span>
          </p>
          <h2 className="mt-1 font-mono text-lg font-semibold">{sampleKey}</h2>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={!prevKey}
            onClick={() => prevKey && router.push(sampleUrl(prevKey))}
            aria-label="Previous frame"
          >
            <ChevronLeft className="size-3.5" />
            Prev
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={!nextKey}
            onClick={() => nextKey && router.push(sampleUrl(nextKey))}
            aria-label="Next frame"
          >
            Next
            <ChevronRight className="size-3.5" />
          </Button>
          <Link
            href={`/scenes/${encodeURIComponent(dataset)}/${encodeURIComponent(shardName)}/${frameIdx ?? 0}`}
            className="inline-flex h-7 items-center gap-1 rounded-md border border-slate-700 bg-slate-900 px-2.5 text-[0.8rem] text-slate-200 transition-colors hover:border-slate-500"
          >
            <Play className="size-3.5" />
            Player
          </Link>
        </div>
      </div>

      {/* 7-camera mosaic: 2 rows x 4 cols, last cell = metadata */}
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        {Array.from({ length: CAMERA_COUNT }).map((_, cam) => (
          <div
            key={cam}
            className="overflow-hidden rounded-md border border-slate-800"
          >
            <CameraImage
              dataset={dataset}
              shard={shardName}
              sampleKey={sampleKey}
              cam={cam}
              className="aspect-video w-full"
            />
            <p className="bg-slate-950 px-2 py-1 font-mono text-[10px] text-slate-400">
              cam_{cam}
            </p>
          </div>
        ))}
        <div className="overflow-hidden rounded-md border border-slate-800 bg-slate-950/50 p-3">
          <p className="mb-2 text-[10px] uppercase tracking-wider text-slate-500">
            Metadata
          </p>
          {loading ? (
            <Skeleton className="h-24 w-full" />
          ) : (
            <pre className="max-h-40 overflow-auto font-mono text-[10px] leading-relaxed text-slate-300">
              {JSON.stringify(
                {
                  episode_id: data?.episode_id,
                  frame_idx: data?.frame_idx,
                  cameras: data?.cameras,
                  ...(data?.meta ?? {}),
                },
                null,
                2,
              )}
            </pre>
          )}
        </div>
      </div>

      <Card className="border-slate-800 bg-slate-950/50">
        <CardHeader>
          <CardTitle className="text-sm">Ego Signal (ego.npy)</CardTitle>
        </CardHeader>
        <CardContent>
          {error ? (
            <ErrorState error={error} onRetry={reload} />
          ) : loading ? (
            <Skeleton className="h-40 w-full" />
          ) : (
            <EgoSignal
              history={data?.ego_history ?? []}
              future={data?.ego_future ?? []}
            />
          )}
        </CardContent>
      </Card>

      <Card className="border-slate-800 bg-slate-950/50">
        <CardHeader>
          <CardTitle className="text-sm">Reasoning Label</CardTitle>
        </CardHeader>
        <CardContent>
          {reasoning.loading ? (
            <Skeleton className="h-32 w-full" />
          ) : reasoning.data ? (
            <ReasoningTimeline label={reasoning.data} />
          ) : (
            <p className="text-sm text-slate-500">
              No reasoning label for this sample.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
