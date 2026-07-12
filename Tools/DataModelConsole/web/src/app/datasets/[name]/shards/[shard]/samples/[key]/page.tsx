"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, use, useEffect, useMemo } from "react";
import { ChevronLeft, ChevronRight, Loader2, Play } from "lucide-react";

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
import {
  ApiError,
  getReasoningLabel,
  getSample,
  getShardIndex,
} from "@/lib/api";

// Sample keys come in two forms: "ep0_000064" (episode id + "_" + 6-digit
// frame) and "s00000139" (flat index, no underscore). Step the trailing number
// by delta, preserving its zero-pad width, for either form. Used only as a
// fallback while the full key list loads.
function siblingKey(key: string, delta: number): string | null {
  const m = key.match(/^(.*?)(\d+)$/);
  if (!m) return null;
  const width = m[2].length;
  const frame = parseInt(m[2], 10) + delta;
  if (frame < 0) return null;
  return `${m[1]}${String(frame).padStart(width, "0")}`;
}

function SampleDetailInner({
  dataset,
  shardName,
  sampleKey,
}: {
  dataset: string;
  shardName: string;
  sampleKey: string;
}) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const version = searchParams.get("version") ?? "";
  const promptVersion = searchParams.get("prompt_version") ?? "";
  const versionQuery = (() => {
    const q = new URLSearchParams();
    if (version) q.set("version", version);
    if (promptVersion) q.set("prompt_version", promptVersion);
    const s = q.toString();
    return s ? `?${s}` : "";
  })();

  const { data, error, loading, reload } = useApi(
    () => getSample(dataset, shardName, sampleKey, version || undefined),
    [dataset, shardName, sampleKey, version],
  );

  // Reasoning label is a separate endpoint; 404 simply means "no label".
  // Pin prompt_version (from the URL) so the shown label matches the run the
  // user was browsing, not an arbitrary partition.
  const reasoning = useApi(
    () =>
      getReasoningLabel(
        dataset,
        sampleKey,
        promptVersion || undefined,
      ).catch((err: unknown) => {
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }),
    [dataset, sampleKey, promptVersion],
  );

  // Bound forward/backward nav to the shard's FULL, ordered sample list so
  // "Next" is disabled on the true last frame (no 404 dead-end) and every
  // interior frame steps correctly. Use the shard index (all keys in order,
  // and cached) rather than listSamples (server-capped at 50, which stranded
  // every sample past index 49). Falls back to sibling arithmetic while loading.
  const samples = useApi(
    () => getShardIndex(dataset, shardName, version || undefined),
    [dataset, shardName, version],
  );
  const keys = useMemo(
    () => (samples.data?.samples ?? []).map((s) => s.key),
    [samples.data],
  );
  const idx = useMemo(() => keys.indexOf(sampleKey), [keys, sampleKey]);
  const prevKey = useMemo(
    () => (idx > 0 ? keys[idx - 1] : siblingKey(sampleKey, -1)),
    [idx, keys, sampleKey],
  );
  const nextKey = useMemo(() => {
    if (idx >= 0 && idx < keys.length - 1) return keys[idx + 1];
    if (idx === keys.length - 1) return null;
    return siblingKey(sampleKey, +1);
  }, [idx, keys, sampleKey]);

  const sampleUrl = (k: string) =>
    `/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shardName)}/samples/${encodeURIComponent(k)}${versionQuery}`;

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
              href={`/datasets/${encodeURIComponent(dataset)}${versionQuery}`}
              className="font-mono hover:text-slate-300"
            >
              {dataset}
            </Link>{" "}
            /{" "}
            <Link
              href={`/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shardName)}${versionQuery}`}
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
            href={`/scenes/${encodeURIComponent(dataset)}/${encodeURIComponent(shardName)}/${frameIdx ?? 0}${versionQuery}`}
            className="inline-flex h-7 items-center gap-1 rounded-md border border-slate-700 bg-slate-900 px-2.5 text-[0.8rem] text-slate-200 transition-colors hover:border-slate-500"
          >
            <Play className="size-3.5" />
            Player
          </Link>
        </div>
      </div>

      {loading && (
        <p className="flex items-center gap-2 text-xs text-slate-500">
          <Loader2 className="size-3.5 animate-spin" />
          Scanning shard index — first open takes ~10s, then it is cached.
        </p>
      )}

      {/* Camera mosaic: 2 rows x 4 cols, tile count follows the sample's real
          camera list (L2D packs 6, NVIDIA 7); last cell = metadata */}
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        {(data?.cameras ?? []).map((_, cam) => (
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
              version={version || undefined}
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

export default function SampleDetailPage({
  params,
}: {
  params: Promise<{ name: string; shard: string; key: string }>;
}) {
  const { name, shard, key } = use(params);
  const dataset = decodeURIComponent(name);
  const shardName = decodeURIComponent(shard);
  const sampleKey = decodeURIComponent(key);

  return (
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <SampleDetailInner
        dataset={dataset}
        shardName={shardName}
        sampleKey={sampleKey}
      />
    </Suspense>
  );
}
