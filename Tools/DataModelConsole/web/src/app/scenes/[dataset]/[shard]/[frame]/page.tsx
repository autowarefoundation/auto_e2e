"use client";

// ADAS player page: /scenes/{dataset}/{shard}/{frame}
//
// Hosts EpisodePlayer. View state (cam, mode, speed, frame) is mirrored into
// URL query params (debounced router.replace) so any moment of any shard is
// a shareable deep link. "Copy link" copies the canonical URL.

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Suspense,
  use,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { Check, Link2 } from "lucide-react";

import {
  EpisodePlayer,
  type PlayerViewState,
} from "@/components/player/episode-player";
import { ErrorState } from "@/components/error-state";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/hooks/use-api";
import { getShardIndex } from "@/lib/api";

function PlayerPageInner({
  dataset,
  shard,
  frame,
}: {
  dataset: string;
  shard: string;
  frame: number;
}) {
  const router = useRouter();
  const searchParams = useSearchParams();

  const { data, error, loading, reload } = useApi(
    () => getShardIndex(dataset, shard),
    [dataset, shard],
  );

  // Initial view state: path frame + query params (cam, mode, speed).
  const initialState = useRef<Partial<PlayerViewState>>({
    frame,
    cam: parseInt(searchParams.get("cam") ?? "0", 10) || 0,
    mode: searchParams.get("mode") === "focus" ? "focus" : "grid",
    speed: parseFloat(searchParams.get("speed") ?? "1") || 1,
  });

  // Debounced URL sync: keep the path's frame segment and query in step with
  // the player without spamming history (replace, not push).
  const viewStateRef = useRef<PlayerViewState | null>(null);
  const syncTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onViewStateChange = useCallback(
    (state: PlayerViewState) => {
      viewStateRef.current = state;
      if (syncTimerRef.current) clearTimeout(syncTimerRef.current);
      syncTimerRef.current = setTimeout(() => {
        const s = viewStateRef.current;
        if (!s) return;
        const q = new URLSearchParams();
        if (s.cam !== 0) q.set("cam", String(s.cam));
        if (s.mode !== "grid") q.set("mode", s.mode);
        if (Math.abs(s.speed - 1) > 1e-9) q.set("speed", String(s.speed));
        const qs = q.toString();
        router.replace(
          `/scenes/${encodeURIComponent(dataset)}/${encodeURIComponent(shard)}/${s.frame}${qs ? `?${qs}` : ""}`,
          { scroll: false },
        );
      }, 500);
    },
    [router, dataset, shard],
  );
  useEffect(
    () => () => {
      if (syncTimerRef.current) clearTimeout(syncTimerRef.current);
    },
    [],
  );

  const [copied, setCopied] = useState(false);
  const copyLink = useCallback(() => {
    const s = viewStateRef.current;
    const q = new URLSearchParams();
    if (s) {
      if (s.cam !== 0) q.set("cam", String(s.cam));
      if (s.mode !== "grid") q.set("mode", s.mode);
      if (Math.abs(s.speed - 1) > 1e-9) q.set("speed", String(s.speed));
    }
    const qs = q.toString();
    const url = `${window.location.origin}/scenes/${encodeURIComponent(dataset)}/${encodeURIComponent(shard)}/${s?.frame ?? frame}${qs ? `?${qs}` : ""}`;
    void navigator.clipboard.writeText(url).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [dataset, shard, frame]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs text-slate-500">
            <Link href="/scenes" className="hover:text-slate-300">
              Scenes
            </Link>{" "}
            /{" "}
            <Link
              href={`/datasets/${encodeURIComponent(dataset)}`}
              className="font-mono hover:text-slate-300"
            >
              {dataset}
            </Link>{" "}
            / <span className="font-mono">{shard}</span>
          </p>
          <h2 className="mt-1 font-mono text-lg font-semibold">{shard}</h2>
        </div>
        <Button variant="outline" size="sm" onClick={copyLink}>
          {copied ? (
            <Check className="size-3.5 text-emerald-500" />
          ) : (
            <Link2 className="size-3.5" />
          )}
          {copied ? "Copied" : "Copy link"}
        </Button>
      </div>

      {error ? (
        <ErrorState error={error} onRetry={reload} />
      ) : loading || !data ? (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="aspect-video w-full" />
            ))}
          </div>
          <Skeleton className="h-24 w-full" />
        </div>
      ) : (
        <EpisodePlayer
          dataset={dataset}
          index={data}
          initialState={initialState.current}
          onViewStateChange={onViewStateChange}
        />
      )}
    </div>
  );
}

export default function ScenePlayerPage({
  params,
}: {
  params: Promise<{ dataset: string; shard: string; frame: string }>;
}) {
  const p = use(params);
  const dataset = decodeURIComponent(p.dataset);
  const shard = decodeURIComponent(p.shard);
  const frame = Math.max(0, parseInt(decodeURIComponent(p.frame), 10) || 0);

  return (
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <PlayerPageInner dataset={dataset} shard={shard} frame={frame} />
    </Suspense>
  );
}
