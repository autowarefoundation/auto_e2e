"use client";

// A Scene is a 10Hz sequence of frames in a WebDataset shard. This page
// offers dataset entry points and a direct scene locator that opens the
// ADAS player at /scenes/{dataset}/{shard}/{frame}.

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Clapperboard, Play } from "lucide-react";

import { ErrorState } from "@/components/error-state";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/hooks/use-api";
import { listDatasets, listShardsForEpisode } from "@/lib/api";

export default function ScenesPage() {
  const { data, error, loading, reload } = useApi(listDatasets);
  const router = useRouter();

  const [dataset, setDataset] = useState("l2d");
  const [shard, setShard] = useState("");
  const [frame, setFrame] = useState("0");
  const [shardOptions, setShardOptions] = useState<string[]>([]);

  // Fetch the chosen dataset's shards so the shard field can suggest real
  // values instead of relying on the user to type an exact tar name.
  useEffect(() => {
    let cancelled = false;
    listShardsForEpisode(dataset)
      .then((shards) => {
        if (!cancelled) setShardOptions(shards.map((s) => s.name));
      })
      .catch(() => {
        if (!cancelled) setShardOptions([]);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset]);

  function onLocate(e: React.FormEvent) {
    e.preventDefault();
    if (!shard.trim()) return;
    const f = Math.max(0, parseInt(frame, 10) || 0);
    router.push(
      `/scenes/${encodeURIComponent(dataset)}/${encodeURIComponent(shard.trim())}/${f}`,
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Scenes</h2>
        <p className="text-sm text-slate-400">
          A scene is a 10Hz camera sequence in a WebDataset shard. Browse by
          dataset or jump straight into the player at a known shard and frame.
        </p>
      </div>

      {error ? (
        <ErrorState error={error} onRetry={reload} />
      ) : loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 2 }).map((_, i) => (
            <Skeleton key={i} className="h-28 w-full" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {(data ?? []).map((ds) => (
            <Link key={ds.name} href={`/datasets/${encodeURIComponent(ds.name)}`}>
              <Card className="border-slate-800 bg-slate-950/50 transition-colors hover:border-slate-600">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 font-mono text-sm">
                    <Clapperboard className="size-4 text-blue-500" />
                    {ds.name}
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-xs text-slate-400">
                  <span className="font-mono">{ds.version}</span> —{" "}
                  <span className="font-mono text-slate-500">{ds.prefix}</span>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}

      <Card className="border-slate-800 bg-slate-950/50">
        <CardHeader>
          <CardTitle className="text-sm">Scene Locator</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={onLocate} className="flex flex-wrap items-center gap-2">
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className="h-9 rounded-md border border-slate-700 bg-slate-900 px-3 text-sm"
              aria-label="Dataset"
            >
              {(data && data.length > 0
                ? data
                : [{ name: "l2d" }, { name: "nvidia_av" }]
              ).map((ds) => (
                <option key={ds.name} value={ds.name}>
                  {ds.name}
                </option>
              ))}
            </select>
            <input
              value={shard}
              onChange={(e) => setShard(e.target.value)}
              placeholder="shard (e.g. train-000000.tar)"
              list="scene-shard-options"
              className="h-9 w-64 rounded-md border border-slate-700 bg-slate-900 px-3 font-mono text-sm placeholder:text-slate-600"
            />
            <datalist id="scene-shard-options">
              {shardOptions.map((name) => (
                <option key={name} value={name} />
              ))}
            </datalist>
            <input
              value={frame}
              onChange={(e) => setFrame(e.target.value)}
              inputMode="numeric"
              placeholder="frame (e.g. 0)"
              className="h-9 w-32 rounded-md border border-slate-700 bg-slate-900 px-3 font-mono text-sm placeholder:text-slate-600"
              aria-label="Frame index"
            />
            <Button type="submit" size="sm" disabled={!shard.trim()}>
              <Play className="size-3.5" />
              Open
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
