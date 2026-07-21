"use client";

// A Scene is a 10Hz sequence of frames in a WebDataset shard. This page
// offers dataset entry points and a direct scene locator that opens the
// ADAS player at /scenes/{dataset}/{shard}/{frame}.

import Link from "next/link";
import {
  usePathname,
  useRouter,
  useSearchParams,
} from "next/navigation";
import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
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
import {
  listDatasets,
  listDatasetVersions,
  listShardsForEpisode,
} from "@/lib/api";

function ScenesPageInner() {
  const { data, error, loading, reload } = useApi(listDatasets);
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const searchParamsString = searchParams.toString();
  const urlDataset = searchParams.get("dataset") ?? "";
  const urlVersion = searchParams.get("version") ?? "";

  const dataset = useMemo(() => {
    if (!data || data.length === 0) return "";
    return (
      data.find((item) => item.name === urlDataset)?.name ??
      data.find((item) => item.name === "kitscenes")?.name ??
      data[0].name
    );
  }, [data, urlDataset]);
  const versions = useApi(
    async () => ({
      dataset,
      items: await listDatasetVersions(dataset),
    }),
    [dataset],
    Boolean(dataset),
  );
  const version = useMemo(() => {
    const items =
      versions.data?.dataset === dataset ? versions.data.items : [];
    if (items.length === 0) return "";
    const advertised =
      data?.find((item) => item.name === dataset)?.version ?? "";
    return (
      items.find((item) => item.version === urlVersion)?.version ??
      items.find((item) => item.version === advertised)?.version ??
      items[0].version
    );
  }, [data, dataset, urlVersion, versions.data]);

  const [shard, setShard] = useState(
    () => searchParams.get("shard") ?? "",
  );
  const [frame, setFrame] = useState(
    () => searchParams.get("frame") ?? "0",
  );
  const [shardOptions, setShardOptions] = useState<string[]>([]);
  const [shardsLoading, setShardsLoading] = useState(false);
  const [shardsReady, setShardsReady] = useState(false);
  const [shardsError, setShardsError] = useState<Error | null>(null);
  const [shardsReload, setShardsReload] = useState(0);
  const shardRequest = useRef(0);

  useEffect(() => {
    if (
      loading ||
      error ||
      !dataset ||
      versions.loading ||
      versions.error ||
      !version
    ) {
      return;
    }
    const query = new URLSearchParams(searchParamsString);
    let changed = false;
    if (query.get("dataset") !== dataset) {
      query.set("dataset", dataset);
      changed = true;
    }
    if (query.get("version") !== version) {
      query.set("version", version);
      changed = true;
    }
    if (changed) {
      router.replace(`${pathname}?${query.toString()}`, { scroll: false });
    }
  }, [
    dataset,
    error,
    loading,
    pathname,
    router,
    searchParamsString,
    version,
    versions.error,
    versions.loading,
  ]);

  useEffect(() => {
    setShard(searchParams.get("shard") ?? "");
    setFrame(searchParams.get("frame") ?? "0");
  }, [searchParams]);

  const navigateSelection = useCallback(
    (nextDataset: string, nextVersion: string | null) => {
      shardRequest.current += 1;
      setShard("");
      setFrame("0");
      setShardOptions([]);
      setShardsLoading(true);
      setShardsReady(false);
      setShardsError(null);

      const query = new URLSearchParams(searchParamsString);
      query.set("dataset", nextDataset);
      if (nextVersion) {
        query.set("version", nextVersion);
      } else {
        query.delete("version");
      }
      query.delete("shard");
      query.set("frame", "0");
      router.push(`${pathname}?${query.toString()}`, { scroll: false });
    },
    [pathname, router, searchParamsString],
  );

  const updateLocator = useCallback(
    (name: "shard" | "frame", value: string) => {
      if (name === "shard") {
        setShard(value);
      } else {
        setFrame(value);
      }
      const query = new URLSearchParams(window.location.search);
      if (value) {
        query.set(name, value);
      } else {
        query.delete(name);
      }
      window.history.replaceState(
        null,
        "",
        `${pathname}?${query.toString()}`,
      );
    },
    [pathname],
  );

  // Fetch the chosen dataset's shards so the shard field can suggest real
  // values instead of relying on the user to type an exact tar name.
  useEffect(() => {
    const request = ++shardRequest.current;
    let cancelled = false;
    setShardOptions([]);
    setShardsReady(false);
    setShardsError(null);

    if (!dataset || !version || versions.loading || versions.error) {
      setShardsLoading(false);
      return;
    }

    setShardsLoading(true);
    listShardsForEpisode(dataset, version)
      .then((shards) => {
        if (!cancelled && request === shardRequest.current) {
          setShardOptions(shards.map((item) => item.name));
          setShardsLoading(false);
          setShardsReady(true);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled && request === shardRequest.current) {
          setShardOptions([]);
          setShardsLoading(false);
          setShardsReady(false);
          setShardsError(
            err instanceof Error ? err : new Error(String(err)),
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [
    dataset,
    shardsReload,
    version,
    versions.error,
    versions.loading,
  ]);

  const trimmedShard = shard.trim();
  const canOpen =
    Boolean(version) &&
    shardsReady &&
    !shardsLoading &&
    !shardsError &&
    shardOptions.includes(trimmedShard);

  function onLocate(e: React.FormEvent) {
    e.preventDefault();
    if (!canOpen) return;
    const f = Math.max(0, parseInt(frame, 10) || 0);
    router.push(
      `/scenes/${encodeURIComponent(dataset)}/${encodeURIComponent(trimmedShard)}/${f}?version=${encodeURIComponent(version)}`,
    );
  }

  function retryShards() {
    shardRequest.current += 1;
    setShardOptions([]);
    setShardsLoading(true);
    setShardsReady(false);
    setShardsError(null);
    setShardsReload((value) => value + 1);
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
            <Link
              key={ds.name}
              href={`/datasets/${encodeURIComponent(ds.name)}?version=${encodeURIComponent(ds.version)}`}
            >
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
          <form
            onSubmit={onLocate}
            className="flex flex-wrap items-center gap-2"
            aria-busy={shardsLoading}
          >
            <select
              value={dataset}
              onChange={(event) =>
                navigateSelection(event.target.value, null)
              }
              disabled={loading || error !== null || !data?.length}
              className="h-9 rounded-md border border-slate-700 bg-slate-900 px-3 text-sm"
              aria-label="Dataset"
            >
              {!data?.length && <option value="">—</option>}
              {(data ?? []).map((ds) => (
                <option key={ds.name} value={ds.name}>
                  {ds.name}
                </option>
              ))}
            </select>
            <select
              value={version}
              onChange={(event) =>
                navigateSelection(dataset, event.target.value)
              }
              disabled={
                versions.loading ||
                versions.error !== null ||
                versions.data?.dataset !== dataset ||
                !versions.data.items.length
              }
              className="h-9 rounded-md border border-slate-700 bg-slate-900 px-3 font-mono text-sm"
              aria-label="Dataset version"
            >
              {(versions.data?.dataset !== dataset ||
                !versions.data.items.length) && (
                <option value="">—</option>
              )}
              {(versions.data?.dataset === dataset
                ? versions.data.items
                : []
              ).map((item) => (
                <option key={item.version} value={item.version}>
                  {item.version}
                </option>
              ))}
            </select>
            <input
              value={shard}
              onChange={(event) =>
                updateLocator("shard", event.target.value)
              }
              placeholder="shard (e.g. train-000000.tar)"
              list="scene-shard-options"
              className="h-9 w-64 rounded-md border border-slate-700 bg-slate-900 px-3 font-mono text-sm placeholder:text-slate-600"
              aria-label="Shard"
            />
            <datalist id="scene-shard-options">
              {shardOptions.map((name) => (
                <option key={name} value={name} />
              ))}
            </datalist>
            <input
              value={frame}
              onChange={(event) =>
                updateLocator("frame", event.target.value)
              }
              inputMode="numeric"
              placeholder="frame (e.g. 0)"
              className="h-9 w-32 rounded-md border border-slate-700 bg-slate-900 px-3 font-mono text-sm placeholder:text-slate-600"
              aria-label="Frame index"
            />
            <Button type="submit" size="sm" disabled={!canOpen}>
              <Play className="size-3.5" />
              Open
            </Button>
          </form>
          {shardsError && (
            <div className="mt-4">
              <ErrorState
                error={shardsError}
                onRetry={retryShards}
                service="S3"
              />
            </div>
          )}
          {versions.error && (
            <div className="mt-4">
              <ErrorState
                error={versions.error}
                onRetry={versions.reload}
                service="S3"
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default function ScenesPage() {
  return (
    <Suspense fallback={<Skeleton className="h-64 w-full" />}>
      <ScenesPageInner />
    </Suspense>
  );
}
