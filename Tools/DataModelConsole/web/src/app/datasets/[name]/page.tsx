"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  Suspense,
  use,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { ErrorState } from "@/components/error-state";
import { Badge } from "@/components/ui/badge";
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
import {
  getReasoningPromptVersions,
  listDatasetVersions,
  listShards,
} from "@/lib/api";
import { formatBytes, formatNumber } from "@/lib/format";
import type { DatasetVersion, Shard } from "@/types";

const PAGE_SIZE = 50;

// versionLink appends ?version= to an intra-console href so the player and
// sample pages open the SAME version the user pinned here.
function versionLink(href: string, version: string): string {
  return version ? `${href}?version=${encodeURIComponent(version)}` : href;
}

// CompositionSummary shows the WHOLE training composition of the selected
// version: everything a training run at that version would consume.
function CompositionSummary({ v }: { v: DatasetVersion }) {
  const items: { label: string; value: string }[] = [
    {
      label: "Total samples",
      value: v.has_manifest ? formatNumber(v.total_samples) : "—",
    },
    { label: "Shards", value: formatNumber(v.shards) },
    {
      label: "Episodes",
      value: v.has_manifest ? formatNumber(v.episodes) : "—",
    },
    {
      label: "Views / sample",
      value: v.has_manifest ? formatNumber(v.num_views) : "—",
    },
    { label: "Size", value: formatBytes(v.size_bytes) },
    { label: "Map", value: v.has_manifest ? (v.has_map ? "yes" : "no") : "—" },
    {
      label: "World model",
      value: v.has_manifest ? (v.has_world_model ? "yes" : "no") : "—",
    },
  ];
  return (
    <Card className="border-slate-800 bg-slate-950/50">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          Training composition
          <Badge variant="secondary" className="text-[10px]">
            {v.version}
          </Badge>
          {!v.has_manifest && (
            <span className="text-[10px] font-normal text-slate-500">
              no manifest — sample/episode counts unavailable
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4 lg:grid-cols-7">
          {items.map((it) => (
            <div key={it.label} className="min-w-0">
              <dt className="text-[10px] uppercase tracking-wider text-slate-500">
                {it.label}
              </dt>
              <dd className="mt-0.5 font-mono text-sm text-slate-200">
                {it.value}
              </dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
}

function DatasetDetailInner({ dataset }: { dataset: string }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const urlVersion = searchParams.get("version") ?? "";

  const versions = useApi(
    () => listDatasetVersions(dataset),
    [dataset],
  );

  // Resolve the effective version: the URL's version when it exists in the
  // dataset, otherwise the newest (versions are returned newest-first). Empty
  // until the version list loads (the API then auto-resolves newest for the
  // first shard fetch, so nothing renders stale).
  const versionList = useMemo(() => versions.data ?? [], [versions.data]);
  const selected = useMemo<DatasetVersion | null>(() => {
    if (versionList.length === 0) return null;
    return (
      versionList.find((v) => v.version === urlVersion) ?? versionList[0]
    );
  }, [versionList, urlVersion]);
  const selectedVersion = selected?.version ?? "";

  // Normalize the URL to the resolved version once known (so a bare
  // /datasets/l2d becomes /datasets/l2d?version=v2.0 and deep links are
  // canonical). Only replace when it actually differs to avoid a loop.
  useEffect(() => {
    if (selectedVersion && urlVersion !== selectedVersion) {
      router.replace(
        `${pathname}?version=${encodeURIComponent(selectedVersion)}`,
        { scroll: false },
      );
    }
  }, [selectedVersion, urlVersion, pathname, router]);

  const onSelectVersion = useCallback(
    (v: string) => {
      router.replace(`${pathname}?version=${encodeURIComponent(v)}`, {
        scroll: false,
      });
    },
    [pathname, router],
  );

  // Shards for the selected version. Keyed on selectedVersion so a version
  // switch refetches; the first page comes from useApi, extra pages append.
  const shardsApi = useApi(
    () => listShards(dataset, 0, PAGE_SIZE, selectedVersion || undefined),
    [dataset, selectedVersion],
  );
  const [extra, setExtra] = useState<Shard[]>([]);
  const [more, setMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [moreError, setMoreError] = useState<Error | null>(null);
  const pageGeneration = useRef(0);

  useEffect(() => {
    pageGeneration.current++;
    setExtra([]);
    setMore(shardsApi.data?.page?.more ?? false);
    setLoadingMore(false);
    setMoreError(null);
  }, [dataset, selectedVersion, shardsApi.data]);

  const shards = [...(shardsApi.data?.shards ?? []), ...extra];

  const loadMore = useCallback(async () => {
    const generation = ++pageGeneration.current;
    setLoadingMore(true);
    setMoreError(null);
    try {
      const res = await listShards(
        dataset,
        shards.length,
        PAGE_SIZE,
        selectedVersion || undefined,
      );
      if (generation !== pageGeneration.current) return;
      setExtra((prev) => [...prev, ...res.shards]);
      setMore(res.page.more);
    } catch (err) {
      if (generation !== pageGeneration.current) return;
      setMoreError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      if (generation === pageGeneration.current) setLoadingMore(false);
    }
  }, [dataset, shards.length, selectedVersion]);

  const promptVersions = useApi(
    async () => ({
      version: selectedVersion,
      prompts: selectedVersion
        ? await getReasoningPromptVersions(dataset, selectedVersion)
        : [],
    }),
    [dataset, selectedVersion],
  );
  const promptVersionList =
    promptVersions.data?.version === selectedVersion
      ? promptVersions.data.prompts
      : [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs text-slate-500">
            <Link href="/datasets" className="hover:text-slate-300">
              Datasets
            </Link>{" "}
            / <span className="font-mono">{dataset}</span>
          </p>
          <h2 className="mt-1 font-mono text-lg font-semibold">{dataset}</h2>
          <p className="text-sm text-slate-400">
            Select a dataset version to see its full training composition.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label
            htmlFor="version-select"
            className="text-[10px] uppercase tracking-wider text-slate-500"
          >
            Version
          </label>
          <select
            id="version-select"
            value={selectedVersion}
            onChange={(e) => onSelectVersion(e.target.value)}
            disabled={versions.loading || versionList.length === 0}
            className="h-9 rounded-md border border-slate-700 bg-slate-900 px-3 font-mono text-sm disabled:opacity-50"
            aria-label="Dataset version"
          >
            {versionList.length === 0 && <option value="">—</option>}
            {versionList.map((v) => (
              <option key={v.version} value={v.version}>
                {v.version}
                {v.has_manifest ? ` · ${formatNumber(v.total_samples)} samples` : ""}
              </option>
            ))}
          </select>
        </div>
      </div>

      {versions.error ? (
        <ErrorState error={versions.error} onRetry={versions.reload} />
      ) : versions.loading ? (
        <Skeleton className="h-28 w-full" />
      ) : selected ? (
        <CompositionSummary v={selected} />
      ) : (
        <p className="text-sm text-slate-500">No versions found.</p>
      )}

      <Card className="border-slate-800 bg-slate-950/50">
        <CardHeader>
          <CardTitle className="text-sm">
            Shards
            {shardsApi.data?.page
              ? ` (${shardsApi.data.page.total} total)`
              : ""}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {shardsApi.error ? (
            <ErrorState error={shardsApi.error} onRetry={shardsApi.reload} />
          ) : shardsApi.loading ? (
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
                  <TableHead className="text-right">Player</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {shards.map((shard) => (
                  <TableRow key={shard.name}>
                    <TableCell>
                      <Link
                        href={versionLink(
                          `/datasets/${encodeURIComponent(dataset)}/shards/${encodeURIComponent(shard.name)}`,
                          selectedVersion,
                        )}
                        className="font-mono text-xs text-blue-500 hover:underline"
                      >
                        {shard.name}
                      </Link>
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {formatBytes(shard.size_bytes)}
                    </TableCell>
                    <TableCell className="text-right">
                      <Link
                        href={versionLink(
                          `/scenes/${encodeURIComponent(dataset)}/${encodeURIComponent(shard.name)}/0`,
                          selectedVersion,
                        )}
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
                      colSpan={3}
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
          {more && !shardsApi.loading && (
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

      <Card className="border-slate-800 bg-slate-950/50">
        <CardHeader>
          <CardTitle className="text-sm">Reasoning label versions</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-3 text-xs text-slate-500">
            Reasoning prompt_versions are an orthogonal axis to the dataset
            version: offline teacher labels attached per sample. A shard version
            can be trained with any of these label sets.
          </p>
          {promptVersions.error ? (
            <ErrorState
              error={promptVersions.error}
              onRetry={promptVersions.reload}
            />
          ) : promptVersions.loading ? (
            <Skeleton className="h-24 w-full" />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Teacher</TableHead>
                  <TableHead>Prompt version</TableHead>
                  <TableHead className="text-right">Labels</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {promptVersionList.map((pv) => (
                  <TableRow key={`${pv.teacher}/${pv.prompt_version}`}>
                    <TableCell className="font-mono text-xs">
                      {pv.teacher}
                    </TableCell>
                    <TableCell
                      className="max-w-[320px] truncate font-mono text-xs"
                      title={pv.prompt_version}
                    >
                      {pv.prompt_version}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {formatNumber(pv.count)}
                    </TableCell>
                  </TableRow>
                ))}
                {promptVersionList.length === 0 && (
                  <TableRow>
                    <TableCell
                      colSpan={3}
                      className="text-center text-sm text-slate-500"
                    >
                      No reasoning labels for this dataset
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

export default function DatasetDetailPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const { name } = use(params);
  const dataset = decodeURIComponent(name);

  return (
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <DatasetDetailInner dataset={dataset} />
    </Suspense>
  );
}
