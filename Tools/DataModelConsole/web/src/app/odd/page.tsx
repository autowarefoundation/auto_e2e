"use client";

// ODD scene search. ODD labels are scene-level catalog metadata (road,
// environment, traffic, image condition, ego events) from the ODD LabelSet of
// a dataset version. The search state (dataset, version, where terms, offset)
// lives in the URL so a search can be shared.

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useMemo } from "react";
import { Play, X } from "lucide-react";

import { ErrorState } from "@/components/error-state";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
import { ApiError, listDatasets, searchODDScenes } from "@/lib/api";
import { friendlyDataset } from "@/lib/format";
import type { ODDCondition } from "@/types";

const PAGE_SIZE = 50;

function parseWhere(terms: string[]): ODDCondition[] {
  return terms.flatMap((term) => {
    const index = term.indexOf("=");
    return index > 0
      ? [{ key: term.slice(0, index), value: term.slice(index + 1) }]
      : [];
  });
}

function seconds(ns: number): string {
  return `${(ns / 1e9).toFixed(1)} s`;
}

function ODDPageInner() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const datasets = useApi(listDatasets);

  const dataset = useMemo(() => {
    const items = datasets.data ?? [];
    const requested = searchParams.get("dataset");
    return (
      items.find((item) => item.name === requested)?.name ??
      items.find((item) => item.name === "kitscenes")?.name ??
      items[0]?.name ??
      ""
    );
  }, [datasets.data, searchParams]);
  const version =
    searchParams.get("version") ??
    datasets.data?.find((item) => item.name === dataset)?.version ??
    "";
  // A joined string keeps the parsed terms stable across renders.
  const whereKey = searchParams.getAll("where").join("\n");
  const where = useMemo(
    () => parseWhere(whereKey ? whereKey.split("\n") : []),
    [whereKey],
  );
  const offset = Math.max(0, parseInt(searchParams.get("offset") ?? "", 10) || 0);

  const search = useApi(
    () => searchODDScenes(dataset, version, where, offset, PAGE_SIZE),
    [dataset, version, where, offset],
    Boolean(dataset && version),
  );
  const noLabels =
    search.error instanceof ApiError && search.error.status === 404;

  function navigate(next: {
    dataset?: string;
    where?: ODDCondition[];
    offset?: number;
  }) {
    const query = new URLSearchParams();
    const nextDataset = next.dataset ?? dataset;
    query.set("dataset", nextDataset);
    if (next.dataset === undefined && version) query.set("version", version);
    for (const condition of next.where ?? where) {
      query.append("where", `${condition.key}=${condition.value}`);
    }
    const nextOffset = next.offset ?? 0;
    if (nextOffset > 0) query.set("offset", String(nextOffset));
    router.push(`${pathname}?${query.toString()}`, { scroll: false });
  }

  const response = search.data;
  const activeTerms = new Set(where.map((c) => `${c.key}=${c.value}`));

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">ODD Search</h2>
        <p className="text-sm text-slate-400">
          Find scenes by their operational design domain labels. A scene
          matches when it has a valid label for every selected value.
        </p>
      </div>

      {datasets.error ? (
        <ErrorState error={datasets.error} onRetry={datasets.reload} />
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={dataset}
            onChange={(event) =>
              navigate({ dataset: event.target.value, where: [] })
            }
            disabled={datasets.loading || !datasets.data?.length}
            className="h-9 rounded-md border border-slate-700 bg-slate-900 px-3 text-sm"
            aria-label="Dataset"
          >
            {!datasets.data?.length && <option value="">—</option>}
            {(datasets.data ?? []).map((item) => (
              <option key={item.name} value={item.name}>
                {friendlyDataset(item.name)}
              </option>
            ))}
          </select>
          <span className="font-mono text-sm text-slate-400">{version}</span>
        </div>
      )}

      {where.length > 0 && (
        <div className="flex flex-wrap items-center gap-2" aria-label="Active filters">
          {where.map((condition) => (
            <Button
              key={`${condition.key}=${condition.value}`}
              variant="secondary"
              size="sm"
              onClick={() =>
                navigate({
                  where: where.filter(
                    (c) => c.key !== condition.key || c.value !== condition.value,
                  ),
                })
              }
              aria-label={`Remove ${condition.key}=${condition.value}`}
            >
              <span className="font-mono text-xs">
                {condition.key}={condition.value}
              </span>
              <X className="size-3" />
            </Button>
          ))}
          <Button variant="ghost" size="sm" onClick={() => navigate({ where: [] })}>
            Clear
          </Button>
        </div>
      )}

      {noLabels ? (
        <Card className="border-slate-800 bg-slate-950/50">
          <CardContent className="py-6 text-sm text-slate-400">
            No ODD labels for {friendlyDataset(dataset)} {version}.
          </CardContent>
        </Card>
      ) : search.error ? (
        <ErrorState error={search.error} onRetry={search.reload} service="S3" />
      ) : search.loading || !response ? (
        <Skeleton className="h-64 w-full" />
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
            <span>
              {response.page.total} of {response.labeled_scenes} scenes match.
            </span>
            <span>
              Labels from{" "}
              <span className="font-mono">
                {response.dataset} {response.label_version}
              </span>{" "}
              (<span className="font-mono">{response.labelset_id}</span>)
            </span>
            <Badge variant="secondary">
              {response.quality.certification_status}
            </Badge>
            <Badge variant="secondary">{response.quality.audit_status}</Badge>
          </div>

          <Card className="border-slate-800 bg-slate-950/50">
            <CardHeader>
              <CardTitle className="text-sm">Scenes</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Shard</TableHead>
                    <TableHead className="text-right">Duration</TableHead>
                    <TableHead className="text-right">Distance</TableHead>
                    <TableHead>First match</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {response.scenes.map((scene) => (
                    <TableRow key={scene.scene_uid}>
                      <TableCell className="font-mono text-xs">
                        <Link
                          href={`/scenes/${encodeURIComponent(response.dataset)}/${encodeURIComponent(scene.shard)}/0?version=${encodeURIComponent(response.version)}`}
                          className="inline-flex items-center gap-1 text-blue-400 hover:underline"
                        >
                          <Play className="size-3" />
                          {scene.shard}
                        </Link>
                      </TableCell>
                      <TableCell className="text-right text-xs">
                        {seconds(scene.duration_ns)}
                      </TableCell>
                      <TableCell className="text-right text-xs">
                        {scene.distance_m.toFixed(0)} m
                      </TableCell>
                      <TableCell className="text-xs text-slate-400">
                        {scene.matches
                          .map(
                            (m) =>
                              `${m.key}=${m.value} @ ${seconds(m.first_offset_ns)}`,
                          )
                          .join(", ") || "—"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <div className="mt-4 flex items-center gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={response.page.offset === 0}
                  onClick={() =>
                    navigate({
                      offset: Math.max(0, response.page.offset - PAGE_SIZE),
                    })
                  }
                >
                  Previous
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={!response.page.more}
                  onClick={() =>
                    navigate({ offset: response.page.offset + PAGE_SIZE })
                  }
                >
                  Next
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="border-slate-800 bg-slate-950/50">
            <CardHeader>
              <CardTitle className="text-sm">
                Labels in the matched scenes
              </CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {response.facets.map((facet) => (
                <div key={facet.key} className="min-w-0">
                  <div className="mb-1 truncate font-mono text-xs text-slate-300">
                    {facet.key}
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {facet.values.map(({ value, scenes }) => {
                      const term = `${facet.key}=${value}`;
                      return (
                        <Button
                          key={term}
                          variant="outline"
                          size="sm"
                          disabled={activeTerms.has(term)}
                          onClick={() =>
                            navigate({
                              where: [...where, { key: facet.key, value }],
                            })
                          }
                          aria-label={`Filter ${term}`}
                        >
                          <span className="font-mono text-xs">{value}</span>
                          <span className="text-xs text-slate-500">{scenes}</span>
                        </Button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

export default function ODDPage() {
  return (
    <Suspense fallback={<Skeleton className="h-64 w-full" />}>
      <ODDPageInner />
    </Suspense>
  );
}
