"use client";

import { useMemo, useState } from "react";
import { Search } from "lucide-react";

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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useApi } from "@/hooks/use-api";
import { getReasoningLabel, getReasoningLabelStats } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import type { ReasoningLabelRecord, ReasoningStatsEntry } from "@/types";

// The API returns flat {entries:[{dataset,teacher,prompt_version,count}]};
// per-dataset / per-teacher / per-prompt_version groupings are derived here.
function groupCounts(
  entries: ReasoningStatsEntry[],
  by: (e: ReasoningStatsEntry) => string,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const e of entries) {
    const k = by(e) || "(unknown)";
    out[k] = (out[k] ?? 0) + e.count;
  }
  return out;
}

function StatsTable({
  title,
  entries,
}: {
  title: string;
  entries: Record<string, number>;
}) {
  const rows = Object.entries(entries).sort((a, b) => b[1] - a[1]);
  return (
    <Card className="border-slate-800 bg-slate-950/50">
      <CardHeader>
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Key</TableHead>
              <TableHead className="text-right">Labels</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map(([k, v]) => (
              <TableRow key={k}>
                <TableCell className="font-mono text-xs">{k}</TableCell>
                <TableCell className="text-right font-mono text-xs">
                  {formatNumber(v)}
                </TableCell>
              </TableRow>
            ))}
            {rows.length === 0 && (
              <TableRow>
                <TableCell
                  colSpan={2}
                  className="text-center text-sm text-slate-500"
                >
                  No data
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

export default function ReasoningLabelsPage() {
  const stats = useApi(getReasoningLabelStats);

  const [dataset, setDataset] = useState("l2d");
  const [sampleId, setSampleId] = useState("");
  const [label, setLabel] = useState<ReasoningLabelRecord | null>(null);
  const [searchError, setSearchError] = useState<Error | null>(null);
  const [searching, setSearching] = useState(false);

  const entries = useMemo(() => stats.data?.entries ?? [], [stats.data]);
  const byDataset = useMemo(
    () => groupCounts(entries, (e) => e.dataset),
    [entries],
  );
  const byTeacher = useMemo(
    () => groupCounts(entries, (e) => e.teacher),
    [entries],
  );
  const byPromptVersion = useMemo(
    () => groupCounts(entries, (e) => e.prompt_version),
    [entries],
  );

  async function onSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!sampleId.trim()) return;
    setSearching(true);
    setSearchError(null);
    setLabel(null);
    try {
      setLabel(await getReasoningLabel(dataset, sampleId.trim()));
    } catch (err) {
      setSearchError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setSearching(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Reasoning Labels</h2>
        <p className="text-sm text-slate-400">
          Offline teacher-generated reasoning labels (5 horizons per sample).
        </p>
      </div>

      {stats.error ? (
        <ErrorState error={stats.error} onRetry={stats.reload} />
      ) : stats.loading ? (
        <div className="grid gap-4 lg:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-48 w-full" />
          ))}
        </div>
      ) : (
        <>
          <Card className="border-slate-800 bg-slate-950/50">
            <CardContent className="flex items-baseline gap-3">
              <span className="text-xs uppercase tracking-wider text-slate-400">
                Total labels
              </span>
              <span className="font-mono text-2xl font-semibold">
                {formatNumber(stats.data?.total)}
              </span>
            </CardContent>
          </Card>
          <div className="grid gap-4 lg:grid-cols-3">
            <StatsTable title="Per Dataset" entries={byDataset} />
            <StatsTable title="Per Teacher" entries={byTeacher} />
            <StatsTable title="Per Prompt Version" entries={byPromptVersion} />
          </div>
        </>
      )}

      <Card className="border-slate-800 bg-slate-950/50">
        <CardHeader>
          <CardTitle className="text-sm">Label Inspector</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <form onSubmit={onSearch} className="flex flex-wrap items-center gap-2">
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className="h-9 rounded-md border border-slate-700 bg-slate-900 px-3 text-sm"
              aria-label="Dataset"
            >
              <option value="l2d">l2d</option>
              <option value="nvidia_av">nvidia_av</option>
              <option value="mock">mock</option>
            </select>
            <input
              value={sampleId}
              onChange={(e) => setSampleId(e.target.value)}
              placeholder="sample_id (e.g. ep0_000064)"
              className="h-9 w-72 rounded-md border border-slate-700 bg-slate-900 px-3 font-mono text-sm placeholder:text-slate-600"
            />
            <Button type="submit" size="sm" disabled={searching}>
              <Search className="size-3.5" />
              {searching ? "Searching..." : "Search"}
            </Button>
          </form>
          {searchError && <ErrorState error={searchError} />}
          {searching && <Skeleton className="h-40 w-full" />}
          {label && <ReasoningTimeline label={label} />}
        </CardContent>
      </Card>
    </div>
  );
}
