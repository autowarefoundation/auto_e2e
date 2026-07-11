"use client";

import { AlertTriangle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api";

// humanize maps known upstream/transport failures to plain language so the
// dashboard does not surface raw API payloads (e.g. Flyte UPSTREAM_ERROR) as
// the primary message. Pass the upstream service name (MLflow / Flyte / S3)
// so a 502 names the right dependency instead of always blaming Flyte.
export function humanizeError(error: Error, service?: string): string {
  if (error instanceof ApiError) {
    if (error.status === 502 || /UPSTREAM_ERROR|unreachable/i.test(error.message)) {
      return `An upstream service${service ? ` (${service})` : ""} is currently unreachable. Data will appear once it responds.`;
    }
    if (error.status === 0) {
      return "Network error — the API could not be reached.";
    }
  }
  return "Failed to load data.";
}

export function ErrorState({
  error,
  onRetry,
  service,
}: {
  error: Error;
  onRetry?: () => void;
  service?: string;
}) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-lg border border-red-500/30 bg-red-500/5 p-8 text-center">
      <AlertTriangle className="size-6 text-red-500" />
      <p className="text-sm text-slate-300">{humanizeError(error, service)}</p>
      <details className="max-w-lg text-left">
        <summary className="cursor-pointer font-mono text-xs text-slate-500 hover:text-slate-400">
          Details
        </summary>
        <p className="mt-1 font-mono text-xs break-all text-slate-500">
          {error.message}
        </p>
      </details>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}
