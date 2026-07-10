"use client";

import { AlertTriangle } from "lucide-react";

import { Button } from "@/components/ui/button";

export function ErrorState({
  error,
  onRetry,
}: {
  error: Error;
  onRetry?: () => void;
}) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-lg border border-red-500/30 bg-red-500/5 p-8 text-center">
      <AlertTriangle className="size-6 text-red-500" />
      <p className="text-sm text-slate-300">Failed to load data</p>
      <p className="max-w-lg font-mono text-xs break-all text-slate-500">
        {error.message}
      </p>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}
