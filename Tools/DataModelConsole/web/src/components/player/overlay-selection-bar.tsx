"use client";

import { AlertTriangle, Loader2, Route } from "lucide-react";

import type { TrajectoryDisplayMode } from "@/lib/ego";
import type { OverlayModel } from "@/types";

export type OverlayLoadStatus =
  | "loading-models"
  | "no-models"
  | "idle"
  | "loading-overlay"
  | "ready"
  | "error";

function modelLabel(model: OverlayModel): string {
  const name = model.model_name || model.registered_model_name;
  const metrics =
    Number.isFinite(model.eval_ade) && Number.isFinite(model.eval_fde)
      ? ` | ADE ${model.eval_ade.toFixed(2)} | FDE ${model.eval_fde.toFixed(2)}`
      : "";
  return `${name} v${model.model_version}${metrics}`;
}

function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (Math.abs(fraction - 0.5) < 1e-9) {
    return floor % 2 === 0 ? floor : floor + 1;
  }
  return Math.round(value);
}

export function OverlaySelectionBar({
  models,
  selectedModelID,
  onSelectModel,
  displayMode,
  onDisplayModeChange,
  status,
  baseSeeds,
  splitBucket,
}: {
  models: OverlayModel[];
  selectedModelID: string;
  onSelectModel: (id: string) => void;
  displayMode: TrajectoryDisplayMode;
  onDisplayModeChange: (mode: TrajectoryDisplayMode) => void;
  status: OverlayLoadStatus;
  baseSeeds: bigint[];
  splitBucket?: number;
}) {
  const selected = models.find(
    (model) => model.model_artifact_id === selectedModelID,
  );
  let splitLabel = "";
  if (selected) {
    if (selected.val_fraction <= 0) {
      splitLabel = "training set | no hold-out";
    } else if (splitBucket !== undefined) {
      const valBuckets = Math.max(
        1,
        Math.min(9, roundHalfEven(selected.val_fraction * 10)),
      );
      splitLabel =
        splitBucket < valBuckets ? "episode/clip hold-out" : "training set";
    } else {
      splitLabel = `${Math.round(selected.val_fraction * 100)}% hold-out configured`;
    }
  }

  return (
    <div className="flex min-h-12 flex-wrap items-center gap-3 border-y border-slate-800 py-2">
      <Route className="size-4 shrink-0 text-emerald-400" aria-hidden />
      {status === "loading-models" ? (
        <span className="flex items-center gap-1.5 text-xs text-slate-500">
          <Loader2 className="size-3.5 animate-spin" />
          Loading trajectory models
        </span>
      ) : status === "no-models" ? (
        <span className="text-xs text-slate-500">
          No precomputed trajectory overlay for this shard
        </span>
      ) : status === "error" && models.length === 0 ? (
        <span className="flex items-center gap-1.5 text-xs text-amber-500">
          <AlertTriangle className="size-3.5" />
          Trajectory model catalog unavailable
        </span>
      ) : (
        <>
          <label className="sr-only" htmlFor="trajectory-model">
            Trajectory model
          </label>
          <select
            id="trajectory-model"
            value={selectedModelID}
            onChange={(event) => onSelectModel(event.target.value)}
            className="h-8 min-w-0 max-w-full rounded-md border border-slate-700 bg-slate-950 px-2 text-xs text-slate-200 outline-none focus:border-slate-500 sm:min-w-80"
          >
            {models.map((model) => (
              <option
                key={model.model_artifact_id}
                value={model.model_artifact_id}
              >
                {modelLabel(model)}
              </option>
            ))}
          </select>
          <span className="font-mono text-[10px] text-slate-500">
            {splitLabel}
          </span>
          <label
            className="ml-auto flex cursor-pointer items-center gap-2 text-xs text-slate-400"
            title="Post-processes extreme curvature for display and can hide model error"
          >
            <input
              type="checkbox"
              checked={displayMode === "display-limited"}
              onChange={(event) =>
                onDisplayModeChange(
                  event.target.checked ? "display-limited" : "raw",
                )
              }
              className="size-3.5 accent-amber-500"
            />
            Display-limited
          </label>
          {status === "loading-overlay" && (
            <Loader2
              className="size-3.5 animate-spin text-slate-500"
              aria-label="Loading trajectory overlay"
            />
          )}
          {status === "error" && (
            <AlertTriangle
              className="size-3.5 text-amber-500"
              aria-label="Trajectory overlay unavailable"
            />
          )}
          {status === "ready" && (
            <span className="font-mono text-[10px] text-emerald-400">
              {baseSeeds.length === 1
                ? `single sample (base_seed ${baseSeeds[0].toString()})`
                : `${baseSeeds.length} seeds | median`}
            </span>
          )}
        </>
      )}
    </div>
  );
}
