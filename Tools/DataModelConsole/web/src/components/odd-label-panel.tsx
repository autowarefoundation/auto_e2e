"use client";

import { useMemo, useState } from "react";

import { useApi } from "@/hooks/use-api";
import { getODDScene } from "@/lib/api";
import type { ODDObservation } from "@/types";

const STATUS_STYLE: Record<string, string> = {
  valid: "border-emerald-800 bg-emerald-950/30 text-emerald-300",
  unavailable: "border-slate-700 bg-slate-900 text-slate-400",
  not_observable: "border-amber-800 bg-amber-950/30 text-amber-300",
  ambiguous: "border-rose-800 bg-rose-950/30 text-rose-300",
};

function observationValue(observation: ODDObservation): string {
  return observation.status === "valid"
    ? observation.values.join(", ")
    : observation.status;
}

export function ODDLabelPanel({
  dataset,
  version,
  sceneUID,
  timestampNS,
  frame,
  fps,
}: {
  dataset: string;
  version: string;
  sceneUID: string;
  timestampNS?: string;
  frame: number;
  fps: number;
}) {
  const [mode, setMode] = useState<"current" | "whole">("current");
  const result = useApi(
    () => getODDScene(dataset, version, sceneUID),
    [dataset, version, sceneUID],
  );
  const playhead = result.data
    ? timestampNS
      ? Number(timestampNS)
      : result.data.start_timestamp_ns + (frame / fps) * 1e9
    : 0;
  const observations = useMemo(() => {
    if (!result.data) return [];
    if (mode === "current") {
      return result.data.observations.filter(
        (item) =>
          item.start_timestamp_ns <= playhead &&
          playhead < item.end_timestamp_ns,
      );
    }
    const unique = new Map<string, ODDObservation>();
    for (const item of result.data.observations) {
      const identity = [
        item.key,
        item.status,
        item.values.join(","),
        item.source,
        item.camera_id ?? "",
      ].join("|");
      const current = unique.get(identity);
      if (!current || current.confidence < item.confidence) {
        unique.set(identity, item);
      }
    }
    return [...unique.values()];
  }, [result.data, mode, playhead]);

  return (
    <section className="space-y-3 border-y border-slate-800 py-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-[10px] uppercase text-slate-500">ODD Labels</p>
          <p className="mt-1 font-mono text-[10px] text-slate-600">
            scene {sceneUID}
          </p>
        </div>
        <div className="flex border border-slate-800">
          {(["current", "whole"] as const).map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => setMode(item)}
              className={`h-7 px-2.5 text-[11px] ${
                mode === item
                  ? "bg-slate-800 text-slate-100"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {item === "current" ? "Current time" : "Whole scene"}
            </button>
          ))}
        </div>
      </div>
      {result.loading ? (
        <p className="text-xs text-slate-500">Loading ODD labels...</p>
      ) : result.error ? (
        <p className="text-xs text-slate-500">
          No ready ODD LabelSet for this scene.
        </p>
      ) : observations.length === 0 ? (
        <p className="text-xs text-slate-500">
          No observations overlap this time.
        </p>
      ) : (
        <div className="grid gap-2 lg:grid-cols-2">
          {observations.map((item) => (
            <div
              key={item.observation_uid}
              className="grid min-w-0 grid-cols-[1fr_auto] gap-2 border-b border-slate-900 py-2"
            >
              <div className="min-w-0">
                <p className="break-words font-mono text-[11px] text-slate-300">
                  {item.key}
                  {item.camera_id ? ` · ${item.camera_id}` : ""}
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  {item.source} · confidence {item.confidence.toFixed(2)}
                </p>
              </div>
              <span
                className={`h-fit max-w-40 break-words border px-2 py-1 text-right font-mono text-[10px] ${
                  STATUS_STYLE[item.status] ?? STATUS_STYLE.unavailable
                }`}
              >
                {observationValue(item)}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
