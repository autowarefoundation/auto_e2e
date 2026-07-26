"use client";

import { useEffect, useRef } from "react";

import {
  BEV_HEATMAP_NAMES,
  BEV_HEATMAP_SIZE,
  bevHeatmapForRow,
  type BEVHeatmapName,
  type OverlayArtifact,
} from "@/lib/overlay";

const LABELS: Record<BEVHeatmapName, string> = {
  image: "Camera",
  map: "Map only",
  route_delta: "Route delta",
  navigation: "Navigation",
  fusion_delta: "Fusion delta",
  fused: "Fused",
};

const DELTA_HEATMAPS = new Set<BEVHeatmapName>([
  "route_delta",
  "fusion_delta",
]);

const COLOR_STOPS = [
  [12, 15, 22],
  [35, 83, 132],
  [22, 151, 143],
  [236, 196, 70],
  [220, 62, 48],
] as const;

function heatColor(value: number): [number, number, number] {
  const position = (value / 255) * (COLOR_STOPS.length - 1);
  const lower = Math.min(Math.floor(position), COLOR_STOPS.length - 2);
  const fraction = position - lower;
  const start = COLOR_STOPS[lower];
  const end = COLOR_STOPS[lower + 1];
  return [
    Math.round(start[0] + (end[0] - start[0]) * fraction),
    Math.round(start[1] + (end[1] - start[1]) * fraction),
    Math.round(start[2] + (end[2] - start[2]) * fraction),
  ];
}

function HeatmapTile({
  name,
  overlay,
  row,
}: {
  name: BEVHeatmapName;
  overlay: OverlayArtifact;
  row: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatmap = bevHeatmapForRow(overlay, row, name);
  const statistics = heatmap
    ? heatmap.values.reduce(
        (result, value) => ({
          sum: result.sum + value,
          peak: Math.max(result.peak, value),
        }),
        { sum: 0, peak: 0 },
      )
    : null;
  const meanValue =
    statistics && heatmap
      ? (statistics.sum / heatmap.values.length / 255) * heatmap.scale
      : null;
  const peakValue =
    statistics && heatmap
      ? (statistics.peak / 255) * heatmap.scale
      : null;
  const metric =
    overlay.formatVersion >= 4
      ? DELTA_HEATMAPS.has(name)
        ? "delta RMS"
        : "spatial deviation"
      : "channel RMS";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!heatmap) return;

    const image = context.createImageData(
      BEV_HEATMAP_SIZE,
      BEV_HEATMAP_SIZE,
    );
    for (let index = 0; index < heatmap.values.length; index++) {
      const [red, green, blue] = heatColor(heatmap.values[index]);
      const offset = index * 4;
      image.data[offset] = red;
      image.data[offset + 1] = green;
      image.data[offset + 2] = blue;
      image.data[offset + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, [heatmap]);

  return (
    <figure
      className="rounded-md border border-slate-800 bg-slate-950 p-2"
      aria-label={`${LABELS[name]} activation`}
    >
      <figcaption className="mb-2 flex min-h-8 items-start justify-between gap-2">
        <span>
          <span className="block text-xs font-medium text-slate-200">
            {LABELS[name]}
          </span>
          <span className="block text-[9px] leading-3 text-slate-500">
            {metric}
          </span>
        </span>
        <span className="text-right font-mono text-[9px] leading-3 text-slate-500">
          {meanValue !== null && peakValue !== null ? (
            <>
              mean {meanValue.toPrecision(3)}
              <br />
              peak {peakValue.toPrecision(3)}
            </>
          ) : (
            "unavailable"
          )}
        </span>
      </figcaption>
      <div className="grid grid-cols-[14px_1fr] grid-rows-[1fr_14px]">
        <span className="flex items-center text-[9px] text-slate-500 [writing-mode:vertical-rl]">
          +X forward
        </span>
        <div className="relative aspect-square overflow-hidden border border-slate-800 bg-slate-950">
          <canvas
            ref={canvasRef}
            width={BEV_HEATMAP_SIZE}
            height={BEV_HEATMAP_SIZE}
            className="size-full [image-rendering:pixelated]"
          />
          <span
            className="absolute left-1/2 top-2/3 block size-2 -translate-x-1/2 -translate-y-1/2 rotate-45 border-l border-t border-white"
            aria-hidden="true"
          />
        </div>
        <span />
        <span className="text-center text-[9px] leading-[14px] text-slate-500">
          +Y left
        </span>
      </div>
    </figure>
  );
}

export function BEVActivationHeatmap({
  overlay,
  row,
}: {
  overlay: OverlayArtifact | null;
  row: number | undefined;
}) {
  const availableNames =
    overlay && overlay.bevHeatmapNames.length > 0
      ? overlay.bevHeatmapNames
      : BEV_HEATMAP_NAMES;
  const sharedScale =
    overlay &&
    overlay.formatVersion < 4 &&
    row !== undefined &&
    overlay.bevHeatmapScales
      ? overlay.bevHeatmapScales[row]
      : null;

  return (
    <section className="space-y-2" aria-label="BEV activation heatmap">
      <div className="flex items-end justify-between gap-3">
        <h3 className="text-sm font-medium text-slate-200">
          Encoder diagnostics
        </h3>
        <span className="font-mono text-[10px] text-slate-500">
          {overlay?.formatVersion && overlay.formatVersion >= 4
            ? "per-encoder contrast"
            : sharedScale !== null
            ? `shared ${sharedScale.toPrecision(3)} RMS`
            : "unavailable"}
        </span>
      </div>
      {overlay && row !== undefined && overlay.bevHeatmaps ? (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {availableNames.map((name) => (
            <HeatmapTile
              key={name}
              name={name}
              overlay={overlay}
              row={row}
            />
          ))}
        </div>
      ) : (
        <div className="flex min-h-32 items-center justify-center rounded-md border border-dashed border-slate-800 px-5 text-center text-xs text-slate-500">
          No encoder diagnostics for this model.
        </div>
      )}
    </section>
  );
}
