"use client";

import { useEffect, useRef, useState } from "react";

import {
  BEV_HEATMAP_NAMES,
  BEV_HEATMAP_SIZE,
  bevHeatmapForRow,
  type BEVHeatmapName,
  type OverlayArtifact,
} from "@/lib/overlay";

const LABELS: Record<BEVHeatmapName, string> = {
  image: "Camera",
  navigation: "Navigation",
  fused: "Fused",
};

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

export function BEVActivationHeatmap({
  overlay,
  row,
}: {
  overlay: OverlayArtifact | null;
  row: number | undefined;
}) {
  const [mode, setMode] = useState<BEVHeatmapName>("fused");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatmap =
    overlay && row !== undefined
      ? bevHeatmapForRow(overlay, row, mode)
      : null;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !heatmap) return;
    const context = canvas.getContext("2d");
    if (!context) return;

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
    <section
      className="rounded-md border border-slate-800 bg-slate-950 p-2"
      aria-label="BEV activation heatmap"
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="text-xs font-medium text-slate-200">BEV activations</h3>
        <span className="font-mono text-[10px] text-slate-500">
          {heatmap ? `max RMS ${heatmap.scale.toPrecision(3)}` : "unavailable"}
        </span>
      </div>
      <div
        role="tablist"
        aria-label="BEV activation branch"
        className="mb-2 grid grid-cols-3 rounded border border-slate-800 bg-slate-900 p-0.5"
      >
        {BEV_HEATMAP_NAMES.map((name) => (
          <button
            key={name}
            type="button"
            role="tab"
            aria-selected={mode === name}
            onClick={() => setMode(name)}
            className={
              mode === name
                ? "h-6 rounded-sm bg-slate-700 px-1 text-[10px] text-white"
                : "h-6 rounded-sm px-1 text-[10px] text-slate-400 hover:text-slate-200"
            }
          >
            {LABELS[name]}
          </button>
        ))}
      </div>
      {heatmap ? (
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
      ) : (
        <div className="flex aspect-square items-center justify-center border border-dashed border-slate-800 px-5 text-center text-xs text-slate-500">
          This model overlay has no BEV diagnostics.
        </div>
      )}
    </section>
  );
}
