"use client";

// CameraMosaic: canvas tiles fed by a FrameStore.
//
// Grid mode: a bird's-eye layout — each camera sits in the CSS-grid cell that
// matches where it points (front on top, rear on the bottom, left cameras on
// the left, right on the right), with the ego state in the center cell. Focus
// mode: one large camera plus a filmstrip of the rest. Late frames are dropped
// — a tile only draws a resolved bitmap if nothing newer has been drawn.

import { useEffect, useRef, useState } from "react";
import { Grid3x3, ImageOff, Loader2 } from "lucide-react";

import type { FrameStore } from "@/lib/frame-store";
import { camLabel, gridDimensions, rigCam } from "@/lib/rig";
import { cn } from "@/lib/utils";
import type { IndexSample } from "@/types";

function CanvasTile({
  store,
  frame,
  cam,
  label,
  ordinal,
  className,
  onClick,
}: {
  store: FrameStore;
  frame: number;
  cam: string;
  label: string;
  ordinal?: number; // 1-based badge matching the "1-7" focus shortcut
  className?: string;
  onClick?: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawnSeqRef = useRef(-1);
  const seqRef = useRef(0);
  // Overlay state: show a spinner until the first bitmap draws; if nothing has
  // ever drawn and a fetch fails (or times out), show a "no image" hint. Once
  // any frame draws we stay "drawn" (drop-late keeps the previous frame on
  // screen through transient misses, so no flicker back to a spinner).
  const [status, setStatus] = useState<"loading" | "drawn" | "error">(
    "loading",
  );

  useEffect(() => {
    const mySeq = ++seqRef.current;
    let cancelled = false;
    const timeout = setTimeout(() => {
      if (!cancelled && drawnSeqRef.current < 0) setStatus("error");
    }, 8000);
    store
      .getFrame(frame, cam)
      .then((bmp) => {
        if (cancelled && mySeq < seqRef.current) return;
        // Drop-late: never overwrite a newer draw with an older frame.
        if (mySeq < drawnSeqRef.current) return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        try {
          if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
            canvas.width = bmp.width;
            canvas.height = bmp.height;
          }
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          ctx.drawImage(bmp, 0, 0);
          drawnSeqRef.current = mySeq;
          setStatus("drawn");
        } catch {
          // Bitmap may have been evicted/closed between resolve and draw.
        }
      })
      .catch(() => {
        // Fetch/decode failure: keep the previous frame on screen; only flag an
        // error if nothing has ever drawn in this tile.
        if (!cancelled && drawnSeqRef.current < 0) setStatus("error");
      });
    return () => {
      cancelled = true;
      clearTimeout(timeout);
      // No per-tile fetch cancellation: the FrameStore fetches whole windows
      // shared across every camera/frame in them, so a window is never "owned"
      // by one leaving tile. Superseded windows fill the cache for scrubbing;
      // destroy() cancels anything still in flight.
    };
  }, [store, frame, cam]);

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-md border border-slate-800 bg-slate-900",
        onClick && "cursor-pointer transition-colors hover:border-slate-500",
        className,
      )}
      onClick={onClick}
      role={onClick ? "button" : undefined}
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full object-contain bg-slate-950"
      />
      {status === "loading" && (
        <div className="absolute inset-0 flex items-center justify-center text-slate-600">
          <Loader2 className="size-4 animate-spin" />
        </div>
      )}
      {status === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 text-slate-600">
          <ImageOff className="size-4" />
          <span className="font-mono text-[8px]">no image</span>
        </div>
      )}
      <span className="absolute bottom-0 left-0 rounded-tr-md bg-slate-950/80 px-1.5 py-0.5 font-mono text-[9px] text-slate-300">
        {label}
      </span>
      {ordinal !== undefined && ordinal >= 1 && ordinal <= 9 && (
        <span className="absolute right-1 top-1 flex size-4 items-center justify-center rounded bg-slate-950/80 font-mono text-[9px] text-slate-400">
          {ordinal}
        </span>
      )}
    </div>
  );
}

// EgoTile is the center cell of the bird's-eye grid: a compact ego-state
// readout with a small car glyph, so the mosaic reads as "the vehicle, with
// its cameras arranged around it."
function EgoTile({ sample }: { sample?: IndexSample }) {
  const ego = sample?.ego_now ?? [];
  return (
    <div className="flex aspect-video w-full flex-col items-center justify-center gap-0.5 overflow-hidden rounded-md border border-slate-700 bg-slate-800/40 p-2 font-mono text-[9px] leading-tight text-slate-400">
      <span className="mb-0.5 text-base leading-none" aria-hidden>
        🚗
      </span>
      <p className="text-slate-500">
        {sample && sample.trip_frame >= 0
          ? `trip frame ${sample.trip_frame}`
          : `frame ${sample?.frame_idx ?? "-"}`}
      </p>
      <p>v {ego[0]?.toFixed(2) ?? "-"} m/s</p>
      <p>a {ego[1]?.toFixed(2) ?? "-"} m/s²</p>
    </div>
  );
}

export function CameraMosaic({
  store,
  dataset,
  sample,
  frame,
  cams,
  mode,
  focusCam,
  onSelectCam,
  onToggleFocus,
}: {
  store: FrameStore;
  dataset: string;
  sample?: IndexSample;
  frame: number;
  cams: string[];
  mode: "grid" | "focus";
  focusCam: number; // index into cams
  onSelectCam: (idx: number) => void;
  onToggleFocus: () => void;
}) {
  if (mode === "focus") {
    const focusedIdx = Math.min(Math.max(focusCam, 0), cams.length - 1);
    const focused = cams[focusedIdx];
    return (
      <div className="space-y-2">
        <div className="relative">
          <CanvasTile
            store={store}
            frame={frame}
            cam={focused}
            label={camLabel(dataset, focused)}
            className="aspect-video w-full"
            onClick={onToggleFocus}
          />
          <button
            type="button"
            onClick={onToggleFocus}
            title="Back to grid (Esc)"
            aria-label="Back to grid"
            className="absolute right-2 top-2 z-10 flex size-7 items-center justify-center rounded-md bg-slate-950/70 text-slate-300 transition-colors hover:bg-slate-950 hover:text-slate-100"
          >
            <Grid3x3 className="size-4" />
          </button>
        </div>
        <div
          className="grid gap-1.5"
          style={{ gridTemplateColumns: `repeat(${cams.length}, minmax(0,1fr))` }}
        >
          {cams.map((cam, i) => (
            <CanvasTile
              key={cam}
              store={store}
              frame={frame}
              cam={cam}
              label={camLabel(dataset, cam)}
              ordinal={i + 1}
              className={cn(
                "aspect-video w-full",
                cam === focused && "ring-1 ring-blue-500",
              )}
              onClick={() => onSelectCam(i)}
            />
          ))}
        </div>
      </div>
    );
  }

  // Grid: bird's-eye layout. Each camera is placed in the CSS-grid cell that
  // matches where it points; the ego readout sits in the center. Cells with no
  // camera stay empty so the spatial arrangement reads clearly.
  const { rows, cols } = gridDimensions(dataset, cams);
  const egoRow = Math.ceil(rows / 2);
  const egoCol = Math.ceil(cols / 2);
  // Detect a camera that would collide with the ego center cell; if one lands
  // there (rare rigs), the ego tile still renders and the camera overlaps — so
  // we only drop the ego tile into a cell no camera claims.
  const claimed = new Set(
    cams.map((cam, i) => {
      const c = rigCam(dataset, cam, i);
      return `${c.row}:${c.col}`;
    }),
  );
  const egoInFreeCell = !claimed.has(`${egoRow}:${egoCol}`);

  return (
    <div
      className="grid gap-2"
      style={{
        gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${rows}, auto)`,
      }}
    >
      {cams.map((cam, i) => {
        const c = rigCam(dataset, cam, i);
        return (
          <div key={cam} style={{ gridRow: c.row, gridColumn: c.col }}>
            <CanvasTile
              store={store}
              frame={frame}
              cam={cam}
              label={c.label}
              ordinal={i + 1}
              className="aspect-video w-full"
              onClick={() => onSelectCam(i)}
            />
          </div>
        );
      })}
      {egoInFreeCell && (
        <div style={{ gridRow: egoRow, gridColumn: egoCol }}>
          <EgoTile sample={sample} />
        </div>
      )}
    </div>
  );
}
