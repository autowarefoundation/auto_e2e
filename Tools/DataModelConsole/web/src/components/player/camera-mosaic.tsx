"use client";

// CameraMosaic: canvas tiles fed by a FrameStore.
//
// Grid mode: a bird's-eye layout — each camera sits in the CSS-grid cell that
// matches where it points (front on top, rear on the bottom, left cameras on
// the left, right on the right), with the ego state in the center cell. Focus
// mode: one large camera plus a filmstrip of the rest. Late frames are dropped
// — and each tile paints an image and its trajectory overlay as one frame.

import { useEffect, useRef, useState } from "react";
import { Grid3x3, ImageOff, Loader2 } from "lucide-react";

import type { FrameStore } from "@/lib/frame-store";
import type {
  CameraProjectionPaths,
  CameraProjectionRibbons,
  ScreenPoint,
  ScreenRibbon,
} from "@/lib/projection";
import { camLabel, gridDimensions, rigCam } from "@/lib/rig";
import { cn } from "@/lib/utils";
import type { IndexSample } from "@/types";

const FRAME_RETRY_MS = 500;
const FRAME_RETRY_MAX_MS = 4_000;

function paintFrame(
  imageCanvas: HTMLCanvasElement,
  overlayCanvas: HTMLCanvasElement,
  bmp: ImageBitmap,
  predictionPaths?: ScreenPoint[][],
  predictionRibbons?: ScreenRibbon[],
  groundTruthRibbons?: ScreenRibbon[],
): boolean {
  if (imageCanvas.width !== bmp.width || imageCanvas.height !== bmp.height) {
    imageCanvas.width = bmp.width;
    imageCanvas.height = bmp.height;
  }
  if (
    overlayCanvas.width !== bmp.width ||
    overlayCanvas.height !== bmp.height
  ) {
    overlayCanvas.width = bmp.width;
    overlayCanvas.height = bmp.height;
  }
  const imageContext = imageCanvas.getContext("2d");
  const overlayContext = overlayCanvas.getContext("2d");
  if (!imageContext || !overlayContext) return false;

  // Both canvases are updated in one task, so the browser cannot composite an
  // image from one frame with trajectory paths from another.
  imageContext.drawImage(bmp, 0, 0);
  overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  const strokePaths = (
    paths: ScreenPoint[][] | undefined,
    color: string,
    width: number,
  ) => {
    if (!paths) return;
    overlayContext.strokeStyle = color;
    overlayContext.lineWidth = width;
    overlayContext.lineCap = "round";
    overlayContext.lineJoin = "round";
    for (const path of paths) {
      if (path.length < 2) continue;
      overlayContext.beginPath();
      path.forEach((point, index) => {
        const x = point.u * overlayCanvas.width;
        const y = point.v * overlayCanvas.height;
        if (index === 0) overlayContext.moveTo(x, y);
        else overlayContext.lineTo(x, y);
      });
      overlayContext.stroke();
    }
  };

  const strokePoints = (
    points: ScreenPoint[],
    color: string,
    width: number,
  ) => {
    if (points.length < 2) return;
    overlayContext.beginPath();
    points.forEach((point, index) => {
      const x = point.u * overlayCanvas.width;
      const y = point.v * overlayCanvas.height;
      if (index === 0) overlayContext.moveTo(x, y);
      else overlayContext.lineTo(x, y);
    });
    overlayContext.strokeStyle = color;
    overlayContext.lineWidth = width;
    overlayContext.lineCap = "round";
    overlayContext.lineJoin = "round";
    overlayContext.stroke();
  };

  const paintRibbons = (
    ribbons: ScreenRibbon[] | undefined,
    fill: string,
    outline: string,
  ) => {
    if (!ribbons) return;
    for (const ribbon of ribbons) {
      if (ribbon.left.length < 2 || ribbon.right.length < 2) continue;
      overlayContext.beginPath();
      ribbon.left.forEach((point, index) => {
        const x = point.u * overlayCanvas.width;
        const y = point.v * overlayCanvas.height;
        if (index === 0) overlayContext.moveTo(x, y);
        else overlayContext.lineTo(x, y);
      });
      for (let index = ribbon.right.length - 1; index >= 0; index--) {
        const point = ribbon.right[index];
        overlayContext.lineTo(
          point.u * overlayCanvas.width,
          point.v * overlayCanvas.height,
        );
      }
      overlayContext.closePath();
      overlayContext.fillStyle = fill;
      overlayContext.fill("evenodd");

      const boundaries = [
        ribbon.left,
        ribbon.right,
        [ribbon.left[0], ribbon.right[0]],
        [ribbon.left.at(-1)!, ribbon.right.at(-1)!],
      ];
      for (const boundary of boundaries) {
        strokePoints(boundary, "rgba(2, 6, 23, 0.78)", 4);
        strokePoints(boundary, outline, 2);
      }
    }
  };

  strokePaths(predictionPaths, "rgba(52, 211, 153, 0.2)", 1.25);
  paintRibbons(
    groundTruthRibbons,
    "rgba(139, 92, 246, 0.3)",
    "rgba(196, 181, 253, 0.98)",
  );
  paintRibbons(
    predictionRibbons,
    "rgba(34, 197, 94, 0.3)",
    "rgba(110, 231, 183, 0.98)",
  );
  return true;
}

function CanvasTile({
  store,
  frame,
  cam,
  label,
  ordinal,
  predictionPaths,
  predictionRibbons,
  groundTruthRibbons,
  className,
  onClick,
  selected = false,
}: {
  store: FrameStore;
  frame: number;
  cam: string;
  label: string;
  ordinal?: number; // 1-based badge matching the "1-7" focus shortcut
  predictionPaths?: ScreenPoint[][];
  predictionRibbons?: ScreenRibbon[];
  groundTruthRibbons?: ScreenRibbon[];
  className?: string;
  onClick: () => void;
  selected?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
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
    let retryCount = 0;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    const isCurrent = () => !cancelled && mySeq === seqRef.current;
    const timeout = setTimeout(() => {
      if (isCurrent() && drawnSeqRef.current < 0) setStatus("error");
    }, 8000);

    function retry() {
      if (!isCurrent()) return;
      const delay = Math.min(
        FRAME_RETRY_MAX_MS,
        FRAME_RETRY_MS * 2 ** retryCount,
      );
      retryCount = Math.min(retryCount + 1, 3);
      retryTimer = setTimeout(load, delay);
    }
    function load() {
      retryTimer = undefined;
      store.getFrame(frame, cam).then(
        (bmp) => {
          if (!isCurrent()) return;
          const canvas = canvasRef.current;
          const overlay = overlayRef.current;
          if (!canvas || !overlay) return;
          try {
            if (
              !paintFrame(
                canvas,
                overlay,
                bmp,
                predictionPaths,
                predictionRibbons,
                groundTruthRibbons,
              )
            ) {
              return;
            }
            drawnSeqRef.current = mySeq;
            clearTimeout(timeout);
            setStatus("drawn");
          } catch {
            // Bitmap may have been evicted/closed between resolve and draw.
            retry();
          }
        },
        () => {
          if (!isCurrent()) return;
          // Keep the previous image/overlay pair on screen while retrying.
          if (drawnSeqRef.current < 0) setStatus("error");
          retry();
        },
      );
    }
    load();

    return () => {
      cancelled = true;
      clearTimeout(timeout);
      if (retryTimer !== undefined) clearTimeout(retryTimer);
      // No per-tile fetch cancellation: the FrameStore fetches whole windows
      // shared across every camera/frame in them, so a window is never "owned"
      // by one leaving tile. Superseded windows fill the cache for scrubbing;
      // destroy() cancels anything still in flight.
    };
  }, [
    store,
    frame,
    cam,
    predictionPaths,
    predictionRibbons,
    groundTruthRibbons,
  ]);

  return (
    <button
      type="button"
      className={cn(
        "relative block overflow-hidden rounded-md border border-slate-800 bg-slate-900 p-0 text-left transition-colors hover:border-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-slate-300",
        className,
      )}
      onClick={onClick}
      aria-label={`${label} camera`}
      aria-pressed={selected}
      aria-busy={status === "loading"}
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full object-contain bg-slate-950"
      />
      <canvas
        ref={overlayRef}
        className="pointer-events-none absolute inset-0 z-[1] h-full w-full object-contain"
        aria-hidden
      />
      {status === "loading" && (
        <div className="absolute inset-0 z-[2] flex items-center justify-center text-slate-400">
          <Loader2 className="size-4 animate-spin" />
        </div>
      )}
      {status === "error" && (
        <div className="absolute inset-0 z-[2] flex flex-col items-center justify-center gap-1 text-slate-400">
          <ImageOff className="size-4" />
          <span className="font-mono text-[8px]">no image</span>
        </div>
      )}
      <span className="absolute bottom-0 left-0 z-[2] whitespace-nowrap rounded-tr-md bg-slate-950/80 px-1.5 py-0.5 font-mono text-[9px] text-slate-300">
        {label}
      </span>
      {ordinal !== undefined && ordinal >= 1 && ordinal <= 9 && (
        <span className="absolute right-1 top-1 z-[2] flex size-4 items-center justify-center rounded bg-slate-950/80 font-mono text-[9px] text-slate-400">
          {ordinal}
        </span>
      )}
    </button>
  );
}

function TrajectoryLegend({
  prediction,
  groundTruth,
}: {
  prediction: boolean;
  groundTruth: boolean;
}) {
  if (!prediction && !groundTruth) return null;
  return (
    <div
      className="mb-1 flex justify-end gap-3 font-mono text-[10px] text-slate-400"
      aria-label="Camera trajectory legend"
    >
      {groundTruth && (
        <span className="inline-flex items-center gap-1.5">
          <span className="h-0.5 w-5 bg-violet-400" aria-hidden />
          Ground truth
        </span>
      )}
      {prediction && (
        <span className="inline-flex items-center gap-1.5">
          <span className="h-0.5 w-5 bg-emerald-400" aria-hidden />
          Prediction
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
      <p className="text-slate-400">
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
  predictionPaths,
  predictionRibbons,
  groundTruthRibbons,
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
  predictionPaths?: CameraProjectionPaths;
  predictionRibbons?: CameraProjectionRibbons;
  groundTruthRibbons?: CameraProjectionRibbons;
}) {
  const hasPrediction = Object.values(predictionRibbons ?? {}).some(
    (ribbons) => ribbons.length > 0,
  );
  const hasGroundTruth = Object.values(groundTruthRibbons ?? {}).some(
    (ribbons) => ribbons.length > 0,
  );
  if (mode === "focus") {
    const focusedIdx = Math.min(Math.max(focusCam, 0), cams.length - 1);
    const focused = cams[focusedIdx];
    return (
      <div className="min-w-0">
        <TrajectoryLegend
          prediction={hasPrediction}
          groundTruth={hasGroundTruth}
        />
        <div className="relative">
          <CanvasTile
            store={store}
            frame={frame}
            cam={focused}
            label={camLabel(dataset, focused)}
            predictionPaths={predictionPaths?.[focused]}
            predictionRibbons={predictionRibbons?.[focused]}
            groundTruthRibbons={groundTruthRibbons?.[focused]}
            className="aspect-video w-full"
            onClick={onToggleFocus}
            selected
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
          className="mt-2 flex min-w-0 max-w-full gap-1.5 overflow-x-auto overscroll-x-contain pb-1"
          role="group"
          aria-label="Camera filmstrip"
        >
          {cams.map((cam, i) => (
            <CanvasTile
              key={cam}
              store={store}
              frame={frame}
              cam={cam}
              label={camLabel(dataset, cam)}
              ordinal={i + 1}
              predictionPaths={predictionPaths?.[cam]}
              predictionRibbons={predictionRibbons?.[cam]}
              groundTruthRibbons={groundTruthRibbons?.[cam]}
              className={cn(
                "aspect-video min-w-28 basis-28 shrink-0 grow",
                cam === focused && "ring-1 ring-blue-500",
              )}
              onClick={() => onSelectCam(i)}
              selected={cam === focused}
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
    <div className="min-w-0">
      <TrajectoryLegend
        prediction={hasPrediction}
        groundTruth={hasGroundTruth}
      />
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
                predictionPaths={predictionPaths?.[cam]}
                predictionRibbons={predictionRibbons?.[cam]}
                groundTruthRibbons={groundTruthRibbons?.[cam]}
                className="aspect-video w-full"
                onClick={() => onSelectCam(i)}
                selected={false}
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
    </div>
  );
}
