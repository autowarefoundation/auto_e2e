"use client";

// CameraMosaic: canvas tiles fed by a FrameStore.
//
// Grid mode: 2x4 mosaic of all cameras. Focus mode: one large camera plus a
// filmstrip of the rest. Late frames are dropped — a tile only draws a
// resolved bitmap if nothing newer has been drawn already.

import { useEffect, useRef } from "react";

import type { FrameStore } from "@/lib/frame-store";
import { cn } from "@/lib/utils";

function CanvasTile({
  store,
  frame,
  cam,
  className,
  onClick,
}: {
  store: FrameStore;
  frame: number;
  cam: string;
  className?: string;
  onClick?: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawnSeqRef = useRef(-1);
  const seqRef = useRef(0);

  useEffect(() => {
    const mySeq = ++seqRef.current;
    let cancelled = false;
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
        } catch {
          // Bitmap may have been evicted/closed between resolve and draw.
        }
      })
      .catch(() => {
        // Fetch/decode failure: keep the previous frame on screen.
      });
    return () => {
      cancelled = true;
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
        className="absolute inset-0 h-full w-full object-cover"
      />
      <span className="absolute bottom-0 left-0 rounded-tr-md bg-slate-950/80 px-1.5 py-0.5 font-mono text-[9px] text-slate-300">
        {cam}
      </span>
    </div>
  );
}

export function CameraMosaic({
  store,
  frame,
  cams,
  mode,
  focusCam,
  onSelectCam,
  onToggleFocus,
}: {
  store: FrameStore;
  frame: number;
  cams: string[];
  mode: "grid" | "focus";
  focusCam: number; // index into cams
  onSelectCam: (idx: number) => void;
  onToggleFocus: () => void;
}) {
  if (mode === "focus") {
    const focused = cams[Math.min(focusCam, cams.length - 1)];
    return (
      <div className="space-y-2">
        <CanvasTile
          store={store}
          frame={frame}
          cam={focused}
          className="aspect-video w-full"
          onClick={onToggleFocus}
        />
        <div className="grid grid-cols-7 gap-1.5">
          {cams.map((cam, i) => (
            <CanvasTile
              key={cam}
              store={store}
              frame={frame}
              cam={cam}
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

  return (
    <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
      {cams.map((cam, i) => (
        <CanvasTile
          key={cam}
          store={store}
          frame={frame}
          cam={cam}
          className="aspect-video w-full"
          onClick={() => onSelectCam(i)}
        />
      ))}
    </div>
  );
}
