"use client";

// EpisodePlayer: orchestrates FrameStore + usePlayback + camera mosaic +
// timeline scrubber + BEV trajectory + reasoning panel for one shard.
//
// Keyboard-first (bindings on the container, not window):
//   Space        play/pause
//   ArrowLeft/Right  step one frame
//   , / .        step one frame back/forward
//   [ / - and ] / +  slower/faster
//   1-7          focus camera n
//   f            toggle focus/grid
//   Esc          back to grid

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Gauge,
  Pause,
  Play,
  Rewind,
  StepBack,
  StepForward,
} from "lucide-react";

import { CameraMosaic } from "@/components/player/camera-mosaic";
import { TimelineScrubber } from "@/components/player/timeline-scrubber";
import { TrajectoryBEV } from "@/components/player/trajectory-bev";
import { ReasoningTimeline } from "@/components/reasoning-timeline";
import { Button } from "@/components/ui/button";
import { usePlayback, MAX_SPEED, MIN_SPEED } from "@/hooks/use-playback";
import { ApiError, getReasoningLabel } from "@/lib/api";
import { FrameStore } from "@/lib/frame-store";
import type { ReasoningLabelRecord, ShardIndex } from "@/types";

const SPEED_STEPS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16];

export interface PlayerViewState {
  frame: number;
  cam: number;
  mode: "grid" | "focus";
  speed: number;
}

function nextSpeed(current: number, dir: 1 | -1): number {
  const idx = SPEED_STEPS.findIndex((s) => s >= current - 1e-9);
  const i = idx === -1 ? SPEED_STEPS.length - 1 : idx;
  const j = Math.min(SPEED_STEPS.length - 1, Math.max(0, i + dir));
  return Math.min(MAX_SPEED, Math.max(MIN_SPEED, SPEED_STEPS[j]));
}

export function EpisodePlayer({
  dataset,
  index,
  initialState,
  onViewStateChange,
}: {
  dataset: string;
  index: ShardIndex;
  initialState?: Partial<PlayerViewState>;
  onViewStateChange?: (state: PlayerViewState) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  // FrameStore lives for the lifetime of this index.
  const [store, setStore] = useState<FrameStore | null>(null);
  useEffect(() => {
    const s = new FrameStore(index);
    setStore(s);
    return () => s.destroy();
  }, [index]);

  const cams = useMemo(() => {
    const first = index.samples[0];
    if (!first) return [];
    return Object.keys(first.members)
      .filter((m) => m.match(/^cam_\d+\.jpg$/))
      .map((m) => m.replace(/\.jpg$/, ""))
      .sort();
  }, [index]);

  const playback = usePlayback(
    index.samples.length,
    index.fps || 10,
    initialState?.frame ?? 0,
  );
  const { frame, playing, speed, direction, setFrame, toggle, step, setSpeed } =
    playback;

  const [mode, setMode] = useState<"grid" | "focus">(
    initialState?.mode ?? "grid",
  );
  const [focusCam, setFocusCam] = useState(initialState?.cam ?? 0);
  useEffect(() => {
    if (initialState?.speed) setSpeed(initialState.speed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Report view state upward (URL serialization is the page's job).
  useEffect(() => {
    onViewStateChange?.({ frame, cam: focusCam, mode, speed });
  }, [frame, focusCam, mode, speed, onViewStateChange]);

  // Prefetch a look-ahead ring for the visible cameras.
  useEffect(() => {
    if (!store) return;
    const visible = mode === "focus" ? [cams[focusCam] ?? cams[0]] : cams;
    store.prefetch(frame, direction, playing ? speed : 1, visible);
  }, [store, frame, direction, speed, playing, mode, focusCam, cams]);

  // Reasoning label for the current frame (debounced; 404 = no label).
  const sample = index.samples[frame];
  const [reasoning, setReasoning] = useState<ReasoningLabelRecord | null>(null);
  useEffect(() => {
    if (!sample?.has_reasoning) {
      setReasoning(null);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      getReasoningLabel(dataset, sample.key)
        .then((label) => {
          if (!cancelled) setReasoning(label);
        })
        .catch((err: unknown) => {
          if (!cancelled && err instanceof ApiError && err.status === 404) {
            setReasoning(null);
          }
        });
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [dataset, sample?.key, sample?.has_reasoning, sample]);

  const focusCamera = useCallback(
    (idx: number) => {
      if (idx < 0 || idx >= cams.length) return;
      setFocusCam(idx);
      setMode("focus");
    },
    [cams.length],
  );

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
      switch (e.key) {
        case " ":
          e.preventDefault();
          toggle();
          break;
        case "ArrowLeft":
        case ",":
          e.preventDefault();
          step(-1);
          break;
        case "ArrowRight":
        case ".":
          e.preventDefault();
          step(1);
          break;
        case "[":
        case "-":
          e.preventDefault();
          setSpeed(nextSpeed(speed, -1));
          break;
        case "]":
        case "+":
        case "=":
          e.preventDefault();
          setSpeed(nextSpeed(speed, 1));
          break;
        case "f":
          e.preventDefault();
          setMode((m) => (m === "grid" ? "focus" : "grid"));
          break;
        case "Escape":
          e.preventDefault();
          setMode("grid");
          break;
        default: {
          const n = parseInt(e.key, 10);
          if (n >= 1 && n <= 7) {
            e.preventDefault();
            focusCamera(n - 1);
          }
        }
      }
    },
    [toggle, step, setSpeed, speed, focusCamera],
  );

  // Focus the container on mount so keys work immediately.
  useEffect(() => {
    containerRef.current?.focus();
  }, []);

  if (!store || cams.length === 0) {
    return (
      <p className="text-sm text-slate-500">
        Empty shard index — nothing to play.
      </p>
    );
  }

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      onKeyDown={onKeyDown}
      className="space-y-4 outline-none focus-visible:ring-1 focus-visible:ring-slate-600 rounded-lg"
      aria-label="Episode player (keyboard: space, arrows, 1-7, f)"
    >
      <div className="grid gap-4 xl:grid-cols-[1fr_300px]">
        <CameraMosaic
          store={store}
          frame={frame}
          cams={cams}
          mode={mode}
          focusCam={focusCam}
          onSelectCam={focusCamera}
          onToggleFocus={() => setMode((m) => (m === "grid" ? "focus" : "grid"))}
        />
        <div className="space-y-3">
          <TrajectoryBEV samples={index.samples} frame={frame} />
          <div className="rounded-md border border-slate-800 bg-slate-900/60 p-2 font-mono text-[10px] leading-relaxed text-slate-400">
            <p>key: {sample?.key ?? "-"}</p>
            <p>
              speed {sample?.ego_now?.[0]?.toFixed(2) ?? "-"} m/s | accel{" "}
              {sample?.ego_now?.[1]?.toFixed(2) ?? "-"} m/s^2
            </p>
            <p>
              yaw_rate {sample?.ego_now?.[2]?.toFixed(3) ?? "-"} rad/s | kappa{" "}
              {sample?.ego_now?.[3]?.toFixed(4) ?? "-"} 1/m
            </p>
          </div>
        </div>
      </div>

      <TimelineScrubber
        samples={index.samples}
        fps={index.fps || 10}
        frame={frame}
        onSeek={setFrame}
      />

      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="outline"
          size="icon-sm"
          onClick={() => setFrame(0)}
          aria-label="Back to start"
        >
          <Rewind className="size-3.5" />
        </Button>
        <Button
          variant="outline"
          size="icon-sm"
          onClick={() => step(-1)}
          aria-label="Step back one frame"
        >
          <StepBack className="size-3.5" />
        </Button>
        <Button
          size="icon-sm"
          onClick={toggle}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? (
            <Pause className="size-3.5" />
          ) : (
            <Play className="size-3.5" />
          )}
        </Button>
        <Button
          variant="outline"
          size="icon-sm"
          onClick={() => step(1)}
          aria-label="Step forward one frame"
        >
          <StepForward className="size-3.5" />
        </Button>
        <span className="mx-1 h-4 w-px bg-slate-800" />
        <Gauge className="size-3.5 text-slate-500" />
        {SPEED_STEPS.map((s) => (
          <button
            key={s}
            onClick={() => setSpeed(s)}
            className={`rounded px-1.5 py-0.5 font-mono text-[10px] transition-colors ${
              Math.abs(speed - s) < 1e-9
                ? "bg-blue-600 text-white"
                : "bg-slate-900 text-slate-400 hover:text-slate-200"
            }`}
          >
            {s}x
          </button>
        ))}
        <span className="ml-auto font-mono text-[10px] text-slate-500">
          keys: space, arrows, ,/., [/], 1-7, f, esc
        </span>
      </div>

      <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
        <p className="mb-3 text-[10px] uppercase tracking-wider text-slate-500">
          Reasoning label
        </p>
        {sample?.has_reasoning && reasoning ? (
          <ReasoningTimeline label={reasoning} />
        ) : sample?.has_reasoning ? (
          <p className="text-sm text-slate-500">Loading label…</p>
        ) : (
          <p className="text-sm text-slate-500">
            No reasoning label at this frame. Amber ticks on the timeline mark
            labelled frames.
          </p>
        )}
      </div>
    </div>
  );
}
