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
  Keyboard,
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
import { MAX_YAW_RATE, MAX_CURVATURE } from "@/lib/ego";
import type { ReasoningLabelRecord, ShardIndex } from "@/types";

const SPEED_STEPS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16];

type LabelState = { key: string; label: ReasoningLabelRecord } | null;

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
  shard,
  index,
  initialState,
  onViewStateChange,
  version,
}: {
  dataset: string;
  shard: string;
  index: ShardIndex;
  initialState?: Partial<PlayerViewState>;
  onViewStateChange?: (state: PlayerViewState) => void;
  version?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  // FrameStore lives for the lifetime of this index.
  const [store, setStore] = useState<FrameStore | null>(null);
  useEffect(() => {
    const s = new FrameStore(index, dataset, shard, undefined, version);
    setStore(s);
    return () => s.destroy();
  }, [index, dataset, shard, version]);

  const fps = index.fps || 10;
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
  const [focusCam, setFocusCam] = useState(Math.max(0, initialState?.cam ?? 0));
  useEffect(() => {
    if (initialState?.speed) setSpeed(initialState.speed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Report view state upward (URL serialization is the page's job).
  useEffect(() => {
    onViewStateChange?.({ frame, cam: focusCam, mode, speed });
  }, [frame, focusCam, mode, speed, onViewStateChange]);

  const visibleCams = useMemo(
    () => (mode === "focus" ? [cams[focusCam] ?? cams[0]].filter(Boolean) : cams),
    [mode, cams, focusCam],
  );

  // Prefetch a look-ahead ring for the visible cameras.
  useEffect(() => {
    if (!store) return;
    store.prefetch(frame, direction, playing ? speed : 1, visibleCams);
  }, [store, frame, direction, speed, playing, visibleCams]);

  // Buffering indicator: while playing, sample whether the next second of
  // frames for the visible cameras is cached ahead of the playhead. Fetching
  // 6 cameras/frame over the network cannot always sustain 10Hz, so rather
  // than silently stuttering we surface a "buffering" chip when the look-ahead
  // runs dry — the clock keeps running (drop-late) and the last frame stays on
  // screen, so it still reads as video, just with an honest buffering hint.
  const [buffering, setBuffering] = useState(false);
  useEffect(() => {
    if (!store || !playing) {
      setBuffering(false);
      return;
    }
    const need = Math.max(3, Math.round((fps || 10) * 0.6));
    const id = setInterval(() => {
      setBuffering(store.cachedCount(frame, direction, need, visibleCams) < 2);
    }, 200);
    return () => clearInterval(id);
  }, [store, playing, frame, direction, visibleCams, fps]);

  // Reasoning label for the current frame (debounced; 404 = no label). The
  // label is bound to the sample key it was fetched for so an in-flight
  // response for a prior frame can never render on the current one, and a
  // discrete status drives the panel (never hangs on 404/5xx, never shows a
  // stale card for a frame that is still loading).
  const sample = index.samples[frame];
  const [reasoning, setReasoning] = useState<LabelState>(null);
  const [labelStatus, setLabelStatus] = useState<
    "idle" | "loading" | "ready" | "absent" | "error"
  >("idle");
  useEffect(() => {
    if (!sample?.has_reasoning) {
      setReasoning(null);
      setLabelStatus("idle");
      return;
    }
    const key = sample.key;
    setLabelStatus("loading"); // clear any stale card immediately
    let cancelled = false;
    const timer = setTimeout(() => {
      getReasoningLabel(dataset, key)
        .then((label) => {
          if (cancelled) return;
          setReasoning({ key, label });
          setLabelStatus("ready");
        })
        .catch((err: unknown) => {
          if (cancelled) return;
          if (err instanceof ApiError && err.status === 404) {
            setReasoning(null);
            setLabelStatus("absent");
          } else {
            console.warn("reasoning label fetch failed", err);
            setReasoning(null);
            setLabelStatus("error");
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

  const [showHelp, setShowHelp] = useState(false);

  // One-time discoverability hint: shown until the user opens the help or
  // explicitly dismisses it, persisted in localStorage so it appears once.
  const HINT_KEY = "adas-player-shortcut-hint-dismissed";
  const [showHint, setShowHint] = useState(false);
  useEffect(() => {
    try {
      if (localStorage.getItem(HINT_KEY) !== "1") setShowHint(true);
    } catch {
      // localStorage may be unavailable (private mode); skip the hint.
    }
  }, []);
  const dismissHint = useCallback(() => {
    setShowHint(false);
    try {
      localStorage.setItem(HINT_KEY, "1");
    } catch {
      // ignore persistence failure
    }
  }, []);

  // Bind shortcuts at the window level so they work regardless of which DOM
  // node has focus (clicking a control no longer traps Space/arrows). The
  // INPUT/TEXTAREA guard keeps typing intact; the BUTTON guard for Space stops
  // the browser's native activation from double-firing alongside toggle().
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
      // The timeline scrubber (role="slider") owns Arrow/Home/End when focused
      // and seeks itself; this window listener is a NATIVE listener, so the
      // scrubber's React stopPropagation cannot suppress it — skip those keys
      // here to avoid double-stepping the frame.
      if (t && t.getAttribute("role") === "slider") {
        if (
          e.key === "ArrowLeft" ||
          e.key === "ArrowRight" ||
          e.key === "Home" ||
          e.key === "End"
        ) {
          return;
        }
      }
      switch (e.key) {
        case " ":
          e.preventDefault(); // also suppresses native BUTTON activation
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
        case "?":
          e.preventDefault();
          setShowHelp((v) => !v);
          dismissHint();
          break;
        case "Escape":
          e.preventDefault();
          setShowHelp(false);
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
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, step, setSpeed, speed, focusCamera, dismissHint]);

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
      className="space-y-4 outline-none focus-visible:ring-1 focus-visible:ring-slate-600 rounded-lg"
      aria-label="Episode player (keyboard: space, arrows, 1-7, f, ? for help)"
    >
      <div className="grid gap-4 xl:grid-cols-[1fr_300px]">
        <CameraMosaic
          store={store}
          dataset={dataset}
          sample={sample}
          frame={frame}
          cams={cams}
          mode={mode}
          focusCam={focusCam}
          onSelectCam={focusCamera}
          onToggleFocus={() => setMode((m) => (m === "grid" ? "focus" : "grid"))}
        />
        <div className="space-y-3">
          <TrajectoryBEV
            samples={index.samples}
            frame={frame}
            fps={index.fps || 10}
            reasoning={
              reasoning?.key === sample?.key ? reasoning.label : null
            }
          />
          <div className="rounded-md border border-slate-800 bg-slate-900/60 p-2 font-mono text-[10px] leading-relaxed text-slate-400">
            <p>
              ep {sample?.episode_id || "-"} ·{" "}
              {sample && sample.trip_frame >= 0
                ? `trip frame ${sample.trip_frame}`
                : `frame ${sample?.frame_idx ?? "-"}`}
            </p>
            <p>key: {sample?.key ?? "-"}</p>
            <p>
              speed {sample?.ego_now?.[0]?.toFixed(2) ?? "-"} m/s | accel{" "}
              {sample?.ego_now?.[1]?.toFixed(2) ?? "-"} m/s^2
            </p>
            <p>
              yaw_rate{" "}
              <span
                className={
                  Math.abs(sample?.ego_now?.[2] ?? 0) > MAX_YAW_RATE
                    ? "text-amber-500"
                    : ""
                }
              >
                {sample?.ego_now?.[2]?.toFixed(3) ?? "-"}
              </span>{" "}
              rad/s | kappa{" "}
              <span
                className={
                  Math.abs(sample?.ego_now?.[3] ?? 0) > MAX_CURVATURE
                    ? "text-amber-500"
                    : ""
                }
              >
                {sample?.ego_now?.[3]?.toFixed(4) ?? "-"}
              </span>{" "}
              1/m
              {(Math.abs(sample?.ego_now?.[2] ?? 0) > MAX_YAW_RATE ||
                Math.abs(sample?.ego_now?.[3] ?? 0) > MAX_CURVATURE) && (
                <span className="text-amber-600"> · non-physical (clamped in BEV)</span>
              )}
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
          title="Back to start"
        >
          <Rewind className="size-3.5" />
        </Button>
        <Button
          variant="outline"
          size="icon-sm"
          onClick={() => step(-1)}
          aria-label="Step back one frame"
          title="Step back one frame (← or ,)"
        >
          <StepBack className="size-3.5" />
        </Button>
        <Button
          size="icon-sm"
          onClick={toggle}
          aria-label={playing ? "Pause" : "Play"}
          title="Play/Pause (Space)"
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
          title="Step forward one frame (→ or .)"
        >
          <StepForward className="size-3.5" />
        </Button>
        <span className="mx-1 h-4 w-px bg-slate-800" />
        <Gauge className="size-3.5 text-slate-500" />
        {/* Always-visible readout: usePlayback clamps but accepts off-preset
            speeds (e.g. ?speed=3), which would otherwise light no chip. */}
        <span className="font-mono text-[10px] text-slate-400">{speed}x</span>
        {SPEED_STEPS.map((s) => (
          <button
            key={s}
            onClick={() => setSpeed(s)}
            title={`Set speed ${s}x ([ / ] to change)`}
            className={`rounded px-1.5 py-0.5 font-mono text-[10px] transition-colors ${
              Math.abs(speed - s) < 1e-9
                ? "bg-blue-600 text-white"
                : "bg-slate-900 text-slate-400 hover:text-slate-200"
            }`}
          >
            {s}x
          </button>
        ))}
        {buffering && (
          <span
            className="ml-auto flex items-center gap-1 rounded bg-amber-950/60 px-2 py-0.5 font-mono text-[10px] text-amber-400"
            title="Fetching frames ahead of the playhead"
          >
            <span className="size-1.5 animate-pulse rounded-full bg-amber-400" />
            buffering
          </span>
        )}
        <button
          onClick={() => {
            setShowHelp((v) => !v);
            dismissHint();
          }}
          title="Keyboard shortcuts (?)"
          aria-label="Keyboard shortcuts"
          className={`${buffering ? "" : "ml-auto"} flex items-center gap-1 rounded bg-slate-900 px-2 py-1 font-mono text-[11px] text-slate-400 transition-colors hover:text-slate-200`}
        >
          <Keyboard className="size-3.5" />
          Shortcuts (?)
        </button>
      </div>

      {showHint && (
        <div className="flex items-center justify-between rounded-md border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-[11px] text-slate-400">
          <span>
            Press <kbd className="rounded bg-slate-800 px-1">?</kbd> for keyboard
            shortcuts.
          </span>
          <button
            onClick={dismissHint}
            className="font-mono text-slate-500 hover:text-slate-300"
            aria-label="Dismiss hint"
          >
            dismiss
          </button>
        </div>
      )}

      {showHelp && (
        <div className="rounded-lg border border-slate-700 bg-slate-950/80 p-4">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-wider text-slate-500">
              Keyboard shortcuts
            </p>
            <button
              onClick={() => setShowHelp(false)}
              className="font-mono text-[10px] text-slate-500 hover:text-slate-300"
            >
              close (esc)
            </button>
          </div>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-[11px] text-slate-400 sm:grid-cols-3">
            {[
              ["Space", "play / pause"],
              ["← / ,", "step back one frame"],
              ["→ / .", "step forward one frame"],
              ["[ / -", "slower"],
              ["] / +", "faster"],
              ["1 - 7", "focus camera n"],
              ["f", "toggle focus / grid"],
              ["Esc", "back to grid"],
              ["?", "toggle this help"],
            ].map(([k, desc]) => (
              <div key={k} className="flex items-baseline gap-2">
                <kbd className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-200">
                  {k}
                </kbd>
                <span>{desc}</span>
              </div>
            ))}
          </dl>
        </div>
      )}

      <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
        <p className="mb-3 text-[10px] uppercase tracking-wider text-slate-500">
          Reasoning label
        </p>
        {labelStatus === "ready" && reasoning?.key === sample?.key ? (
          <ReasoningTimeline label={reasoning.label} />
        ) : labelStatus === "loading" ? (
          <p className="text-sm text-slate-500">Loading label…</p>
        ) : labelStatus === "error" ? (
          <p className="text-sm text-amber-500">
            Could not load the label for this frame. It may retry on the next
            step.
          </p>
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
