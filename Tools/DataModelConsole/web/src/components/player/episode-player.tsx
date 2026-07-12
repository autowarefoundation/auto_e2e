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
import { MAX_YAW_RATE, MAX_CURVATURE, yawRateFrom } from "@/lib/ego";
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
  // Pick the nearest preset strictly in the requested direction. This is
  // correct whether `current` is on a preset (steps to the neighbor) or off it
  // (e.g. ?speed=3): "faster" from 3 gives 4, not 8. The old index+dir jumped
  // past the immediately-higher preset for off-preset values.
  if (dir === 1) {
    const next = SPEED_STEPS.find((s) => s > current + 1e-9);
    return Math.min(MAX_SPEED, next ?? current);
  }
  let prev = current;
  for (const s of SPEED_STEPS) {
    if (s < current - 1e-9) prev = s;
  }
  return Math.max(MIN_SPEED, prev);
}

export function EpisodePlayer({
  dataset,
  shard,
  index,
  initialState,
  onViewStateChange,
  version,
  promptVersion,
}: {
  dataset: string;
  shard: string;
  index: ShardIndex;
  initialState?: Partial<PlayerViewState>;
  onViewStateChange?: (state: PlayerViewState) => void;
  version?: string;
  promptVersion?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  // FrameStore lives for the lifetime of this index.
  const [store, setStore] = useState<FrameStore | null>(null);
  useEffect(() => {
    const s = new FrameStore(index, dataset, shard, undefined, version);
    setStore(s);
    return () => s.destroy();
  }, [index, dataset, shard, version]);

  const cams = useMemo(() => {
    const first = index.samples[0];
    if (!first) return [];
    return Object.keys(first.members)
      .filter((m) => m.match(/^cam_\d+\.jpg$/))
      .map((m) => m.replace(/\.jpg$/, ""))
      .sort();
  }, [index]);

  // Buffer-readiness predicate for the buffer-gated clock: a frame is ready
  // when every currently-visible camera has a decoded bitmap for it. Defined
  // via refs the player keeps current (store + visibleCams) so the identity is
  // stable and the playback hook never re-subscribes its rAF loop.
  const storeRef = useRef<FrameStore | null>(null);
  storeRef.current = store;
  const visibleCamsRef = useRef<string[]>([]);
  const frameReady = useCallback((f: number) => {
    const s = storeRef.current;
    const cams = visibleCamsRef.current;
    if (!s || cams.length === 0) return true; // nothing to gate on yet
    return s.cachedCount(f, 1, 1, cams) >= 1;
  }, []);

  const playback = usePlayback(
    index.samples.length,
    index.fps || 10,
    initialState?.frame ?? 0,
    frameReady,
  );
  const {
    frame,
    playing,
    speed,
    direction,
    stalled,
    setFrame,
    toggle,
    step,
    setSpeed,
    pause,
  } = playback;

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
  visibleCamsRef.current = visibleCams;

  // Prefetch a look-ahead ring for the visible cameras.
  useEffect(() => {
    if (!store) return;
    store.prefetch(frame, direction, playing ? speed : 1, visibleCams);
  }, [store, frame, direction, speed, playing, visibleCams]);

  // Current-frame readiness: whether every visible camera has a decoded bitmap
  // for the frame on screen. When paused and the user scrubs into an unbuffered
  // frame, the mosaic keeps the previous frame (drop-late) with no indication;
  // this flag drives a loading hint for that case too, not only mid-playback
  // stalls. Probed on a short interval only while NOT ready (self-clearing, no
  // steady-state cost) since the cache fills asynchronously.
  const [currentReady, setCurrentReady] = useState(true);
  useEffect(() => {
    if (!store) return;
    const check = () =>
      setCurrentReady(store.cachedCount(frame, 1, 1, visibleCams) >= 1);
    check();
    const id = setInterval(check, 250);
    return () => clearInterval(id);
  }, [store, frame, visibleCams]);

  // Buffering indicator: the clock is buffer-gated (see usePlayback), so it
  // reports `stalled` exactly when it is holding for the next frame to decode.
  // We also surface it when the frame currently on screen is not yet drawable
  // (e.g. a paused scrub into an unbuffered frame), so the UI is never silently
  // frozen. The picture is never behind a running counter — the clock holds on
  // the last drawn frame until the buffer catches up.
  const buffering = (playing && stalled) || !currentReady;

  // Stall recovery: while the clock is held, `frame` is frozen, so the prefetch
  // effect above (keyed on `frame`) never re-fires. If the window fetch for the
  // gated frame failed transiently, nothing would ever re-request it and the
  // player would wedge on "buffering" forever. Re-issue the look-ahead on an
  // interval while stalled so a swallowed failure is retried; the clock resumes
  // the instant the frame decodes. Only runs during a stall, so no steady-state cost.
  useEffect(() => {
    if (!store || !buffering) return;
    const id = setInterval(() => {
      store.prefetch(frame, direction, speed, visibleCams);
    }, 500);
    return () => clearInterval(id);
  }, [store, buffering, frame, direction, speed, visibleCams]);

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
      getReasoningLabel(dataset, key, promptVersion)
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
  }, [dataset, promptVersion, sample?.key, sample?.has_reasoning, sample]);

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
              {/* The BEV integrates heading via yawRateFrom(speed, kappa),
                  which clamps BOTH kappa and the resulting yaw rate v*kappa to
                  MAX_YAW_RATE. So the "clamped in BEV" note fires whenever that
                  actually saturates for this sample — either kappa out of range
                  OR the in-range kappa still yielding an over-limit yaw rate at
                  this speed. A raw yaw_rate channel spike (not what the BEV
                  integrates) is flagged separately as non-physical. */}
              {(() => {
                const v = sample?.ego_now?.[0] ?? 0;
                const kappa = sample?.ego_now?.[3] ?? 0;
                const yaw = sample?.ego_now?.[2] ?? 0;
                const bevClamped =
                  Math.abs(kappa) > MAX_CURVATURE ||
                  Math.abs(yawRateFrom(v, kappa)) >= MAX_YAW_RATE - 1e-9;
                if (bevClamped) {
                  return (
                    <span className="text-amber-600"> · non-physical (clamped in BEV)</span>
                  );
                }
                if (Math.abs(yaw) > MAX_YAW_RATE) {
                  return (
                    <span className="text-amber-600"> · yaw_rate non-physical</span>
                  );
                }
                return null;
              })()}
            </p>
          </div>
        </div>
      </div>

      <TimelineScrubber
        samples={index.samples}
        fps={index.fps || 10}
        frame={frame}
        onSeek={setFrame}
        onScrubStart={pause}
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
        {/* Group the buffering chip and Shortcuts button under one ml-auto so
            the chip toggling does not reflow (shift) the Shortcuts button. */}
        <div className="ml-auto flex items-center gap-2">
          {buffering && (
            <span
              className="flex items-center gap-1 rounded bg-amber-950/60 px-2 py-0.5 font-mono text-[10px] text-amber-400"
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
            className="flex items-center gap-1 rounded bg-slate-900 px-2 py-1 font-mono text-[11px] text-slate-400 transition-colors hover:text-slate-200"
          >
            <Keyboard className="size-3.5" />
            Shortcuts (?)
          </button>
        </div>
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
