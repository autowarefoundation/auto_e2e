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
import {
  OverlaySelectionBar,
  type OverlayLoadStatus,
} from "@/components/player/overlay-selection-bar";
import { SceneMap } from "@/components/player/scene-map";
import { TimelineScrubber } from "@/components/player/timeline-scrubber";
import { TrajectoryBEV } from "@/components/player/trajectory-bev";
import { ReasoningTimeline } from "@/components/reasoning-timeline";
import { Button } from "@/components/ui/button";
import { usePlayback, MAX_SPEED, MIN_SPEED } from "@/hooks/use-playback";
import {
  ApiError,
  getReasoningLabel,
  getShardRigProjection,
  getShardOverlay,
  listShardOverlayModels,
} from "@/lib/api";
import { FrameStore } from "@/lib/frame-store";
import {
  integrateInterleavedControl,
  MAX_YAW_RATE,
  MAX_CURVATURE,
  trajectoryCurvatureSign,
  yawRateFrom,
} from "@/lib/ego";
import type {
  TrajectoryDisplayMode,
  TrajectoryPoint,
} from "@/lib/ego";
import {
  controlsForRow,
  parseOverlay,
  resolveOverlayRows,
} from "@/lib/overlay";
import type { OverlayArtifact } from "@/lib/overlay";
import {
  projectTrajectoriesToCameras,
  projectTrajectoryToCameras,
} from "@/lib/projection";
import type {
  OverlayModel,
  ReasoningLabelRecord,
  RigProjectionDocument,
  ShardIndex,
} from "@/types";

const SPEED_STEPS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16];
const SPACE_OWNING_ELEMENTS = [
  "a[href]",
  "button",
  "input",
  "select",
  "textarea",
  "summary",
  "audio[controls]",
  "video[controls]",
  '[contenteditable]:not([contenteditable="false"])',
  '[role="button"]',
  '[role="checkbox"]',
  '[role="combobox"]',
  '[role="link"]',
  '[role="menuitem"]',
  '[role="option"]',
  '[role="radio"]',
  '[role="searchbox"]',
  '[role="slider"]',
  '[role="spinbutton"]',
  '[role="switch"]',
  '[role="tab"]',
  '[role="textbox"]',
  '[role="treeitem"]',
].join(",");

type LabelState = { key: string; label: ReasoningLabelRecord } | null;

export interface PlayerViewState {
  frame: number;
  cam: number;
  mode: "grid" | "focus";
  speed: number;
  model: string;
  predictionMode: TrajectoryDisplayMode;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function medianTrajectory(paths: TrajectoryPoint[][]): TrajectoryPoint[] {
  if (paths.length === 0) return [];
  const steps = Math.min(...paths.map((path) => path.length));
  const result = new Array<TrajectoryPoint>(steps);
  for (let step = 0; step < steps; step++) {
    result[step] = {
      x: median(paths.map((path) => path[step].x)),
      y: median(paths.map((path) => path[step].y)),
      heading: median(paths.map((path) => path[step].heading)),
    };
  }
  return result;
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
  teacher,
  promptVersion,
}: {
  dataset: string;
  shard: string;
  index: ShardIndex;
  initialState?: Partial<PlayerViewState>;
  onViewStateChange?: (state: PlayerViewState) => void;
  version?: string;
  teacher?: string;
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

  const [overlayModels, setOverlayModels] = useState<OverlayModel[]>([]);
  const [selectedModelID, setSelectedModelID] = useState(
    initialState?.model ?? "",
  );
  const [predictionMode, setPredictionMode] =
    useState<TrajectoryDisplayMode>(
      initialState?.predictionMode ?? "raw",
    );
  const [overlayStatus, setOverlayStatus] =
    useState<OverlayLoadStatus>("loading-models");
  const [overlay, setOverlay] = useState<OverlayArtifact | null>(null);
  const [overlayRows, setOverlayRows] = useState<Map<string, number>>(
    new Map(),
  );
  const [rigProjection, setRigProjection] =
    useState<RigProjectionDocument | null>(null);

  useEffect(() => {
    let cancelled = false;
    setOverlayStatus("loading-models");
    setOverlayModels([]);
    setOverlay(null);
    setOverlayRows(new Map());
    listShardOverlayModels(dataset, shard, version)
      .then((response) => {
        if (cancelled) return;
        const models = response.models ?? [];
        setOverlayModels(models);
        if (models.length === 0) {
          setSelectedModelID("");
          setOverlayStatus("no-models");
          return;
        }
        setSelectedModelID((current) =>
          models.some((model) => model.model_artifact_id === current)
            ? current
            : models[0].model_artifact_id,
        );
        setOverlayStatus("idle");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        console.warn("trajectory model listing failed", err);
        setOverlayStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [dataset, shard, version]);

  useEffect(() => {
    let cancelled = false;
    setRigProjection(null);
    getShardRigProjection(dataset, shard, version)
      .then((projection) => {
        if (!cancelled) setRigProjection(projection);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (!(err instanceof ApiError && err.status === 404)) {
          console.warn("rig projection fetch failed", err);
        }
        setRigProjection(null);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset, shard, version]);

  useEffect(() => {
    if (!selectedModelID) return;
    let cancelled = false;
    setOverlayStatus("loading-overlay");
    setOverlay(null);
    setOverlayRows(new Map());
    getShardOverlay(dataset, shard, selectedModelID, version)
      .then((buffer) => {
        const parsed = parseOverlay(buffer);
        return resolveOverlayRows(
          parsed,
          index.samples.map((entry) => entry.sample_uid),
        ).then((rows) => ({ parsed, rows }));
      })
      .then(({ parsed, rows }) => {
        if (cancelled) return;
        setOverlay(parsed);
        setOverlayRows(rows);
        setOverlayStatus("ready");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        console.warn("trajectory overlay fetch failed", err);
        setOverlayStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [dataset, shard, selectedModelID, version, index.samples]);

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

  // Live playhead + speed for the lookahead tick's setInterval closure, which
  // would otherwise capture a stale `frame`/`speed`.
  const frameRef = useRef(frame);
  frameRef.current = frame;
  const speedRef = useRef(speed);
  speedRef.current = speed;

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
    onViewStateChange?.({
      frame,
      cam: focusCam,
      mode,
      speed,
      model: selectedModelID,
      predictionMode,
    });
  }, [
    frame,
    focusCam,
    mode,
    speed,
    selectedModelID,
    predictionMode,
    onViewStateChange,
  ]);

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

  // Buffer-health-driven refill: while playing, keep the next few window buffers
  // fetched ahead of the playhead on a steady tick. The prefetch effect above
  // fires only on `frame` change, but the buffer-gated clock FREEZES `frame`
  // when the buffer starves — so without this the refill would never re-fire and
  // playback stalls (measured: 2.3fps, buffer draining faster than it refilled).
  // A steady tick keeps MAX_INFLIGHT window GETs saturated regardless of whether
  // the clock is advancing. Runs only during playback (no steady-state cost).
  useEffect(() => {
    if (!store || !playing) return;
    const tick = () => {
      const f = frameRef.current;
      // Fetch the next window buffers (cheap, no decode) AND decode the visible
      // cameras ahead of the playhead (prefetch → getFrame). Buffer-only refill
      // is not enough: the clock gates on DECODED frames, so an arrived-but-
      // undecoded window still stalls it — the decode pressure must also be kept
      // up on the live playhead, not just on frame-change.
      store.ensureLookahead(f, direction);
      store.prefetch(f, direction, speedRef.current, visibleCamsRef.current);
    };
    tick();
    const id = setInterval(tick, 200);
    return () => clearInterval(id);
  }, [store, playing, direction]);

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
  const curvatureSign = trajectoryCurvatureSign(dataset);
  const predictionTrajectories = useMemo(() => {
    if (!overlay || !sample) return [];
    const row = overlayRows.get(sample.sample_uid);
    if (row === undefined) return [];
    const paths = new Array<TrajectoryPoint[]>(overlay.seedCount);
    for (let seed = 0; seed < overlay.seedCount; seed++) {
      paths[seed] = integrateInterleavedControl(
        overlay.v0[row],
        controlsForRow(overlay, row, seed),
        0.1,
        predictionMode,
        curvatureSign,
      );
    }
    return paths;
  }, [overlay, overlayRows, sample, predictionMode, curvatureSign]);
  const medianPrediction = useMemo(
    () => medianTrajectory(predictionTrajectories),
    [predictionTrajectories],
  );
  const predictionFan = useMemo(
    () => (predictionTrajectories.length > 1 ? predictionTrajectories : []),
    [predictionTrajectories],
  );
  const cameraPredictionPaths = useMemo(
    () => projectTrajectoriesToCameras(rigProjection, predictionFan),
    [rigProjection, predictionFan],
  );
  const cameraMedianPredictionPaths = useMemo(
    () => projectTrajectoryToCameras(rigProjection, medianPrediction),
    [rigProjection, medianPrediction],
  );
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
      getReasoningLabel(dataset, key, promptVersion, version, teacher)
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
  }, [
    dataset,
    version,
    teacher,
    promptVersion,
    sample?.key,
    sample?.has_reasoning,
    sample,
  ]);

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

  // Bind shortcuts at the window level so they work outside the player too.
  // Text-entry controls retain every key, while semantic interactive elements
  // retain Space for their native activation instead of also toggling playback.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null;
      if (
        t?.closest(
          'input, select, textarea, [contenteditable]:not([contenteditable="false"])',
        )
      ) {
        return;
      }
      // Never hijack a browser/OS accelerator: if a command/control/alt
      // modifier is held (e.g. Cmd+R reload, Ctrl+L address bar), let the
      // browser handle it. Shift is allowed — none of our keys use it, and
      // Shift+'?' is how '?' is typed on many layouts.
      if (e.metaKey || e.ctrlKey || e.altKey) return;
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
          if (t?.closest(SPACE_OWNING_ELEMENTS)) return;
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
      <p
        role="status"
        aria-live="polite"
        className="text-sm text-slate-500"
      >
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
      <OverlaySelectionBar
        models={overlayModels}
        selectedModelID={selectedModelID}
        onSelectModel={setSelectedModelID}
        displayMode={predictionMode}
        onDisplayModeChange={setPredictionMode}
        status={overlayStatus}
        baseSeeds={overlay?.baseSeeds ?? []}
        splitBucket={sample?.split_bucket}
      />
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
          predictionPaths={cameraPredictionPaths}
          medianPredictionPaths={cameraMedianPredictionPaths}
        />
        <div className="space-y-3">
          <TrajectoryBEV
            samples={index.samples}
            frame={frame}
            fps={index.fps || 10}
            reasoning={
              reasoning?.key === sample?.key ? reasoning.label : null
            }
            predictionTrajectories={predictionFan}
            medianPrediction={medianPrediction}
            curvatureSign={curvatureSign}
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

      <SceneMap
        dataset={dataset}
        version={version}
        sample={sample}
        predictionTrajectories={predictionFan}
        medianPrediction={medianPrediction}
        curvatureSign={curvatureSign}
      />

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
              role="status"
              aria-live="polite"
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
          <p
            role="status"
            aria-live="polite"
            className="text-sm text-slate-500"
          >
            Loading label…
          </p>
        ) : labelStatus === "error" ? (
          <p role="alert" className="text-sm text-amber-500">
            Could not load the label for this frame. It may retry on the next
            step.
          </p>
        ) : (
          <p
            role="status"
            aria-live="polite"
            className="text-sm text-slate-500"
          >
            No reasoning label at this frame
            {promptVersion ? " for the selected teacher and prompt version" : ""}. Amber
            ticks on the timeline mark frames labelled in any run, so a ticked
            frame may still be unlabelled in this one.
          </p>
        )}
      </div>
    </div>
  );
}
