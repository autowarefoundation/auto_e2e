"use client";

// useCarryForwardLabel: keeps a reasoning label on screen continuously during
// playback. Labels attach at ~1 Hz while video runs at 10 Hz, so instead of
// showing a label only on its own frame (blanking the other ~9), this fetches
// the sparse anchor frames lazily into a scope-keyed cache and always displays
// the nearest present label at or before the playhead — carry-forward.
//
// The selection + fetch-target logic is the pure core in
// lib/reasoning-carry-forward; this hook owns the cache, the fetches, and the
// React state.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, getReasoningLabel } from "@/lib/api";
import {
  anchorFramesOf,
  carryForwardFetchTargets,
  selectCarryForwardLabel,
  type CacheEntry,
  type CarrySelection,
} from "@/lib/reasoning-carry-forward";
import type { IndexSample, ReasoningLabelRecord } from "@/types";

export interface CarryForwardLabel extends CarrySelection<ReasoningLabelRecord> {
  // Frames elapsed since the displayed label's anchor (>= 0; 0 on the anchor
  // frame). Zero when there is no displayed label.
  elapsedFrames: number;
  elapsedSec: number;
}

export function useCarryForwardLabel({
  samples,
  frame,
  fps,
  dataset,
  version,
  teacher,
  promptVersion,
}: {
  samples: IndexSample[];
  frame: number;
  fps: number;
  dataset: string;
  version?: string;
  teacher?: string;
  promptVersion?: string;
}): CarryForwardLabel {
  const cacheRef = useRef(new Map<string, CacheEntry<ReasoningLabelRecord>>());
  // Fetch-generation token: bumped whenever the scope changes so in-flight
  // responses resolving after a scope switch are discarded instead of poisoning
  // the fresh cache with cross-scope labels.
  const scopeRef = useRef(0);
  // Re-render trigger when a fetch resolves (the cache is a ref, not state).
  // `tick` is a selector dependency so a resolution re-selects even while
  // paused, when `frame` is not changing to drive the recompute on its own.
  const [tick, setTick] = useState(0);
  const rerender = useCallback(() => setTick((t) => t + 1), []);

  const anchorFrames = useMemo(() => anchorFramesOf(samples), [samples]);

  const keyForFrame = useCallback(
    (f: number) => samples[f]?.key,
    [samples],
  );

  // Reset on scope change: clear the cache, invalidate in-flight fetches, and
  // re-render so the panel falls back to loading/none under the new scope. A new
  // Map identity (not .clear()) keeps any captured closure from mutating the old
  // one after the swap.
  useEffect(() => {
    cacheRef.current = new Map();
    scopeRef.current += 1;
    rerender();
  }, [dataset, version, teacher, promptVersion, rerender]);

  // Computed inline (not memoized) because it reads the mutable cacheRef: the
  // component re-renders on every `frame` change and on every `tick` bump (a
  // fetch resolving), which are exactly the moments the selection can change, so
  // a fresh scan each render is both correct and cheap — the backward walk is
  // over the sparse ~1Hz anchors, not every frame. `tick` is referenced so the
  // dependency is explicit to readers even though the value isn't used directly.
  void tick;
  const selection = selectCarryForwardLabel(
    anchorFrames,
    frame,
    keyForFrame,
    cacheRef.current,
  );

  // Fetch driver: at most one backward (fill the label to show) + one forward
  // (look-ahead so the next anchor swaps with no gap) fetch per frame move.
  // Cache-status dedupe throttles it — no debounce timer, which is correct under
  // continuous 10 Hz playback where `frame` changes every tick.
  useEffect(() => {
    const cache = cacheRef.current;
    const { backward, forward } = carryForwardFetchTargets(
      anchorFrames,
      frame,
      keyForFrame,
      cache,
    );
    const targets = [backward, forward].filter(
      (f): f is number => f !== undefined,
    );
    if (targets.length === 0) return;

    const myScope = scopeRef.current;
    for (const af of targets) {
      const key = keyForFrame(af);
      if (!key || cache.get(key)) continue; // already loading/present/absent
      cache.set(key, { status: "loading" });
      getReasoningLabel(dataset, key, promptVersion, version, teacher)
        .then((label) => {
          if (scopeRef.current !== myScope) return; // scope changed mid-flight
          cacheRef.current.set(key, { status: "present", label });
          rerender();
        })
        .catch((err: unknown) => {
          if (scopeRef.current !== myScope) return;
          if (err instanceof ApiError && err.status === 404) {
            // Not labelled in this scope: negative-cache so the selector skips
            // it and the playhead crossing it does not blank the panel.
            cacheRef.current.set(key, { status: "absent" });
          } else {
            // Transient failure: drop back to uncached so the next frame move
            // retries, instead of dead-ending on an error state.
            cacheRef.current.delete(key);
            console.warn("reasoning label fetch failed", err);
          }
          rerender();
        });
    }
    // A resolution mutates cacheRef and calls rerender(); the parent re-renders
    // and this effect re-runs (frame typically also changed), re-planning the
    // next fetch target. Scope inputs re-run it when the partition changes.
  }, [
    anchorFrames,
    frame,
    keyForFrame,
    dataset,
    version,
    teacher,
    promptVersion,
    rerender,
  ]);

  const elapsedFrames =
    selection.label && selection.anchorFrame >= 0
      ? Math.max(0, frame - selection.anchorFrame)
      : 0;

  return {
    ...selection,
    elapsedFrames,
    elapsedSec: elapsedFrames / (fps || 10),
  };
}
