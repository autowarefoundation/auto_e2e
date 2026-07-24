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
  // responses resolving after a scope switch are discarded instead of writing a
  // stale entry into the freshly-cleared cache.
  const scopeRef = useRef(0);
  // Re-render trigger when a fetch resolves (the cache is a ref, not state).
  // `tick` is a dependency of BOTH the selector and the fetch effect: a
  // resolution must re-select the shown label AND re-plan the next fetch even
  // while paused, when `frame` is not advancing to drive either on its own.
  const [tick, setTick] = useState(0);
  const rerender = useCallback(() => setTick((t) => t + 1), []);

  const anchorFrames = useMemo(() => anchorFramesOf(samples), [samples]);

  // The cache is keyed by scope + sample so a render that happens BEFORE the
  // scope-reset effect commits cannot read a prior scope's label (React runs
  // render before effects): under the new scope the lookup key differs and
  // misses, instead of momentarily painting the old teacher/prompt's label.
  const scopeKey = useMemo(
    () => [dataset, version ?? "", teacher ?? "", promptVersion ?? ""].join(" "),
    [dataset, version, teacher, promptVersion],
  );
  // Raw WebDataset sample key (for the API call) vs the scope-qualified cache
  // key (for the Map). The pure core treats keyForFrame as an opaque cache key,
  // so it gets the scope-qualified one.
  const sampleKeyForFrame = useCallback(
    (f: number) => samples[f]?.key,
    [samples],
  );
  const cacheKeyForFrame = useCallback(
    (f: number) => {
      const k = samples[f]?.key;
      return k === undefined ? undefined : `${scopeKey} ${k}`;
    },
    [samples, scopeKey],
  );

  // Reset on scope change: drop the old scope's entries (memory) and invalidate
  // in-flight fetches via the generation token. Correctness of what's SHOWN does
  // not depend on this effect's timing — the scope-qualified keys already make a
  // pre-commit render miss the old scope — so this is pure hygiene.
  useEffect(() => {
    cacheRef.current = new Map();
    scopeRef.current += 1;
    rerender();
  }, [scopeKey, rerender]);

  // Computed inline (not memoized) because it reads the mutable cacheRef: the
  // component re-renders on every `frame` change and every `tick` bump (a fetch
  // resolving), which are exactly the moments the selection can change, so a
  // fresh scan each render is both correct and cheap — the backward walk is over
  // the sparse ~1Hz anchors, not every frame.
  const selection = selectCarryForwardLabel(
    anchorFrames,
    frame,
    cacheKeyForFrame,
    cacheRef.current,
  );

  // Fetch driver: one backward (the label to show) + one forward (look-ahead so
  // the next anchor swaps with no gap) fetch per invocation. It re-runs on every
  // `frame` change AND every `tick` bump, so the backward walk past scoped-404
  // (absent) anchors advances one step per resolution even while PAUSED — not
  // only while `frame` is ticking. Cache-status dedupe throttles it (no timer).
  useEffect(() => {
    const cache = cacheRef.current;
    const { backward, forward } = carryForwardFetchTargets(
      anchorFrames,
      frame,
      cacheKeyForFrame,
      cache,
    );
    const targets = [backward, forward].filter(
      (f): f is number => f !== undefined,
    );
    if (targets.length === 0) return;

    const myScope = scopeRef.current;
    for (const af of targets) {
      const cacheKey = cacheKeyForFrame(af);
      const sampleKey = sampleKeyForFrame(af);
      if (!cacheKey || !sampleKey || cache.get(cacheKey)) continue; // deduped
      cache.set(cacheKey, { status: "loading" });
      getReasoningLabel(dataset, sampleKey, promptVersion, version, teacher)
        .then((label) => {
          if (scopeRef.current !== myScope) return; // scope changed mid-flight
          cacheRef.current.set(cacheKey, { status: "present", label });
          rerender();
        })
        .catch((err: unknown) => {
          if (scopeRef.current !== myScope) return;
          if (err instanceof ApiError && err.status === 404) {
            // Not labelled in this scope: negative-cache so the selector skips
            // it and the playhead crossing it does not blank the panel.
            // rerender() so the tick-keyed effect re-runs and the backward walk
            // advances to the next older anchor even while paused.
            cacheRef.current.set(cacheKey, { status: "absent" });
            rerender();
          } else {
            // Transient failure: drop back to uncached so a later fetch retries.
            // Do NOT rerender here — with `tick` in the effect deps that would
            // re-run the effect and immediately re-fetch the same key, hammering
            // a persistent 5xx in a tight loop while paused. The display is
            // unchanged (loading -> uncached both read as pending), so the retry
            // waits for the next frame move instead.
            cacheRef.current.delete(cacheKey);
            console.warn("reasoning label fetch failed", err);
          }
        });
    }
  }, [
    anchorFrames,
    frame,
    tick,
    cacheKeyForFrame,
    sampleKeyForFrame,
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
