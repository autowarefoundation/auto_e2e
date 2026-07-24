import { expect, test } from "@playwright/test";

import {
  anchorFramesOf,
  carryForwardFetchTargets,
  selectCarryForwardLabel,
  type CacheEntry,
} from "../src/lib/reasoning-carry-forward";

// A minimal label stand-in; the selector only cares about identity/presence.
type Label = { id: string };

// Build a cache keyed by frame->key = `f${frame}`. `spec` maps a frame to its
// entry so tests read declaratively.
function cacheOf(
  spec: Record<number, CacheEntry<Label>>,
): Map<string, CacheEntry<Label>> {
  const m = new Map<string, CacheEntry<Label>>();
  for (const [frame, entry] of Object.entries(spec)) {
    m.set(`f${frame}`, entry);
  }
  return m;
}
const keyForFrame = (f: number) => `f${f}`;

test("anchorFramesOf returns ascending has_reasoning indices", () => {
  const samples = [
    { has_reasoning: true },
    { has_reasoning: false },
    { has_reasoning: false },
    { has_reasoning: true },
    { has_reasoning: false },
  ];
  expect(anchorFramesOf(samples)).toEqual([0, 3]);
});

test("carry-forward holds the last present label between 1Hz anchors", () => {
  const anchors = [0, 10, 20];
  const cache = cacheOf({ 0: { status: "present", label: { id: "a0" } } });
  // Frames 0..9 all before the next anchor at 10: label a0 carries forward.
  for (const frame of [0, 1, 5, 9]) {
    const sel = selectCarryForwardLabel(anchors, frame, keyForFrame, cache);
    expect(sel.label?.id).toBe("a0");
    expect(sel.anchorFrame).toBe(0);
    expect(sel.isAnchorFrame).toBe(frame === 0);
  }
});

test("a newer present anchor wins immediately once crossed", () => {
  const anchors = [0, 10, 20];
  const cache = cacheOf({
    0: { status: "present", label: { id: "a0" } },
    10: { status: "present", label: { id: "a10" } },
  });
  expect(selectCarryForwardLabel(anchors, 9, keyForFrame, cache).label?.id).toBe(
    "a0",
  );
  const at10 = selectCarryForwardLabel(anchors, 10, keyForFrame, cache);
  expect(at10.label?.id).toBe("a10");
  expect(at10.isAnchorFrame).toBe(true);
  expect(
    selectCarryForwardLabel(anchors, 15, keyForFrame, cache).label?.id,
  ).toBe("a10");
});

test("an absent (scoped 404) nearer anchor falls back to the older present one", () => {
  const anchors = [0, 10, 20];
  const cache = cacheOf({
    0: { status: "present", label: { id: "a0" } },
    10: { status: "absent" },
  });
  // Playhead past the absent anchor 10: keep showing a0 rather than blanking,
  // and it never reports pending for an absent anchor.
  const sel = selectCarryForwardLabel(anchors, 12, keyForFrame, cache);
  expect(sel.label?.id).toBe("a0");
  expect(sel.anchorFrame).toBe(0);
  expect(sel.pending).toBe(false);
});

test("no present label yet but an anchor is loading => pending (loading, not none)", () => {
  const anchors = [0, 10];
  const cache = cacheOf({ 0: { status: "loading" } });
  const sel = selectCarryForwardLabel(anchors, 3, keyForFrame, cache);
  expect(sel.label).toBeNull();
  expect(sel.pending).toBe(true);
});

test("before the first anchor there is nothing to carry and nothing pending", () => {
  const anchors = [10, 20];
  const sel = selectCarryForwardLabel(anchors, 5, keyForFrame, cacheOf({}));
  expect(sel.label).toBeNull();
  expect(sel.anchorFrame).toBe(-1);
  expect(sel.pending).toBe(false);
});

test("a future anchor's label never renders on an earlier frame", () => {
  const anchors = [0, 10];
  const cache = cacheOf({
    10: { status: "present", label: { id: "a10" } },
  });
  // At frame 5 the only present label is anchored at 10 (future): must not show.
  expect(selectCarryForwardLabel(anchors, 5, keyForFrame, cache).label).toBeNull();
});

test("fetch targets: closest uncached backward + immediate forward look-ahead", () => {
  const anchors = [0, 10, 20];
  // Nothing cached at frame 12: backward = closest anchor <= 12 (=10),
  // forward = immediate anchor > 12 (=20).
  const t = carryForwardFetchTargets(anchors, 12, keyForFrame, cacheOf({}));
  expect(t.backward).toBe(10);
  expect(t.forward).toBe(20);
});

test("fetch targets: stop backward scan at the first present fallback", () => {
  const anchors = [0, 10, 20];
  // Anchor 10 present, so no need to fetch anything older; frame 15 has no
  // forward-uncached beyond 20.
  const t = carryForwardFetchTargets(
    anchors,
    15,
    keyForFrame,
    cacheOf({ 10: { status: "present", label: { id: "a10" } } }),
  );
  expect(t.backward).toBeUndefined();
  expect(t.forward).toBe(20);
});

test("fetch targets: skip absent anchors backward to the next uncached", () => {
  const anchors = [0, 10, 20];
  // Anchor 20 absent (scoped 404), 10 uncached: at frame 22 the backward target
  // is 10 (skipping absent 20), no forward anchor exists.
  const t = carryForwardFetchTargets(
    anchors,
    22,
    keyForFrame,
    cacheOf({ 20: { status: "absent" } }),
  );
  expect(t.backward).toBe(10);
  expect(t.forward).toBeUndefined();
});

test("fetch targets: nothing to do when the nearest backward anchor is loading", () => {
  const anchors = [0, 10, 20];
  // 10 loading (in-flight), 20 present forward-of-frame irrelevant. At frame 12,
  // backward stops at the loading anchor (already in-flight) -> undefined.
  const t = carryForwardFetchTargets(
    anchors,
    12,
    keyForFrame,
    cacheOf({ 10: { status: "loading" } }),
  );
  expect(t.backward).toBeUndefined();
  expect(t.forward).toBe(20);
});

// driveToStable models the hook's fetch loop at a FIXED frame (i.e. paused):
// repeatedly plan a target, resolve it into the cache, and re-plan — exactly
// what the hook must do across resolutions. `labelAt` returns a label id for a
// present anchor or null for a scoped-404 (absent). Returns the number of
// backward fetches issued so a runaway (fetch-storm) shows up as a large count.
// This is the regression guard for the paused-stall bug: the pure contract must
// converge to a present fallback within the sparse anchor set, and the hook (via
// `tick` in its effect deps) must keep invoking it until it does.
function driveToStable(
  anchors: number[],
  frame: number,
  labelAt: (f: number) => string | null,
): { cache: Map<string, CacheEntry<Label>>; backwardFetches: number } {
  const cache = new Map<string, CacheEntry<Label>>();
  let backwardFetches = 0;
  // Bounded well above the anchor count; a correct loop converges in <= anchors.
  for (let guard = 0; guard < anchors.length + 5; guard++) {
    const { backward, forward } = carryForwardFetchTargets(
      anchors,
      frame,
      keyForFrame,
      cache,
    );
    if (backward === undefined && forward === undefined) break;
    for (const af of [backward, forward]) {
      if (af === undefined) continue;
      if (af === backward) backwardFetches++;
      const id = labelAt(af);
      cache.set(
        keyForFrame(af),
        id === null ? { status: "absent" } : { status: "present", label: { id } },
      );
    }
  }
  return { cache, backwardFetches };
}

test("paused backfill converges past scoped-404 anchors to the older present label", () => {
  // Regression for the paused-stall bug: anchors 50 and 40 are candidates
  // (has_reasoning any-run) but 404 in this scope; 30 has a label. Paused at 55,
  // the loop must walk 50 -> 40 -> 30 and end showing the label at 30.
  const anchors = [0, 10, 20, 30, 40, 50, 60];
  const present = new Set([0, 10, 20, 30, 60]); // 40, 50 are scoped-404
  const labelAt = (f: number) => (present.has(f) ? `a${f}` : null);

  const { cache, backwardFetches } = driveToStable(anchors, 55, labelAt);
  const sel = selectCarryForwardLabel(anchors, 55, keyForFrame, cache);
  expect(sel.label?.id).toBe("a30");
  expect(sel.anchorFrame).toBe(30);
  expect(sel.pending).toBe(false);
  // 50 (absent) + 40 (absent) + 30 (present) = 3 backward fetches, not the whole
  // prior-anchor set — the storm guard holds.
  expect(backwardFetches).toBe(3);
});

test("paused backfill reports none (not stuck loading) when no label precedes the playhead", () => {
  // All anchors <= frame are scoped-404: the loop must terminate with a settled
  // "none", never an indefinite pending/loading.
  const anchors = [10, 20, 30];
  const labelAt = () => null; // every anchor 404s in this scope
  const { cache } = driveToStable(anchors, 35, labelAt);
  const sel = selectCarryForwardLabel(anchors, 35, keyForFrame, cache);
  expect(sel.label).toBeNull();
  expect(sel.pending).toBe(false);
});
