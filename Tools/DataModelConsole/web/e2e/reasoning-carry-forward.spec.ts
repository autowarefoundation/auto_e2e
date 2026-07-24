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
