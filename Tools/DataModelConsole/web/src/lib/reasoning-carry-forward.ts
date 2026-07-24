// Carry-forward selection for reasoning labels during playback.
//
// Reasoning labels attach at ~1 Hz (Platform label_stride=10: every frame where
// frame_idx % 10 == 0, plus each split group's first valid sample), while video
// plays at 10 Hz. So most frames have no label of their own. To QA label
// accuracy during smooth playback we keep the LAST-KNOWN label on screen and
// swap to a newer one the instant the playhead crosses its anchor — the panel
// never blanks between the ~1 Hz anchors.
//
// This module is the pure, framework-free core (selection + fetch-target
// planning) so it is unit-testable; the React wiring lives in the
// use-carry-forward-label hook.

// A frame carries a reasoning label member (candidate for fetching). The shard
// index's has_reasoning is derived from a packed reasoning.json and is NOT
// scoped to a selected (teacher, prompt_version), so a scoped fetch may still
// 404 — that negative result is cached as "absent" and skipped over.
export interface HasReasoningFrame {
  has_reasoning: boolean;
}

export type CacheStatus = "loading" | "present" | "absent";

// One anchor frame's fetch state. `label` is set only when status === "present".
// A transient (non-404) fetch error is NOT represented here: the hook clears the
// key back to uncached so it retries, rather than dead-ending on an error state.
export interface CacheEntry<Label> {
  status: CacheStatus;
  label?: Label;
}

export interface CarrySelection<Label> {
  // The label to display (nearest present at or before the playhead), or null.
  label: Label | null;
  // The frame the displayed label was anchored to (-1 when label is null).
  anchorFrame: number;
  // True when the displayed label's anchor IS the current frame (elapsed 0).
  // Drives whether the BEV may pin horizon dots (they are only geometrically
  // valid on the anchor frame's own plan).
  isAnchorFrame: boolean;
  // No present label at/before the playhead yet, but an anchor <= frame is still
  // loading or not-yet-fetched — so the correct UI is "loading", not "none".
  pending: boolean;
}

// anchorFramesOf returns the ascending frame indices that carry a reasoning
// label member (the candidate anchors).
export function anchorFramesOf(samples: HasReasoningFrame[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < samples.length; i++) {
    if (samples[i].has_reasoning) out.push(i);
  }
  return out;
}

// selectCarryForwardLabel picks the nearest cached-present label at or before
// `frame`, scanning anchors newest-first so a newer label always wins and an
// out-of-scope (absent) or still-loading nearer anchor is skipped in favour of
// an older present one — the carry-forward. It considers only anchors <= frame,
// so a future anchor's label can never render on an earlier frame (this
// structurally removes the "in-flight response for a prior frame renders on the
// current one" class of bug the old effect guarded against by hand).
export function selectCarryForwardLabel<Label>(
  anchorFrames: number[],
  frame: number,
  keyForFrame: (f: number) => string | undefined,
  cache: Map<string, CacheEntry<Label>>,
): CarrySelection<Label> {
  let pending = false;
  for (let i = anchorFrames.length - 1; i >= 0; i--) {
    const af = anchorFrames[i];
    if (af > frame) continue;
    const key = keyForFrame(af);
    const entry = key ? cache.get(key) : undefined;
    if (entry?.status === "present" && entry.label != null) {
      return {
        label: entry.label,
        anchorFrame: af,
        isAnchorFrame: af === frame,
        pending: false,
      };
    }
    // The nearest actionable anchor is still resolving (or not yet fetched):
    // remember we're pending, but keep scanning older anchors for a present
    // fallback so playback shows the last-known label instead of blanking.
    if (!entry || entry.status === "loading") pending = true;
    // status "absent" (scoped 404) → skip and continue to the older anchor.
  }
  return { label: null, anchorFrame: -1, isAnchorFrame: false, pending };
}

// carryForwardFetchTargets returns which anchors the driver should fetch now:
//  - backward: the closest uncached anchor <= frame, stopping the scan once a
//    present anchor is reached (a fresher fallback already exists closer to the
//    playhead, so older-than-present anchors are irrelevant). Absent anchors are
//    skipped so a scoped 404 falls through to the next real label.
//  - forward: the immediate next anchor > frame, if uncached, so crossing it
//    swaps with no loading gap (look-ahead prefetch).
// Anchors already loading/present/absent are never re-fetched; that cache-status
// dedupe is the throttle (no timer needed, correct under continuous 10 Hz play).
export function carryForwardFetchTargets<Label>(
  anchorFrames: number[],
  frame: number,
  keyForFrame: (f: number) => string | undefined,
  cache: Map<string, CacheEntry<Label>>,
): { backward?: number; forward?: number } {
  let backward: number | undefined;
  for (let i = anchorFrames.length - 1; i >= 0; i--) {
    const af = anchorFrames[i];
    if (af > frame) continue;
    const key = keyForFrame(af);
    const entry = key ? cache.get(key) : undefined;
    if (entry?.status === "present") break; // fresher fallback exists; stop
    if (entry?.status === "absent" || entry?.status === "loading") continue;
    backward = af; // closest uncached anchor <= frame
    break;
  }

  let forward: number | undefined;
  for (let i = 0; i < anchorFrames.length; i++) {
    const af = anchorFrames[i];
    if (af <= frame) continue;
    const key = keyForFrame(af);
    if (key && !cache.get(key)) forward = af;
    break; // only the immediate next anchor > frame is prefetched
  }

  return { backward, forward };
}
