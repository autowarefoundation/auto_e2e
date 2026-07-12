// FrameStore: random-access JPEG frame source over a WebDataset shard.
//
// Each frame/camera is fetched through the console API's image endpoint
// (getSampleImageUrl → /api/v1/.../image/cam_N), which is same-origin behind
// CloudFront in production (so no browser CORS and no presigned S3 URL leaked
// to the client) and CORS-allowed from the API in local dev. The API streams
// exactly one tar member; CloudFront caches the immutable JPEGs. An LRU cache
// (bitmaps are GPU-resident, so bounded and close()d on eviction) plus a
// direction-aware look-ahead ring makes 10Hz playback smooth.
//
// The shard index is still used for frame ordering, per-frame ego_now, and
// hazard markers; only the byte-range-against-presigned-tar fetch was dropped.

import { getSampleImageUrl } from "@/lib/api";
import type { IndexSample, ShardIndex } from "@/types";

const DEFAULT_MAX_ENTRIES = 500;
// Keep client concurrency at or below the server's image Throttle so prefetch
// bursts don't get 429'd; the server allows 64 concurrent image GETs, but a
// single player only needs a small look-ahead ring in flight at once.
const MAX_INFLIGHT = 8;
const PREFETCH_BEHIND = 4;

export class FrameStore {
  private readonly index: ShardIndex;
  private readonly dataset: string;
  private readonly shard: string;
  // Optional pinned dataset version; threaded onto every image URL so the
  // player renders the SAME version selected on the detail page (else the API
  // auto-resolves the newest).
  private readonly version?: string;
  private readonly byFrame = new Map<number, IndexSample>();
  // Map iteration order = insertion order; entries are re-inserted on access
  // so the first key is always the least recently used.
  private readonly cache = new Map<string, ImageBitmap>();
  private readonly inflight = new Map<string, Promise<ImageBitmap>>();
  // One AbortController per in-flight fetch so destroy() can cancel pending
  // network requests instead of leaving them to resolve into closed bitmaps.
  private readonly controllers = new Map<string, AbortController>();
  private readonly maxEntries: number;
  private destroyed = false;
  private active = 0;
  private readonly waiters: Array<() => void> = [];

  constructor(
    index: ShardIndex,
    dataset: string,
    shard: string,
    maxEntries = DEFAULT_MAX_ENTRIES,
    version?: string,
  ) {
    this.index = index;
    this.dataset = dataset;
    this.shard = shard;
    this.maxEntries = maxEntries;
    this.version = version;
    for (const s of index.samples) this.byFrame.set(s.frame_idx, s);
  }

  get frameCount(): number {
    return this.index.samples.length;
  }

  get fps(): number {
    return this.index.fps || 10;
  }

  // sampleAt resolves a playback position to its index entry. The playback
  // clock produces a 0..N-1 sequential position, so array position is the
  // authoritative lookup — this is robust even when sample keys carry a flat
  // s%08d global index (all frame_idx could otherwise collide). byFrame is
  // only a fallback for callers that pass a semantic frame_idx.
  sampleAt(pos: number): IndexSample | undefined {
    return this.index.samples[pos] ?? this.byFrame.get(pos);
  }

  // cachedCount returns how many of the next `n` frames (from `frame`, in
  // `dir`) already have ALL `cams` decoded in cache — used by the player to
  // gate playback start on a filled look-ahead buffer so it visibly plays
  // instead of stuttering while frames stream in.
  cachedCount(frame: number, dir: 1 | -1, n: number, cams: string[]): number {
    let ready = 0;
    for (let i = 0; i < n; i++) {
      const f = frame + i * dir;
      if (f < 0 || f >= this.frameCount) break;
      if (cams.every((c) => this.cache.has(`${f}:${c}`))) ready++;
      else break; // contiguous run only — a gap stalls playback there
    }
    return ready;
  }

  // withSlot gates work behind a single global inflight budget shared by the
  // draw-path getFrame and prefetch, so bursts never exceed the server's image
  // Throttle (and the browser's per-host socket limit). It queues instead of
  // firing, so a fast scrub can't unleash a connection storm.
  private async withSlot<T>(fn: () => Promise<T>): Promise<T> {
    if (this.active >= MAX_INFLIGHT) {
      await new Promise<void>((res) => this.waiters.push(res));
    }
    this.active++;
    try {
      return await fn();
    } finally {
      this.active--;
      this.waiters.shift()?.();
    }
  }

  // getFrame returns the decoded bitmap for (frame, cam), deduplicating
  // concurrent requests and populating the LRU cache.
  getFrame(frameIdx: number, cam: string): Promise<ImageBitmap> {
    const key = `${frameIdx}:${cam}`;
    const hit = this.cache.get(key);
    if (hit) {
      // bump recency
      this.cache.delete(key);
      this.cache.set(key, hit);
      return Promise.resolve(hit);
    }
    const pending = this.inflight.get(key);
    if (pending) return pending;

    const controller = new AbortController();
    this.controllers.set(key, controller);
    const p = this.withSlot(() =>
      this.fetchBitmap(frameIdx, cam, controller.signal),
    )
      .then((bmp) => {
        // Guard by identity: a superseded fetch must not delete a newer
        // same-key fetch's tracking (that would break dedup and lose the new
        // fetch's cancellation). Only clear entries still pointing at us.
        if (this.inflight.get(key) === p) this.inflight.delete(key);
        if (this.controllers.get(key) === controller)
          this.controllers.delete(key);
        if (this.destroyed) {
          bmp.close();
          throw new Error("FrameStore destroyed");
        }
        this.put(key, bmp);
        return bmp;
      })
      .catch((err: unknown) => {
        if (this.inflight.get(key) === p) this.inflight.delete(key);
        if (this.controllers.get(key) === controller)
          this.controllers.delete(key);
        throw err;
      });
    this.inflight.set(key, p);
    return p;
  }

  // prefetch warms the cache around the playhead: a look-ahead ring in the
  // playback direction (longer at higher speed) plus a short tail behind.
  prefetch(
    centerFrame: number,
    direction: 1 | -1,
    speed: number,
    cams: string[],
  ): void {
    if (this.destroyed || cams.length === 0) return;
    const ahead = Math.min(
      48,
      Math.max(8, Math.ceil(Math.max(speed, 0.1) * 12)),
    );
    // Nearest frames first so imminent draws win the bandwidth.
    const offsets: number[] = [];
    for (let d = 1; d <= ahead; d++) offsets.push(d * direction);
    for (let d = 1; d <= PREFETCH_BEHIND; d++) offsets.push(-d * direction);

    for (const off of offsets) {
      if (this.inflight.size >= MAX_INFLIGHT) return;
      const f = centerFrame + off;
      if (f < 0 || f >= this.frameCount) continue;
      for (const cam of cams) {
        if (this.inflight.size >= MAX_INFLIGHT) return;
        const key = `${f}:${cam}`;
        if (this.cache.has(key) || this.inflight.has(key)) continue;
        void this.getFrame(f, cam).catch(() => {
          // Prefetch failures are non-fatal; the draw path retries on demand.
        });
      }
    }
  }

  // abort releases the connection slot for a (frame,cam) that scrolled past
  // instead of letting a superseded fetch run to completion. A queued-but-not-
  // started fetch self-corrects: when its slot frees, fetchBitmap runs with an
  // already-aborted signal and rejects immediately. The rejected fetch's own
  // .catch also deletes these entries; a double-delete is harmless.
  abort(frameIdx: number, cam: string): void {
    const key = `${frameIdx}:${cam}`;
    this.controllers.get(key)?.abort();
    this.controllers.delete(key);
    this.inflight.delete(key);
  }

  // destroy closes every cached bitmap. Pending fetches close their bitmaps
  // on arrival (see getFrame).
  destroy(): void {
    this.destroyed = true;
    for (const controller of this.controllers.values()) controller.abort();
    this.controllers.clear();
    for (const bmp of this.cache.values()) bmp.close();
    this.cache.clear();
    this.inflight.clear();
  }

  private put(key: string, bmp: ImageBitmap): void {
    // A re-request of a key already in cache (e.g. abort → prefetch-behind)
    // would set() over the existing bitmap and orphan its GPU memory. Close
    // and drop the prior bitmap before inserting the new one.
    const prev = this.cache.get(key);
    if (prev && prev !== bmp) {
      prev.close();
      this.cache.delete(key);
    }
    while (this.cache.size >= this.maxEntries) {
      const oldest = this.cache.keys().next().value;
      if (oldest === undefined) break;
      this.cache.get(oldest)?.close();
      this.cache.delete(oldest);
    }
    this.cache.set(key, bmp);
  }

  private async fetchBitmap(
    frameIdx: number,
    cam: string,
    signal?: AbortSignal,
  ): Promise<ImageBitmap> {
    const sample = this.sampleAt(frameIdx);
    if (!sample) throw new Error(`no sample at frame ${frameIdx}`);
    const range = sample.members[`${cam}.jpg`];
    if (!range) {
      throw new Error(`no member ${cam}.jpg in ${sample.key}`);
    }

    // cam is "cam_N"; the API endpoint takes the numeric index. Passing the
    // known tar byte range lets the API serve it via a bounded S3 range GET.
    const camNum = Number(cam.replace(/^cam_/, ""));
    const url = getSampleImageUrl(
      this.dataset,
      this.shard,
      sample.key,
      camNum,
      range,
      this.version,
    );
    const res = await fetch(url, { signal });
    if (!res.ok) {
      throw new Error(`image fetch failed: ${res.status} ${res.statusText}`);
    }
    const blob = await res.blob();
    return createImageBitmap(blob);
  }
}
