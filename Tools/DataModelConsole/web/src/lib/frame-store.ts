// FrameStore: random-access JPEG frame source over a WebDataset shard.
//
// Playback is gated by network round trips, not bandwidth: fetching each
// camera JPEG as its own range GET means ~6 requests per frame, and over a
// high-latency link (e.g. a browser far from the S3 region) the per-request
// latency — not throughput — caps the fill rate well below 10Hz. Parallelizing
// the tiny GETs makes it worse (TLS/slow-start contention).
//
// So instead we fetch ONE contiguous byte range covering a whole WINDOW of
// consecutive frames' camera members in a single request (the /blob endpoint),
// then slice each JPEG out of the returned buffer using the per-member offsets
// from the shard index and decode on demand. One round trip is amortized across
// WINDOW_FRAMES × cameras, so the fill rate becomes bandwidth-bound (near real
// time) rather than latency-bound. Decoded bitmaps are GPU-resident, so they
// live in a bounded LRU and are close()d on eviction; the raw window buffers
// are held in a small separate LRU just long enough to decode from.

import { getShardBlobUrl } from "@/lib/api";
import type { IndexSample, ShardIndex } from "@/types";

const DEFAULT_MAX_ENTRIES = 500;
// Frames per contiguous fetch window. Larger windows amortize more round trips
// but delay the first draw and over-fetch on a scrub; 8 balances both.
const WINDOW_FRAMES = 8;
// Raw window buffers retained for slicing (each ~0.6-1MB). A few is enough to
// decode the current and adjacent windows; bitmaps are the real cache.
const MAX_BUFFERS = 4;
// Concurrent window GETs. These are large reads, so a handful is plenty and
// keeps us from re-introducing the many-tiny-connections contention.
const MAX_INFLIGHT = 3;

interface WindowBuffer {
  buf: ArrayBuffer;
  spanStart: number;
}

export class FrameStore {
  private readonly index: ShardIndex;
  private readonly dataset: string;
  private readonly shard: string;
  // Optional pinned dataset version; threaded onto every fetch so the player
  // renders the SAME version selected on the detail page (else the API
  // auto-resolves the newest).
  private readonly version?: string;
  private readonly cams: string[];
  private readonly byFrame = new Map<number, IndexSample>();
  // Decoded bitmaps, keyed "frame:cam". Map iteration order = insertion order;
  // entries are re-inserted on access so the first key is the LRU victim.
  private readonly cache = new Map<string, ImageBitmap>();
  private readonly bitmapInflight = new Map<string, Promise<ImageBitmap>>();
  // Raw contiguous window buffers, keyed by window index, small LRU.
  private readonly buffers = new Map<number, WindowBuffer>();
  private readonly bufInflight = new Map<number, Promise<WindowBuffer>>();
  // One AbortController per in-flight window fetch so destroy() cancels pending
  // network requests instead of leaving them to resolve into a destroyed store.
  private readonly controllers = new Map<number, AbortController>();
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
    // Camera set is stable across the shard; take it from the first sample.
    const first = index.samples[0];
    this.cams = first
      ? Object.keys(first.members)
          .filter((m) => /^cam_\d+\.jpg$/.test(m))
          .map((m) => m.replace(/\.jpg$/, ""))
          .sort()
      : [];
  }

  get frameCount(): number {
    return this.index.samples.length;
  }

  get fps(): number {
    return this.index.fps || 10;
  }

  // sampleAt resolves a playback position to its index entry. The playback
  // clock produces a 0..N-1 sequential position, so array position is the
  // authoritative lookup — robust even when sample keys carry a flat s%08d
  // global index. byFrame is a fallback for callers passing a semantic frame_idx.
  sampleAt(pos: number): IndexSample | undefined {
    return this.index.samples[pos] ?? this.byFrame.get(pos);
  }

  // cachedCount returns how many of the next `n` frames (from `frame`, in
  // `dir`) already have ALL `cams` decoded in cache — the player gates its
  // playback clock on this so it advances only into drawable frames (smooth,
  // never stuttering ahead of the buffer).
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

  // withSlot gates work behind a single global inflight budget so concurrent
  // window fetches never exceed MAX_INFLIGHT (and queue instead of firing).
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

  private windowOf(frame: number): number {
    return Math.floor(frame / WINDOW_FRAMES);
  }

  // windowSpan computes the contiguous byte range covering every camera member
  // of every frame in the window, so any (frame, cam) in it can be sliced from
  // a single fetched buffer.
  private windowSpan(winIdx: number): { start: number; end: number } | null {
    const lo = winIdx * WINDOW_FRAMES;
    const hi = Math.min(this.frameCount, lo + WINDOW_FRAMES);
    let start = Infinity;
    let end = -Infinity;
    for (let f = lo; f < hi; f++) {
      const s = this.sampleAt(f);
      if (!s) continue;
      for (const cam of this.cams) {
        const m = s.members[`${cam}.jpg`];
        if (!m) continue;
        start = Math.min(start, m.offset);
        end = Math.max(end, m.offset + m.size);
      }
    }
    if (start === Infinity) return null;
    return { start, end };
  }

  // fetchBuffer fetches (once, deduplicated) the contiguous byte range for a
  // window and caches it for slicing.
  private fetchBuffer(winIdx: number): Promise<WindowBuffer> {
    const existing = this.buffers.get(winIdx);
    if (existing) return Promise.resolve(existing);
    const pend = this.bufInflight.get(winIdx);
    if (pend) return pend;
    const span = this.windowSpan(winIdx);
    if (!span) return Promise.reject(new Error(`empty window ${winIdx}`));

    const controller = new AbortController();
    this.controllers.set(winIdx, controller);
    const p = this.withSlot(async () => {
      const url = getShardBlobUrl(
        this.dataset,
        this.shard,
        span.start,
        span.end - span.start,
        this.version,
      );
      const res = await fetch(url, { signal: controller.signal });
      if (!res.ok) {
        throw new Error(`blob fetch failed: ${res.status} ${res.statusText}`);
      }
      const buf = await res.arrayBuffer();
      return { buf, spanStart: span.start };
    })
      .then((entry) => {
        if (this.bufInflight.get(winIdx) === p) this.bufInflight.delete(winIdx);
        if (this.controllers.get(winIdx) === controller)
          this.controllers.delete(winIdx);
        if (this.destroyed) throw new Error("FrameStore destroyed");
        this.putBuffer(winIdx, entry);
        return entry;
      })
      .catch((err: unknown) => {
        if (this.bufInflight.get(winIdx) === p) this.bufInflight.delete(winIdx);
        if (this.controllers.get(winIdx) === controller)
          this.controllers.delete(winIdx);
        throw err;
      });
    this.bufInflight.set(winIdx, p);
    return p;
  }

  // getFrame returns the decoded bitmap for (frame, cam): fetches the enclosing
  // window buffer (shared across all cameras/frames in it), slices out this
  // member's JPEG and decodes it. Concurrent requests are deduplicated.
  getFrame(frameIdx: number, cam: string): Promise<ImageBitmap> {
    const key = `${frameIdx}:${cam}`;
    const hit = this.cache.get(key);
    if (hit) {
      this.cache.delete(key);
      this.cache.set(key, hit);
      return Promise.resolve(hit);
    }
    const pending = this.bitmapInflight.get(key);
    if (pending) return pending;

    const winIdx = this.windowOf(frameIdx);
    const p = this.fetchBuffer(winIdx)
      .then(async (entry) => {
        const sample = this.sampleAt(frameIdx);
        const m = sample?.members[`${cam}.jpg`];
        if (!sample || !m) {
          throw new Error(`no member ${cam}.jpg at frame ${frameIdx}`);
        }
        const begin = m.offset - entry.spanStart;
        if (begin < 0 || begin + m.size > entry.buf.byteLength) {
          throw new Error(`member ${cam}.jpg out of window buffer range`);
        }
        const slice = entry.buf.slice(begin, begin + m.size);
        const bmp = await createImageBitmap(
          new Blob([slice], { type: "image/jpeg" }),
        );
        if (this.bitmapInflight.get(key) === p) this.bitmapInflight.delete(key);
        if (this.destroyed) {
          bmp.close();
          throw new Error("FrameStore destroyed");
        }
        this.put(key, bmp);
        return bmp;
      })
      .catch((err: unknown) => {
        if (this.bitmapInflight.get(key) === p) this.bitmapInflight.delete(key);
        throw err;
      });
    this.bitmapInflight.set(key, p);
    return p;
  }

  // prefetch warms the buffers and decodes the visible cameras for a look-ahead
  // run in the playback direction (longer at higher speed) plus a short tail
  // behind, so the draw path and cachedCount find frames ready.
  prefetch(
    centerFrame: number,
    direction: 1 | -1,
    speed: number,
    cams: string[],
  ): void {
    if (this.destroyed || cams.length === 0) return;
    const ahead = Math.min(
      WINDOW_FRAMES * 3,
      Math.max(WINDOW_FRAMES * 2, Math.ceil(Math.max(speed, 0.1) * 12)),
    );
    // Decode nearest frames first so imminent draws win the bandwidth. getFrame
    // deduplicates and shares the underlying window fetches.
    for (let d = 0; d <= ahead; d++) {
      const f = centerFrame + d * direction;
      if (f < 0 || f >= this.frameCount) break;
      for (const cam of cams) {
        const key = `${f}:${cam}`;
        if (this.cache.has(key) || this.bitmapInflight.has(key)) continue;
        void this.getFrame(f, cam).catch(() => {
          // Prefetch failures are non-fatal; the draw path retries on demand.
        });
      }
    }
    // A small tail behind for scrub-back / reverse.
    const behindWin = this.windowOf(centerFrame) - direction;
    if (
      behindWin >= 0 &&
      behindWin * WINDOW_FRAMES < this.frameCount &&
      !this.buffers.has(behindWin) &&
      !this.bufInflight.has(behindWin)
    ) {
      void this.fetchBuffer(behindWin).catch(() => {});
    }
  }

  // abort is a no-op under windowed fetching: fetches are window-scoped and
  // shared across every tile/frame in the window, so cancelling on one tile's
  // unmount would strand the others. Superseded windows are cheap to let finish
  // (they fill the cache for scrubbing); destroy() cancels everything.
  abort(): void {
    // intentionally empty — see doc comment
  }

  // destroy closes every cached bitmap and cancels pending fetches.
  destroy(): void {
    this.destroyed = true;
    for (const controller of this.controllers.values()) controller.abort();
    this.controllers.clear();
    for (const bmp of this.cache.values()) bmp.close();
    this.cache.clear();
    this.bitmapInflight.clear();
    this.buffers.clear();
    this.bufInflight.clear();
  }

  private putBuffer(winIdx: number, entry: WindowBuffer): void {
    this.buffers.delete(winIdx);
    this.buffers.set(winIdx, entry);
    while (this.buffers.size > MAX_BUFFERS) {
      const oldest = this.buffers.keys().next().value;
      if (oldest === undefined) break;
      this.buffers.delete(oldest);
    }
  }

  private put(key: string, bmp: ImageBitmap): void {
    // A re-request of a key already in cache would set() over the existing
    // bitmap and orphan its GPU memory. Close and drop the prior one first.
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
}
