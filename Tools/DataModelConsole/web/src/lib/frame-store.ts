// FrameStore: random-access JPEG frame source over a WebDataset shard.
//
// A shard is a tar of per-frame JPEGs. The shard index gives a presigned URL
// for the whole tar plus byte ranges per member, so any frame/camera can be
// fetched with a single HTTP Range request and decoded to an ImageBitmap.
// An LRU cache (bitmaps are GPU-resident, so bounded and close()d on
// eviction) plus a direction-aware look-ahead ring makes 10Hz playback
// smooth.

import type { IndexSample, ShardIndex } from "@/types";

const DEFAULT_MAX_ENTRIES = 500;
const MAX_INFLIGHT = 24;
const PREFETCH_BEHIND = 4;

export class FrameStore {
  private readonly index: ShardIndex;
  private readonly byFrame = new Map<number, IndexSample>();
  // Map iteration order = insertion order; entries are re-inserted on access
  // so the first key is always the least recently used.
  private readonly cache = new Map<string, ImageBitmap>();
  private readonly inflight = new Map<string, Promise<ImageBitmap>>();
  private readonly maxEntries: number;
  private destroyed = false;

  constructor(index: ShardIndex, maxEntries = DEFAULT_MAX_ENTRIES) {
    this.index = index;
    this.maxEntries = maxEntries;
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

  has(frameIdx: number, cam: string): boolean {
    return this.cache.has(`${frameIdx}:${cam}`);
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

    const p = this.fetchBitmap(frameIdx, cam)
      .then((bmp) => {
        this.inflight.delete(key);
        if (this.destroyed) {
          bmp.close();
          throw new Error("FrameStore destroyed");
        }
        this.put(key, bmp);
        return bmp;
      })
      .catch((err: unknown) => {
        this.inflight.delete(key);
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

  // destroy closes every cached bitmap. Pending fetches close their bitmaps
  // on arrival (see getFrame).
  destroy(): void {
    this.destroyed = true;
    for (const bmp of this.cache.values()) bmp.close();
    this.cache.clear();
    this.inflight.clear();
  }

  private put(key: string, bmp: ImageBitmap): void {
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
  ): Promise<ImageBitmap> {
    const sample = this.sampleAt(frameIdx);
    if (!sample) throw new Error(`no sample at frame ${frameIdx}`);
    const member = sample.members[`${cam}.jpg`];
    if (!member) throw new Error(`no member ${cam}.jpg in ${sample.key}`);

    const res = await fetch(this.index.presigned_tar_url, {
      headers: {
        Range: `bytes=${member.offset}-${member.offset + member.size - 1}`,
      },
    });
    if (!res.ok) {
      throw new Error(`range fetch failed: ${res.status} ${res.statusText}`);
    }
    const blob = await res.blob();
    return createImageBitmap(blob);
  }
}
