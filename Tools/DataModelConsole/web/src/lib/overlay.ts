const MAGIC = "AOVL";
const HEADER_BYTES = 20;
const DIRECTORY_ENTRY_BYTES = 12;
const HORIZON = 64;
const DIMS = 2;
export const BEV_HEATMAP_SIZE = 32;
const LEGACY_BEV_HEATMAP_NAMES = ["image", "navigation", "fused"] as const;
export const BEV_HEATMAP_NAMES = [
  "image",
  "map",
  "route_delta",
  "navigation",
  "fusion_delta",
  "fused",
] as const;
export type BEVHeatmapName = (typeof BEV_HEATMAP_NAMES)[number];

export interface OverlayArtifact {
  formatVersion: number;
  flags: number;
  sampleCount: number;
  seedCount: number;
  horizon: number;
  dims: number;
  baseSeeds: bigint[];
  directory: Map<bigint, number>;
  controls: Float32Array;
  v0: Float32Array;
  bevHeatmaps: Uint8Array | null;
  bevHeatmapScales: Float32Array | null;
  bevHeatmapNames: readonly BEVHeatmapName[];
}

export function parseOverlay(buffer: ArrayBuffer): OverlayArtifact {
  if (buffer.byteLength < HEADER_BYTES) {
    throw new Error("Overlay is shorter than its header");
  }
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== MAGIC) {
    throw new Error(`Invalid overlay magic ${JSON.stringify(magic)}`);
  }

  const view = new DataView(buffer);
  const formatVersion = view.getUint16(4, true);
  const flags = view.getUint16(6, true);
  const sampleCount = view.getUint32(8, true);
  const seedCount = view.getUint16(12, true);
  const horizon = view.getUint16(14, true);
  const dims = view.getUint16(16, true);
  const reserved = view.getUint16(18, true);
  const bevHeatmapNames =
    formatVersion === 2
      ? LEGACY_BEV_HEATMAP_NAMES
      : formatVersion === 3
        ? BEV_HEATMAP_NAMES
        : [];
  const heatmapCount = bevHeatmapNames.length;
  if (
    (formatVersion !== 1 &&
      formatVersion !== 2 &&
      formatVersion !== 3) ||
    horizon !== HORIZON ||
    dims !== DIMS ||
    reserved !== heatmapCount
  ) {
    throw new Error("Unsupported overlay format");
  }
  if (sampleCount === 0 || seedCount === 0) {
    throw new Error("Overlay must contain samples and seeds");
  }

  const seedsOffset = HEADER_BYTES;
  const directoryOffset = seedsOffset + seedCount * 8;
  const controlsOffset =
    directoryOffset + sampleCount * DIRECTORY_ENTRY_BYTES;
  const controlsLength = sampleCount * seedCount * horizon * dims;
  const speedsOffset = controlsOffset + controlsLength * 4;
  const heatmapScalesOffset = speedsOffset + sampleCount * 4;
  const heatmapsOffset = heatmapScalesOffset + sampleCount * 4;
  const heatmapsLength =
    sampleCount *
    heatmapCount *
    BEV_HEATMAP_SIZE *
    BEV_HEATMAP_SIZE;
  const expectedBytes =
    heatmapCount > 0
      ? heatmapsOffset + heatmapsLength
      : heatmapScalesOffset;
  if (buffer.byteLength !== expectedBytes) {
    throw new Error(
      `Overlay size mismatch: expected ${expectedBytes}, got ${buffer.byteLength}`,
    );
  }

  const baseSeeds = new Array<bigint>(seedCount);
  for (let i = 0; i < seedCount; i++) {
    baseSeeds[i] = view.getBigInt64(seedsOffset + i * 8, true);
  }

  const directory = new Map<bigint, number>();
  const seenRows = new Set<number>();
  let previousHash = BigInt(-1);
  for (let i = 0; i < sampleCount; i++) {
    const offset = directoryOffset + i * DIRECTORY_ENTRY_BYTES;
    const hash = view.getBigUint64(offset, true);
    const row = view.getUint32(offset + 8, true);
    if (hash <= previousHash || row >= sampleCount || seenRows.has(row)) {
      throw new Error("Invalid overlay directory");
    }
    directory.set(hash, row);
    seenRows.add(row);
    previousHash = hash;
  }

  return {
    formatVersion,
    flags,
    sampleCount,
    seedCount,
    horizon,
    dims,
    baseSeeds,
    directory,
    controls: new Float32Array(buffer, controlsOffset, controlsLength),
    v0: new Float32Array(buffer, speedsOffset, sampleCount),
    bevHeatmaps:
      heatmapCount > 0
        ? new Uint8Array(buffer, heatmapsOffset, heatmapsLength)
        : null,
    bevHeatmapScales:
      heatmapCount > 0
        ? new Float32Array(buffer, heatmapScalesOffset, sampleCount)
        : null,
    bevHeatmapNames,
  };
}

export function controlsForRow(
  overlay: OverlayArtifact,
  row: number,
  seedIndex: number,
): Float32Array {
  if (
    row < 0 ||
    row >= overlay.sampleCount ||
    seedIndex < 0 ||
    seedIndex >= overlay.seedCount
  ) {
    throw new RangeError("Overlay row or seed is out of bounds");
  }
  const stride = overlay.horizon * overlay.dims;
  const begin = (row * overlay.seedCount + seedIndex) * stride;
  return overlay.controls.subarray(begin, begin + stride);
}

export function bevHeatmapForRow(
  overlay: OverlayArtifact,
  row: number,
  name: BEVHeatmapName,
): { values: Uint8Array; scale: number } | null {
  if (
    row < 0 ||
    row >= overlay.sampleCount ||
    overlay.bevHeatmaps === null ||
    overlay.bevHeatmapScales === null
  ) {
    return null;
  }
  const heatmapIndex = overlay.bevHeatmapNames.indexOf(name);
  if (heatmapIndex < 0) return null;
  const pixels = BEV_HEATMAP_SIZE * BEV_HEATMAP_SIZE;
  const begin =
    (row * overlay.bevHeatmapNames.length + heatmapIndex) * pixels;
  return {
    values: overlay.bevHeatmaps.subarray(begin, begin + pixels),
    scale: overlay.bevHeatmapScales[row],
  };
}

export async function sampleUIDHash(sampleUID: string): Promise<bigint> {
  const digest = new Uint8Array(
    await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(sampleUID),
    ),
  );
  let result = BigInt(0);
  for (let i = 7; i >= 0; i--) {
    result = (result << BigInt(8)) | BigInt(digest[i]);
  }
  return result;
}

export async function resolveOverlayRows(
  overlay: OverlayArtifact,
  sampleUIDs: string[],
): Promise<Map<string, number>> {
  const hashes = await Promise.all(sampleUIDs.map(sampleUIDHash));
  const rows = new Map<string, number>();
  hashes.forEach((hash, index) => {
    const row = overlay.directory.get(hash);
    if (row !== undefined) rows.set(sampleUIDs[index], row);
  });
  return rows;
}
