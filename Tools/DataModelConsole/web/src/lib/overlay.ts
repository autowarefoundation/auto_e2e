const MAGIC = "AOVL";
const HEADER_BYTES = 20;
const DIRECTORY_ENTRY_BYTES = 12;
const HORIZON = 64;
const DIMS = 2;

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
  if (formatVersion !== 1 || horizon !== HORIZON || dims !== DIMS || reserved !== 0) {
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
  const expectedBytes = speedsOffset + sampleCount * 4;
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
