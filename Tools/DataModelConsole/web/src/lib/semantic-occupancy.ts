const MAGIC = "ASOC";
const FORMAT_VERSION = 1;
const HEADER_BYTES = 20;
const DIRECTORY_ENTRY_BYTES = 12;
const FLAG_TEACHER_PRESENT = 1 << 0;

export const SEMANTIC_OCCUPANCY_CLASS_NAMES = [
  "drivable_area",
  "lane_area",
  "intersection",
  "crosswalk",
  "stop_line",
  "vehicle",
  "vulnerable_road_user",
  "other_obstacle",
] as const;

export type SemanticOccupancyClassName =
  (typeof SEMANTIC_OCCUPANCY_CLASS_NAMES)[number];

export interface SemanticOccupancyArtifact {
  formatVersion: number;
  flags: number;
  sampleCount: number;
  classCount: number;
  height: number;
  width: number;
  directory: { hashHigh: number; hashLow: number; row: number }[];
  probability: Uint8Array;
  teacher: Uint8Array | null;
  validBits: Uint8Array | null;
}

function readMagic(view: DataView): string {
  return String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  );
}

export function parseSemanticOccupancy(
  buffer: ArrayBuffer,
): SemanticOccupancyArtifact {
  if (buffer.byteLength < HEADER_BYTES) {
    throw new Error("Semantic occupancy is shorter than its header");
  }
  const view = new DataView(buffer);
  const magic = readMagic(view);
  const formatVersion = view.getUint16(4, true);
  const flags = view.getUint16(6, true);
  const sampleCount = view.getUint32(8, true);
  const classCount = view.getUint16(12, true);
  const height = view.getUint16(14, true);
  const width = view.getUint16(16, true);
  const reserved = view.getUint16(18, true);
  if (
    magic !== MAGIC ||
    formatVersion !== FORMAT_VERSION ||
    flags & ~FLAG_TEACHER_PRESENT ||
    sampleCount < 1 ||
    classCount !== SEMANTIC_OCCUPANCY_CLASS_NAMES.length ||
    height < 1 ||
    width < 1 ||
    reserved !== 0
  ) {
    throw new Error("Unsupported semantic occupancy header");
  }

  const directoryBytes = sampleCount * DIRECTORY_ENTRY_BYTES;
  const cellCount = sampleCount * classCount * height * width;
  const validBytes = Math.ceil(cellCount / 8);
  const hasTeacher = Boolean(flags & FLAG_TEACHER_PRESENT);
  const expectedBytes =
    HEADER_BYTES +
    directoryBytes +
    cellCount +
    (hasTeacher ? cellCount + validBytes : 0);
  if (!Number.isSafeInteger(expectedBytes) || buffer.byteLength !== expectedBytes) {
    throw new Error(
      `Semantic occupancy size mismatch: expected ${expectedBytes}, got ${buffer.byteLength}`,
    );
  }

  const directory = new Array<{
    hashHigh: number;
    hashLow: number;
    row: number;
  }>(sampleCount);
  let cursor = HEADER_BYTES;
  let previousHashHigh = -1;
  let previousHashLow = -1;
  const seenRows = new Uint8Array(sampleCount);
  for (let index = 0; index < sampleCount; index++) {
    const hashLow = view.getUint32(cursor, true);
    const hashHigh = view.getUint32(cursor + 4, true);
    const row = view.getUint32(cursor + 8, true);
    cursor += DIRECTORY_ENTRY_BYTES;
    const ordered =
      hashHigh > previousHashHigh ||
      (hashHigh === previousHashHigh && hashLow > previousHashLow);
    if (!ordered || row >= sampleCount || seenRows[row]) {
      throw new Error("Semantic occupancy directory is invalid");
    }
    previousHashHigh = hashHigh;
    previousHashLow = hashLow;
    seenRows[row] = 1;
    directory[index] = { hashHigh, hashLow, row };
  }

  const probability = new Uint8Array(buffer, cursor, cellCount);
  cursor += cellCount;
  const teacher = hasTeacher
    ? new Uint8Array(buffer, cursor, cellCount)
    : null;
  if (teacher) cursor += cellCount;
  const validBits = hasTeacher
    ? new Uint8Array(buffer, cursor, validBytes)
    : null;
  return {
    formatVersion,
    flags,
    sampleCount,
    classCount,
    height,
    width,
    directory,
    probability,
    teacher,
    validBits,
  };
}

interface Uint64Parts {
  high: number;
  low: number;
}

async function sampleUIDHash(sampleUID: string): Promise<Uint64Parts> {
  const encoded = new TextEncoder().encode(sampleUID);
  const digest = await crypto.subtle.digest("SHA-256", encoded);
  const view = new DataView(digest);
  return {
    low: view.getUint32(0, true),
    high: view.getUint32(4, true),
  };
}

function rowForHash(
  artifact: SemanticOccupancyArtifact,
  target: Uint64Parts,
): number | undefined {
  let low = 0;
  let high = artifact.directory.length - 1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    const entry = artifact.directory[middle];
    if (
      entry.hashHigh === target.high &&
      entry.hashLow === target.low
    ) {
      return entry.row;
    }
    if (
      entry.hashHigh < target.high ||
      (entry.hashHigh === target.high && entry.hashLow < target.low)
    ) {
      low = middle + 1;
    }
    else high = middle - 1;
  }
  return undefined;
}

export async function resolveSemanticOccupancyRows(
  artifact: SemanticOccupancyArtifact,
  sampleUIDs: string[],
): Promise<Map<string, number>> {
  const hashes = await Promise.all(sampleUIDs.map(sampleUIDHash));
  const rows = new Map<string, number>();
  for (let index = 0; index < sampleUIDs.length; index++) {
    const row = rowForHash(artifact, hashes[index]);
    if (row !== undefined) rows.set(sampleUIDs[index], row);
  }
  return rows;
}

export function semanticOccupancyValue(
  values: Uint8Array,
  artifact: SemanticOccupancyArtifact,
  row: number,
  classIndex: number,
  rasterRow: number,
  rasterCol: number,
): number {
  if (
    row < 0 ||
    row >= artifact.sampleCount ||
    classIndex < 0 ||
    classIndex >= artifact.classCount ||
    rasterRow < 0 ||
    rasterRow >= artifact.height ||
    rasterCol < 0 ||
    rasterCol >= artifact.width
  ) {
    return 0;
  }
  const index =
    (((row * artifact.classCount + classIndex) * artifact.height + rasterRow) *
      artifact.width) +
    rasterCol;
  return values[index] / 255;
}

export function semanticOccupancyValid(
  artifact: SemanticOccupancyArtifact,
  row: number,
  classIndex: number,
  rasterRow: number,
  rasterCol: number,
): boolean {
  if (!artifact.validBits) return false;
  const index =
    (((row * artifact.classCount + classIndex) * artifact.height + rasterRow) *
      artifact.width) +
    rasterCol;
  return Boolean(
    artifact.validBits[index >> 3] & (1 << (index & 7)),
  );
}
