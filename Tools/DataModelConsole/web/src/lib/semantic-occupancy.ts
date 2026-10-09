const MAGIC = "ASOC";
const FORMAT_VERSION = 1;
const HEADER_BYTES = 20;
const DIRECTORY_ENTRY_BYTES = 12;
const FLAG_TEACHER_PRESENT = 1 << 0;

export const SEMANTIC_OCCUPANCY_CLASS_NAMES = [
  "drivable_area",
  "lane_boundary",
  "intersection",
  "crosswalk",
  "stop_line",
  "vehicle",
  "vulnerable_road_user",
  "other_obstacle",
] as const;

export type SemanticOccupancyClassName =
  (typeof SEMANTIC_OCCUPANCY_CLASS_NAMES)[number];

export type SemanticOccupancyDisplayMode =
  | "prediction"
  | "teacher"
  | "error";

export type SemanticOccupancyErrorKind = "fp" | "fn";

export interface SemanticOccupancyComponent {
  classIndex: number;
  className: SemanticOccupancyClassName;
  errorKind: SemanticOccupancyErrorKind | null;
  cellCount: number;
  centroidRow: number;
  centroidCol: number;
  minRow: number;
  maxRow: number;
  minCol: number;
  maxCol: number;
  meanConfidence: number;
  peakConfidence: number;
  principalAxisRadians: number;
  majorSpanCells: number;
  minorSpanCells: number;
}

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

interface ExtractSemanticOccupancyComponentsOptions {
  artifact: SemanticOccupancyArtifact;
  row: number;
  classIndices: readonly number[];
  mode: SemanticOccupancyDisplayMode;
  threshold: number;
  minCells?: number;
  maxComponents?: number;
}

function cellIndex(
  artifact: SemanticOccupancyArtifact,
  row: number,
  classIndex: number,
  rasterRow: number,
  rasterCol: number,
): number {
  return (
    (((row * artifact.classCount + classIndex) * artifact.height + rasterRow) *
      artifact.width) +
    rasterCol
  );
}

export function extractSemanticOccupancyComponents({
  artifact,
  row,
  classIndices,
  mode,
  threshold,
  minCells = 1,
  maxComponents = 80,
}: ExtractSemanticOccupancyComponentsOptions): SemanticOccupancyComponent[] {
  if (
    row < 0 ||
    row >= artifact.sampleCount ||
    maxComponents <= 0 ||
    minCells <= 0
  ) {
    return [];
  }

  const width = artifact.width;
  const height = artifact.height;
  const pixelCount = width * height;
  const components: SemanticOccupancyComponent[] = [];

  for (const classIndex of classIndices) {
    if (classIndex < 0 || classIndex >= artifact.classCount) continue;
    const visited = new Uint8Array(pixelCount);
    const queue = new Int32Array(pixelCount);
    const classOffset =
      (row * artifact.classCount + classIndex) * pixelCount;

    const stateAt = (
      rasterIndex: number,
    ): { kind: 0 | 1 | 2; confidence: number } => {
      const prediction = artifact.probability[classOffset + rasterIndex] / 255;
      if (mode === "prediction") {
        return prediction >= threshold
          ? { kind: 1, confidence: prediction }
          : { kind: 0, confidence: 0 };
      }
      if (!artifact.teacher || !artifact.validBits) {
        return { kind: 0, confidence: 0 };
      }
      const valueIndex = cellIndex(
        artifact,
        row,
        classIndex,
        Math.floor(rasterIndex / width),
        rasterIndex % width,
      );
      if (
        !(artifact.validBits[valueIndex >> 3] & (1 << (valueIndex & 7)))
      ) {
        return { kind: 0, confidence: 0 };
      }
      const target = artifact.teacher[classOffset + rasterIndex] / 255;
      if (mode === "teacher") {
        return target >= 0.5
          ? { kind: 1, confidence: target }
          : { kind: 0, confidence: 0 };
      }
      const predictedPositive = prediction >= threshold;
      const targetPositive = target >= 0.5;
      if (predictedPositive === targetPositive) {
        return { kind: 0, confidence: 0 };
      }
      return {
        kind: predictedPositive ? 1 : 2,
        confidence: Math.abs(prediction - target),
      };
    };

    for (let start = 0; start < pixelCount; start++) {
      if (visited[start]) continue;
      const startState = stateAt(start);
      if (startState.kind === 0) continue;

      let queueStart = 0;
      let queueEnd = 1;
      queue[0] = start;
      visited[start] = 1;
      let count = 0;
      let sumRow = 0;
      let sumCol = 0;
      let sumRowSquared = 0;
      let sumColSquared = 0;
      let sumRowCol = 0;
      let confidenceSum = 0;
      let peakConfidence = 0;
      let minRow = height;
      let maxRow = -1;
      let minCol = width;
      let maxCol = -1;

      while (queueStart < queueEnd) {
        const rasterIndex = queue[queueStart++];
        const rasterRow = Math.floor(rasterIndex / width);
        const rasterCol = rasterIndex - rasterRow * width;
        const state = stateAt(rasterIndex);
        count++;
        sumRow += rasterRow;
        sumCol += rasterCol;
        sumRowSquared += rasterRow * rasterRow;
        sumColSquared += rasterCol * rasterCol;
        sumRowCol += rasterRow * rasterCol;
        confidenceSum += state.confidence;
        peakConfidence = Math.max(peakConfidence, state.confidence);
        minRow = Math.min(minRow, rasterRow);
        maxRow = Math.max(maxRow, rasterRow);
        minCol = Math.min(minCol, rasterCol);
        maxCol = Math.max(maxCol, rasterCol);

        for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
          const neighborRow = rasterRow + rowOffset;
          if (neighborRow < 0 || neighborRow >= height) continue;
          for (let colOffset = -1; colOffset <= 1; colOffset++) {
            if (rowOffset === 0 && colOffset === 0) continue;
            const neighborCol = rasterCol + colOffset;
            if (neighborCol < 0 || neighborCol >= width) continue;
            const neighborIndex = neighborRow * width + neighborCol;
            if (visited[neighborIndex]) continue;
            const neighborState = stateAt(neighborIndex);
            if (neighborState.kind !== startState.kind) continue;
            visited[neighborIndex] = 1;
            queue[queueEnd++] = neighborIndex;
          }
        }
      }

      if (count < minCells) continue;
      const centroidRow = sumRow / count;
      const centroidCol = sumCol / count;
      const rowVariance =
        sumRowSquared / count - centroidRow * centroidRow;
      const colVariance =
        sumColSquared / count - centroidCol * centroidCol;
      const covariance =
        sumRowCol / count - centroidRow * centroidCol;
      const principalAxisRadians =
        count > 1
          ? 0.5 *
            Math.atan2(
              2 * covariance,
              rowVariance - colVariance,
            )
          : 0;
      const axisRow = Math.cos(principalAxisRadians);
      const axisCol = Math.sin(principalAxisRadians);
      const cellProjectionSpan = Math.abs(axisRow) + Math.abs(axisCol);
      let minMajorProjection = Number.POSITIVE_INFINITY;
      let maxMajorProjection = Number.NEGATIVE_INFINITY;
      let minMinorProjection = Number.POSITIVE_INFINITY;
      let maxMinorProjection = Number.NEGATIVE_INFINITY;
      for (let index = 0; index < queueEnd; index++) {
        const rasterIndex = queue[index];
        const rasterRow = Math.floor(rasterIndex / width);
        const rasterCol = rasterIndex - rasterRow * width;
        const rowOffset = rasterRow - centroidRow;
        const colOffset = rasterCol - centroidCol;
        const majorProjection =
          rowOffset * axisRow + colOffset * axisCol;
        const minorProjection =
          -rowOffset * axisCol + colOffset * axisRow;
        minMajorProjection = Math.min(
          minMajorProjection,
          majorProjection,
        );
        maxMajorProjection = Math.max(
          maxMajorProjection,
          majorProjection,
        );
        minMinorProjection = Math.min(
          minMinorProjection,
          minorProjection,
        );
        maxMinorProjection = Math.max(
          maxMinorProjection,
          minorProjection,
        );
      }
      components.push({
        classIndex,
        className: SEMANTIC_OCCUPANCY_CLASS_NAMES[classIndex],
        errorKind:
          mode === "error"
            ? startState.kind === 1
              ? "fp"
              : "fn"
            : null,
        cellCount: count,
        centroidRow,
        centroidCol,
        minRow,
        maxRow,
        minCol,
        maxCol,
        meanConfidence: confidenceSum / count,
        peakConfidence,
        principalAxisRadians,
        majorSpanCells:
          maxMajorProjection -
          minMajorProjection +
          cellProjectionSpan,
        minorSpanCells:
          maxMinorProjection -
          minMinorProjection +
          cellProjectionSpan,
      });
    }
  }

  return components
    .sort(
      (left, right) =>
        right.cellCount - left.cellCount ||
        right.meanConfidence - left.meanConfidence,
    )
    .slice(0, maxComponents);
}
