"use client";

import { Box, Map as MapIcon } from "lucide-react";
import {
  type MouseEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  SEMANTIC_OCCUPANCY_CLASS_NAMES,
  semanticOccupancyValid,
  semanticOccupancyValue,
  type SemanticOccupancyArtifact,
} from "@/lib/semantic-occupancy";

const CLASS_LABELS = [
  "Drivable",
  "Lane",
  "Intersection",
  "Crosswalk",
  "Stop line",
  "Vehicle",
  "VRU",
  "Obstacle",
] as const;

const CLASS_COLORS = [
  [72, 148, 186],
  [87, 198, 160],
  [245, 183, 61],
  [226, 232, 240],
  [244, 78, 91],
  [95, 214, 101],
  [224, 105, 196],
  [242, 139, 55],
] as const;

type DisplayMode = "prediction" | "teacher" | "error";
type ProjectionMode = "top-down" | "isometric";

interface PointerReading {
  rasterRow: number;
  rasterCol: number;
  values: number[];
}

function rasterCoordinates(
  event: MouseEvent<HTMLCanvasElement>,
  artifact: SemanticOccupancyArtifact,
  projection: ProjectionMode,
): [number, number] | null {
  const rect = event.currentTarget.getBoundingClientRect();
  const canvasX =
    ((event.clientX - rect.left) / rect.width) * event.currentTarget.width;
  const canvasY =
    ((event.clientY - rect.top) / rect.height) * event.currentTarget.height;
  let row = canvasY;
  let col = canvasX;
  if (projection === "isometric") {
    const vertical = (canvasY - 126) / 0.42;
    col = (canvasX - vertical) / 2;
    row = (canvasX + vertical) / 2;
  }
  row = Math.floor(row);
  col = Math.floor(col);
  return row >= 0 &&
    row < artifact.height &&
    col >= 0 &&
    col < artifact.width
    ? [row, col]
    : null;
}

function sourceImage(
  artifact: SemanticOccupancyArtifact,
  row: number,
  mode: DisplayMode,
  threshold: number,
  opacity: number,
  enabled: boolean[],
): ImageData {
  const pixels = new Uint8ClampedArray(
    artifact.height * artifact.width * 4,
  );
  const teacher = artifact.teacher;
  for (let rasterRow = 0; rasterRow < artifact.height; rasterRow++) {
    for (let rasterCol = 0; rasterCol < artifact.width; rasterCol++) {
      let selectedClass = -1;
      let selectedValue = 0;
      let errorKind: "fp" | "fn" | null = null;
      for (let classIndex = 0; classIndex < artifact.classCount; classIndex++) {
        if (!enabled[classIndex]) continue;
        const prediction = semanticOccupancyValue(
          artifact.probability,
          artifact,
          row,
          classIndex,
          rasterRow,
          rasterCol,
        );
        if (mode === "prediction") {
          if (prediction >= threshold && prediction > selectedValue) {
            selectedClass = classIndex;
            selectedValue = prediction;
          }
          continue;
        }
        if (
          !teacher ||
          !semanticOccupancyValid(
            artifact,
            row,
            classIndex,
            rasterRow,
            rasterCol,
          )
        ) {
          continue;
        }
        const target = semanticOccupancyValue(
          teacher,
          artifact,
          row,
          classIndex,
          rasterRow,
          rasterCol,
        );
        if (mode === "teacher") {
          if (target >= 0.5 && target > selectedValue) {
            selectedClass = classIndex;
            selectedValue = target;
          }
          continue;
        }
        const predictedPositive = prediction >= threshold;
        const targetPositive = target >= 0.5;
        if (predictedPositive === targetPositive) continue;
        const confidence = Math.abs(prediction - target);
        if (confidence > selectedValue) {
          selectedClass = classIndex;
          selectedValue = confidence;
          errorKind = predictedPositive ? "fp" : "fn";
        }
      }
      if (selectedClass < 0) continue;
      const color =
        mode === "error"
          ? errorKind === "fp"
            ? ([244, 63, 94] as const)
            : ([34, 211, 238] as const)
          : CLASS_COLORS[selectedClass];
      const offset = (rasterRow * artifact.width + rasterCol) * 4;
      pixels[offset] = color[0];
      pixels[offset + 1] = color[1];
      pixels[offset + 2] = color[2];
      pixels[offset + 3] = Math.round(
        255 * opacity * Math.max(0.25, selectedValue),
      );
    }
  }
  return new ImageData(pixels, artifact.width, artifact.height);
}

export function SemanticOccupancyView({
  artifact,
  row,
  status,
}: {
  artifact: SemanticOccupancyArtifact | null;
  row: number | undefined;
  status: "idle" | "loading" | "ready" | "unavailable" | "error";
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<DisplayMode>("prediction");
  const [projection, setProjection] =
    useState<ProjectionMode>("top-down");
  const [threshold, setThreshold] = useState(0.5);
  const [opacity, setOpacity] = useState(0.8);
  const [enabled, setEnabled] = useState(
    SEMANTIC_OCCUPANCY_CLASS_NAMES.map(() => true),
  );
  const [pointer, setPointer] = useState<PointerReading | null>(null);
  const hasTeacher = Boolean(artifact?.teacher && artifact.validBits);

  useEffect(() => {
    if (!hasTeacher && mode !== "prediction") setMode("prediction");
  }, [hasTeacher, mode]);

  const image = useMemo(
    () =>
      artifact && row !== undefined
        ? sourceImage(
            artifact,
            row,
            mode,
            threshold,
            opacity,
            enabled,
          )
        : null,
    [artifact, enabled, mode, opacity, row, threshold],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#080b10";
    context.fillRect(0, 0, canvas.width, canvas.height);
    if (!image || !artifact) return;
    const source = document.createElement("canvas");
    source.width = artifact.width;
    source.height = artifact.height;
    const sourceContext = source.getContext("2d");
    if (!sourceContext) return;
    sourceContext.putImageData(image, 0, 0);
    sourceContext.strokeStyle = "rgba(255,255,255,0.92)";
    sourceContext.lineWidth = 2;
    sourceContext.beginPath();
    sourceContext.moveTo(149.5, 292);
    sourceContext.lineTo(143.5, 306);
    sourceContext.lineTo(155.5, 306);
    sourceContext.closePath();
    sourceContext.stroke();

    context.imageSmoothingEnabled = false;
    if (projection === "top-down") {
      context.drawImage(source, 0, 0);
      return;
    }
    context.save();
    context.setTransform(1, -0.42, 1, 0.42, 0, 126);
    context.drawImage(source, 0, 0);
    context.restore();
  }, [artifact, image, projection]);

  const pointerRows = pointer
    ? SEMANTIC_OCCUPANCY_CLASS_NAMES.map((name, index) => ({
        name,
        label: CLASS_LABELS[index],
        color: CLASS_COLORS[index],
        value: pointer.values[index],
      }))
        .filter((entry, index) => enabled[index])
        .sort((a, b) => b.value - a.value)
    : [];

  return (
    <section className="space-y-3" aria-label="2D BEV semantic occupancy">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium text-slate-200">
            2D BEV semantic occupancy
          </h3>
          <p className="font-mono text-[10px] text-slate-500">
            180 m × 120 m · 0.4 m/px
          </p>
        </div>
        <div className="flex items-center gap-1 rounded-md border border-slate-800 p-1">
          <button
            type="button"
            className={`rounded px-2 py-1 text-[10px] ${
              projection === "top-down"
                ? "bg-slate-700 text-white"
                : "text-slate-400"
            }`}
            onClick={() => setProjection("top-down")}
            title="Top-down view"
            aria-label="Top-down semantic occupancy"
          >
            <MapIcon className="size-3.5" />
          </button>
          <button
            type="button"
            className={`rounded px-2 py-1 text-[10px] ${
              projection === "isometric"
                ? "bg-slate-700 text-white"
                : "text-slate-400"
            }`}
            onClick={() => setProjection("isometric")}
            title="Isometric view"
            aria-label="Isometric 2D semantic occupancy"
          >
            <Box className="size-3.5" />
          </button>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px]">
        <div className="relative min-h-64 overflow-hidden border border-slate-800 bg-[#080b10]">
          {artifact && row !== undefined ? (
            <canvas
              ref={canvasRef}
              width={projection === "top-down" ? artifact.width : 750}
              height={projection === "top-down" ? artifact.height : 315}
              className="mx-auto block max-h-[620px] w-full object-contain [image-rendering:pixelated]"
              onMouseLeave={() => setPointer(null)}
              onMouseMove={(event) => {
                const coordinates = rasterCoordinates(
                  event,
                  artifact,
                  projection,
                );
                if (!coordinates) {
                  setPointer(null);
                  return;
                }
                const [rasterRow, rasterCol] = coordinates;
                setPointer({
                  rasterRow,
                  rasterCol,
                  values: SEMANTIC_OCCUPANCY_CLASS_NAMES.map(
                    (_, classIndex) =>
                      semanticOccupancyValue(
                        artifact.probability,
                        artifact,
                        row,
                        classIndex,
                        rasterRow,
                        rasterCol,
                      ),
                  ),
                });
              }}
            />
          ) : (
            <div
              role="status"
              className="flex min-h-64 items-center justify-center px-6 text-center text-xs text-slate-500"
            >
              {status === "loading"
                ? "Loading semantic occupancy..."
                : status === "error"
                  ? "Semantic occupancy failed validation."
                  : "No semantic occupancy artifact for this model."}
            </div>
          )}
          {pointer && (
            <div className="pointer-events-none absolute left-2 top-2 min-w-40 border border-slate-700 bg-slate-950/95 p-2 font-mono text-[9px] text-slate-300">
              <p className="mb-1 text-slate-500">
                x {(120 - (pointer.rasterRow + 0.5) * 0.4).toFixed(1)} m · y{" "}
                {(60 - (pointer.rasterCol + 0.5) * 0.4).toFixed(1)} m
              </p>
              {pointerRows.slice(0, 4).map((entry) => (
                <p
                  key={entry.name}
                  className="flex items-center justify-between gap-3"
                >
                  <span className="flex items-center gap-1">
                    <span
                      className="size-2"
                      style={{
                        backgroundColor: `rgb(${entry.color.join(",")})`,
                      }}
                    />
                    {entry.label}
                  </span>
                  <span>{entry.value.toFixed(3)}</span>
                </p>
              ))}
            </div>
          )}
        </div>

        <div className="space-y-4">
          <div
            className="grid grid-cols-3 border border-slate-800"
            role="tablist"
            aria-label="Semantic occupancy source"
          >
            {(["prediction", "teacher", "error"] as const).map((value) => (
              <button
                key={value}
                type="button"
                role="tab"
                disabled={value !== "prediction" && !hasTeacher}
                aria-selected={mode === value}
                className={`px-2 py-1.5 text-[10px] capitalize ${
                  mode === value
                    ? "bg-slate-700 text-white"
                    : "text-slate-400"
                } disabled:cursor-not-allowed disabled:text-slate-700`}
                onClick={() => setMode(value)}
              >
                {value}
              </button>
            ))}
          </div>

          <label className="block space-y-1 text-[10px] text-slate-400">
            <span className="flex justify-between">
              <span>Confidence</span>
              <span className="font-mono">{threshold.toFixed(2)}</span>
            </span>
            <input
              className="w-full accent-sky-500"
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
            />
          </label>

          <label className="block space-y-1 text-[10px] text-slate-400">
            <span className="flex justify-between">
              <span>Opacity</span>
              <span className="font-mono">{opacity.toFixed(2)}</span>
            </span>
            <input
              className="w-full accent-sky-500"
              type="range"
              min="0.1"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(event) => setOpacity(Number(event.target.value))}
            />
          </label>

          <fieldset className="grid grid-cols-2 gap-x-3 gap-y-2">
            <legend className="mb-2 text-[10px] text-slate-500">
              Classes
            </legend>
            {CLASS_LABELS.map((label, index) => (
              <label
                key={label}
                className="flex min-w-0 items-center gap-2 text-[10px] text-slate-300"
              >
                <input
                  type="checkbox"
                  checked={enabled[index]}
                  onChange={() =>
                    setEnabled((current) =>
                      current.map((value, currentIndex) =>
                        currentIndex === index ? !value : value,
                      ),
                    )
                  }
                  className="sr-only"
                />
                <span
                  className={`size-3 shrink-0 border ${
                    enabled[index]
                      ? "border-transparent"
                      : "border-slate-600 bg-transparent"
                  }`}
                  style={
                    enabled[index]
                      ? {
                          backgroundColor: `rgb(${CLASS_COLORS[index].join(",")})`,
                        }
                      : undefined
                  }
                  aria-hidden="true"
                />
                <span className="truncate">{label}</span>
              </label>
            ))}
          </fieldset>

          {mode === "error" && hasTeacher && (
            <div className="flex gap-4 font-mono text-[9px] text-slate-500">
              <span className="flex items-center gap-1">
                <span className="size-2 bg-rose-500" /> FP
              </span>
              <span className="flex items-center gap-1">
                <span className="size-2 bg-cyan-400" /> FN
              </span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
