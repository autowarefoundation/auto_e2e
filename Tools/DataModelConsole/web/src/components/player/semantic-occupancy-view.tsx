"use client";

import { OrbitControls } from "@react-three/drei";
import {
  Canvas,
  type ThreeEvent,
  useFrame,
  useThree,
} from "@react-three/fiber";
import { CarFront, Map as MapIcon, Orbit } from "lucide-react";
import {
  type ComponentProps,
  type MutableRefObject,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

import {
  EgoVehicle,
  SemanticObstacle,
  SemanticPedestrian,
  SemanticVehicle,
} from "@/components/player/semantic-scene-models";
import {
  SceneEnvironment,
  ScenePostProcessing,
  SemanticGround,
} from "@/components/player/semantic-scene-environment";
import {
  extractSemanticOccupancyComponents,
  SEMANTIC_OCCUPANCY_CLASS_NAMES,
  semanticOccupancyValid,
  semanticOccupancyValue,
  type SemanticOccupancyArtifact,
  type SemanticOccupancyComponent,
  type SemanticOccupancyDisplayMode,
} from "@/lib/semantic-occupancy";

const METERS_PER_CELL = 0.4;
const MAX_SCENE_OBJECTS = 56;
const OBJECT_CLASS_INDICES = [5, 6, 7] as const;

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

const OBJECT_COLORS = ["#62df7b", "#ef73cf", "#f39a42"] as const;
const ERROR_COLORS = { fp: "#ff365f", fn: "#35dff4" } as const;

type CameraPreset = "orbit" | "top" | "ego";

function classLabel(classIndex: number, laneLabel: string): string {
  return classIndex === 1 ? laneLabel : CLASS_LABELS[classIndex];
}

function classNameForIndex(
  classIndex: number,
  supportedClassNames: ReadonlySet<string>,
): string {
  if (classIndex !== 1) {
    return SEMANTIC_OCCUPANCY_CLASS_NAMES[classIndex];
  }
  return supportedClassNames.has("lane_boundary")
    ? "lane_boundary"
    : supportedClassNames.has("lane_area")
      ? "lane_area"
      : "lane_boundary";
}

interface PointerReading {
  rasterRow: number;
  rasterCol: number;
  values: number[];
}

interface CameraPresetDefinition {
  fov: number;
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
}

const CAMERA_PRESETS: Record<CameraPreset, CameraPresetDefinition> = {
  orbit: {
    position: [28, 24, -20],
    target: [0, 0.8, 26],
    up: [0, 1, 0],
    fov: 43,
  },
  top: {
    position: [0, 142, 30.01],
    target: [0, 0, 30],
    up: [0, 0, 1],
    fov: 43,
  },
  ego: {
    position: [0, 3.8, -9.2],
    target: [0, 0.9, 18],
    up: [0, 1, 0],
    fov: 50,
  },
};

function sourcePixels(
  artifact: SemanticOccupancyArtifact,
  row: number,
  mode: SemanticOccupancyDisplayMode,
  threshold: number,
  opacity: number,
  enabled: boolean[],
): Uint8Array {
  const pixels = new Uint8Array(artifact.height * artifact.width * 4);
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
        mode === "error" && errorKind
          ? errorKind === "fp"
            ? ([255, 54, 95] as const)
            : ([53, 223, 244] as const)
          : CLASS_COLORS[selectedClass];
      const offset = (rasterRow * artifact.width + rasterCol) * 4;
      pixels[offset] = color[0];
      pixels[offset + 1] = color[1];
      pixels[offset + 2] = color[2];
      pixels[offset + 3] = Math.round(255 * opacity * selectedValue);
    }
  }
  return pixels;
}

function CameraRig({ preset }: { preset: CameraPreset }) {
  const controlsRef =
    useRef<OrbitControlsImpl>(null) as MutableRefObject<OrbitControlsImpl | null>;
  const { camera, invalidate, size } = useThree();
  const transitionRef = useRef<{
    active: boolean;
    elapsed: number;
    fromFov: number;
    fromPosition: THREE.Vector3;
    fromTarget: THREE.Vector3;
    fromUp: THREE.Vector3;
    toFov: number;
    toPosition: THREE.Vector3;
    toTarget: THREE.Vector3;
    toUp: THREE.Vector3;
  } | null>(null);

  useEffect(() => {
    const definition = CAMERA_PRESETS[preset];
    const aspect = size.width / Math.max(size.height, 1);
    const distanceScale = aspect < 0.9 ? 1.38 : 1;
    const position = definition.position.map(
      (coordinate, index) =>
        definition.target[index] +
        (coordinate - definition.target[index]) * distanceScale,
    ) as [number, number, number];
    transitionRef.current = {
      active: true,
      elapsed: 0,
      fromFov:
        camera instanceof THREE.PerspectiveCamera
          ? camera.fov
          : definition.fov,
      fromPosition: camera.position.clone(),
      fromTarget:
        controlsRef.current?.target.clone() ??
        new THREE.Vector3(...definition.target),
      fromUp: camera.up.clone(),
      toFov: definition.fov,
      toPosition: new THREE.Vector3(...position),
      toTarget: new THREE.Vector3(...definition.target),
      toUp: new THREE.Vector3(...definition.up),
    };
    invalidate();
  }, [camera, invalidate, preset, size.height, size.width]);

  useFrame((_, delta) => {
    const transition = transitionRef.current;
    if (!transition?.active) return;
    transition.elapsed = Math.min(transition.elapsed + delta, 0.72);
    const progress = transition.elapsed / 0.72;
    const eased =
      progress < 0.5
        ? 4 * progress ** 3
        : 1 - (-2 * progress + 2) ** 3 / 2;
    camera.position.lerpVectors(
      transition.fromPosition,
      transition.toPosition,
      eased,
    );
    camera.up
      .lerpVectors(transition.fromUp, transition.toUp, eased)
      .normalize();
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.fov = THREE.MathUtils.lerp(
        transition.fromFov,
        transition.toFov,
        eased,
      );
      camera.updateProjectionMatrix();
    }
    controlsRef.current?.target.lerpVectors(
      transition.fromTarget,
      transition.toTarget,
      eased,
    );
    controlsRef.current?.update();
    if (progress >= 1) {
      transition.active = false;
      return;
    }
    invalidate();
  });

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      enablePan
      enableRotate
      enableZoom
      minDistance={2.5}
      maxDistance={190}
      maxPolarAngle={Math.PI * 0.495}
      screenSpacePanning
      onChange={() => invalidate()}
      onStart={() => {
        if (transitionRef.current) {
          transitionRef.current.active = false;
        }
      }}
    />
  );
}

function componentPosition(
  artifact: SemanticOccupancyArtifact,
  component: SemanticOccupancyComponent,
): [number, number, number] {
  return [
    (artifact.width / 2 - (component.centroidCol + 0.5)) *
      METERS_PER_CELL,
    0,
    (artifact.height * (2 / 3) - (component.centroidRow + 0.5)) *
      METERS_PER_CELL,
  ];
}

function SceneObject({
  artifact,
  component,
  decorativeVariants,
  opacity,
}: {
  artifact: SemanticOccupancyArtifact;
  component: SemanticOccupancyComponent;
  decorativeVariants: boolean;
  opacity: number;
}) {
  const majorSpan = component.majorSpanCells * METERS_PER_CELL;
  const minorSpan = component.minorSpanCells * METERS_PER_CELL;
  const position = componentPosition(artifact, component);
  const color = component.errorKind
    ? ERROR_COLORS[component.errorKind]
    : OBJECT_COLORS[component.classIndex - OBJECT_CLASS_INDICES[0]];

  if (component.classIndex === 5) {
    return (
      <SemanticVehicle
        color={color}
        confidence={component.meanConfidence}
        errorKind={component.errorKind}
        length={majorSpan}
        opacity={opacity}
        position={position}
        width={minorSpan}
        yaw={component.principalAxisRadians}
      />
    );
  }
  if (component.classIndex === 6) {
    return (
      <SemanticPedestrian
        color={color}
        confidence={component.meanConfidence}
        errorKind={component.errorKind}
        length={majorSpan}
        opacity={opacity}
        position={position}
        width={minorSpan}
        yaw={component.principalAxisRadians}
      />
    );
  }
  return (
    <SemanticObstacle
      color={color}
      confidence={component.meanConfidence}
      decorativeVariants={decorativeVariants}
      errorKind={component.errorKind}
      length={majorSpan}
      opacity={opacity}
      position={position}
      width={minorSpan}
      yaw={component.principalAxisRadians}
    />
  );
}

function GroundInteraction({
  artifact,
  onRead,
}: {
  artifact: SemanticOccupancyArtifact;
  onRead: (rasterRow: number, rasterCol: number) => void;
}) {
  const groundWidth = artifact.width * METERS_PER_CELL;
  const groundLength = artifact.height * METERS_PER_CELL;
  const groundCenterZ =
    (artifact.height * (2 / 3) - artifact.height / 2) *
    METERS_PER_CELL;
  const handlePointer = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    const rasterRow = Math.floor(
      artifact.height * (2 / 3) - event.point.z / METERS_PER_CELL,
    );
    const rasterCol = Math.floor(
      artifact.width / 2 - event.point.x / METERS_PER_CELL,
    );
    if (
      rasterRow >= 0 &&
      rasterRow < artifact.height &&
      rasterCol >= 0 &&
      rasterCol < artifact.width
    ) {
      onRead(rasterRow, rasterCol);
    }
  };

  return (
    <mesh
      position={[0, 0.08, groundCenterZ]}
      rotation={[-Math.PI / 2, 0, 0]}
      onPointerMove={handlePointer}
    >
      <planeGeometry args={[groundWidth, groundLength]} />
      <meshBasicMaterial transparent opacity={0} depthWrite={false} />
    </mesh>
  );
}

function OccupancyScene({
  artifact,
  cameraPreset,
  components,
  objectOpacity,
  onPointerRead,
  pixels,
  showDemoStreetscape,
}: {
  artifact: SemanticOccupancyArtifact;
  cameraPreset: CameraPreset;
  components: SemanticOccupancyComponent[];
  objectOpacity: number;
  onPointerRead: (rasterRow: number, rasterCol: number) => void;
  pixels: Uint8Array;
  showDemoStreetscape: boolean;
}) {
  const groundWidth = artifact.width * METERS_PER_CELL;
  const groundLength = artifact.height * METERS_PER_CELL;
  const groundCenterZ =
    (artifact.height * (2 / 3) - artifact.height / 2) *
    METERS_PER_CELL;

  return (
    <>
      <SceneEnvironment
        groundCenterZ={groundCenterZ}
        groundLength={groundLength}
        groundWidth={groundWidth}
        showDemoStreetscape={showDemoStreetscape}
        viewMode={cameraPreset}
      />
      <SemanticGround
        artifact={artifact}
        metersPerCell={METERS_PER_CELL}
        pixels={pixels}
      />
      {components.map((component, index) => (
        <SceneObject
          key={`${component.classIndex}:${component.errorKind}:${component.centroidRow}:${component.centroidCol}:${index}`}
          artifact={artifact}
          component={component}
          decorativeVariants={showDemoStreetscape}
          opacity={objectOpacity}
        />
      ))}
      <EgoVehicle viewMode={cameraPreset} />
      <GroundInteraction artifact={artifact} onRead={onPointerRead} />
      <CameraRig preset={cameraPreset} />
      <ScenePostProcessing viewMode={cameraPreset} />
    </>
  );
}

function CameraButton({
  active,
  children,
  label,
  ...props
}: {
  active: boolean;
  children: React.ReactNode;
  label: string;
} & ComponentProps<"button">) {
  return (
    <button
      type="button"
      aria-label={label}
      aria-pressed={active}
      title={label}
      className={`group relative flex size-8 items-center justify-center rounded border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 ${
        active
          ? "border-cyan-400/60 bg-cyan-400/15 text-cyan-100"
          : "border-slate-700 bg-slate-950/80 text-slate-400 hover:border-slate-500 hover:text-white"
      }`}
      {...props}
    >
      {children}
      <span
        role="tooltip"
        className="pointer-events-none absolute top-9 right-0 z-20 whitespace-nowrap rounded border border-slate-700 bg-slate-950 px-2 py-1 text-[9px] font-normal text-slate-200 opacity-0 shadow-xl transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100"
      >
        {label}
      </span>
    </button>
  );
}

export function SemanticOccupancyView({
  artifact,
  demoEnvironment = false,
  laneLabel = "Lane",
  row,
  status,
  supportedClasses,
}: {
  artifact: SemanticOccupancyArtifact | null;
  demoEnvironment?: boolean;
  laneLabel?: string;
  row: number | undefined;
  status: "idle" | "loading" | "ready" | "unavailable" | "error";
  supportedClasses?: readonly string[];
}) {
  const [mode, setMode] =
    useState<SemanticOccupancyDisplayMode>("prediction");
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>("orbit");
  const [threshold, setThreshold] = useState(0.2);
  const [opacity, setOpacity] = useState(0.8);
  const [enabled, setEnabled] = useState(
    SEMANTIC_OCCUPANCY_CLASS_NAMES.map(() => true),
  );
  const [compactViewport, setCompactViewport] = useState(false);
  const [pointer, setPointer] = useState<PointerReading | null>(null);
  const hasTeacher = Boolean(artifact?.teacher && artifact.validBits);
  const supportedClassNames = useMemo(
    () =>
      new Set(
        supportedClasses ?? SEMANTIC_OCCUPANCY_CLASS_NAMES,
      ),
    [supportedClasses],
  );
  const effectiveEnabled = useMemo(
    () =>
      enabled.map(
        (isEnabled, index) =>
          isEnabled &&
          supportedClassNames.has(
            classNameForIndex(index, supportedClassNames),
          ),
      ),
    [enabled, supportedClassNames],
  );
  const visibleClassIndices = useMemo(
    () =>
      SEMANTIC_OCCUPANCY_CLASS_NAMES.flatMap((_, index) =>
        supportedClassNames.has(
          classNameForIndex(index, supportedClassNames),
        )
          ? [index]
          : [],
      ),
    [supportedClassNames],
  );

  useEffect(() => {
    if (!hasTeacher && mode !== "prediction") setMode("prediction");
  }, [hasTeacher, mode]);

  useEffect(() => {
    const updateViewport = () => setCompactViewport(window.innerWidth < 700);
    updateViewport();
    window.addEventListener("resize", updateViewport);
    return () => window.removeEventListener("resize", updateViewport);
  }, []);

  const pixels = useMemo(
    () =>
      artifact && row !== undefined
        ? sourcePixels(
            artifact,
            row,
            mode,
            threshold,
            opacity,
            effectiveEnabled,
          )
        : null,
    [artifact, effectiveEnabled, mode, opacity, row, threshold],
  );
  const objectClassIndices = useMemo(
    () =>
      OBJECT_CLASS_INDICES.filter(
        (classIndex) => effectiveEnabled[classIndex],
      ),
    [effectiveEnabled],
  );
  const objectLegend = useMemo(
    () =>
      OBJECT_CLASS_INDICES.flatMap((classIndex, colorIndex) =>
        supportedClassNames.has(SEMANTIC_OCCUPANCY_CLASS_NAMES[classIndex])
          ? [
              {
                classIndex,
                color: OBJECT_COLORS[colorIndex],
              },
            ]
          : [],
      ),
    [supportedClassNames],
  );
  const components = useMemo(
    () =>
      artifact && row !== undefined
        ? extractSemanticOccupancyComponents({
            artifact,
            row,
            classIndices: objectClassIndices,
            mode,
            threshold,
            minCells: 1,
            maxComponents: MAX_SCENE_OBJECTS,
          })
        : [],
    [artifact, mode, objectClassIndices, row, threshold],
  );

  const pointerRows = pointer
    ? SEMANTIC_OCCUPANCY_CLASS_NAMES.map((_, index) => ({
        name: classNameForIndex(index, supportedClassNames),
        label: classLabel(index, laneLabel),
        color: CLASS_COLORS[index],
        value: pointer.values[index],
      }))
        .filter((_, index) => effectiveEnabled[index])
        .sort((left, right) => right.value - left.value)
    : [];

  const updatePointer = (rasterRow: number, rasterCol: number) => {
    if (!artifact || row === undefined) return;
    setPointer({
      rasterRow,
      rasterCol,
      values: SEMANTIC_OCCUPANCY_CLASS_NAMES.map((_, classIndex) =>
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
  };

  return (
    <section className="min-w-0 space-y-3" aria-label="3D semantic occupancy">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium text-slate-200">
            3D semantic occupancy
          </h3>
          <p className="font-mono text-[10px] text-slate-500">
            {artifact
              ? `${(artifact.height * METERS_PER_CELL).toFixed(0)} m × ${(artifact.width * METERS_PER_CELL).toFixed(0)} m`
              : "180 m × 120 m"}{" "}
            · {METERS_PER_CELL.toFixed(1)} m/cell
          </p>
        </div>
        <span className="font-mono text-[9px] uppercase text-slate-500">
          {components.length} {components.length === 1 ? "object" : "objects"}
        </span>
      </div>

      <div className="grid min-w-0 gap-4 xl:grid-cols-[minmax(0,1fr)_260px]">
        <div
          className="relative h-[430px] min-w-0 overflow-hidden rounded-md border border-slate-800 bg-[#05090d] sm:h-[540px] xl:h-[620px]"
          onPointerLeave={() => setPointer(null)}
        >
          {artifact && row !== undefined && pixels ? (
            <>
              <Canvas
                aria-label="Interactive 3D semantic occupancy scene"
                camera={{
                  position: CAMERA_PRESETS.orbit.position,
                  fov: CAMERA_PRESETS.orbit.fov,
                  near: 0.1,
                  far: 420,
                }}
                dpr={compactViewport ? [1, 1.15] : [1, 1.5]}
                frameloop="demand"
                gl={{
                  alpha: false,
                  antialias: true,
                  powerPreference: "high-performance",
                  preserveDrawingBuffer: true,
                }}
                shadows
                onCreated={({ gl }) => {
                  gl.domElement.setAttribute("role", "img");
                  gl.domElement.setAttribute(
                    "aria-label",
                    "Interactive 3D semantic occupancy scene",
                  );
                  gl.outputColorSpace = THREE.SRGBColorSpace;
                  gl.toneMapping = THREE.ACESFilmicToneMapping;
                  gl.toneMappingExposure = 1.08;
                }}
              >
                <OccupancyScene
                  artifact={artifact}
                  cameraPreset={cameraPreset}
                  components={components}
                  objectOpacity={opacity}
                  onPointerRead={updatePointer}
                  pixels={pixels}
                  showDemoStreetscape={demoEnvironment}
                />
              </Canvas>

              <div
                className="absolute top-2 right-2 flex gap-1"
                role="group"
                aria-label="Semantic occupancy camera"
              >
                <CameraButton
                  active={cameraPreset === "orbit"}
                  label="Orbit view"
                  onClick={() => setCameraPreset("orbit")}
                >
                  <Orbit className="size-4" />
                </CameraButton>
                <CameraButton
                  active={cameraPreset === "top"}
                  label="Top view"
                  onClick={() => setCameraPreset("top")}
                >
                  <MapIcon className="size-4" />
                </CameraButton>
                <CameraButton
                  active={cameraPreset === "ego"}
                  label="Ego view"
                  onClick={() => setCameraPreset("ego")}
                >
                  <CarFront className="size-4" />
                </CameraButton>
              </div>

              {objectLegend.length > 0 && (
                <div className="pointer-events-none absolute bottom-2 left-2 flex flex-wrap gap-3 rounded border border-slate-700/80 bg-slate-950/80 px-2 py-1.5 font-mono text-[9px] text-slate-400 backdrop-blur">
                  {objectLegend.map((entry) => (
                    <span
                      key={entry.classIndex}
                      className="flex items-center gap-1"
                    >
                      <span
                        className="size-2"
                        style={{ backgroundColor: entry.color }}
                      />
                      {classLabel(entry.classIndex, laneLabel)}
                    </span>
                  ))}
                </div>
              )}

              {pointer && (
                <div className="pointer-events-none absolute top-2 left-2 min-w-40 rounded border border-slate-700 bg-slate-950/92 p-2 font-mono text-[9px] text-slate-300 shadow-xl backdrop-blur">
                  <p className="mb-1 text-slate-500">
                    x{" "}
                    {(
                      (artifact.height * (2 / 3) -
                        (pointer.rasterRow + 0.5)) *
                      METERS_PER_CELL
                    ).toFixed(1)}{" "}
                    m · y{" "}
                    {(
                      (artifact.width / 2 -
                        (pointer.rasterCol + 0.5)) *
                      METERS_PER_CELL
                    ).toFixed(1)}{" "}
                    m
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
            </>
          ) : (
            <div
              role="status"
              className="flex h-full min-h-64 items-center justify-center px-6 text-center text-xs text-slate-500"
            >
              {status === "loading"
                ? "Loading semantic occupancy..."
                : status === "error"
                  ? "Semantic occupancy failed validation."
                  : "No semantic occupancy artifact for this model."}
            </div>
          )}
        </div>

        <div className="min-w-0 space-y-4">
          <div
            className="grid grid-cols-3 overflow-hidden rounded border border-slate-800"
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
                className={`min-w-0 px-2 py-2 text-[10px] capitalize transition-colors ${
                  mode === value
                    ? "bg-slate-700 text-white"
                    : "text-slate-400 hover:bg-slate-900"
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
              aria-label="Semantic confidence threshold"
              className="w-full accent-cyan-400"
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
              aria-label="Semantic layer opacity"
              className="w-full accent-cyan-400"
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
            {visibleClassIndices.map((index) => (
              <label
                key={classNameForIndex(index, supportedClassNames)}
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
                  className={`size-3 shrink-0 rounded-sm border ${
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
                <span className="truncate">
                  {classLabel(index, laneLabel)}
                </span>
              </label>
            ))}
          </fieldset>

          {mode === "error" && hasTeacher && (
            <div className="flex gap-4 font-mono text-[9px] text-slate-500">
              <span className="flex items-center gap-1">
                <span className="size-2 bg-[#ff365f]" /> FP
              </span>
              <span className="flex items-center gap-1">
                <span className="size-2 bg-[#35dff4]" /> FN
              </span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
