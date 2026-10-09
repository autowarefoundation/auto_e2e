"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Box,
  Camera,
  ChevronLeft,
  ChevronRight,
  Database,
  Loader2,
  ShieldCheck,
} from "lucide-react";

import { CameraImage } from "@/components/camera-image";
import { ErrorState } from "@/components/error-state";
import { SemanticOccupancyView } from "@/components/player/semantic-occupancy-view";
import { Button } from "@/components/ui/button";
import { useApi } from "@/hooks/use-api";
import {
  getShardIndex,
  getShardSemanticOccupancy,
  listDatasets,
  listDatasetVersions,
  listShardsForEpisode,
  listShardSemanticOccupancyModels,
} from "@/lib/api";
import { friendlyDataset } from "@/lib/format";
import {
  parseSemanticOccupancy,
  resolveSemanticOccupancyRows,
  type SemanticOccupancyArtifact,
} from "@/lib/semantic-occupancy";
import { gridDimensions, rigCam } from "@/lib/rig";
import type {
  SemanticOccupancyModel,
  ShardIndex,
} from "@/types";

const SELECT_CLASS =
  "h-9 min-w-0 rounded border border-slate-700 bg-slate-950 px-2 text-xs text-slate-200 outline-none focus:border-cyan-500 disabled:cursor-not-allowed disabled:text-slate-600";

type OccupancyStatus =
  | "idle"
  | "loading"
  | "ready"
  | "unavailable"
  | "error";

function shortDigest(value: string): string {
  return value.length > 16
    ? `${value.slice(0, 10)}...${value.slice(-6)}`
    : value;
}

export default function OccupancyPage() {
  const datasets = useApi(listDatasets);
  const [dataset, setDataset] = useState("");
  const [version, setVersion] = useState("");
  const [shard, setShard] = useState("");
  const [modelID, setModelID] = useState("");
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const available = datasets.data ?? [];
    if (!available.length) return;
    if (!available.some((entry) => entry.name === dataset)) {
      setDataset(
        available.find((entry) => entry.name === "kitscenes")?.name ??
          available[0].name,
      );
    }
  }, [dataset, datasets.data]);

  const versions = useApi(
    async () => ({
      dataset,
      items: await listDatasetVersions(dataset),
    }),
    [dataset],
    Boolean(dataset),
  );
  const versionItems = useMemo(
    () =>
      versions.data?.dataset === dataset ? versions.data.items : [],
    [dataset, versions.data],
  );

  useEffect(() => {
    if (!versionItems.length) return;
    if (!versionItems.some((entry) => entry.version === version)) {
      const advertised = datasets.data?.find(
        (entry) => entry.name === dataset,
      )?.version;
      setVersion(
        versionItems.find((entry) => entry.version === advertised)
          ?.version ?? versionItems[0].version,
      );
    }
  }, [dataset, datasets.data, version, versionItems]);

  const shards = useApi(
    async () => ({
      dataset,
      version,
      items: await listShardsForEpisode(dataset, version),
    }),
    [dataset, version],
    Boolean(dataset && version),
  );
  const shardItems = useMemo(
    () =>
      shards.data?.dataset === dataset && shards.data.version === version
        ? shards.data.items
        : [],
    [dataset, shards.data, version],
  );

  useEffect(() => {
    if (!shardItems.length) return;
    if (!shardItems.some((entry) => entry.name === shard)) {
      setShard(shardItems[0].name);
    }
  }, [shard, shardItems]);

  const indexResult = useApi(
    async () => ({
      dataset,
      version,
      shard,
      index: await getShardIndex(dataset, shard, version),
    }),
    [dataset, version, shard],
    Boolean(dataset && version && shard),
  );
  const index: ShardIndex | null =
    indexResult.data?.dataset === dataset &&
    indexResult.data.version === version &&
    indexResult.data.shard === shard
      ? indexResult.data.index
      : null;

  const models = useApi(
    () => listShardSemanticOccupancyModels(dataset, shard, version),
    [dataset, version, shard],
    Boolean(dataset && version && shard),
  );
  const modelItems = useMemo(
    () =>
      models.data?.dataset === dataset &&
      models.data.version === version &&
      models.data.shard === shard
        ? models.data.models
        : [],
    [dataset, models.data, shard, version],
  );

  useEffect(() => {
    if (!modelItems.length) {
      setModelID("");
      return;
    }
    if (!modelItems.some((entry) => entry.model_artifact_id === modelID)) {
      setModelID(modelItems[0].model_artifact_id);
    }
  }, [modelID, modelItems]);

  useEffect(() => {
    setFrame((current) =>
      Math.min(Math.max(current, 0), Math.max(0, (index?.samples.length ?? 1) - 1)),
    );
  }, [index]);

  const [artifact, setArtifact] =
    useState<SemanticOccupancyArtifact | null>(null);
  const [rows, setRows] = useState<Map<string, number>>(new Map());
  const [occupancyStatus, setOccupancyStatus] =
    useState<OccupancyStatus>("idle");
  const [occupancyError, setOccupancyError] = useState<Error | null>(null);
  const [artifactGeneration, setArtifactGeneration] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setArtifact(null);
    setRows(new Map());
    setOccupancyError(null);
    if (!dataset || !version || !shard || !modelID || !index) {
      setOccupancyStatus(
        models.loading ? "loading" : modelItems.length ? "idle" : "unavailable",
      );
      return;
    }
    setOccupancyStatus("loading");
    getShardSemanticOccupancy(dataset, shard, modelID, version)
      .then(async (buffer) => {
        const parsed = parseSemanticOccupancy(buffer);
        const resolvedRows = await resolveSemanticOccupancyRows(
          parsed,
          index.samples.map((sample) => sample.sample_uid || sample.key),
        );
        if (cancelled) return;
        setArtifact(parsed);
        setRows(resolvedRows);
        setOccupancyStatus("ready");
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setArtifact(null);
        setRows(new Map());
        setOccupancyError(
          error instanceof Error ? error : new Error(String(error)),
        );
        setOccupancyStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [
    artifactGeneration,
    dataset,
    index,
    modelID,
    modelItems.length,
    models.loading,
    shard,
    version,
  ]);

  const selectedModel: SemanticOccupancyModel | undefined =
    modelItems.find((entry) => entry.model_artifact_id === modelID);
  const sample = index?.samples[frame];
  const sampleUID = sample?.sample_uid || sample?.key;
  const row = sampleUID ? rows.get(sampleUID) : undefined;
  const cameras = useMemo(
    () =>
      Object.keys(sample?.members ?? {})
        .map((member) => /^cam_(\d+)\.jpg$/.exec(member))
        .filter((match): match is RegExpExecArray => Boolean(match))
        .map((match) => Number(match[1]))
        .sort((left, right) => left - right),
    [sample],
  );
  const cameraMembers = useMemo(
    () => cameras.map((camera) => `cam_${camera}`),
    [cameras],
  );
  const cameraGrid = useMemo(
    () =>
      gridDimensions(
        dataset,
        cameraMembers,
        cameraMembers.length,
      ),
    [cameraMembers, dataset],
  );

  const resetCoordinates = (
    nextDataset: string,
    nextVersion = "",
    nextShard = "",
  ) => {
    setDataset(nextDataset);
    setVersion(nextVersion);
    setShard(nextShard);
    setModelID("");
    setFrame(0);
    setArtifact(null);
    setRows(new Map());
  };

  return (
    <div className="min-w-0 space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">
            Occupancy
          </h2>
          <p className="text-sm text-slate-400">
            KITScenes camera evidence and model-native BEV occupancy at the
            same frame.
          </p>
        </div>
        <div className="flex items-center gap-2 font-mono text-[10px] text-slate-500">
          <ShieldCheck className="size-3.5 text-emerald-400" />
          no viewer geometry correction
        </div>
      </div>

      <section
        aria-label="Occupancy selection"
        className="grid min-w-0 gap-3 border-y border-slate-800 py-4 sm:grid-cols-2 xl:grid-cols-[1fr_1fr_1.5fr_1.5fr]"
      >
        <label className="min-w-0 space-y-1 text-[10px] text-slate-500">
          <span className="flex items-center gap-1">
            <Database className="size-3" /> Dataset
          </span>
          <select
            aria-label="Occupancy dataset"
            className={`${SELECT_CLASS} w-full`}
            value={dataset}
            disabled={datasets.loading || Boolean(datasets.error)}
            onChange={(event) => resetCoordinates(event.target.value)}
          >
            {!datasets.data?.length && <option value="">Unavailable</option>}
            {(datasets.data ?? []).map((entry) => (
              <option key={entry.name} value={entry.name}>
                {friendlyDataset(entry.name)}
              </option>
            ))}
          </select>
        </label>
        <label className="min-w-0 space-y-1 text-[10px] text-slate-500">
          <span>Version</span>
          <select
            aria-label="Occupancy dataset version"
            className={`${SELECT_CLASS} w-full font-mono`}
            value={version}
            disabled={versions.loading || Boolean(versions.error)}
            onChange={(event) =>
              resetCoordinates(dataset, event.target.value)
            }
          >
            {!versionItems.length && <option value="">Unavailable</option>}
            {versionItems.map((entry) => (
              <option key={entry.version} value={entry.version}>
                {entry.version}
              </option>
            ))}
          </select>
        </label>
        <label className="min-w-0 space-y-1 text-[10px] text-slate-500">
          <span>Shard</span>
          <select
            aria-label="Occupancy shard"
            className={`${SELECT_CLASS} w-full font-mono`}
            value={shard}
            disabled={shards.loading || Boolean(shards.error)}
            onChange={(event) =>
              resetCoordinates(dataset, version, event.target.value)
            }
          >
            {!shardItems.length && <option value="">Unavailable</option>}
            {shardItems.map((entry) => (
              <option key={entry.name} value={entry.name}>
                {entry.name}
              </option>
            ))}
          </select>
        </label>
        <label className="min-w-0 space-y-1 text-[10px] text-slate-500">
          <span>Model</span>
          <select
            aria-label="Occupancy model"
            className={`${SELECT_CLASS} w-full`}
            value={modelID}
            disabled={models.loading || Boolean(models.error)}
            onChange={(event) => setModelID(event.target.value)}
          >
            {!modelItems.length && <option value="">Unavailable</option>}
            {modelItems.map((entry) => (
              <option
                key={entry.model_artifact_id}
                value={entry.model_artifact_id}
              >
                {entry.display_name}
              </option>
            ))}
          </select>
        </label>
      </section>

      {datasets.error ? (
        <ErrorState error={datasets.error} onRetry={datasets.reload} />
      ) : versions.error ? (
        <ErrorState error={versions.error} onRetry={versions.reload} />
      ) : shards.error ? (
        <ErrorState error={shards.error} onRetry={shards.reload} />
      ) : indexResult.error ? (
        <ErrorState error={indexResult.error} onRetry={indexResult.reload} />
      ) : models.error ? (
        <ErrorState error={models.error} onRetry={models.reload} />
      ) : null}

      <section aria-label="Frame evidence" className="min-w-0 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="flex items-center gap-2 text-sm font-medium text-slate-200">
              <Camera className="size-4 text-cyan-400" />
              Camera evidence
            </h3>
            <p className="font-mono text-[10px] text-slate-500">
              {sampleUID ?? "Waiting for shard index"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon-sm"
              aria-label="Previous occupancy frame"
              title="Previous frame"
              disabled={!index || frame <= 0}
              onClick={() => setFrame((value) => Math.max(0, value - 1))}
            >
              <ChevronLeft className="size-4" />
            </Button>
            <label className="flex min-w-36 items-center gap-2 font-mono text-[10px] text-slate-400">
              <input
                aria-label="Occupancy frame"
                className="min-w-0 flex-1 accent-cyan-400"
                type="range"
                min={0}
                max={Math.max(0, (index?.samples.length ?? 1) - 1)}
                value={frame}
                disabled={!index?.samples.length}
                onChange={(event) => setFrame(Number(event.target.value))}
              />
              <span className="w-20 text-right">
                {index?.samples.length
                  ? `${frame + 1}/${index.samples.length}`
                  : "-/-"}
              </span>
            </label>
            <Button
              variant="outline"
              size="icon-sm"
              aria-label="Next occupancy frame"
              title="Next frame"
              disabled={!index || frame >= index.samples.length - 1}
              onClick={() =>
                setFrame((value) =>
                  Math.min((index?.samples.length ?? 1) - 1, value + 1),
                )
              }
            >
              <ChevronRight className="size-4" />
            </Button>
          </div>
        </div>

        {indexResult.loading || !index ? (
          <div className="flex h-32 items-center justify-center border-y border-slate-800 text-slate-500">
            <Loader2 className="size-4 animate-spin" />
          </div>
        ) : (
          <div
            className="grid min-w-0 gap-2"
            role="group"
            aria-label="Camera evidence mosaic"
            style={{
              gridTemplateColumns: `repeat(${cameraGrid.cols}, minmax(0, 1fr))`,
              gridTemplateRows: `repeat(${cameraGrid.rows}, auto)`,
            }}
          >
            {cameras.map((camera, index) => {
              const member = `cam_${camera}`;
              const rig = rigCam(
                dataset,
                member,
                index,
                cameraMembers.length,
              );
              return (
                <figure
                  key={camera}
                  className="min-w-0 overflow-hidden rounded-md border border-slate-800 bg-slate-950"
                  style={{ gridRow: rig.row, gridColumn: rig.col }}
                >
                  <CameraImage
                    dataset={dataset}
                    version={version}
                    shard={shard}
                    sampleKey={sample!.key}
                    cam={camera}
                    range={sample!.members[`${member}.jpg`]}
                    className="aspect-square w-full"
                  />
                  <figcaption className="border-t border-slate-800 px-2 py-1 font-mono text-[9px] text-slate-500">
                    {rig.label}
                  </figcaption>
                </figure>
              );
            })}
          </div>
        )}
      </section>

      {occupancyError && (
        <ErrorState
          error={occupancyError}
          onRetry={() => setArtifactGeneration((value) => value + 1)}
        />
      )}
      <SemanticOccupancyView
        artifact={artifact}
        laneLabel={
          selectedModel?.model_family === "HENet"
            ? "Lane divider"
            : undefined
        }
        row={row}
        status={
          occupancyStatus === "ready" && row === undefined
            ? "unavailable"
            : occupancyStatus
        }
        supportedClasses={selectedModel?.supported_classes}
      />

      {selectedModel && (
        <section
          aria-label="Occupancy model provenance"
          className="grid min-w-0 gap-4 border-y border-slate-800 py-4 lg:grid-cols-[minmax(0,1fr)_minmax(280px,0.7fr)]"
        >
          <div className="min-w-0">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <Box className="size-4 text-emerald-400" />
              <h3 className="text-sm font-medium text-slate-100">
                {selectedModel.display_name}
              </h3>
              <span className="border border-slate-700 px-1.5 py-0.5 font-mono text-[9px] uppercase text-slate-400">
                {selectedModel.artifact_kind}
              </span>
            </div>
            <dl className="grid min-w-0 gap-x-4 gap-y-1 font-mono text-[10px] sm:grid-cols-2">
              <dt className="text-slate-500">Artifact</dt>
              <dd className="min-w-0 truncate text-slate-300">
                {shortDigest(selectedModel.model_artifact_id)}
              </dd>
              <dt className="text-slate-500">Weight</dt>
              <dd className="min-w-0 truncate text-slate-300">
                {shortDigest(selectedModel.model_source.weight_sha256)}
              </dd>
              <dt className="text-slate-500">Head</dt>
              <dd className="min-w-0 break-all text-slate-300">
                {selectedModel.head_version}
              </dd>
              <dt className="text-slate-500">Input</dt>
              <dd className="min-w-0 break-all text-slate-300">
                {selectedModel.input_contract}
              </dd>
              <dt className="text-slate-500">Code license</dt>
              <dd className="text-slate-300">
                {selectedModel.model_source.code_license_spdx}
              </dd>
              <dt className="text-slate-500">Weight license</dt>
              <dd className="text-slate-300">
                {selectedModel.model_source.license_spdx}
              </dd>
              <dt className="text-slate-500">Training data license</dt>
              <dd className="text-slate-300">
                {selectedModel.model_source.training_data_license_spdx}
              </dd>
              <dt className="text-slate-500">Weight source</dt>
              <dd className="min-w-0 break-all text-slate-300">
                {selectedModel.model_source.weight_source_url.startsWith(
                  "https://",
                ) ? (
                  <a
                    className="text-cyan-400 hover:text-cyan-300"
                    href={selectedModel.model_source.weight_source_url}
                    rel="noreferrer"
                    target="_blank"
                  >
                    source
                  </a>
                ) : (
                  selectedModel.model_source.weight_source_url
                )}
              </dd>
              <dt className="text-slate-500">Repository</dt>
              <dd className="min-w-0 truncate">
                <a
                  className="text-cyan-400 hover:text-cyan-300"
                  href={selectedModel.model_source.repository}
                  rel="noreferrer"
                  target="_blank"
                >
                  {selectedModel.model_family}
                </a>
              </dd>
              <dt className="text-slate-500">Config</dt>
              <dd className="min-w-0 break-all text-slate-300">
                {selectedModel.model_source.config}
              </dd>
              <dt className="text-slate-500">Producer</dt>
              <dd className="min-w-0 break-all text-slate-300">
                {JSON.stringify(selectedModel.producer_config)}
              </dd>
              <dt className="text-slate-500">Teacher / Error</dt>
              <dd className="text-slate-300">
                {selectedModel.teacher_available ? "available" : "unavailable"}
              </dd>
            </dl>
          </div>
          <div className="min-w-0">
            <p className="mb-1 text-[10px] uppercase text-slate-500">
              Scientific limitations
            </p>
            <ul className="space-y-1 text-xs leading-relaxed text-slate-400">
              {selectedModel.limitations.map((limitation) => (
                <li key={limitation}>{limitation}</li>
              ))}
            </ul>
          </div>
        </section>
      )}
    </div>
  );
}
