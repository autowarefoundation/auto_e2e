"use client";

import { useMemo, useState } from "react";
import { AlertTriangle, Database, X } from "lucide-react";

import { useApi } from "@/hooks/use-api";
import { getODDEvidence, getODDScene } from "@/lib/api";
import type {
  ODDLabelEvidence,
  ODDObservation,
  ODDStatus,
} from "@/types";

const STATUS_STYLE: Record<ODDStatus, string> = {
  valid: "border-emerald-800 bg-emerald-950/30 text-emerald-300",
  unavailable: "border-slate-700 bg-slate-900 text-slate-400",
  not_observable: "border-amber-800 bg-amber-950/30 text-amber-300",
  ambiguous: "border-rose-800 bg-rose-950/30 text-rose-300",
};
const CATEGORIES = [
  { id: "odd", label: "ODD" },
  { id: "event", label: "Events" },
  { id: "perception", label: "Perception" },
] as const;

type Category = (typeof CATEGORIES)[number]["id"];

function observationValue(observation: ODDObservation): string {
  return observation.status === "valid"
    ? observation.values.join(", ")
    : observation.status;
}

function evidenceValue(evidence: ODDLabelEvidence): string {
  if (evidence.status === "valid") return evidence.values.join(", ");
  if (evidence.candidate_values.length > 0) {
    return evidence.candidate_values
      .map((candidate) => `${candidate.value} ${candidate.score.toFixed(2)}`)
      .join(", ");
  }
  return evidence.status;
}

function EvidenceRows({
  title,
  evidence,
  conflict = false,
}: {
  title: string;
  evidence: ODDLabelEvidence[];
  conflict?: boolean;
}) {
  if (evidence.length === 0) return null;
  return (
    <section>
      <p
        className={`text-[10px] uppercase ${
          conflict ? "text-rose-500" : "text-slate-600"
        }`}
      >
        {title}
      </p>
      <div className="mt-1 divide-y divide-slate-900">
        {evidence.map((item) => (
          <div
            key={item.evidence_uid}
            className="grid gap-2 py-2 sm:grid-cols-[minmax(0,1fr)_auto]"
          >
            <div className="min-w-0">
              <p className="font-mono text-[11px] text-slate-300">
                {item.source} · {item.provenance.labeler_name}
              </p>
              <p className="mt-1 break-words font-mono text-[10px] text-slate-500">
                {item.provenance.model_name
                  ? `${item.provenance.model_provider} / ${item.provenance.model_name} / ${item.provenance.model_revision}`
                  : item.provenance.labeler_version}
              </p>
              {item.measurements.length > 0 && (
                <p className="mt-1 text-[10px] text-slate-600">
                  {item.measurements
                    .map(
                      (measurement) =>
                        `${measurement.name}=${measurement.value} ${measurement.unit}`,
                    )
                    .join(" · ")}
                </p>
              )}
            </div>
            <div className="text-right">
              <p className="font-mono text-[11px] text-slate-300">
                {evidenceValue(item)}
              </p>
              <p className="mt-1 font-mono text-[10px] text-slate-600">
                {item.confidence.toFixed(2)}
              </p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function EvidenceDetail({
  dataset,
  version,
  sceneUID,
  observation,
  onClose,
}: {
  dataset: string;
  version: string;
  sceneUID: string;
  observation: ODDObservation;
  onClose: () => void;
}) {
  const result = useApi(
    () =>
      getODDEvidence(
        dataset,
        version,
        sceneUID,
        observation.observation_uid,
      ),
    [dataset, version, sceneUID, observation.observation_uid],
  );
  return (
    <aside className="border-l-2 border-cyan-900 bg-slate-950/60 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="break-words font-mono text-xs text-slate-200">
            {observation.key}
          </p>
          <p className="mt-1 font-mono text-[10px] text-slate-500">
            {observation.observation_uid}
          </p>
        </div>
        <button
          type="button"
          title="Close evidence"
          aria-label="Close evidence"
          onClick={onClose}
          className="grid size-7 shrink-0 place-items-center text-slate-600 hover:text-slate-200"
        >
          <X className="size-4" />
        </button>
      </div>
      {result.loading ? (
        <p className="mt-3 text-xs text-slate-500">Loading evidence...</p>
      ) : result.error ? (
        <div className="mt-3 flex items-center gap-2 text-xs text-rose-400">
          <AlertTriangle className="size-4" />
          Evidence unavailable
        </div>
      ) : result.data ? (
        <div className="mt-4 space-y-4">
          <EvidenceRows
            title="Supporting evidence"
            evidence={result.data.supporting_evidence}
          />
          <EvidenceRows
            title="Conflicting evidence"
            evidence={result.data.conflicting_evidence}
            conflict
          />
          {result.data.related_events.map((event) => (
            <section key={event.event_uid}>
              <p className="text-[10px] uppercase text-slate-600">
                Event phases
              </p>
              <div className="mt-2 flex min-h-6">
                {event.phases.map((phase) => {
                  const eventDuration = Math.max(
                    1,
                    event.end_timestamp_ns - event.start_timestamp_ns,
                  );
                  return (
                    <span
                      key={phase.phase}
                      className="grid place-items-center border-r border-slate-950 bg-amber-950 px-1 font-mono text-[9px] text-amber-300"
                      style={{
                        width: `${
                          ((phase.end_timestamp_ns -
                            phase.start_timestamp_ns) /
                            eventDuration) *
                          100
                        }%`,
                      }}
                    >
                      {phase.phase}
                    </span>
                  );
                })}
              </div>
            </section>
          ))}
          <p className="break-all font-mono text-[9px] text-slate-700">
            LabelSet {result.data.labelset_id} · manifest{" "}
            {result.data.manifest_sha256}
          </p>
        </div>
      ) : null}
    </aside>
  );
}

export function ODDLabelPanel({
  dataset,
  version,
  sceneUID,
  timestampNS,
  frame,
  fps,
  onSeek,
}: {
  dataset: string;
  version: string;
  sceneUID: string;
  timestampNS?: string;
  frame: number;
  fps: number;
  onSeek?: (frame: number) => void;
}) {
  const [mode, setMode] = useState<"current" | "whole">("current");
  const [category, setCategory] = useState<Category>("odd");
  const [selected, setSelected] = useState<ODDObservation | null>(null);
  const result = useApi(
    () => getODDScene(dataset, version, sceneUID),
    [dataset, version, sceneUID],
  );
  const playhead = result.data
    ? timestampNS
      ? Number(timestampNS)
      : result.data.start_timestamp_ns + (frame / fps) * 1e9
    : 0;
  const categoryObservations = useMemo(
    () =>
      result.data?.observations.filter((item) =>
        item.key.startsWith(`${category}.`),
      ) ?? [],
    [result.data, category],
  );
  const observations = useMemo(() => {
    if (mode === "whole") return categoryObservations;
    return categoryObservations.filter(
      (item) =>
        item.start_timestamp_ns <= playhead &&
        playhead < item.end_timestamp_ns,
    );
  }, [categoryObservations, mode, playhead]);
  const routeAction = result.data?.observations.find(
    (item) =>
      item.key === "odd.route.action" &&
      (mode === "whole" ||
        (item.start_timestamp_ns <= playhead &&
          playhead < item.end_timestamp_ns)),
  );
  const egoManeuver = result.data?.observations.find(
    (item) =>
      item.key === "event.ego.maneuver" &&
      (mode === "whole" ||
        (item.start_timestamp_ns <= playhead &&
          playhead < item.end_timestamp_ns)),
  );

  function selectObservation(observation: ODDObservation) {
    setSelected(observation);
    if (result.data && onSeek) {
      const seekFrame =
        ((observation.start_timestamp_ns - result.data.start_timestamp_ns) /
          1e9) *
        fps;
      onSeek(seekFrame);
    }
  }

  return (
    <section className="space-y-4 border-y border-slate-800 py-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-[10px] uppercase text-slate-500">ODD Labels</p>
          <p className="mt-1 font-mono text-[10px] text-slate-600">
            scene {sceneUID}
          </p>
        </div>
        <div className="flex border border-slate-800">
          {(["current", "whole"] as const).map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => setMode(item)}
              className={`h-7 px-2.5 text-[11px] ${
                mode === item
                  ? "bg-slate-800 text-slate-100"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {item === "current" ? "Current time" : "Whole scene"}
            </button>
          ))}
        </div>
      </div>

      {(routeAction || egoManeuver) && (
        <div className="grid gap-3 border-y border-slate-900 py-3 sm:grid-cols-2">
          <div>
            <p className="text-[9px] uppercase text-slate-600">
              Planned route
            </p>
            <p className="mt-1 font-mono text-xs text-cyan-300">
              {routeAction ? observationValue(routeAction) : "not observed"}
            </p>
          </div>
          <div>
            <p className="text-[9px] uppercase text-slate-600">
              Executed maneuver
            </p>
            <p className="mt-1 font-mono text-xs text-amber-300">
              {egoManeuver ? observationValue(egoManeuver) : "not observed"}
            </p>
          </div>
        </div>
      )}

      <div className="flex max-w-full overflow-x-auto border-b border-slate-900">
        {CATEGORIES.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => {
              setCategory(item.id);
              setSelected(null);
            }}
            className={`h-8 shrink-0 border-b-2 px-3 text-xs ${
              category === item.id
                ? "border-cyan-500 text-slate-100"
                : "border-transparent text-slate-500"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      {result.loading ? (
        <p className="text-xs text-slate-500">Loading ODD labels...</p>
      ) : result.error ? (
        <p className="text-xs text-slate-500">
          No ready ODD LabelSet for this scene.
        </p>
      ) : observations.length === 0 ? (
        <p className="text-xs text-slate-500">
          No {category} observations overlap this scope.
        </p>
      ) : (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(20rem,0.7fr)]">
          <div className="divide-y divide-slate-900">
            {observations.map((item) => {
              const sceneDuration = Math.max(
                1,
                (result.data?.end_timestamp_ns ?? 1) -
                  (result.data?.start_timestamp_ns ?? 0),
              );
              const left =
                ((item.start_timestamp_ns -
                  (result.data?.start_timestamp_ns ?? 0)) /
                  sceneDuration) *
                100;
              const width =
                ((item.end_timestamp_ns - item.start_timestamp_ns) /
                  sceneDuration) *
                100;
              return (
                <button
                  key={item.observation_uid}
                  type="button"
                  onClick={() => selectObservation(item)}
                  className={`grid w-full min-w-0 gap-2 py-3 text-left sm:grid-cols-[minmax(0,1fr)_10rem] ${
                    selected?.observation_uid === item.observation_uid
                      ? "bg-slate-950"
                      : ""
                  }`}
                >
                  <div className="min-w-0">
                    <p className="break-words font-mono text-[11px] text-slate-300">
                      {item.key}
                      {item.camera_id ? ` · ${item.camera_id}` : ""}
                      {item.actor_track_uid
                        ? ` · ${item.actor_track_uid}`
                        : ""}
                    </p>
                    <p className="mt-1 text-xs text-slate-500">
                      {item.source} · confidence {item.confidence.toFixed(2)}
                    </p>
                    <div className="relative mt-2 h-1 bg-slate-900">
                      <span
                        className="absolute top-0 h-full bg-cyan-700"
                        style={{
                          left: `${Math.max(0, left)}%`,
                          width: `${Math.max(
                            1,
                            Math.min(100 - left, width),
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                  <span
                    className={`h-fit max-w-full break-words border px-2 py-1 text-right font-mono text-[10px] ${STATUS_STYLE[item.status]}`}
                  >
                    {observationValue(item)}
                  </span>
                </button>
              );
            })}
          </div>
          {selected ? (
            <EvidenceDetail
              dataset={dataset}
              version={version}
              sceneUID={sceneUID}
              observation={selected}
              onClose={() => setSelected(null)}
            />
          ) : (
            <div className="hidden min-h-32 place-items-center border-l border-slate-900 text-xs text-slate-700 xl:grid">
              <span className="flex items-center gap-2">
                <Database className="size-4" />
                Evidence
              </span>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
