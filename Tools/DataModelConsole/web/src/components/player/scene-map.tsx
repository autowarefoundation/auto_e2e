"use client";

import { useEffect, useMemo, useState } from "react";

import {
  SlippyMap,
  type MapPath,
} from "@/components/map/slippy-map";
import { getEpisodeGPSPath } from "@/lib/api";
import { decodeEgo, integrateTrajectory } from "@/lib/ego";
import type { TrajectoryPoint } from "@/lib/ego";
import { decodeEpisodePath, egoTrajectoryToGeo } from "@/lib/geo";
import type { GeoPoint } from "@/lib/geo";
import type { IndexSample } from "@/types";

export function SceneMap({
  dataset,
  version,
  sample,
  predictionTrajectories,
  medianPrediction,
  curvatureSign,
}: {
  dataset: string;
  version?: string;
  sample?: IndexSample;
  predictionTrajectories: TrajectoryPoint[][];
  medianPrediction: TrajectoryPoint[];
  curvatureSign: 1 | -1;
}) {
  const pose = sample?.pose_current;
  const hasExactPose =
    pose !== undefined &&
    Number.isFinite(pose.latitude_deg) &&
    Number.isFinite(pose.longitude_deg) &&
    Number.isFinite(pose.heading_deg_cw_from_north);
  const [episodePath, setEpisodePath] = useState<GeoPoint[]>([]);

  useEffect(() => {
    if (!hasExactPose || !sample?.episode_id) {
      setEpisodePath([]);
      return;
    }
    let cancelled = false;
    getEpisodeGPSPath(dataset, sample.episode_id, version)
      .then(decodeEpisodePath)
      .then((path) => {
        if (!cancelled) setEpisodePath(path);
      })
      .catch(() => {
        if (!cancelled) setEpisodePath([]);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset, version, sample?.episode_id, hasExactPose]);

  const origin = useMemo<GeoPoint | null>(
    () =>
      hasExactPose && pose
        ? {
            latitude: pose.latitude_deg,
            longitude: pose.longitude_deg,
          }
        : null,
    [hasExactPose, pose],
  );
  const recordedFuture = useMemo(() => {
    if (!sample?.ego_future?.length) return [];
    const { future } = decodeEgo([], sample.ego_future);
    return integrateTrajectory(
      sample.ego_now?.[0] ?? 0,
      future.accel,
      future.curvature,
      0.1,
      "raw",
      curvatureSign,
    );
  }, [sample, curvatureSign]);

  if (!origin || !pose) return null;
  const heading = pose.heading_deg_cw_from_north;

  const paths: MapPath[] = [];
  if (episodePath.length > 1) {
    paths.push({
      id: "episode",
      points: episodePath,
      color: "#f59e0b",
      width: 2,
      opacity: 0.72,
    });
  }
  if (recordedFuture.length > 1) {
    paths.push({
      id: "recorded-future",
      points: egoTrajectoryToGeo(origin, heading, recordedFuture),
      color: "#3b82f6",
      width: 2,
      opacity: 0.9,
    });
  }
  predictionTrajectories.forEach((trajectory, index) => {
    paths.push({
      id: `prediction-${index}`,
      points: egoTrajectoryToGeo(origin, heading, trajectory),
      color: "#34d399",
      width: 1.25,
      opacity: 0.3,
    });
  });
  if (medianPrediction.length > 1) {
    paths.push({
      id: "prediction-median",
      points: egoTrajectoryToGeo(origin, heading, medianPrediction),
      color: "#6ee7b7",
      width: 3,
      opacity: 0.95,
    });
  }

  return (
    <section className="space-y-2" aria-label="Scene geographic map">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] text-slate-500">
        <span className="text-slate-300">Scene map</span>
        {episodePath.length > 1 && (
          <span className="text-amber-500">driven route</span>
        )}
        <span className="text-blue-500">recorded future</span>
        {medianPrediction.length > 1 && (
          <span className="text-emerald-400">model prediction</span>
        )}
        <span className="ml-auto">heading {heading.toFixed(1)} deg</span>
      </div>
      <SlippyMap
        center={origin}
        zoom={17}
        paths={paths}
        markers={[
          {
            id: "ego",
            point: origin,
            color: "#f8fafc",
            radius: 5,
            label: "Current ego position",
          },
        ]}
        minZoom={13}
        maxZoom={19}
        followCenter
        viewKey={`${dataset}:${version ?? ""}:${sample.episode_id}`}
        className="h-80"
        ariaLabel="Driven route with recorded and predicted trajectories"
      />
    </section>
  );
}
