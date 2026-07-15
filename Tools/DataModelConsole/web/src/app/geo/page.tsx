"use client";

import { useEffect, useMemo, useState } from "react";
import { Loader2, MapPinned, ShieldCheck } from "lucide-react";

import {
  SlippyMap,
  type MapMarker,
} from "@/components/map/slippy-map";
import { ErrorState } from "@/components/error-state";
import { Skeleton } from "@/components/ui/skeleton";
import {
  getGeoHeatmap,
  getGeoStats,
  listDatasets,
  listDatasetVersions,
} from "@/lib/api";
import { fitGeoBounds } from "@/lib/geo";
import type {
  Dataset,
  DatasetVersion,
  GeoJSONFeatureCollection,
  GeoStats,
} from "@/types";

interface GeoPageState {
  stats: GeoStats | null;
  heatmap: GeoJSONFeatureCollection | null;
  loading: boolean;
  error: Error | null;
}

export default function GeoPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [dataset, setDataset] = useState("");
  const [version, setVersion] = useState("");
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState<Error | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const [geo, setGeo] = useState<GeoPageState>({
    stats: null,
    heatmap: null,
    loading: false,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    setCatalogLoading(true);
    setCatalogError(null);
    listDatasets()
      .then((items) => {
        if (cancelled) return;
        setDatasets(items);
        setDataset((current) =>
          items.some((item) => item.name === current)
            ? current
            : (items.find((item) => item.name === "l2d")?.name ??
              items[0]?.name ??
              ""),
        );
        setCatalogLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setCatalogError(
          err instanceof Error ? err : new Error(String(err)),
        );
        setCatalogLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [reloadToken]);

  useEffect(() => {
    if (!dataset) {
      setVersions([]);
      setVersion("");
      return;
    }
    let cancelled = false;
    listDatasetVersions(dataset)
      .then((items) => {
        if (cancelled) return;
        setVersions(items);
        setVersion((current) =>
          items.some((item) => item.version === current && item.has_gps)
            ? current
            : (items.find((item) => item.has_gps)?.version ?? ""),
        );
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setVersions([]);
        setVersion("");
        setGeo({
          stats: null,
          heatmap: null,
          loading: false,
          error: err instanceof Error ? err : new Error(String(err)),
        });
      });
    return () => {
      cancelled = true;
    };
  }, [dataset]);

  useEffect(() => {
    if (!dataset || !version) {
      setGeo({ stats: null, heatmap: null, loading: false, error: null });
      return;
    }
    let cancelled = false;
    setGeo({ stats: null, heatmap: null, loading: true, error: null });
    getGeoStats(dataset, version)
      .then(async (stats) => {
        const heatmap = stats.heatmap_url
          ? await getGeoHeatmap(stats.heatmap_url)
          : { type: "FeatureCollection" as const, features: [] };
        if (!cancelled) {
          setGeo({ stats, heatmap, loading: false, error: null });
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setGeo({
          stats: null,
          heatmap: null,
          loading: false,
          error: err instanceof Error ? err : new Error(String(err)),
        });
      });
    return () => {
      cancelled = true;
    };
  }, [dataset, version, reloadToken]);

  const mapView = useMemo(() => {
    const bbox = geo.stats?.summary.bbox;
    return bbox
      ? fitGeoBounds(bbox, 960, 520, 5, 14)
      : {
          center: { latitude: 0, longitude: 0 },
          zoom: 5,
        };
  }, [geo.stats]);

  const markers = useMemo<MapMarker[]>(() => {
    const features = geo.heatmap?.features ?? [];
    let maxCount = 1;
    for (const feature of features) {
      maxCount = Math.max(maxCount, feature.properties.sample_count);
    }
    return features.map((feature, index) => ({
      id: `cell-${index}`,
      point: {
        longitude: feature.geometry.coordinates[0],
        latitude: feature.geometry.coordinates[1],
      },
      color: "#10b981",
      radius:
        4 + 12 * Math.sqrt(feature.properties.sample_count / maxCount),
      opacity: 0.35,
      label: `${feature.properties.sample_count.toLocaleString()} samples | ${feature.properties.episode_count} episodes`,
    }));
  }, [geo.heatmap]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <MapPinned className="size-5 text-emerald-400" />
            <h2 className="text-lg font-semibold">Geographic coverage</h2>
          </div>
          <p className="mt-1 text-sm text-slate-400">
            Privacy-preserving ODD coverage from published dataset versions.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <label className="sr-only" htmlFor="geo-dataset">
            Dataset
          </label>
          <select
            id="geo-dataset"
            value={dataset}
            onChange={(event) => setDataset(event.target.value)}
            disabled={catalogLoading}
            className="h-9 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-slate-200"
          >
            {datasets.map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}
              </option>
            ))}
          </select>
          <label className="sr-only" htmlFor="geo-version">
            Dataset version
          </label>
          <select
            id="geo-version"
            value={version}
            onChange={(event) => setVersion(event.target.value)}
            disabled={versions.every((item) => !item.has_gps)}
            className="h-9 rounded-md border border-slate-700 bg-slate-950 px-3 font-mono text-sm text-slate-200"
          >
            {versions
              .filter((item) => item.has_gps)
              .map((item) => (
                <option key={item.version} value={item.version}>
                  {item.version}
                </option>
              ))}
          </select>
        </div>
      </div>

      {catalogError ? (
        <ErrorState
          error={catalogError}
          onRetry={() => setReloadToken((value) => value + 1)}
        />
      ) : catalogLoading || geo.loading ? (
        <div className="space-y-4">
          <p className="flex items-center gap-2 text-xs text-slate-500">
            <Loader2 className="size-3.5 animate-spin" />
            Loading aggregate geospatial statistics
          </p>
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-[520px] w-full" />
        </div>
      ) : !version ? (
        <p className="border-y border-slate-800 py-8 text-sm text-slate-500">
          This dataset has no published GPS-enabled version.
        </p>
      ) : geo.error ? (
        <ErrorState
          error={geo.error}
          onRetry={() => setReloadToken((value) => value + 1)}
        />
      ) : geo.stats ? (
        <>
          <div className="grid border-y border-slate-800 sm:grid-cols-3">
            {[
              [
                "Samples",
                geo.stats.summary.sample_pose_count.toLocaleString(),
              ],
              ["Episodes", geo.stats.summary.episode_count.toLocaleString()],
              [
                "Route points",
                geo.stats.summary.path_point_count.toLocaleString(),
              ],
            ].map(([label, value]) => (
              <div
                key={label}
                className="border-b border-slate-800 px-4 py-3 last:border-b-0 sm:border-r sm:border-b-0 sm:last:border-r-0"
              >
                <p className="text-[10px] uppercase text-slate-500">{label}</p>
                <p className="mt-1 font-mono text-lg text-slate-100">{value}</p>
              </div>
            ))}
          </div>

          <section className="space-y-2">
            <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
              <span className="text-slate-300">k-anonymous coverage cells</span>
              <span>k &gt;= {geo.stats.summary.privacy?.k_anonymity ?? "-"}</span>
              <span>{markers.length.toLocaleString()} published cells</span>
              <span className="ml-auto flex items-center gap-1 text-emerald-400">
                <ShieldCheck className="size-3.5" />
                endpoints excluded
              </span>
            </div>
            <SlippyMap
              center={mapView.center}
              zoom={mapView.zoom}
              markers={markers}
              minZoom={5}
              maxZoom={16}
              viewKey={`${dataset}:${version}`}
              className="h-[520px]"
              ariaLabel="Aggregate geographic dataset coverage"
            />
          </section>
        </>
      ) : null}
    </div>
  );
}
