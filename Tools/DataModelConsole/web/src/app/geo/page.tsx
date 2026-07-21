"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
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

interface VersionCatalogState {
  dataset: string;
  items: DatasetVersion[];
  loading: boolean;
  error: Error | null;
}

interface GeoPageState {
  dataset: string;
  version: string;
  stats: GeoStats | null;
  heatmap: GeoJSONFeatureCollection | null;
  loading: boolean;
  error: Error | null;
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function LoadingState({ label }: { label: string }) {
  return (
    <div className="space-y-4">
      <p className="flex items-center gap-2 text-xs text-slate-500">
        <Loader2 className="size-3.5 animate-spin" />
        {label}
      </p>
      <Skeleton className="h-20 w-full" />
      <Skeleton className="h-[520px] w-full" />
    </div>
  );
}

function GeoPageInner() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const searchParamsString = searchParams.toString();
  const urlDataset = searchParams.get("dataset") ?? "";
  const urlVersion = searchParams.get("version") ?? "";

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState<Error | null>(null);
  const [catalogReload, setCatalogReload] = useState(0);
  const [versionCatalog, setVersionCatalog] =
    useState<VersionCatalogState>({
      dataset: "",
      items: [],
      loading: false,
      error: null,
    });
  const [versionsReload, setVersionsReload] = useState(0);
  const [geoReload, setGeoReload] = useState(0);
  const [geo, setGeo] = useState<GeoPageState>({
    dataset: "",
    version: "",
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
        setCatalogLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setDatasets([]);
        setCatalogError(toError(err));
        setCatalogLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [catalogReload]);

  const dataset = useMemo(() => {
    if (catalogLoading || catalogError || datasets.length === 0) return "";
    return (
      datasets.find((item) => item.name === urlDataset)?.name ??
      datasets.find((item) => item.name === "kitscenes")?.name ??
      datasets[0].name
    );
  }, [catalogError, catalogLoading, datasets, urlDataset]);

  useEffect(() => {
    if (!dataset) {
      setVersionCatalog({
        dataset: "",
        items: [],
        loading: false,
        error: null,
      });
      return;
    }
    let cancelled = false;
    setVersionCatalog({
      dataset,
      items: [],
      loading: true,
      error: null,
    });
    listDatasetVersions(dataset)
      .then((items) => {
        if (cancelled) return;
        setVersionCatalog({
          dataset,
          items,
          loading: false,
          error: null,
        });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setVersionCatalog({
          dataset,
          items: [],
          loading: false,
          error: toError(err),
        });
      });
    return () => {
      cancelled = true;
    };
  }, [dataset, versionsReload]);

  const hasCurrentVersionCatalog = versionCatalog.dataset === dataset;
  const versionsLoading =
    dataset !== "" &&
    (!hasCurrentVersionCatalog || versionCatalog.loading);
  const versionsError = hasCurrentVersionCatalog
    ? versionCatalog.error
    : null;
  const versions = useMemo(
    () =>
      hasCurrentVersionCatalog &&
      !versionCatalog.loading &&
      !versionCatalog.error
        ? versionCatalog.items.filter((item) => item.has_gps)
        : [],
    [hasCurrentVersionCatalog, versionCatalog],
  );
  const version = useMemo(
    () =>
      versions.find((item) => item.version === urlVersion)?.version ??
      versions[0]?.version ??
      "",
    [urlVersion, versions],
  );

  useEffect(() => {
    if (
      catalogLoading ||
      catalogError ||
      !dataset ||
      versionsLoading ||
      versionsError
    ) {
      return;
    }

    const query = new URLSearchParams(searchParamsString);
    let changed = false;
    if (query.get("dataset") !== dataset) {
      query.set("dataset", dataset);
      changed = true;
    }
    if (version) {
      if (query.get("version") !== version) {
        query.set("version", version);
        changed = true;
      }
    } else if (query.has("version")) {
      query.delete("version");
      changed = true;
    }
    if (changed) {
      router.replace(`${pathname}?${query.toString()}`, { scroll: false });
    }
  }, [
    catalogError,
    catalogLoading,
    dataset,
    pathname,
    router,
    searchParamsString,
    version,
    versionsError,
    versionsLoading,
  ]);

  const navigateSelection = useCallback(
    (nextDataset: string, nextVersion: string | null) => {
      const query = new URLSearchParams(searchParamsString);
      query.set("dataset", nextDataset);
      if (nextVersion) {
        query.set("version", nextVersion);
      } else {
        query.delete("version");
      }
      router.push(`${pathname}?${query.toString()}`, { scroll: false });
    },
    [pathname, router, searchParamsString],
  );

  useEffect(() => {
    if (!dataset || !version) {
      return;
    }
    let cancelled = false;
    setGeo({
      dataset,
      version,
      stats: null,
      heatmap: null,
      loading: true,
      error: null,
    });
    void (async () => {
      try {
        const stats = await getGeoStats(dataset, version);
        if (cancelled) return;
        const heatmap = stats.heatmap_url
          ? await getGeoHeatmap(stats.heatmap_url)
          : { type: "FeatureCollection" as const, features: [] };
        if (cancelled) return;
        setGeo({
          dataset,
          version,
          stats,
          heatmap,
          loading: false,
          error: null,
        });
      } catch (err: unknown) {
        if (cancelled) return;
        setGeo({
          dataset,
          version,
          stats: null,
          heatmap: null,
          loading: false,
          error: toError(err),
        });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [dataset, geoReload, version]);

  const activeGeo =
    geo.dataset === dataset && geo.version === version ? geo : null;
  const geoLoading =
    version !== "" && (!activeGeo || activeGeo.loading);

  const mapView = useMemo(() => {
    const bbox = activeGeo?.stats?.summary.bbox;
    return bbox
      ? fitGeoBounds(bbox, 960, 520, 5, 14)
      : {
          center: { latitude: 0, longitude: 0 },
          zoom: 5,
        };
  }, [activeGeo?.stats]);

  const markers = useMemo<MapMarker[]>(() => {
    const features = activeGeo?.heatmap?.features ?? [];
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
  }, [activeGeo?.heatmap]);

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
            onChange={(event) =>
              navigateSelection(event.target.value, null)
            }
            disabled={catalogLoading || catalogError !== null}
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
            onChange={(event) =>
              navigateSelection(dataset, event.target.value)
            }
            disabled={
              versionsLoading ||
              versionsError !== null ||
              versions.length === 0
            }
            className="h-9 rounded-md border border-slate-700 bg-slate-950 px-3 font-mono text-sm text-slate-200"
          >
            {versions.map((item) => (
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
          onRetry={() => setCatalogReload((value) => value + 1)}
        />
      ) : catalogLoading ? (
        <LoadingState label="Loading datasets" />
      ) : versionsError ? (
        <ErrorState
          error={versionsError}
          onRetry={() => setVersionsReload((value) => value + 1)}
        />
      ) : versionsLoading ? (
        <LoadingState label="Loading dataset versions" />
      ) : !version ? (
        <p className="border-y border-slate-800 py-8 text-sm text-slate-500">
          This dataset has no published GPS-enabled version.
        </p>
      ) : activeGeo?.error ? (
        <ErrorState
          error={activeGeo.error}
          onRetry={() => setGeoReload((value) => value + 1)}
        />
      ) : geoLoading ? (
        <LoadingState label="Loading aggregate geospatial statistics" />
      ) : activeGeo?.stats ? (
        <>
          <div className="grid border-y border-slate-800 sm:grid-cols-3">
            {[
              [
                "Samples",
                activeGeo.stats.summary.sample_pose_count.toLocaleString(),
              ],
              [
                "Episodes",
                activeGeo.stats.summary.episode_count.toLocaleString(),
              ],
              [
                "Route points",
                activeGeo.stats.summary.path_point_count.toLocaleString(),
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
              <span>
                k &gt;={" "}
                {activeGeo.stats.summary.privacy?.k_anonymity ?? "-"}
              </span>
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

export default function GeoPage() {
  return (
    <Suspense fallback={<LoadingState label="Loading geographic coverage" />}>
      <GeoPageInner />
    </Suspense>
  );
}
