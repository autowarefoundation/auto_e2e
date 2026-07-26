"use client";

import { useEffect, useMemo, useState } from "react";

import { getSampleNavigationMapURL } from "@/lib/api";
import type { IndexSample } from "@/types";

export function NavigationMap({
  dataset,
  shard,
  version,
  sample,
}: {
  dataset: string;
  shard: string;
  version?: string;
  sample: IndexSample | undefined;
}) {
  const hasMap = Boolean(
    sample &&
      (sample.members["map.jpg"] ||
        (sample.members["map_semantic.npz"] &&
          sample.members["route_mask.npz"])),
  );
  const source = useMemo(
    () =>
      sample && hasMap
        ? getSampleNavigationMapURL(dataset, shard, sample.key, version)
        : "",
    [dataset, hasMap, sample, shard, version],
  );
  const [status, setStatus] = useState<"loading" | "ready" | "error">(
    "loading",
  );

  useEffect(() => {
    setStatus("loading");
  }, [source]);

  return (
    <section
      className="rounded-md border border-slate-800 bg-slate-950 p-2"
      aria-label="Rendered navigation map"
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="text-xs font-medium text-slate-200">Navigation map</h3>
        <div className="flex items-center gap-2 text-[9px] text-slate-500">
          <span className="flex items-center gap-1">
            <span className="size-2 bg-teal-500" aria-hidden="true" />
            Route
          </span>
          <span className="flex items-center gap-1">
            <span className="size-2 bg-slate-300" aria-hidden="true" />
            Lane
          </span>
        </div>
      </div>
      {source ? (
        <div className="grid grid-cols-[14px_1fr] grid-rows-[1fr_14px]">
          <span className="flex items-center text-[9px] text-slate-500 [writing-mode:vertical-rl]">
            +X forward
          </span>
          <div className="relative aspect-square overflow-hidden border border-slate-800 bg-slate-950">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              key={source}
              src={source}
              alt="Ego-frame semantic navigation raster"
              className="size-full object-contain [image-rendering:pixelated]"
              onLoad={() => setStatus("ready")}
              onError={() => setStatus("error")}
            />
            {status === "loading" && (
              <div
                role="status"
                className="absolute inset-0 flex items-center justify-center bg-slate-950 text-xs text-slate-500"
              >
                Loading map...
              </div>
            )}
            {status === "error" && (
              <div
                role="status"
                className="absolute inset-0 flex items-center justify-center bg-slate-950 px-4 text-center text-xs text-rose-400"
              >
                Map render failed.
              </div>
            )}
            {status === "ready" && (
              <span
                className="absolute left-1/2 top-2/3 block size-2 -translate-x-1/2 -translate-y-1/2 rotate-45 border-l border-t border-white"
                aria-hidden="true"
              />
            )}
          </div>
          <span />
          <span className="text-center text-[9px] leading-[14px] text-slate-500">
            +Y left
          </span>
        </div>
      ) : (
        <div className="flex aspect-square items-center justify-center border border-dashed border-slate-800 px-5 text-center text-xs text-slate-500">
          No navigation raster for this sample.
        </div>
      )}
    </section>
  );
}
