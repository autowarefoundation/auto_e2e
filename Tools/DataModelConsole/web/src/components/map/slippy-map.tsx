"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { LocateFixed, ZoomIn, ZoomOut } from "lucide-react";

import { geoToWorldPixel, worldPixelToGeo } from "@/lib/geo";
import type { GeoPoint, WorldPixel } from "@/lib/geo";
import { cn } from "@/lib/utils";

const TILE_SIZE = 256;

export interface MapPath {
  id: string;
  points: GeoPoint[];
  color: string;
  width?: number;
  opacity?: number;
  dash?: string;
}

export interface MapMarker {
  id: string;
  point: GeoPoint;
  color: string;
  radius?: number;
  opacity?: number;
  label?: string;
}

function nearestWrappedX(x: number, centerX: number, worldSize: number): number {
  let result = x;
  while (result - centerX > worldSize / 2) result -= worldSize;
  while (result - centerX < -worldSize / 2) result += worldSize;
  return result;
}

export function SlippyMap({
  center,
  zoom: initialZoom,
  paths = [],
  markers = [],
  minZoom = 2,
  maxZoom = 19,
  followCenter = false,
  viewKey,
  className,
  ariaLabel,
}: {
  center: GeoPoint;
  zoom: number;
  paths?: MapPath[];
  markers?: MapMarker[];
  minZoom?: number;
  maxZoom?: number;
  followCenter?: boolean;
  viewKey?: string;
  className?: string;
  ariaLabel: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 960, height: 480 });
  const [viewCenter, setViewCenter] = useState(center);
  const [zoom, setZoom] = useState(
    Math.max(minZoom, Math.min(maxZoom, Math.round(initialZoom))),
  );
  const dragRef = useRef<{
    x: number;
    y: number;
    center: WorldPixel;
  } | null>(null);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const resize = new ResizeObserver(([entry]) => {
      if (!entry) return;
      setSize({
        width: Math.max(1, entry.contentRect.width),
        height: Math.max(1, entry.contentRect.height),
      });
    });
    resize.observe(node);
    return () => resize.disconnect();
  }, []);

  useEffect(() => {
    setViewCenter(center);
    setZoom(Math.max(minZoom, Math.min(maxZoom, Math.round(initialZoom))));
    // viewKey resets a panned map when the dataset or episode changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewKey, initialZoom, minZoom, maxZoom]);

  useEffect(() => {
    if (followCenter && !dragRef.current) setViewCenter(center);
  }, [center, followCenter]);

  const centerPixel = useMemo(
    () => geoToWorldPixel(viewCenter, zoom),
    [viewCenter, zoom],
  );
  const worldSize = TILE_SIZE * 2 ** zoom;

  const tiles = useMemo(() => {
    const firstX = Math.floor((centerPixel.x - size.width / 2) / TILE_SIZE);
    const lastX = Math.floor((centerPixel.x + size.width / 2) / TILE_SIZE);
    const firstY = Math.floor((centerPixel.y - size.height / 2) / TILE_SIZE);
    const lastY = Math.floor((centerPixel.y + size.height / 2) / TILE_SIZE);
    const count = 2 ** zoom;
    const result: {
      key: string;
      x: number;
      y: number;
      tileX: number;
      tileY: number;
    }[] = [];
    for (let x = firstX; x <= lastX; x++) {
      for (let y = firstY; y <= lastY; y++) {
        if (y < 0 || y >= count) continue;
        const wrappedX = ((x % count) + count) % count;
        result.push({
          key: `${x}:${y}`,
          x: x * TILE_SIZE - centerPixel.x + size.width / 2,
          y: y * TILE_SIZE - centerPixel.y + size.height / 2,
          tileX: wrappedX,
          tileY: y,
        });
      }
    }
    return result;
  }, [centerPixel, size, zoom]);

  const toScreen = (point: GeoPoint) => {
    const world = geoToWorldPixel(point, zoom);
    const x = nearestWrappedX(world.x, centerPixel.x, worldSize);
    return {
      x: x - centerPixel.x + size.width / 2,
      y: world.y - centerPixel.y + size.height / 2,
    };
  };

  const svgPaths = paths.map((path) => ({
    ...path,
    d: path.points
      .map((point, index) => {
        const screen = toScreen(point);
        return `${index === 0 ? "M" : "L"}${screen.x.toFixed(1)},${screen.y.toFixed(1)}`;
      })
      .join(" "),
  }));

  const changeZoom = (delta: number) => {
    setZoom((current) =>
      Math.max(minZoom, Math.min(maxZoom, current + delta)),
    );
  };

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative min-h-72 w-full touch-none overflow-hidden border border-slate-800 bg-slate-900",
        className,
      )}
      role="img"
      aria-label={ariaLabel}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        dragRef.current = {
          x: event.clientX,
          y: event.clientY,
          center: centerPixel,
        };
      }}
      onPointerMove={(event) => {
        const drag = dragRef.current;
        if (!drag) return;
        setViewCenter(
          worldPixelToGeo(
            {
              x: drag.center.x - (event.clientX - drag.x),
              y: drag.center.y - (event.clientY - drag.y),
            },
            zoom,
          ),
        );
      }}
      onPointerUp={() => {
        dragRef.current = null;
      }}
      onPointerCancel={() => {
        dragRef.current = null;
      }}
      onWheel={(event) => {
        event.preventDefault();
        changeZoom(event.deltaY < 0 ? 1 : -1);
      }}
      onDoubleClick={() => changeZoom(1)}
    >
      <div className="absolute inset-0 bg-slate-900">
        {tiles.map((tile) => (
          // OSM raster tiles are the map itself, not decorative page imagery.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={tile.key}
            src={`https://tile.openstreetmap.org/${zoom}/${tile.tileX}/${tile.tileY}.png`}
            alt=""
            draggable={false}
            className="pointer-events-none absolute size-64 select-none"
            style={{ left: tile.x, top: tile.y }}
          />
        ))}
      </div>
      <svg
        className="pointer-events-none absolute inset-0 size-full"
        viewBox={`0 0 ${size.width} ${size.height}`}
        preserveAspectRatio="none"
        aria-hidden
      >
        {svgPaths.map((path) =>
          path.d ? (
            <path
              key={path.id}
              d={path.d}
              fill="none"
              stroke={path.color}
              strokeWidth={path.width ?? 2}
              strokeOpacity={path.opacity ?? 1}
              strokeDasharray={path.dash}
              strokeLinecap="round"
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
          ) : null,
        )}
        {markers.map((marker) => {
          const screen = toScreen(marker.point);
          return (
            <g key={marker.id}>
              <circle
                cx={screen.x}
                cy={screen.y}
                r={marker.radius ?? 5}
                fill={marker.color}
                fillOpacity={marker.opacity ?? 0.85}
                stroke="#f8fafc"
                strokeWidth="1.5"
                vectorEffect="non-scaling-stroke"
              />
              {marker.label && <title>{marker.label}</title>}
            </g>
          );
        })}
      </svg>

      <div
        className="absolute right-2 top-2 z-10 flex flex-col overflow-hidden border border-slate-700 bg-slate-950/90"
        onPointerDown={(event) => event.stopPropagation()}
      >
        <button
          type="button"
          className="flex size-8 items-center justify-center text-slate-300 hover:bg-slate-800"
          onClick={() => changeZoom(1)}
          title="Zoom in"
          aria-label="Zoom in"
        >
          <ZoomIn className="size-4" />
        </button>
        <button
          type="button"
          className="flex size-8 items-center justify-center border-t border-slate-700 text-slate-300 hover:bg-slate-800"
          onClick={() => changeZoom(-1)}
          title="Zoom out"
          aria-label="Zoom out"
        >
          <ZoomOut className="size-4" />
        </button>
        <button
          type="button"
          className="flex size-8 items-center justify-center border-t border-slate-700 text-slate-300 hover:bg-slate-800"
          onClick={() => {
            setViewCenter(center);
            setZoom(
              Math.max(minZoom, Math.min(maxZoom, Math.round(initialZoom))),
            );
          }}
          title="Reset map"
          aria-label="Reset map"
        >
          <LocateFixed className="size-4" />
        </button>
      </div>

      <div className="absolute bottom-0 right-0 z-10 bg-slate-950/80 px-1.5 py-0.5 text-[9px] text-slate-300">
        ©{" "}
        <a
          href="https://www.openstreetmap.org/copyright"
          target="_blank"
          rel="noreferrer"
          className="underline"
        >
          OpenStreetMap contributors
        </a>
      </div>
    </div>
  );
}
