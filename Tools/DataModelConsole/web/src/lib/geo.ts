import type { TrajectoryPoint } from "@/lib/ego";

const EARTH_RADIUS_M = 6_378_137;
const TILE_SIZE = 256;

export interface GeoPoint {
  latitude: number;
  longitude: number;
}

export interface WorldPixel {
  x: number;
  y: number;
}

// At heading 0 (north), positive ego y-left moves west.
export function egoTrajectoryToGeo(
  origin: GeoPoint,
  headingDegreesCWFromNorth: number,
  trajectory: ArrayLike<TrajectoryPoint>,
): GeoPoint[] {
  const heading = (headingDegreesCWFromNorth * Math.PI) / 180;
  const lat0 = (origin.latitude * Math.PI) / 180;
  const cosLat = Math.cos(lat0);
  const out = new Array<GeoPoint>(trajectory.length);
  for (let i = 0; i < trajectory.length; i++) {
    const point = trajectory[i];
    const east =
      point.x * Math.sin(heading) - point.y * Math.cos(heading);
    const north =
      point.x * Math.cos(heading) + point.y * Math.sin(heading);
    out[i] = {
      latitude:
        origin.latitude +
        ((north / EARTH_RADIUS_M) * 180) / Math.PI,
      longitude:
        origin.longitude +
        ((east / (EARTH_RADIUS_M * cosLat)) * 180) / Math.PI,
    };
  }
  return out;
}

export function geoToWorldPixel(
  point: GeoPoint,
  zoom: number,
): WorldPixel {
  const scale = TILE_SIZE * 2 ** zoom;
  const sinLat = Math.sin(
    (Math.max(-85.05112878, Math.min(85.05112878, point.latitude)) *
      Math.PI) /
      180,
  );
  return {
    x: ((point.longitude + 180) / 360) * scale,
    y:
      (0.5 -
        Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) *
      scale,
  };
}

export function worldPixelToGeo(
  pixel: WorldPixel,
  zoom: number,
): GeoPoint {
  const scale = TILE_SIZE * 2 ** zoom;
  const longitude = (pixel.x / scale) * 360 - 180;
  const mercator = Math.PI * (1 - (2 * pixel.y) / scale);
  return {
    latitude:
      (Math.atan(Math.sinh(mercator)) * 180) / Math.PI,
    longitude: ((longitude + 540) % 360) - 180,
  };
}

export function fitGeoBounds(
  bbox: [number, number, number, number],
  width = 960,
  height = 480,
  minZoom = 2,
  maxZoom = 16,
): { center: GeoPoint; zoom: number } {
  const [minLon, minLat, maxLon, maxLat] = bbox;
  const center = {
    latitude: (minLat + maxLat) / 2,
    longitude: (minLon + maxLon) / 2,
  };
  const northWest = geoToWorldPixel(
    { latitude: maxLat, longitude: minLon },
    0,
  );
  const southEast = geoToWorldPixel(
    { latitude: minLat, longitude: maxLon },
    0,
  );
  const spanX = Math.max(Math.abs(southEast.x - northWest.x), 1e-9);
  const spanY = Math.max(Math.abs(southEast.y - northWest.y), 1e-9);
  const zoom = Math.floor(
    Math.log2(
      Math.min(
        Math.max(1, width - 64) / spanX,
        Math.max(1, height - 64) / spanY,
      ),
    ),
  );
  return {
    center,
    zoom: Math.max(minZoom, Math.min(maxZoom, zoom)),
  };
}

export function decodeEpisodePath(buffer: ArrayBuffer): GeoPoint[] {
  if (buffer.byteLength % 32 !== 0) {
    throw new Error("Episode path payload is not a sequence of four float64s");
  }
  const view = new DataView(buffer);
  const points = new Array<GeoPoint>(buffer.byteLength / 32);
  for (let i = 0; i < points.length; i++) {
    const offset = i * 32;
    const latitude = view.getFloat64(offset, true);
    const longitude = view.getFloat64(offset + 8, true);
    if (
      !Number.isFinite(latitude) ||
      !Number.isFinite(longitude) ||
      latitude < -90 ||
      latitude > 90 ||
      longitude < -180 ||
      longitude > 180
    ) {
      throw new Error("Episode path contains an invalid coordinate");
    }
    points[i] = { latitude, longitude };
  }
  return points;
}
