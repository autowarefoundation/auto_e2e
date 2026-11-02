"""Convert one nuPlan log window into RDS-HQ clips for the WSM4 renderer.

Each clip is the same window with the ego displaced: a lateral offset and a yaw that ramp in
smoothly from zero, so the first frame still matches the recorded camera image used as the
model's initial RGB. Clip 0 is always the unshifted log, as the reference.

Reads the official nuPlan open-data archives anonymously with HTTP range requests. Their size
and ETag are checked against the Platform raw snapshot manifest, so the bytes are the snapshot's.

Outputs under <out>/<clip_id>/: rds-hq/<component>/<variant>.tar, rgb/<camera_key>.jpg (undistorted
first frame), gt/<camera_key>/<frame>.jpg (undistorted 10 Hz frames) and meta.json.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sqlite3
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pyogrio
import s3fs
import shapely
from pyproj import Transformer

BUCKET = "motional-nuplan/public/nuplan-v1.1/"
# key: (size, ETag) recorded as size_bytes / source_etag in nuplan-v1.1-mini-group0-20260809/manifest.json
ARCHIVES = {
    "db": ("nuplan-v1.1_mini.zip", 8550100030, "08abc074db9227e758cc41c6b1ee223c-1020"),
    "camera": ("sensor_blobs/mini_set/nuplan-v1.1_mini_camera_0.zip", 52219710368, "e211f3b90eabb608fce75f7d70df0b19-6226"),
    "maps": ("nuplan-maps-v1.1.zip", 970997691, "0efaac6b7c603ae6f341021b35059bcc-116"),
}
# nuPlan channel -> nearest physical camera of the reference rig, whose key selects the rig-view embedding.
CAMERAS = {
    "CAM_F0": "camera_front_wide_120fov",
    "CAM_L0": "camera_cross_left_120fov",
    "CAM_R0": "camera_cross_right_120fov",
    "CAM_L2": "camera_rear_left_70fov",
    "CAM_R2": "camera_rear_right_70fov",
    "CAM_B0": "camera_rear_tele_30fov",
}
OBJECT_TYPES = {"vehicle": "Car", "bicycle": "Cyclist", "pedestrian": "Pedestrian", "generic_object": "Others"}
FPS = 30
MAP_MARGIN_M = 120.0


class _Values(list):
    """Stand-in for the devkit's list subclasses; the pickle fills it with APPENDS."""


class _NuPlanTypes(pickle.Unpickler):
    """Camera calibration is pickled as nuplan-devkit list subclasses; read them as plain lists without the devkit."""

    def find_class(self, module: str, name: str):
        if module == "nuplan.database.common.data_types":
            return _Values
        return super().find_class(module, name)


def unpickle(blob: bytes):
    return _NuPlanTypes(io.BytesIO(blob)).load()


def open_archive(fs: s3fs.S3FileSystem, key: str) -> zipfile.ZipFile:
    name, size, etag = ARCHIVES[key]
    info = fs.info(BUCKET + name)
    if int(info["size"]) != size or info["ETag"].strip('"') != etag:
        raise ValueError(f"{name} differs from the raw snapshot manifest")
    return zipfile.ZipFile(fs.open(BUCKET + name, block_size=8 << 20))


def extract(archive: zipfile.ZipFile, member: str, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(archive.read(member))
    return dst


def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def rigid(rot: np.ndarray, t) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rot, t
    return m


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pinhole_as_ftheta(k: np.ndarray, w: int, h: int) -> tuple[list[float], float]:
    """F-theta intrinsics equal to an undistorted pinhole: the forward polynomial r(theta) fits f*tan(theta)
    up to the image corner. The renderer draws boxes only through the F-theta path. Returns the 11 values and
    the worst radius error in pixels."""
    f, cx, cy = (k[0, 0] + k[1, 1]) / 2, k[0, 2], k[1, 2]
    theta_max = np.arctan(np.hypot(max(cx, w - cx), max(cy, h - cy)) / f) + np.radians(2)
    theta = np.linspace(0, theta_max, 500)
    basis = np.stack([theta ** p for p in range(1, 6)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, f * np.tan(theta), rcond=None)
    error = float(np.abs(basis @ coef - f * np.tan(theta)).max())
    return [cx, cy, w, h, 0.0, *coef, 0.0], error


def ego_poses(db: sqlite3.Connection, times_us: np.ndarray) -> list[np.ndarray]:
    """Global vehicle-to-world poses at the requested times, interpolated from the ~100 Hz ego_pose table."""
    rows = np.array(db.execute("SELECT timestamp, x, y, z, qw, qx, qy, qz FROM ego_pose ORDER BY timestamp").fetchall())
    poses = []
    for t in times_us:
        i = int(np.clip(np.searchsorted(rows[:, 0], t), 1, len(rows) - 1))
        a, b = rows[i - 1], rows[i]
        w = (t - a[0]) / (b[0] - a[0])
        q = (1 - w) * a[4:] + w * (b[4:] if np.dot(a[4:], b[4:]) >= 0 else -b[4:])
        poses.append(rigid(quat_to_rot(*(q / np.linalg.norm(q))), (1 - w) * a[1:4] + w * b[1:4]))
    return poses


def pick_start(db: sqlite3.Connection, duration_s: float) -> int:
    """First time, 2 s after the camera data starts, from which the ego drives straight at 5 m/s or more."""
    first, last = db.execute("SELECT min(timestamp), max(timestamp) FROM image").fetchone()
    for t0 in range(first + 2_000_000, last - int(duration_s * 1e6), 500_000):
        times = t0 + np.linspace(0, duration_s * 1e6, 11)
        poses = ego_poses(db, times)
        xy = np.array([p[:2, 3] for p in poses])
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / (duration_s / 10)
        yaw = [np.arctan2(p[1, 0], p[0, 0]) for p in poses]
        turn = abs(np.angle(np.exp(1j * (yaw[-1] - yaw[0]))))
        if speed.min() >= 5.0 and np.degrees(turn) < 5.0:
            return t0
    raise ValueError("no straight window of 5 m/s or more in this log")


def shifted(vehicle_to_world: list[np.ndarray], lateral_m: float, yaw_deg: float, ramp_s: float) -> list[np.ndarray]:
    """Displace the ego to its left by lateral_m and rotate it by yaw_deg, ramping in over ramp_s."""
    out = []
    for i, pose in enumerate(vehicle_to_world):
        s = np.clip(i / FPS / ramp_s, 0.0, 1.0)
        s = s * s * (3 - 2 * s)
        out.append(pose @ rigid(rot_z(np.radians(yaw_deg * s)), [0.0, lateral_m * s, 0.0]))
    return out


def tar_write(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def npy(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(array, dtype=np.float64))
    return buf.getvalue()


def map_layers(gpkg: Path, world_from_global: np.ndarray, region, ground_z: float) -> dict[str, list]:
    """WSM map labels in the clip world frame. nuPlan has no paint classes or elevation, so lane dividers
    are WHITE/OTHER, edges of the drivable surface are road_boundary, and all vertices sit on ground_z."""
    # Geometry is stored in WGS84; ego poses use the projected CRS named in the map's meta table.
    (crs,) = sqlite3.connect(gpkg).execute("SELECT value FROM meta WHERE key = 'projectedCoordSystem'").fetchone()
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def read(layer):
        geoms = np.asarray(pyogrio.read_dataframe(gpkg, layer=layer, columns=[]).geometry.values)
        return shapely.transform(geoms, lambda xy: np.column_stack(to_utm.transform(xy[:, 0], xy[:, 1])))

    def to_world(coords):
        xy = np.asarray(coords)[:, :2]
        pts = np.column_stack([xy, np.zeros(len(xy)), np.ones(len(xy))]) @ world_from_global.T
        pts[:, 2] = ground_z
        return pts[:, :3].round(3).tolist()

    def lines(geoms):
        for g in shapely.intersection(geoms, region):
            for part in getattr(g, "geoms", [g]):
                if isinstance(part, shapely.LineString) and part.length > 0.5:
                    yield to_world(part.coords)

    drivable = shapely.union_all(np.concatenate([read(name) for name in (
        "lane_groups_polygons", "lane_group_connectors", "intersections", "generic_drivable_areas", "carpark_areas")]))
    edges = shapely.boundary(drivable)
    dividers = shapely.difference(read("boundaries"), shapely.buffer(edges, 0.3))
    crosswalks = [g for g in shapely.intersection(read("crosswalks"), region) for g in getattr(g, "geoms", [g])
                  if isinstance(g, shapely.Polygon) and g.area > 1.0]
    return {
        "3d_lanelines/lanelines.json": [{"labelData": {"shape3d": {"attributes": [
            {"name": "colors", "enumsList": {"enumsList": ["WHITE"]}},
            {"name": "styles", "enumsList": {"enumsList": ["OTHER"]}}], "polyline3d": {"vertices": v}}}}
            for v in lines(dividers)],
        "3d_road_boundaries/road_boundaries.json": [{"labelData": {"shape3d": {"attributes": [
            {"name": "boundary_type", "text": "road_boundary"}], "polyline3d": {"vertices": v}}}}
            for v in lines(np.array([edges]))],
        "3d_crosswalks/crosswalks.json": [{"labelData": {"shape3d": {"surface": {
            "vertices": to_world(shapely.simplify(g, 0.1).exterior.coords[:-1])}}}} for g in crosswalks],
        **{f"{c}/{f}": [] for c, f in (
            ("3d_wait_lines", "wait_lines.json"), ("3d_poles", "poles.json"), ("3d_traffic_signs", "traffic_signs.json"),
            ("3d_road_markings", "road_markings.json"), ("3d_traffic_lights", "traffic_lights.json"))},
    }


def objects(db: sqlite3.Connection, times_us: np.ndarray, world_from_global: np.ndarray) -> list[dict]:
    """Boxes of the nearest 20 Hz lidar sweep for every 30 Hz frame, in the clip world frame."""
    sweeps = np.array(db.execute("SELECT timestamp FROM lidar_pc ORDER BY timestamp").fetchall())[:, 0]
    tokens = [r[0] for r in db.execute("SELECT token FROM lidar_pc ORDER BY timestamp")]
    frames, centers = [], {}
    for t in times_us:
        token = tokens[int(np.abs(sweeps - t).argmin())]
        frame = {}
        for track, name, x, y, z, yaw, length, width, height in db.execute(
            "SELECT hex(b.track_token), c.name, b.x, b.y, b.z, b.yaw, b.length, b.width, b.height FROM lidar_box b "
            "JOIN track t ON t.token = b.track_token JOIN category c ON c.token = t.category_token "
            "WHERE b.lidar_pc_token = ?", (token,)):
            if name not in OBJECT_TYPES:
                continue
            pose = world_from_global @ rigid(rot_z(yaw), [x, y, z])
            frame[track] = {"object_type": OBJECT_TYPES[name], "object_lwh": [length, width, height], "object_to_world": pose.tolist()}
            centers.setdefault(track, []).append(pose[:3, 3])
        frames.append(frame)
    moving = {k: bool(np.ptp(np.array(v), axis=0)[:2].max() > 1.0) for k, v in centers.items()}
    for frame in frames:
        for track, record in frame.items():
            record["object_is_moving"] = moving[track]
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log", default="2021.05.12.22.28.35_veh-35_00620_01164")
    p.add_argument("--start-us", type=int, help="window start (image timestamp clock); default picks a straight window")
    p.add_argument("--frames", type=int, default=121, help="30 Hz frames, 4k+1")
    p.add_argument("--ramp-s", type=float, default=2.0, help="seconds over which each shift ramps in")
    p.add_argument("--shifts", default="0:0,0.5:0,1.0:0,1.0:3",
                   help="comma-separated lateral_m:yaw_deg pairs; positive is to the ego's left")
    args = p.parse_args()

    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "ap-northeast-1"})
    cache = args.out / "_nuplan"
    with open_archive(fs, "db") as z:
        db_path = extract(z, f"data/cache/mini/{args.log}.db", cache / f"{args.log}.db")
    db = sqlite3.connect(db_path)
    (location,) = db.execute("SELECT map_version FROM log").fetchone()
    with open_archive(fs, "maps") as z:
        member = next(n for n in z.namelist() if f"/{location}/" in n and n.endswith("map.gpkg"))
        gpkg = extract(z, member, cache / f"{location}.gpkg")

    duration_s = (args.frames - 1) / FPS
    t0 = args.start_us or pick_start(db, duration_s)
    times = t0 + np.round(np.arange(args.frames) * 1e6 / FPS).astype(np.int64)
    global_poses = ego_poses(db, times)
    p0 = global_poses[0]
    world_from_global = np.linalg.inv(rigid(rot_z(np.arctan2(p0[1, 0], p0[0, 0])), p0[:3, 3]))
    vehicle_to_world = [world_from_global @ g for g in global_poses]
    frames = objects(db, times, world_from_global)
    # nuPlan maps are 2D; put the road on the median bottom of the vehicle boxes near the ego.
    bottoms = [r["object_to_world"][2][3] - r["object_lwh"][2] / 2 for f in frames[:30] for r in f.values()
               if r["object_type"] == "Car" and np.hypot(r["object_to_world"][0][3], r["object_to_world"][1][3]) < 30]
    ground_z = float(np.median(bottoms)) if bottoms else 0.0
    xy = np.array([v[:2, 3] for v in vehicle_to_world])
    global_xy = np.column_stack([xy, np.zeros(len(xy)), np.ones(len(xy))]) @ np.linalg.inv(world_from_global).T
    region = shapely.buffer(shapely.LineString(global_xy[:, :2]), MAP_MARGIN_M)
    layers = map_layers(gpkg, world_from_global, region, ground_z)

    clip = f"nuplan_{args.log.replace('.', '')}_{t0}"
    root = args.out / clip
    cams = {r[0]: r[1:] for r in db.execute(
        "SELECT channel, token, translation, rotation, intrinsic, distortion, width, height FROM camera")}
    camera_to_vehicle, intrinsics, undistort, ftheta_error_px = {}, {}, {}, {}
    for channel, key in CAMERAS.items():
        _, trans, rot, k, dist, w, h = cams[channel]
        k = np.asarray(unpickle(k), dtype=np.float64).reshape(3, 3)
        dist = np.asarray(unpickle(dist), dtype=np.float64).reshape(-1)
        new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, (w, h), 0.0, (w, h))  # same rectification as the pack
        rot = np.asarray(unpickle(rot), dtype=np.float64).reshape(4)  # w, x, y, z
        camera_to_vehicle[key] = rigid(quat_to_rot(*rot), np.asarray(unpickle(trans), dtype=np.float64).reshape(3))
        intrinsics[key], ftheta_error_px[key] = pinhole_as_ftheta(new_k, w, h)
        undistort[key] = (k, dist, new_k)

    variants = []
    for spec in args.shifts.split(","):
        lateral_m, yaw_deg = (float(v) for v in spec.split(":"))
        # L/R: to the ego's left/right; the log name keeps its own characters.
        suffix = f"lat{lateral_m:+.1f}_yaw{yaw_deg:+.1f}".replace(".", "p").replace("+", "L").replace("-", "R")
        name = f"{clip}_{suffix}"
        poses = shifted(vehicle_to_world, lateral_m, yaw_deg, args.ramp_s)
        tar_write(root / "rds-hq/pose" / f"{name}.tar", {
            f"{name}.{i:06d}.pose.{key}.npy": npy(pose @ c2v)
            for i, pose in enumerate(poses) for key, c2v in camera_to_vehicle.items()})
        tar_write(root / "rds-hq/ftheta_intrinsic" / f"{name}.tar",
                  {f"{name}.ftheta_intrinsic.{key}.npy": npy(v) for key, v in intrinsics.items()})
        tar_write(root / "rds-hq/all_object_info" / f"{name}.tar",
                  {f"{name}.{i:06d}.all_object_info.json": json.dumps(f).encode() for i, f in enumerate(frames)})
        for member, labels in layers.items():
            component, key = member.split("/")
            tar_write(root / "rds-hq" / component / f"{name}.tar", {f"{name}.{key}": json.dumps({"labels": labels}).encode()})
        tar_write(root / "rds-hq/3d_traffic_lights_lightnet_status" / f"{name}.tar",
                  {f"{name}.aggregated_states.json": json.dumps({"state_to_legacy_render_state": {}, "traffic_light_states": {}}).encode()})
        variants.append({"name": name, "lateral_m": lateral_m, "yaw_deg": yaw_deg})

    # Recorded 10 Hz images of the window, undistorted, for the first-frame conditioning and the comparison.
    images = {}
    for channel, key in CAMERAS.items():
        rows = db.execute("SELECT img.timestamp, img.filename_jpg FROM image img JOIN camera c ON c.token = img.camera_token "
                          "WHERE c.channel = ? AND img.timestamp BETWEEN ? AND ? ORDER BY img.timestamp",
                          (channel, int(times[0]) - 50_000, int(times[-1]) + 50_000)).fetchall()
        images[key] = rows
    with open_archive(fs, "camera") as z:
        for key, rows in images.items():
            k, dist, new_k = undistort[key]
            for ts, filename in rows:
                frame = int(round((ts - t0) * FPS / 1e6))
                if not 0 <= frame < args.frames:
                    continue
                raw = cv2.imdecode(np.frombuffer(z.read(f"nuplan-v1.1_mini_camera_0/{filename}"), np.uint8), cv2.IMREAD_COLOR)
                rect = cv2.undistort(raw, k, dist, None, new_k)
                (root / "gt" / key).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(root / "gt" / key / f"{frame:06d}.jpg"), rect, [cv2.IMWRITE_JPEG_QUALITY, 95])
            first = sorted((root / "gt" / key).glob("*.jpg"))[0]
            (root / "rgb").mkdir(exist_ok=True)
            (root / "rgb" / f"{key}.jpg").write_bytes(first.read_bytes())

    meta = {"log": args.log, "location": location, "start_us": int(t0), "frames": args.frames, "fps": FPS,
            "ramp_s": args.ramp_s, "ground_z": ground_z, "cameras": CAMERAS, "variants": variants,
            "ftheta_fit_max_error_px": ftheta_error_px,
            "first_rgb_frame": {k: sorted((root / "gt" / k).glob("*.jpg"))[0].stem for k in CAMERAS.values()},
            "map_label_counts": {k: len(v) for k, v in layers.items()},
            "object_tracks": len({t for f in frames for t in f})}
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
