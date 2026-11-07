"""AutoBEV-only packers on tiny synthetic upstream files: members, class rules, frames."""

from __future__ import annotations

import io
import itertools
import json
import math

import numpy as np
import pytest
from PIL import Image

from data_parsing.argoverse2 import pack as av2
from data_parsing.nuscenes import pack as nuscenes
from data_parsing.once import pack as once
from data_parsing.simbev import pack as simbev
from data_processing.autobev_members import (
    BOX_CLASSES,
    camera_member,
    decode_boxes,
    decode_npy,
    history_camera_member,
    manifest_contract,
)
from data_processing.autobev_pack import SAMPLE_KIND, bev_segmentation, write_partition
from data_processing.reactive_training_artifacts import BEV_SEGMENTATION_CLASSES, decode_bev_segmentation
from model_components.autobev.geometry import AUTOBEV_GEOMETRY, HISTORY_FRAMES, IMAGE_H, IMAGE_W, NUM_VIEWS

WIDTH, HEIGHT, FOCAL = 160, 90, 80.0
K = [[FOCAL, 0.0, WIDTH / 2], [0.0, FOCAL, HEIGHT / 2], [0.0, 0.0, 1.0]]
SLOT_YAWS = (0.0, 55.0, -55.0, 125.0, 180.0, -125.0)
# Camera axes (z forward, x right, y down) in a forward-looking ego frame (x forward, y left, z up).
EGO_FROM_FORWARD_CAMERA = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (WIDTH, HEIGHT), (90, 120, 150)).save(out, format="JPEG")
    return out.getvalue()


def _rz(yaw_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _camera_rotation(yaw_deg: float) -> np.ndarray:
    return _rz(yaw_deg) @ EGO_FROM_FORWARD_CAMERA


def _quaternion(rotation: np.ndarray) -> list[float]:
    """wxyz of a rotation matrix."""
    m = rotation
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2
    y = math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2
    z = math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2
    x = math.copysign(x, m[2, 1] - m[1, 2])
    y = math.copysign(y, m[0, 2] - m[2, 0])
    z = math.copysign(z, m[1, 0] - m[0, 1])
    return [w, x, y, z]


def _points(z: float = 0.0) -> np.ndarray:
    """A ground disc of LiDAR returns around the sensor, every 2 m out to 60 m."""
    xs, ys = np.meshgrid(np.arange(-60.0, 61.0, 2.0), np.arange(-60.0, 61.0, 2.0))
    keep = np.hypot(xs, ys) > 2.0
    return np.stack([xs[keep], ys[keep], np.full(keep.sum(), z)], axis=1)


def _check_members(key: str, members: dict[str, bytes]) -> tuple[np.ndarray, np.ndarray]:
    expected = {camera_member(v) for v in range(NUM_VIEWS)}
    expected |= {history_camera_member(h, v) for h in range(HISTORY_FRAMES) for v in range(NUM_VIEWS)}
    expected |= {"projection.npy", "history_projection.npy", "bev_segmentation.npz", "depth.npz",
                 "boxes.npz", "meta.json"}
    assert set(members) == expected, key
    with Image.open(io.BytesIO(members["cam_0.jpg"])) as image:
        assert image.size == (IMAGE_W, IMAGE_H)
    assert decode_npy(members["projection.npy"]).shape == (NUM_VIEWS, 3, 4)
    assert decode_npy(members["history_projection.npy"]).shape == (HISTORY_FRAMES, NUM_VIEWS, 3, 4)
    return decode_bev_segmentation(members["bev_segmentation.npz"])


def _class_valid(valid: np.ndarray) -> dict[str, bool]:
    return {name: bool(valid[index].any()) for index, name in enumerate(BEV_SEGMENTATION_CLASSES)}


# --- nuScenes -----------------------------------------------------------------


def _nuscenes_files() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    tables: dict[str, list[dict]] = {name: [] for name in nuscenes._TABLES}
    tables["log"].append({"token": "log", "location": "boston-seaport"})
    tables["scene"].append({"token": "scene", "name": "scene-0001", "log_token": "log", "first_sample_token": "s0"})
    for index, channel in enumerate((*nuscenes.CAMERAS, nuscenes.LIDAR)):
        tables["sensor"].append({"token": f"sensor-{channel}", "channel": channel})
        camera = index < NUM_VIEWS
        rotation = _camera_rotation(SLOT_YAWS[index]) if camera else np.eye(3)
        tables["calibrated_sensor"].append({
            "token": f"calib-{channel}", "sensor_token": f"sensor-{channel}",
            "translation": [1.0, 0.0, 1.6], "rotation": _quaternion(rotation),
            "camera_intrinsic": K if camera else [],
        })
    jpeg = _jpeg()
    cloud = np.c_[_points(-1.8), np.zeros((len(_points()), 2))].astype(np.float32).tobytes()
    for frame in range(8):
        token = f"s{frame}"
        tables["sample"].append({
            "token": token, "timestamp": 1_000_000 + frame * 500_000,
            "prev": f"s{frame - 1}" if frame else "", "next": f"s{frame + 1}" if frame < 7 else "",
        })
        tables["ego_pose"].append({"token": f"pose-{frame}", "translation": [100.0 + 5 * frame, 200.0, 0.0],
                                   "rotation": [1.0, 0.0, 0.0, 0.0]})
        for channel in (*nuscenes.CAMERAS, nuscenes.LIDAR):
            name = f"samples/{channel}/{token}.{'jpg' if channel != nuscenes.LIDAR else 'pcd.bin'}"
            files[name] = jpeg if channel != nuscenes.LIDAR else cloud
            tables["sample_data"].append({
                "token": f"{token}-{channel}", "sample_token": token, "ego_pose_token": f"pose-{frame}",
                "calibrated_sensor_token": f"calib-{channel}", "filename": name, "is_key_frame": True,
                "height": HEIGHT, "width": WIDTH,
            })
    tables["category"] += [{"token": "car", "name": "vehicle.car"}, {"token": "ped", "name": "human.pedestrian.adult"}]
    tables["instance"] += [{"token": "i-car", "category_token": "car"}, {"token": "i-ped", "category_token": "ped"}]
    tables["sample_annotation"] += [
        {"token": "a-car", "sample_token": "s7", "instance_token": "i-car", "translation": [145.0, 202.0, 0.8],
         "size": [2.0, 4.5, 1.6], "rotation": [1.0, 0.0, 0.0, 0.0], "num_lidar_pts": 20, "num_radar_pts": 0},
        {"token": "a-ped", "sample_token": "s7", "instance_token": "i-ped", "translation": [150.0, 195.0, 0.9],
         "size": [0.6, 0.6, 1.8], "rotation": [1.0, 0.0, 0.0, 0.0], "num_lidar_pts": 0, "num_radar_pts": 0},
    ]
    for name, rows in tables.items():
        files[f"{nuscenes.VERSION}/{name}.json"] = json.dumps(rows).encode()
    nodes = {"n0": (120.0, 190.0), "n1": (200.0, 190.0), "n2": (200.0, 210.0), "n3": (120.0, 210.0),
             "c0": (150.0, 190.0), "c1": (155.0, 190.0), "c2": (155.0, 210.0), "c3": (150.0, 210.0)}
    layers = {
        "node": [{"token": t, "x": x, "y": y} for t, (x, y) in nodes.items()],
        "polygon": [{"token": "road", "exterior_node_tokens": ["n0", "n1", "n2", "n3"], "holes": []},
                    {"token": "cross", "exterior_node_tokens": ["c0", "c1", "c2", "c3"], "holes": []}],
        "drivable_area": [{"token": "d", "polygon_tokens": ["road"]}],
        "ped_crossing": [{"token": "p", "polygon_token": "cross"}],
        "stop_line": [{"token": "s", "polygon_token": "cross"}],
        "road_segment": [{"token": "r", "polygon_token": "cross", "is_intersection": True}],
        "lane": [{"token": "l", "left_lane_divider_segment_nodes": [{"x": 120.0, "y": 202.0}, {"x": 200.0, "y": 202.0}],
                  "right_lane_divider_segment_nodes": [{"x": 120.0, "y": 198.0}, {"x": 200.0, "y": 198.0}]}],
    }
    files["maps/expansion/boston-seaport.json"] = json.dumps(layers).encode()
    return files


def test_nuscenes_packs_every_class_and_marks_unseen_boxes_unknown():
    (key, members), = itertools.islice(nuscenes.iter_samples(_nuscenes_files(), 0, 1), 1)
    assert key == "nusc-scene-0001-s7"
    target, valid = _check_members(key, members)
    assert all(_class_valid(valid).values())
    boxes, labels = decode_boxes(members["boxes.npz"])
    # The reference pose is (135, 200): the car is 10 m ahead and 2 m left, on the road.
    np.testing.assert_allclose(boxes[0, :3], [10.0, 2.0, 0.8], atol=1e-4)
    assert labels.tolist() == [BOX_CLASSES.index("VEHICLE")]
    vru = BEV_SEGMENTATION_CLASSES.index("vulnerable_road_user")
    row, col = np.rint(AUTOBEV_GEOMETRY.ego_to_pixel(np.array([[15.0, -5.0]]))[0]).astype(int)
    assert not valid[vru, row, col] and target[vru, row, col] == 0.0


# --- Argoverse 2 ----------------------------------------------------------------

AV2_LOG = "log-a"
AV2_START = 315_000_000_000_000_000


def _feather(frame) -> bytes:
    out = io.BytesIO()
    frame.reset_index(drop=True).to_feather(out)
    return out.getvalue()


def _av2_files(log: str = AV2_LOG) -> dict[str, bytes]:
    import pandas as pd

    root = f"train/{log}"
    ring = ("ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left",
            "ring_rear_left", "ring_rear_right", "ring_side_right")
    yaws = dict(zip(ring, (0.0, 45.0, -45.0, 100.0, 150.0, -150.0, -100.0)))
    files = {
        f"{root}/calibration/intrinsics.feather": _feather(pd.DataFrame([
            {"sensor_name": c, "fx_px": FOCAL, "fy_px": FOCAL, "cx_px": WIDTH / 2, "cy_px": HEIGHT / 2,
             "k1": 0.0, "k2": 0.0, "k3": 0.0, "width_px": WIDTH, "height_px": HEIGHT} for c in ring])),
        f"{root}/calibration/egovehicle_SE3_sensor.feather": _feather(pd.DataFrame([
            dict(zip(("qw", "qx", "qy", "qz"), _quaternion(_camera_rotation(yaws[c]))),
                 sensor_name=c, tx_m=1.0, ty_m=0.0, tz_m=1.6) for c in ring])),
    }
    stamps = [AV2_START + k * 50_000_000 for k in range(81)]
    files[f"{root}/city_SE3_egovehicle.feather"] = _feather(pd.DataFrame([
        {"timestamp_ns": t, "tx_m": 10.0 + i * 0.25, "ty_m": 0.0, "tz_m": 0.0, "qw": 1.0, "qx": 0.0, "qy": 0.0,
         "qz": 0.0} for i, t in enumerate(stamps)]))
    annotated = stamps[::2]
    files[f"{root}/annotations.feather"] = _feather(pd.DataFrame([
        {"timestamp_ns": t, "category": "REGULAR_VEHICLE", "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
         "tx_m": 12.0, "ty_m": 1.0, "tz_m": 0.9, "length_m": 4.5, "width_m": 2.0, "height_m": 1.7,
         "num_interior_pts": 30} for t in annotated]))
    files[f"{root}/map/log_map_archive_{log}____MIA_city_1.json"] = json.dumps({
        "drivable_areas": {"1": {"area_boundary": [{"x": 0.0, "y": -20.0}, {"x": 120.0, "y": -20.0},
                                                   {"x": 120.0, "y": 20.0}, {"x": 0.0, "y": 20.0}]}},
        "pedestrian_crossings": {"1": {"edge1": [{"x": 40.0, "y": -10.0}, {"x": 40.0, "y": 10.0}],
                                       "edge2": [{"x": 44.0, "y": -10.0}, {"x": 44.0, "y": 10.0}]}},
        "lane_segments": {"1": {"is_intersection": False, "lane_type": "VEHICLE",
                                "left_lane_boundary": [{"x": 0.0, "y": 2.0}, {"x": 120.0, "y": 2.0}],
                                "right_lane_boundary": [{"x": 0.0, "y": -2.0}, {"x": 120.0, "y": -2.0}]}},
    }).encode()
    jpeg = _jpeg()
    for camera in ring:
        for t in stamps:
            files[f"{root}/sensors/cameras/{camera}/{t}.jpg"] = jpeg
    for t in annotated:
        files[f"{root}/sensors/lidar/{t}.feather"] = _feather(pd.DataFrame(_points(-1.8), columns=["x", "y", "z"]))
    return files


def test_argoverse2_never_labels_intersection_or_stop_line_and_keeps_box_z():
    files = _av2_files()
    samples = list(itertools.islice(av2.iter_samples(files, 0, 1), 1))
    (key, members), = samples
    target, valid = _check_members(key, members)
    assert _class_valid(valid) == {name: name not in av2.NEVER_LABELLED for name in BEV_SEGMENTATION_CLASSES}
    boxes, _ = decode_boxes(members["boxes.npz"])
    assert boxes[0, 2] == pytest.approx(0.9)
    assert json.loads(members["meta.json"])["cameras"] == list(av2.cameras(AV2_LOG))


def test_argoverse2_samples_every_half_second_after_the_history_and_finds_them_again_from_the_subset():
    files = _av2_files()
    times = {}
    for name in files:
        if "/sensors/cameras/" in name:
            camera, stamp = name.split("/sensors/cameras/")[1].removesuffix(".jpg").split("/")
            times.setdefault(camera, []).append(int(stamp))
    annotated = sorted({int(n.rsplit("/", 1)[1].split(".")[0]) for n in files if "/sensors/lidar/" in n})
    samples = av2.plan_samples(AV2_LOG, annotated, times)
    assert [s["timestamp"] - AV2_START for s in samples] == [3_500_000_000, 4_000_000_000]
    needed = set(av2.needed_files(samples))
    subset = {camera: [t for t in stamps if f"sensors/cameras/{camera}/{t}.jpg" in needed]
              for camera, stamps in times.items()}
    assert av2.plan_samples(AV2_LOG, annotated, subset) == samples


def test_argoverse2_holds_out_pittsburgh_and_alternates_the_rear_camera():
    maps = {"train-a": "log_map_archive_a____PIT_city_1.json", "train-b": "log_map_archive_b____MIA_city_2.json",
            "val-c": "log_map_archive_c____WDC_city_3.json"}
    assert av2.training_logs(maps) == ["train-b", "val-c"]
    rears = {av2.cameras(f"log-{i}")[4] for i in range(20)}
    assert rears == {"ring_rear_left", "ring_rear_right"}


# --- ONCE -----------------------------------------------------------------------

ONCE_SEQ = "000001"


def _once_files(with_lanes: bool = True) -> dict[str, bytes]:
    yaws = {"cam01": 0.0, "cam03": 0.5, "cam05": 133.0, "cam06": 58.0, "cam07": -58.0, "cam08": -128.0,
            "cam09": -179.0}
    calib = {}
    for camera, yaw in yaws.items():
        lidar_from_camera = np.eye(4)
        lidar_from_camera[:3, :3] = _camera_rotation(yaw)
        lidar_from_camera[:3, 3] = [0.5, 0.0, -0.3]
        calib[camera] = {"cam_to_velo": lidar_from_camera.tolist(), "cam_intrinsic": K,
                         "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]}
    frames = []
    for index in range(10):
        frame = {"frame_id": str(1_600_000_000_000 + index * 500),
                 "pose": [0.0, 0.0, 0.0, 1.0, 3.0 * index, 0.0, 0.0]}
        if index == 8:
            # A car standing on the road 10 m ahead: its centre is 0.75 m above the road.
            frame["annos"] = {"names": ["Car", "Cone"],
                              "boxes_3d": [[10.0, 0.0, -once.LIDAR_ABOVE_ROAD_M + 0.75, 4.0, 2.0, 1.5, 0.0],
                                           [15.0, 3.0, -1.5, 0.4, 0.4, 0.6, 0.0]]}
        frames.append(frame)
    files = {f"data/{ONCE_SEQ}/{ONCE_SEQ}.json": json.dumps({"calib": calib, "frames": frames}).encode()}
    jpeg = _jpeg()
    for frame in frames:
        for camera in once.CAMERAS:
            files[f"data/{ONCE_SEQ}/{camera}/{frame['frame_id']}.jpg"] = jpeg
    reference = frames[8]["frame_id"]
    points = np.c_[_points(-once.LIDAR_ABOVE_ROAD_M), np.zeros(len(_points()))].astype(np.float32)
    files[f"data/{ONCE_SEQ}/{once.LIDAR}/{reference}.bin"] = points.tobytes()
    if with_lanes:
        # Lane points in the cam01 frame (x right, y down, z forward), on the road.
        height = once.LIDAR_ABOVE_ROAD_M - 0.3
        lanes = [[[side, height, depth] for depth in range(5, 41, 5)] for side in (-1.8, 1.8)]
        files[f"{once.LANES_ROOT}/{ONCE_SEQ}/cam01/{reference}.json"] = json.dumps({"lanes": lanes}).encode()
    return files


def test_once_moves_the_origin_to_the_road_at_pack_time():
    (key, members), = list(once.iter_samples(_once_files(), 0, 1))
    assert key == f"once-{ONCE_SEQ}-{1_600_000_000_000 + 8 * 500}"
    boxes, labels = decode_boxes(members["boxes.npz"])
    np.testing.assert_allclose(boxes[0, :3], [10.0, 0.0, 0.75], atol=1e-5)
    assert labels.tolist() == [BOX_CLASSES.index("VEHICLE"), BOX_CLASSES.index("OTHER")]
    # The front camera sits 1.78 - 0.3 m above the road, so a road point 20 m
    # ahead projects below the principal row.
    projection = decode_npy(members["projection.npy"])[0]
    u, v, w = projection @ np.array([20.0, 0.0, 0.0, 1.0])
    assert w == pytest.approx(19.5, abs=1e-4)
    assert v / w > IMAGE_H / 2


def test_once_classes_follow_the_readme_table():
    _, members = next(once.iter_samples(_once_files(), 0, 1))
    _, valid = _check_members("once", members)
    assert _class_valid(valid) == {name: name not in once.NEVER_LABELLED for name in BEV_SEGMENTATION_CLASSES}
    _, members = next(once.iter_samples(_once_files(with_lanes=False), 0, 1))
    _, valid = _check_members("once", members)
    assert not _class_valid(valid)["lane_boundary"]


# --- SimBEV ---------------------------------------------------------------------


def _detections(objects: list[dict]) -> bytes:
    out = io.BytesIO()
    np.save(out, np.array(objects, dtype=object), allow_pickle=True)
    return out.getvalue()


def _box(cx: float, cy: float, tag: int, *, valid: bool) -> dict:
    corners = [[cx + dx, cy + dy, dz] for dx in (-2.0, 2.0) for dy in (-1.0, 1.0) for dz in (0.0, 1.5)]
    return {"bounding_box": corners, "semantic_tags": [tag], "valid_flag": valid, "attributes": {"role_name": ""}}


def _simbev_files() -> dict[str, bytes]:
    metadata = {"camera_intrinsics": K,
                "LIDAR": {"sensor2ego_translation": [0.0, 0.0, 1.8], "sensor2ego_rotation": [1.0, 0.0, 0.0, 0.0]}}
    for camera, yaw in zip(simbev.CAMERAS, SLOT_YAWS):
        metadata[camera] = {"sensor2ego_translation": [0.6, 0.0, 1.6],
                            "sensor2ego_rotation": _quaternion(_camera_rotation(yaw))}
    files: dict[str, bytes] = {}
    jpeg = _jpeg()
    frames = []
    for index in range(simbev.FRAMES_PER_SCENE):
        stem = f"SimBEV-scene-0000-frame-{index:04d}"
        frame = {"ego2global_translation": [index * 0.5, 0.0, 0.0], "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
                 "LIDAR": f"/dataset/simbev/sweeps/LIDAR/{stem}-LIDAR.npz",
                 "GT_SEG": f"/dataset/simbev/ground-truth/seg/{stem}-GT_SEG.npz",
                 "GT_DET": f"/dataset/simbev/ground-truth/det/{stem}-GT_DET.bin"}
        for camera in simbev.CAMERAS:
            frame[f"RGB-{camera}"] = f"/dataset/simbev/sweeps/RGB-{camera}/{stem}-RGB-{camera}.jpg"
            if simbev.needed_member(frame[f"RGB-{camera}"]):
                files[f"RGB-{camera}/{stem}-RGB-{camera}.jpg"] = jpeg
        if index in simbev.SAMPLE_FRAMES:
            out = io.BytesIO()
            np.savez(out, data=np.c_[_points(-1.8), np.zeros(len(_points()))])
            files[f"LIDAR/{stem}-LIDAR.npz"] = out.getvalue()
            out = io.BytesIO()
            np.savez(out, data=np.ones((3, simbev.SIMBEV_BEV_CELLS, simbev.SIMBEV_BEV_CELLS), np.uint8))
            files[f"seg/{stem}-GT_SEG.npz"] = out.getvalue()
            files[f"det/{stem}-GT_DET.bin"] = _detections([
                _box(index * 0.5 + 12.0, 0.0, 14, valid=True), _box(index * 0.5 + 20.0, 6.0, 12, valid=False),
            ])
        frames.append(frame)
    files["simbev_infos_train.json"] = json.dumps({"metadata": metadata, "data": {
        "scene_0000": {"scene_info": {"map": "Town01"}, "scene_data": frames}}}).encode()
    for split in ("val", "test"):
        files[f"simbev_infos_{split}.json"] = json.dumps({"metadata": metadata, "data": {}}).encode()
    return files


def test_simbev_labels_road_to_72_m_and_objects_only():
    files = _simbev_files()
    assert all(simbev.needed_member(name) for name in files)
    key, members = next(simbev.iter_samples(files, 0, 1))
    assert key == f"simbev-scene_0000-{simbev.FIRST_SAMPLE:04d}"
    _, valid = _check_members(key, members)
    assert _class_valid(valid) == {name: name not in simbev.NEVER_LABELLED for name in BEV_SEGMENTATION_CLASSES}
    drivable = BEV_SEGMENTATION_CLASSES.index("drivable_area")
    row, _ = np.rint(AUTOBEV_GEOMETRY.ego_to_pixel(np.array([[76.0, 0.0]]))[0]).astype(int)
    assert not valid[drivable, row].any()
    boxes, labels = decode_boxes(members["boxes.npz"])
    np.testing.assert_allclose(boxes[0, :6], [12.0, 0.0, 0.75, 4.0, 2.0, 1.5], atol=1e-5)
    assert labels.tolist() == [BOX_CLASSES.index("VEHICLE")]


def test_simbev_reads_only_sample_frames():
    assert simbev.needed_member("x/SimBEV-scene-0007-frame-0070-LIDAR.npz")
    assert not simbev.needed_member("x/SimBEV-scene-0007-frame-0080-LIDAR.npz")
    assert simbev.needed_member("RGB-CAM_BACK/SimBEV-scene-0007-frame-0000-RGB-CAM_BACK.jpg")
    assert not simbev.needed_member("RGB-CAM_BACK/SimBEV-scene-0007-frame-0005-RGB-CAM_BACK.jpg")
    assert simbev.needed_member("simbev_infos_val.json")
    assert len(simbev.SAMPLE_FRAMES) == 13


# --- shared ---------------------------------------------------------------------


def test_a_never_labelled_class_cannot_be_labelled():
    shape = (AUTOBEV_GEOMETRY.height_px, AUTOBEV_GEOMETRY.width_px)
    labelled = {"stop_line": (np.zeros(shape, np.float32), np.ones(shape, bool))}
    with pytest.raises(ValueError, match="class rules"):
        bev_segmentation(labelled, never_labelled=frozenset({"stop_line"}))


def test_write_partition_records_the_autobev_contract(tmp_path):
    samples = list(itertools.islice(nuscenes.iter_samples(_nuscenes_files(), 0, 1), 1))
    manifest = write_partition(
        tmp_path, partition_id="nuscenes-0000", dataset=nuscenes.DATASET, pack_contract=nuscenes.PACK_CONTRACT,
        cameras=nuscenes.CAMERAS, never_labelled=nuscenes.NEVER_LABELLED, samples=samples,
    )
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    assert manifest["sample_kind"] == SAMPLE_KIND
    assert {key: manifest[key] for key in manifest_contract()} == json.loads(json.dumps(manifest_contract()))
    assert manifest["total_samples"] == 1 and manifest["shard_names"] == ["nuscenes-0000-00000.tar"]
    import tarfile

    with tarfile.open(tmp_path / "nuscenes-0000-00000.tar") as archive:
        names = archive.getnames()
    assert len(names) == len(samples[0][1]) and all(n.startswith("nusc-scene-0001-s7.") for n in names)
