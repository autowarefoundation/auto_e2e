# ruff: noqa
"""Deterministic sample scan of the public KITScenes Val v3.5 dashboard artifacts.

Protocol (documented in the paper): shards sorted by artifact name; for each of the
first N shards take the sample at the middle index of the shard's index list; fetch
map_semantic / route_mask / trajectory_xy / navigation_meta / cam_0 and all four
published control overlays (schema v5, seed_count 1). Nothing is filtered by error.
"""
import io, json, os, sys, urllib.request, urllib.parse
import numpy as np
sys.path.insert(0, "/tmp/autoe2e_paper/figgen")
import aovl

BASE = "https://d2itskdqq39tx1.cloudfront.net/api/v1/datasets/kitscenes-val/shards"
OUT = "/tmp/autoe2e_paper/qual_cache"
os.makedirs(OUT, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

MODELS = {
    "kit_ep5": "120a21639d9767d512eea645be0a83b01476c9e0366a8b957ac51b7ead1bd762",
    "kit_ep7": "a1e6b1621018740485b3e1e43dda23713d913667eb39a65b77fcbf895dcbcefd",
    "nuplan_ep5": "ca8b43d7a777d6fd9195bb253d1452f3cd645f7aab3bd0e36b1f74ffb31df29b",
    "nuplan_ep4": "ed00e072471aae51b4cf48fa54c3fd9c03c61e0fd6e2978829cd9068dfd6da68",
}


def get(url, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
    return None


shards = sorted(json.load(open("/tmp/autoe2e_paper/dash_val_shards_all.json")), key=lambda s: s["name"])
records = []
for shard in shards[:N]:
    name = shard["name"]
    idx = json.loads(get(f"{BASE}/{name}/index"))
    samples = idx["samples"]
    s = samples[len(samples) // 2]
    key = s["key"]
    mem = s["members"]
    rec_dir = os.path.join(OUT, key)
    os.makedirs(rec_dir, exist_ok=True)

    def member(mname, fname):
        path = os.path.join(rec_dir, fname)
        if not os.path.exists(path):
            m = mem[mname]
            open(path, "wb").write(get(f"{BASE}/{name}/blob?offset={m['offset']}&size={m['size']}"))
        return path

    member("map_semantic.npz", "map_semantic.npz")
    member("route_mask.npz", "route_mask.npz")
    member("trajectory_xy.npz", "trajectory_xy.npz")
    member("navigation_meta.json", "navigation_meta.json")
    cam = os.path.join(rec_dir, "cam_0.jpg")
    if not os.path.exists(cam):
        m = mem["cam_0.jpg"]
        open(cam, "wb").write(get(f"{BASE}/{name}/samples/{key}/image/cam_0?offset={m['offset']}&size={m['size']}"))
    preds = {}
    for label, mid in MODELS.items():
        opath = os.path.join(rec_dir, f"overlay_{label}.bin")
        if not os.path.exists(opath):
            open(opath, "wb").write(get(f"{BASE}/{name}/overlays/{mid}"))
        ov = aovl.decode(open(opath, "rb").read())
        row = ov["directory"][aovl.sample_uid_hash(key)]
        preds[label] = {"controls": ov["controls"][row, 0].tolist(), "v0": float(ov["v0"][row])}
    meta = json.load(open(os.path.join(rec_dir, "navigation_meta.json")))
    traj = np.load(os.path.join(rec_dir, "trajectory_xy.npz"))
    gt = traj["trajectory_xy_m"]; valid = traj["trajectory_valid"].astype(bool)
    errs = {}
    for label, p in preds.items():
        xy, _, _ = aovl.rollout(np.asarray(p["controls"]), p["v0"])
        e = np.linalg.norm(xy - gt, axis=1)
        errs[label] = {"fde5": float(e[49]) if valid[49] else None,
                       "ade5": float(e[:50][valid[:50]].mean()) if valid[:50].any() else None}
    rec = {"shard": name, "key": key, "ego_now": s["ego_now"], "n_valid": int(valid.sum()),
           "maneuver": meta.get("route_maneuver"), "intersection": meta.get("route_intersection"),
           "destination_visible": meta.get("destination_visible"), "route_valid": meta.get("route_valid"),
           "map_valid": meta.get("map_valid"), "geometry_id": meta.get("geometry_id"),
           "route_confidence": meta.get("route_confidence"), "errors": errs, "preds": preds}
    records.append(rec)
    print(f"{name[:22]} {key[-8:]} v0={s['ego_now'][0]:.2f} man={rec['maneuver']} inter={rec['intersection']} "
          f"dest={rec['destination_visible']} fde5(ep5)={errs['kit_ep5']['fde5']} geom={rec['geometry_id']}")
json.dump(records, open(os.path.join(OUT, "records.json"), "w"))
print("saved", len(records))
