# ruff: noqa
"""Real-data figures from the public KITScenes Val v3.5 artifacts (deterministic protocol)."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, "/tmp/autoe2e_paper/figgen")
import aovl

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none"})
OUT = "/tmp/autoe2e_paper/vivlio/figures/"
CACHE = "/tmp/autoe2e_paper/qual_cache"
records = json.load(open(os.path.join(CACHE, "records.json")))

# geometry autoe2e-bev-450x300-0p4m-v1: rows = X from +120 (row 0) to -60 ; cols = Y from +60 (col 0) to -60
X_MAX, X_MIN, Y_MAX, Y_MIN, MPP = 120.0, -60.0, 60.0, -60.0, 0.4
EXTENT_IMG = (Y_MAX, Y_MIN, X_MIN, X_MAX)  # imshow extent (left,right,bottom,top) with x-axis = Y(left+), y-axis = X(forward)

CH_NAMES = ["0 drivable area", "1 lane boundary", "2 lane centerline", "3 intersection", "4 crosswalk",
            "5 stop line", "6 static traffic signal", "7 traffic-direction sin", "8 traffic-direction cos",
            "9 traffic-direction valid", "10 known-map area", "11 road level", "12 road-level valid",
            "13 overlapping-level ambiguity", "R0 selected route corridor", "R1 destination marker"]


def load(rec):
    d = os.path.join(CACHE, rec["key"])
    m = np.load(os.path.join(d, "map_semantic.npz"))["array"]
    r = np.load(os.path.join(d, "route_mask.npz"))["array"]
    t = np.load(os.path.join(d, "trajectory_xy.npz"))
    return m, r, t["trajectory_xy_m"], t["trajectory_valid"].astype(bool), os.path.join(d, "cam_0.jpg")


def to_img_coords(xy):
    """ego XY (x forward, y left) -> plot coords (horizontal = y with left positive to the LEFT, vertical = x)."""
    return xy[:, 1], xy[:, 0]


# ---------------- Figure: channels ----------------
# deterministic choice: the scanned sample whose Map+Route rasters have the largest number of non-empty
# channels (ties -> earliest in scan order)
def nonempty(rec):
    m, r, *_ = load(rec)
    return int(sum(float((m[i] > 0).mean()) > 0 for i in range(14)) + sum(float((r[i] > 0).mean()) > 0 for i in range(2)))
sel = max(records, key=lambda rec: (nonempty(rec), -records.index(rec)))
m, r, gt, valid, cam = load(sel)
fig, axes = plt.subplots(4, 4, figsize=(7.0, 9.4))
for i, ax in enumerate(axes.flat):
    arr = m[i] if i < 14 else r[i - 14].astype(np.float32)
    cmap = "viridis" if i in (7, 8, 11) else "gray_r"
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, extent=EXTENT_IMG, interpolation="nearest", aspect="equal")
    n = int(valid.sum()); gx, gy = to_img_coords(gt[:n]); ax.plot(gx, gy, color="#d62728", lw=0.9, alpha=0.9)
    ax.plot([0], [0], marker="^", color="#d62728", ms=4.5, mew=0)
    ax.set_title(CH_NAMES[i], fontsize=7.4, pad=2)
    ax.set_xticks([40, 0, -40]); ax.set_xticklabels(["+40", "0", "-40"], fontsize=5.5)
    ax.set_yticks([-40, 0, 40, 80, 120]); ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlim(60, -60)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.text(58, -57, f"occupied {float((arr > 0).mean()) * 100:.1f}%", fontsize=5.6, ha="left", va="bottom",
            color="#333", bbox=dict(facecolor="white", alpha=0.8, lw=0, pad=1))
fig.suptitle("HD Map (14 channels) and Route (2 channels) rasters of one KITScenes Val v3.5 sample\n"
             "450 x 300 cells at 0.4 m/px, X: -60 ... 120 m (up), Y: -60 ... 60 m (left positive to the left); "
             "red: ego and recorded 5 s trajectory", fontsize=7.8, y=0.995)
fig.text(0.5, 0.004, f"sample {sel['key']} (route maneuver: {sel['maneuver']}, intersection ahead on route: {sel['intersection']}). "
         "Channels 7/8 store (sin+1)/2 and (cos+1)/2 of the lane direction where channel 9 is 1; channel 11 stores (level+8)/16 where channel 12 is 1.",
         ha="center", fontsize=6.0, wrap=True)
fig.tight_layout(rect=(0, 0.015, 1, 0.965))
fig.savefig(OUT + "fig_channels_real.svg", bbox_inches="tight"); plt.close(fig)
print("channels sample:", sel["key"], "nonempty", nonempty(sel))

# ---------------- Figure: qualitative ----------------
MODELS = [("kit_ep5", "KITScenes Ep. 5", "#1b7f3b", "-"), ("kit_ep7", "KITScenes Ep. 7", "#e07b00", "-"),
          ("nuplan_ep5", "nuPlan Ep. 5", "#2456a4", "--"), ("nuplan_ep4", "nuPlan Ep. 4", "#7f7f7f", "--")]

by_fde = sorted([r for r in records if r["errors"]["kit_ep5"]["fde5"] is not None], key=lambda r: r["errors"]["kit_ep5"]["fde5"])
panels = [
    ("(a) lowest Ep.5 FDE@5s of the scan", by_fde[0]),
    ("(b) highest Ep.5 FDE@5s of the scan", by_fde[-1]),
    ("(c) first 'left' maneuver", next(r for r in records if r["maneuver"] == "left")),
    ("(d) first 'right' maneuver", next(r for r in records if r["maneuver"] == "right")),
    ("(e) first straight, intersection ahead", next(r for r in records if r["maneuver"] == "straight" and r["intersection"])),
    ("(f) first straight, no intersection", next(r for r in records if r["maneuver"] == "straight" and not r["intersection"])),
]
fig = plt.figure(figsize=(7.0, 7.9))
gs = fig.add_gridspec(4, 3, height_ratios=[0.5, 1.0, 0.5, 1.0], hspace=0.22, wspace=0.14, top=0.945, bottom=0.10)
summary_rows = []
for k, (title, rec) in enumerate(panels):
    col = k % 3; rowblock = (k // 3) * 2
    m, r, gt, valid, cam = load(rec)
    ax_cam = fig.add_subplot(gs[rowblock, col]); ax = fig.add_subplot(gs[rowblock + 1, col])
    img = mpimg.imread(cam); h = img.shape[0]
    ax_cam.imshow(img[h // 4: h * 3 // 4 + h // 8]); ax_cam.set_xticks([]); ax_cam.set_yticks([])
    ax_cam.set_title(title, fontsize=7.4, pad=2)
    # BEV background: drivable (ch0), intersection (ch3), corridor, destination, lane boundaries (ch1)
    rgb = np.ones((*m.shape[1:], 3))
    rgb[m[0] > 0] = (0.90, 0.90, 0.90)
    rgb[m[3] > 0] = (0.84, 0.84, 0.90)
    rgb[r[0] > 0] = (0.72, 0.86, 0.98)
    rgb[m[1] > 0] = (0.55, 0.55, 0.55)
    rgb[m[5] > 0] = (0.80, 0.30, 0.30)
    rgb[r[1] > 0] = (0.85, 0.35, 0.85)
    ax.imshow(rgb, extent=EXTENT_IMG, interpolation="nearest", aspect="equal")
    n = int(valid.sum())
    gx, gy = to_img_coords(gt[:n]); ax.plot(gx, gy, color="k", lw=1.6, label="recorded (GT, 5 s)")
    fdes = {}
    for key, lab, c, ls in MODELS:
        p = rec["preds"][key]
        xy, _, _ = aovl.rollout(np.asarray(p["controls"]), p["v0"])
        px, py = to_img_coords(xy[:n]); ax.plot(px, py, ls, color=c, lw=1.2, label=lab)
        fdes[lab] = float(np.linalg.norm(xy[n - 1] - gt[n - 1]))
    ax.plot([0], [0], marker="^", color="#d62728", ms=6, mew=0)
    ax.set_xlim(30, -30); ax.set_ylim(-12, 62)  # y left positive on the left; x forward up
    ax.set_xticks([20, 0, -20]); ax.set_xticklabels(["+20", "0", "-20"], fontsize=6)
    ax.set_yticks([0, 20, 40, 60]); ax.tick_params(axis="y", labelsize=6)
    if col == 0:
        ax.set_ylabel("x forward [m]", fontsize=6.5)
    if k >= 3:
        ax.set_xlabel("y left [m]", fontsize=6.5, labelpad=1)
    ax.text(29, 60, f"v0 = {rec['ego_now'][0]:.1f} m/s\nFDE@5s Ep5 {fdes['KITScenes Ep. 5']:.2f} m | Ep7 {fdes['KITScenes Ep. 7']:.2f} m\n"
            f"nuPlan Ep5 {fdes['nuPlan Ep. 5']:.2f} m | Ep4 {fdes['nuPlan Ep. 4']:.2f} m",
            fontsize=5.3, va="top", ha="left", bbox=dict(facecolor="white", alpha=0.85, lw=0, pad=1.5))
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    summary_rows.append((title, rec["key"], rec["maneuver"], rec["intersection"], rec["ego_now"][0], fdes))
handles = [Line2D([], [], color="k", lw=1.6, label="recorded trajectory (GT, 5.0 s valid)")] + \
          [Line2D([], [], color=c, ls=ls, lw=1.2, label=lab) for _, lab, c, ls in MODELS] + \
          [Patch(facecolor=(0.90, 0.90, 0.90), label="drivable area"), Patch(facecolor=(0.84, 0.84, 0.90), label="intersection"),
           Patch(facecolor=(0.72, 0.86, 0.98), label="selected route corridor"), Patch(facecolor=(0.85, 0.35, 0.85), label="destination marker"),
           Patch(facecolor=(0.55, 0.55, 0.55), label="lane boundary"), Patch(facecolor=(0.80, 0.30, 0.30), label="stop line")]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Qualitative KITScenes Val v3.5 samples: published control overlays rolled out with the unicycle model (first 50 steps)\n"
             "over the sample's own Map/Route rasters; top: model-input front tile (centre crop)", fontsize=7.4, y=0.995)
fig.savefig(OUT + "fig_qualitative.svg", bbox_inches="tight"); plt.close(fig)
json.dump(summary_rows, open("/tmp/autoe2e_paper/qual_summary.json", "w"), indent=1)
for row in summary_rows:
    print(row)

# ---------------- Figure: BEV geometry ----------------
fig, ax = plt.subplots(figsize=(3.4, 3.9))
ax.add_patch(Rectangle((-128, -85.5), 256, 256, fill=False, ec="#999", ls=":", lw=1.0))
ax.add_patch(Rectangle((-60, -60), 120, 180, fill=False, ec="#2456a4", lw=1.4))
ax.add_patch(Rectangle((-60, -60), 120, 180, fill=True, fc="#2456a4", alpha=0.06))
for yy in np.arange(-60, 60.1, 20):
    ax.plot([yy, yy], [-60, 120], color="#2456a4", lw=0.3, alpha=0.5)
for xx in np.arange(-60, 120.1, 20):
    ax.plot([-60, 60], [xx, xx], color="#2456a4", lw=0.3, alpha=0.5)
ax.plot(0, 0, marker="^", color="#d62728", ms=8, mew=0)
ax.annotate("ego (0,0)", (0, 0), xytext=(8, -14), textcoords="offset points", fontsize=7)
ax.text(0, 124, "camera BEV latent 300 x 200 @ 0.6 m and Map/Route raster 450 x 300 @ 0.4 m\n"
        "X: -60 ... +120 m (forward), Y: -60 ... +60 m (left)", ha="center", va="bottom", fontsize=6.6, color="#2456a4")
ax.text(0, -92, "dotted: published KITScenes v3 audit geometry 256 x 256 @ 1.0 m\n(X: -85.5 ... 170.5 m, Y: -128 ... 128 m); not the model-facing raster",
        ha="center", va="top", fontsize=6.2, color="#666")
ax.set_xlim(135, -135); ax.set_ylim(-140, 185)
ax.set_xlabel("y (left positive) [m]", fontsize=7); ax.set_ylabel("x (forward) [m]", fontsize=7)
ax.tick_params(labelsize=6.5); ax.set_aspect("equal")
ax.set_title("BEV extents (top view)", fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "fig_geometry.svg", bbox_inches="tight"); plt.close(fig)
print("done")
