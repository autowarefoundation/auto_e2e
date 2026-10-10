# ruff: noqa
"""Result figures from the reported evaluation numbers (no fabrication)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.8, "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
OUT = "/tmp/autoe2e_paper/vivlio/figures/"
H = [1, 2, 3, 5]
MODELS = ["KITScenes Ep. 5", "KITScenes Ep. 7", "nuPlan Ep. 5", "nuPlan Ep. 4"]
COL = {"KITScenes Ep. 5": "#1b7f3b", "KITScenes Ep. 7": "#e07b00", "nuPlan Ep. 5": "#2456a4", "nuPlan Ep. 4": "#7f7f7f"}
MK = {"KITScenes Ep. 5": "o", "KITScenes Ep. 7": "s", "nuPlan Ep. 5": "^", "nuPlan Ep. 4": "D"}
LS = {"KITScenes Ep. 5": "-", "KITScenes Ep. 7": "-", "nuPlan Ep. 5": "--", "nuPlan Ep. 4": "--"}

VAL_ADE = {"KITScenes Ep. 5": [0.1472, 0.3729, 0.7449, 1.9405], "KITScenes Ep. 7": [0.1347, 0.3886, 0.8035, 2.0952],
           "nuPlan Ep. 5": [0.1951, 0.4723, 0.9147, 2.2951], "nuPlan Ep. 4": [0.2030, 0.5018, 0.9503, 2.2919]}
VAL_FDE = {"KITScenes Ep. 5": [0.2850, 0.9239, 2.0196, 5.5645], "KITScenes Ep. 7": [0.2831, 1.0101, 2.2127, 5.9509],
           "nuPlan Ep. 5": [0.3779, 1.1331, 2.4226, 6.3885], "nuPlan Ep. 4": [0.4066, 1.1935, 2.4508, 6.2272]}
TEST_ADE = {"KITScenes Ep. 5": [0.1463, 0.4121, 0.8263, 2.1042], "KITScenes Ep. 7": [0.1680, 0.4923, 0.9831, 2.4699],
            "nuPlan Ep. 5": [0.1602, 0.3979, 0.7924, 2.1120], "nuPlan Ep. 4": [0.1916, 0.4817, 0.9181, 2.2840]}
TEST_FDE = {"KITScenes Ep. 5": [0.3107, 1.0453, 2.2199, 5.9422], "KITScenes Ep. 7": [0.3754, 1.2512, 2.6242, 6.8964],
            "nuPlan Ep. 5": [0.3134, 0.9727, 2.1621, 6.1751], "nuPlan Ep. 4": [0.3899, 1.1514, 2.3895, 6.4266]}
VAL_LAT = {"KITScenes Ep. 5": 0.9844, "KITScenes Ep. 7": 1.0220, "nuPlan Ep. 5": 1.1911, "nuPlan Ep. 4": 1.2082}
VAL_LON = {"KITScenes Ep. 5": 1.4025, "KITScenes Ep. 7": 1.5659, "nuPlan Ep. 5": 1.6633, "nuPlan Ep. 4": 1.6448}
TEST_LAT = {"KITScenes Ep. 5": 1.2044, "KITScenes Ep. 7": 1.4730, "nuPlan Ep. 5": 1.0584, "nuPlan Ep. 4": 1.2167}
TEST_LON = {"KITScenes Ep. 5": 1.4051, "KITScenes Ep. 7": 1.6598, "nuPlan Ep. 5": 1.5496, "nuPlan Ep. 4": 1.6287}


def curves(ade, fde, title, fname, n):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55))
    for ax, data, name in zip(axes, (ade, fde), ("ADE [m]", "FDE [m]")):
        for m in MODELS:
            ax.plot(H, data[m], LS[m], marker=MK[m], ms=4, lw=1.4, color=COL[m], label=m)
            dy = {"KITScenes Ep. 5": -7, "KITScenes Ep. 7": -1, "nuPlan Ep. 5": 6, "nuPlan Ep. 4": 0}[m]
            ax.annotate(f"{data[m][-1]:.2f}", (5, data[m][-1]), textcoords="offset points", xytext=(7, dy), fontsize=6.5, color=COL[m])
        ax.set_xlabel("horizon [s]"); ax.set_ylabel(name); ax.set_xticks(H); ax.grid(alpha=0.3, lw=0.5)
        ax.set_xlim(0.7, 5.9)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"{title}  (n = {n} samples per model)", fontsize=9.5, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT + fname, bbox_inches="tight"); plt.close(fig)


curves(VAL_ADE, VAL_FDE, "KITScenes Val v3.5 — Camera + HD Map + Route (external overlay replay)", "fig_results_val.svg", "11,035")
curves(TEST_ADE, TEST_FDE, "KITScenes Test v1.0 — Camera only (external overlay replay)", "fig_results_test.svg", "23,690")

# lateral / longitudinal bars
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.4), sharey=True)
for ax, lat, lon, title in zip(axes, (VAL_LAT, TEST_LAT), (VAL_LON, TEST_LON),
                               ("Val v3.5: Camera + Map + Route (n = 11,035)", "Test v1.0: Camera only (n = 23,690)")):
    x = np.arange(len(MODELS)); w = 0.38
    b1 = ax.bar(x - w / 2, [lat[m] for m in MODELS], w, color=[COL[m] for m in MODELS], alpha=0.55, label="mean |lateral| (y)")
    b2 = ax.bar(x + w / 2, [lon[m] for m in MODELS], w, color=[COL[m] for m in MODELS], alpha=1.0, label="mean |longitudinal| (x)",
                hatch="///", edgecolor="white")
    for rect in list(b1) + list(b2):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02, f"{rect.get_height():.2f}",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([m.replace(" Ep. ", "\nEp. ") for m in MODELS], fontsize=7.5)
    ax.set_title(title); ax.grid(axis="y", alpha=0.3, lw=0.5); ax.set_ylim(0, 2.35)
axes[0].set_ylabel("mean absolute error over valid steps [m]")
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(facecolor="#888", alpha=0.55, label="mean |lateral| error"),
                        Patch(facecolor="#888", hatch="///", edgecolor="white", label="mean |longitudinal| error")],
               frameon=False, loc="upper center", ncol=2, fontsize=7.2)
fig.tight_layout(); fig.savefig(OUT + "fig_results_latlon.svg", bbox_inches="tight"); plt.close(fig)

# ---- Epoch 5 vs Epoch 7 internal validation trade-off (n = 3,820 samples) ----
metrics = [
    ("ADE 6.4 s [m]", 2.6326, 2.7548, "lower"),
    ("FDE 6.4 s [m]", 7.5567, 7.8574, "lower"),
    ("Route corridor compliance", 0.3959, 0.4236, "higher"),
    ("Route corridor success", 0.0608, 0.0763, "higher"),
    ("Route progress proxy [m]", 44.7857, 45.5856, "higher"),
    ("Route progress ratio", 0.8725, 0.8982, "higher"),
    ("Drivable-area compliance", 0.7758, 0.7988, "higher"),
    ("Drivable-area success", 0.4442, 0.4139, "higher"),
    ("Comfortable rate", 0.7542, 0.7466, "higher"),
    ("Route reconstruction IoU", 0.9991, 0.9990, "higher"),
]
fig, ax = plt.subplots(figsize=(6.9, 2.9))
labels = [m[0] for m in metrics]
rel = []
better = []
for name, e5, e7, direction in metrics:
    r = (e7 - e5) / e5 * 100.0
    rel.append(r)
    improved = (r < 0) if direction == "lower" else (r > 0)
    better.append(improved)
y = np.arange(len(metrics))[::-1]
colors = ["#1b7f3b" if b else "#b8443f" for b in better]
ax.barh(y, rel, color=colors, alpha=0.85, height=0.62)
for yi, r, (name, e5, e7, d) in zip(y, rel, metrics):
    ax.text(max(r, 0.0) + 0.3, yi, f"{r:+.1f}%   ({e5:.4g} -> {e7:.4g})", va="center", ha="left", fontsize=7)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("relative change from KITScenes Epoch 5 to Epoch 7 [%]  (green = improvement in the metric's preferred direction)")
ax.set_xlim(-12, 30); ax.grid(axis="x", alpha=0.3, lw=0.5)
ax.set_title("Checkpoint-internal validation, KITScenes frozen split (n = 3,820; route metrics on 2,911 route-valid samples)", fontsize=8.8)
fig.tight_layout(); fig.savefig(OUT + "fig_tradeoff_ep5_ep7.svg", bbox_inches="tight"); plt.close(fig)

# ---- parameter budget ----
parts = [
    ("T8 temporal fusion", 30_809_856, "frozen"),
    ("ResNet-50 backbone", 23_508_032, "frozen"),
    ("BEV queries (300x200x256)", 15_360_000, "frozen"),
    ("Encoder layers x6", 4_940_928, "frozen"),
    ("FPN (4 levels)", 3_278_592, "frozen"),
    ("Front cross-attention copy", 328_960, "frozen"),
    ("Pos./level/camera embeddings", 66_560, "frozen"),
    ("BEV segmentation head", 165_000, "frozen"),
    ("GRU control planner", 835_380, "trainable"),
    ("Deformable navigation fusion", 350_288, "trainable"),
    ("Map encoder E_map", 167_200, "trainable"),
    ("Route encoder E_route", 77_984, "trainable"),
    ("Route reconstruction head", 17_218, "trainable"),
    ("Route gate + front gate", 512, "trainable"),
]
fig, ax = plt.subplots(figsize=(6.9, 3.0))
names = [p[0] for p in parts]; vals = [p[1] for p in parts]; kinds = [p[2] for p in parts]
y = np.arange(len(parts))[::-1]
ax.barh(y, vals, color=["#6E7B8E" if k == "frozen" else "#2E8B57" for k in kinds], height=0.65)
ax.set_xscale("log"); ax.set_xlim(2e2, 2e8)
for yi, v in zip(y, vals):
    ax.text(v * 1.15, yi, f"{v:,}", va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("parameters (log scale)")
ax.set_title("Parameter budget of the evaluated configuration: 79,906,522 total; 1,448,582 trainable in trajectory stages", fontsize=8.8)
ax.legend(handles=[Patch(facecolor="#6E7B8E", label="frozen in trajectory stages (78.46 M)"),
                   Patch(facecolor="#2E8B57", label="trainable (1.45 M)")], frameon=False, loc="lower right")
ax.grid(axis="x", alpha=0.3, lw=0.5)
fig.tight_layout(); fig.savefig(OUT + "fig_param_budget.svg", bbox_inches="tight"); plt.close(fig)
print("results figures written")
