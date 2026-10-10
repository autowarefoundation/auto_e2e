# ruff: noqa
"""Publication-quality parameter decomposition with non-overlapping panels."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.4,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.4,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 8.0,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig_param_decomposition.svg"

total = 79_906_522
total_parts = [
    ("T8 temporal fusion", 30_809_856, "#4f6685"),
    ("ResNet-50 backbone", 23_508_032, "#667a97"),
    ("BEV queries", 15_360_000, "#7e8fa8"),
    ("BEVFormer encoder x6", 4_940_928, "#94a2b6"),
    ("FPN", 3_278_592, "#aab5c4"),
    ("Other camera-BEV", 395_788, "#c0c8d2"),
    ("GRU planner", 835_380, "#257f48"),
    ("Navigation fusion", 350_288, "#3d9a61"),
    ("Map/Route encoders + gate", 245_440, "#61b47e"),
    ("BEV seg. head (disabled)", 165_000, "#cf9c24"),
    ("Route reconstruction", 17_218, "#8bc99f"),
]
assert sum(v for _, v, _ in total_parts) == total

trainable = 1_448_582
train_parts = [
    ("GRU control planner", 835_380, "#1c7f3d"),
    ("Deformable navigation fusion", 350_288, "#2f9c59"),
    ("Map + Route encoders + gate", 245_440, "#58b778"),
    ("Route reconstruction head", 17_218, "#8bcca2"),
    ("Front residual gate", 256, "#b8ddc4"),
]
assert sum(v for _, v, _ in train_parts) == trainable

fig = plt.figure(figsize=(7.3, 5.35), facecolor="white")
gs = fig.add_gridspec(
    nrows=3,
    ncols=1,
    height_ratios=[1.0, 0.92, 1.85],
    left=0.30,
    right=0.975,
    top=0.91,
    bottom=0.10,
    hspace=0.42,
)
fig.suptitle("Where the parameters are used", fontsize=10.5, fontweight="bold", y=0.975)

# Panel (a): one stacked bar.
ax_top = fig.add_subplot(gs[0])
left = 0.0
for name, value, color in total_parts:
    share = 100.0 * value / total
    ax_top.barh(0, share, left=left, height=0.48, color=color, edgecolor="white", linewidth=0.45)
    if share >= 4.0:
        ax_top.text(
            left + share / 2.0,
            0,
            f"{share:.1f}%",
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color="white" if share >= 10 else "#202020",
        )
    left += share
ax_top.set_xlim(0, 100)
ax_top.set_ylim(-0.6, 0.6)
ax_top.set_yticks([])
ax_top.set_xlabel("share of all 79,906,522 parameters [%]", labelpad=2)
ax_top.set_title(
    "(a) Total capacity — camera-BEV pathway: 78.29 M (97.98%)",
    loc="left",
    fontweight="bold",
    pad=8,
)
ax_top.text(
    100,
    0.47,
    "78,292,940 frozen + 256 trainable",
    ha="right",
    va="bottom",
    fontsize=7.0,
    color="#444",
)
ax_top.grid(axis="x", alpha=0.22, linewidth=0.5)

# Dedicated legend row: never shares the trainable panel's drawing area.
ax_leg = fig.add_subplot(gs[1])
ax_leg.axis("off")
legend_handles = [
    Patch(facecolor=color, edgecolor="none", label=f"{name}: {value/1e6:.2f} M ({100*value/total:.2f}%)")
    for name, value, color in total_parts
]
ax_leg.legend(
    handles=legend_handles,
    ncol=3,
    frameon=False,
    loc="center",
    bbox_to_anchor=(0.50, 0.50),
    fontsize=6.7,
    columnspacing=1.0,
    handlelength=1.2,
    handletextpad=0.45,
    labelspacing=0.65,
)

# Panel (b): trainable allocation.
ax_bottom = fig.add_subplot(gs[2])
labels = [p[0] for p in train_parts][::-1]
values = [p[1] for p in train_parts][::-1]
colors = [p[2] for p in train_parts][::-1]
shares = [100.0 * value / trainable for value in values]
y = list(range(len(labels)))
bars = ax_bottom.barh(y, shares, color=colors, height=0.58)
ax_bottom.set_yticks(y)
ax_bottom.set_yticklabels(labels)
ax_bottom.set_xlim(0, 69)
ax_bottom.set_xlabel("share of 1,448,582 trainable parameters [%]", labelpad=4)
ax_bottom.set_title(
    "(b) Trainable allocation — task-specific planning and navigation modules",
    loc="left",
    fontweight="bold",
    pad=9,
)
ax_bottom.grid(axis="x", alpha=0.25, linewidth=0.5)
ax_bottom.set_axisbelow(True)
for bar, value, share in zip(bars, values, shares):
    ax_bottom.text(
        min(share + 0.8, 64.0),
        bar.get_y() + bar.get_height() / 2.0,
        f"{share:.2f}%  ({value:,})",
        ha="left",
        va="center",
        fontsize=7.4,
        clip_on=False,
    )

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.12)
print(OUT)
