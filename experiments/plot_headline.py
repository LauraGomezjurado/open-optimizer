"""Two headline figures for the optimizer result.

fig_mechanism.png    effective rank vs depth, next to effective rank vs width
fig_matched_loss.png effective rank vs loss, one point per seed

Numbers are read straight out of llm_probe/results/ by scripts/collect below, so
editing this file cannot drift from the raw runs. Run from the repo root:
    python experiments/plot_headline.py
"""
import json
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "llm_probe" / "results"
OUT = ROOT / "results"

# categorical slots 1-3 from the validated palette, light mode
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
SURFACE = "#fcfcfb"

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1.0,
})


def load(name):
    with open(RES / name) as fh:
        d = json.load(fh)
    return d["cfg"], d["final"]


def scan(tag_a, tag_m):
    """Return (adamw_effrank, muon_effrank) for one scan point."""
    return load(tag_a)[1]["mlp_effrank"], load(tag_m)[1]["mlp_effrank"]


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def endlabel(ax, x, y, text, color, dx=0.35, dy=0.0):
    """Colored mark carries identity, the text itself stays in ink."""
    ax.plot([x], [y], "o", color=color, ms=9, zorder=5,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(text, (x, y), xytext=(x + dx, y + dy), color=INK,
                fontsize=10, fontweight="bold", va="center")


# ---------------------------------------------------------------- fig 1
depths = [2, 4, 6, 8, 12]
depth_files = {2: ("flat_ar_seed0_scan_d2.json", "flat_ar_seed0_muon_scan_d2.json"),
               4: ("flat_ar_seed0_scan_d4.json", "flat_ar_seed0_muon_scan_d4.json"),
               6: ("flat_ar_seed0_scan_w384.json", "flat_ar_seed0_muon_scan_w384.json"),
               8: ("flat_ar_seed0_scan_d8.json", "flat_ar_seed0_muon_scan_d8.json"),
               12: ("flat_ar_seed0_scan_d12.json", "flat_ar_seed0_muon_scan_d12.json")}
d_adam, d_muon = zip(*[scan(*depth_files[L]) for L in depths])

widths = [256, 384, 512]
width_files = {256: ("flat_ar_seed0_scan_w256.json", "flat_ar_seed0_muon_scan_w256.json"),
               384: ("flat_ar_seed0_scan_w384.json", "flat_ar_seed0_muon_scan_w384.json"),
               512: ("flat_ar_seed0_scan_w512.json", "flat_ar_seed0_muon_scan_w512.json")}
w_adam, w_muon = zip(*[scan(*width_files[d]) for d in widths])

fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.4, 4.9))

for ax, xs, adam, muon, xlabel, xhi in [
        (axl, depths, d_adam, d_muon, "layers (width fixed at 384)", 16.2),
        (axr, widths, w_adam, w_muon, "width (depth fixed at 6 layers)", 645)]:
    ax.plot(xs, muon, "-", color=ORANGE, lw=2, zorder=3)
    ax.plot(xs, muon, "o", color=ORANGE, ms=8, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.plot(xs, adam, "-", color=BLUE, lw=2, zorder=3)
    ax.plot(xs, adam, "o", color=BLUE, ms=8, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    style(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0.10, 0.66)
    span = max(xs) - min(xs)
    ax.set_xlim(min(xs) - span * 0.08, xhi)
    ax.set_xticks(xs)

axl.set_ylabel("independent directions used\n(effective rank, fraction of width)")
endlabel(axl, depths[-1], d_muon[-1], "Muon", ORANGE, dx=0.35)
endlabel(axl, depths[-1], d_adam[-1], "AdamW", BLUE, dx=0.35)
endlabel(axr, widths[-1], w_muon[-1], "Muon", ORANGE, dx=11)
endlabel(axr, widths[-1], w_adam[-1], "AdamW", BLUE, dx=11)

axl.set_title("Deeper helps Muon only", loc="left", fontsize=11.5,
              fontweight="bold", color=INK, pad=8)
axr.set_title("Wider helps AdamW only", loc="left", fontsize=11.5,
              fontweight="bold", color=INK, pad=8)

# Setup and caveats live in the README next to the image, not in the figure.
fig.text(0.052, 0.955, "Two ways of buying independent directions",
         ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
fig.subplots_adjust(top=0.845, bottom=0.115, left=0.088, right=0.985, wspace=0.26)
fig.savefig(OUT / "fig_mechanism.png", dpi=200, facecolor=SURFACE)
print("wrote", OUT / "fig_mechanism.png")

# ---------------------------------------------------------------- fig 2
# Text position is explicit so labels cannot collide. No synthetic marker is
# drawn: every dot on this figure is one real run, and the label sits beside its
# own cluster so the colored data marks carry identity.
series = [
    ("Muon", ORANGE, ["flat_ar_seed0_muon_optc.json", "flat_ar_seed1_muon_optc.json",
                      "flat_ar_seed2_muon_optc.json"], (2.7052, 0.518)),
    ("Aurora, 1 seed", AQUA, ["flat_ar_seed0_aurora_recomb.json"], (2.7232, 0.4959)),
    ("AdamW", BLUE, ["flat_ar_seed0_optc.json", "flat_ar_seed1_optc.json",
                     "flat_ar_seed2_optc.json"], (2.7605, 0.3505)),
]

fig2, ax = plt.subplots(figsize=(7.8, 4.9))
for name, color, files, (lx, ly) in series:
    pts = [(load(f)[1]["ar_ce"], load(f)[1]["mlp_effrank"]) for f in files]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "o", color=color, ms=11, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(lx, ly, name, color=INK, fontsize=10.5, fontweight="bold",
            va="center", ha="left")

style(ax)
ax.set_xlabel("next-token loss, lower is better")
ax.set_ylabel("independent directions used\n(effective rank, fraction of width)")
ax.set_ylim(0.30, 0.57)
ax.set_xlim(2.7005, 2.788)
fig2.text(0.055, 0.955, "Muon uses more directions at the same loss",
          ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
fig2.subplots_adjust(top=0.875, bottom=0.115, left=0.135, right=0.975)
fig2.savefig(OUT / "fig_matched_loss.png", dpi=200, facecolor=SURFACE)
print("wrote", OUT / "fig_matched_loss.png")
