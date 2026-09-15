"""Render every hidden unit of an evolved genome next to an SGD one.

The original results/genome_neurons_*.png pair had no saved generator, so it
could not be regenerated or retitled. This rebuilds it from
data/fer_genome/*.npz, side by side in one figure so the two are compared
directly. Run from the repo root:
    python experiments/plot_neurons.py --target skull

Method, following the FER paper: feed the coordinate grid through the network and
draw each hidden unit's activation as an image. Red is positive, blue is
negative, white is zero, on a shared scale of -1 to 1. A unit that runs past that
range fills solid. One column per unit, one row per layer.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fer_cppn import load_genome  # noqa: E402

INK, INK2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"
BORDER = np.array([0.62, 0.62, 0.60])  # gutter, so solid cells stay separable

RES = 56       # pixels per unit
GUTTER = 3     # white space between units
FRAME = 1      # border thickness around each unit


def unit_images(name, res=RES):
    """Return (n_layers, width, res, res) of post-activation unit responses."""
    net = load_genome(name)
    X = net.coordinate_grid(res)
    _, (_, As, _) = net.forward(X, cache=True)
    acts = As[1:]                      # drop the input, keep hidden layers
    grid = np.stack([a.reshape(res, res, -1).transpose(2, 0, 1) for a in acts])
    return grid                        # (layers, width, res, res)


def tile(grid, cmap="bwr"):
    """Lay units out as one RGB image, rows are layers and columns are units."""
    n_layers, width = grid.shape[:2]
    cell = RES + 2 * FRAME
    step = cell + GUTTER
    canvas = np.ones((n_layers * step - GUTTER, width * step - GUTTER, 3))
    lut = plt.get_cmap(cmap)
    for li in range(n_layers):
        for ui in range(width):
            img = np.clip(grid[li, ui], -1.0, 1.0)
            rgb = lut((img + 1.0) / 2.0)[..., :3]
            y, x = li * step, ui * step
            canvas[y:y + cell, x:x + cell] = BORDER
            canvas[y + FRAME:y + FRAME + RES, x + FRAME:x + FRAME + RES] = rgb
    return canvas


def main(target):
    evolved = unit_images(target)
    sgd = unit_images(f"sgd_{target}")
    panels = [("Evolved by Picbreeder", tile(evolved)),
              ("Trained by gradient descent", tile(sgd))]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6))
    for ax, (name, canvas) in zip(axes, panels):
        ax.imshow(canvas, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(name, loc="left", fontsize=11.5, fontweight="bold",
                     color=INK, pad=8)

    # How to read the grid belongs in the README next to the image, not here.
    fig.text(0.043, 0.965, f"Both networks draw the same {target}",
             ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.855, bottom=0.02, left=0.02, right=0.985,
                        wspace=0.06)

    out = os.path.join(ROOT, "results", f"fig_neurons_{target}.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="skull",
                    choices=["skull", "apple", "butterfly"])
    main(ap.parse_args().target)
