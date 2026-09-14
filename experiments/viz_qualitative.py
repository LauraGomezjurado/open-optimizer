"""
Qualitative FER-style visualization (borrows the FER paper's own method).

The FER paper (akarshkumar0101/fer) diagnoses fractured-vs-factored by EYE, with
two visual instruments and no scalar metrics:
  1. viz_feature_maps  -- render every hidden neuron's activation over the input
     grid as an image (bwr_r colormap). A factored net shows clean, coherent,
     low-redundancy neuron images; a fractured net shows noisy, redundant,
     high-frequency ones.
  2. sweep_weight      -- perturb a single weight across a range and render the
     output image sequence. A factored net changes smoothly and locally
     ("mouth opens"); a fractured net distorts globally ("skull-crushing").

We reproduce BOTH for two optimizer geometries fit to the same target, so the
qualitative fracture our scalar `sensitivity` tracks is directly visible. This is
the qualitative anchor for the quantitative metrics (which are our own
operationalization -- the FER paper reports no numbers).

Usage:
  python experiments/viz_qualitative.py --target skull
  python experiments/viz_qualitative.py --target butterfly --init_scale 1.5
Emits results/qual_featuremaps_<target>.png and results/qual_weepsweeps_<target>.png
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cppn_numpy import CPPN, make_coordinate_grid   # noqa
from optimizers import SteepestDescent               # noqa

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = 48
N_ITERS = 1200
N_HIDDEN, WIDTH = 6, 32
LR = {"adam": 0.02, "sign": 0.02, "spectral": 0.05, "scalar_adam": 0.02}

# the two geometries we contrast: fractured end vs factored end
GEOMS = [
    ("Adam (per-coord)", "adam", None),
    ("Spectral p=1 (isotropic)", "spectral", 1.0),
]
TARGETS = {"skull": "target_64.png", "apple": "target_apple_64.png",
           "butterfly": "target_butterfly_64.png"}


def cosine_lr(base, it, total):
    return base * 0.5 * (1 + np.cos(np.pi * it / total))


def load_target(target):
    im = Image.open(os.path.join(ROOT, "data", TARGETS[target])).convert("RGB")
    if im.size != (RES, RES):
        im = im.resize((RES, RES), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64).reshape(-1, 3) / 255.0


def fit(kind, p, target, init_scale, seed=0):
    cppn = CPPN(N_HIDDEN, WIDTH, init_scale=init_scale, seed=seed)
    X = make_coordinate_grid(RES)
    base = LR[kind]
    opt = SteepestDescent(cppn.params, kind=kind, p=p, lr=base)
    loss = None
    for it in range(N_ITERS):
        opt.lr = cosine_lr(base, it, N_ITERS)
        loss, grads = cppn.loss_and_grad(X, target)
        cppn.params = opt.step(cppn.params, grads)
    return cppn, X, float(loss)


def feature_maps(cppn, X, layer=-1):
    """Every hidden neuron's activation over the grid, as (width, RES, RES).
    layer=-1 -> last hidden layer (the one the feat-effrank metric reads)."""
    _, (_, As, _) = cppn.forward(X, params=cppn.params, cache=True)
    A = As[1:][layer]                       # skip input; (N, width)
    return A.T.reshape(-1, RES, RES)        # (width, RES, RES)


def plot_feature_maps(target, init_scale):
    tgt = load_target(target)
    nrow = len(GEOMS)
    fig, axes = plt.subplots(nrow, WIDTH,
                             figsize=(WIDTH * 0.42, nrow * 0.55 + 0.8), dpi=150)
    for gi, (label, kind, p) in enumerate(GEOMS):
        cppn, X, loss = fit(kind, p, tgt, init_scale)
        fmaps = feature_maps(cppn, X)
        sat = float(np.mean(np.abs(fmaps) > 0.95))
        for j in range(WIDTH):
            ax = axes[gi, j]
            # RAW activations clipped to [-1,1] (NOT per-neuron normalized): this
            # makes saturation visible -- fractured units clip to solid red/blue,
            # factored units keep graded structure. This is the FER intent.
            ax.imshow(np.clip(fmaps[j], -1, 1), cmap="bwr_r", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
        axes[gi, 0].set_ylabel(f"{label}\nloss={loss:.4f}\nsat={sat:.2f}",
                               fontsize=7, rotation=0, ha="right", va="center",
                               labelpad=28)
    fig.suptitle(f"Last-hidden neuron feature maps  —  target: {target}  "
                 "(solid = saturated/clipped; graded = structured. "
                 "top row fractures, bottom stays structured)", fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.86, bottom=0.02,
                        wspace=0.06, hspace=0.06)
    out = os.path.join(ROOT, "results", f"qual_featuremaps_{target}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Wrote {os.path.relpath(out, ROOT)}")


def plot_weight_sweeps(target, init_scale, n_weights=5, n_steps=7, r=0.6, seed=3):
    """FER's sweep_weight: perturb single weights, render the output sequence.
    Factored -> smooth/local change; fractured -> global 'skull-crushing'."""
    tgt = load_target(target)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(GEOMS) * n_weights, n_steps,
                             figsize=(n_steps * 1.1, len(GEOMS) * n_weights * 1.1),
                             dpi=120)
    ts = np.linspace(-r, r, n_steps)
    for gi, (label, kind, p) in enumerate(GEOMS):
        cppn, X, loss = fit(kind, p, tgt, init_scale)
        # sample weights from the FIRST hidden layer (most global influence)
        W = cppn.params[0]
        picks = [tuple(rng.integers([W.shape[0], W.shape[1]])) for _ in range(n_weights)]
        for wi, (r_, c_) in enumerate(picks):
            row = gi * n_weights + wi
            w0 = cppn.params[0][r_, c_]
            for si, t in enumerate(ts):
                cppn.params[0][r_, c_] = w0 + t
                img = cppn.render(X).reshape(RES, RES, 3)
                cppn.params[0][r_, c_] = w0
                ax = axes[row, si]
                ax.imshow(np.clip(img, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
                if si == 0:
                    ax.set_ylabel(f"{label[:10]}\nw{wi}", fontsize=6)
                if wi == 0 and gi == 0:
                    ax.set_title(f"t={t:+.2f}", fontsize=6)
    fig.suptitle(f"Single-weight sweeps (layer 0)  —  target: {target}  "
                 f"(top {n_weights} rows = {GEOMS[0][0]}, bottom = {GEOMS[1][0]})",
                 fontsize=9)
    fig.tight_layout()
    out = os.path.join(ROOT, "results", f"qual_weightsweeps_{target}.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"Wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS), default="skull")
    ap.add_argument("--init_scale", type=float, default=2.5)
    args = ap.parse_args()
    plot_feature_maps(args.target, args.init_scale)
    plot_weight_sweeps(args.target, args.init_scale)
