"""
Dose-response experiment: does update *geometry* move representational
factoredness when fitting a fixed target image with a CPPN?

We fit the SAME target (the evolved Picbreeder 'skull' image -- the UFR artifact
from the FER paper) with a ladder of optimizer geometries, matched to unit-norm
steps, then score the endpoint with the factoredness readouts. Prediction
(bridge #1): as we walk coordinatewise -> spectral (Adam/Sign -> Spectral p:1->0),
factoredness improves (locality up, feat effrank up, sensitivity/redundancy down).
"""
import os, sys, json, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cppn_numpy import CPPN, make_coordinate_grid   # noqa
from optimizers import SteepestDescent, build_ladder  # noqa
from metrics import factoredness_metrics              # noqa

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = 48
N_ITERS = 1500
N_HIDDEN = 6
WIDTH = 32
SEED = 0
# per-geometry learning rate (unit-norm steps, so these are step sizes)
LR = {"adam": 0.02, "sign": 0.02, "spectral": 0.05}


def cosine_lr(base, it, total):
    return base * 0.5 * (1 + np.cos(np.pi * it / total))


def train(label, kind, p, X, target, seed=SEED):
    cppn = CPPN(n_hidden_layers=N_HIDDEN, width=WIDTH, init_scale=2.5, seed=seed)
    base_lr = LR[kind]
    opt = SteepestDescent(cppn.params, kind=kind, p=(p or 0.0), lr=base_lr)
    losses = []
    for it in range(N_ITERS):
        opt.lr = cosine_lr(base_lr, it, N_ITERS)
        loss, grads = cppn.loss_and_grad(X, target)
        cppn.params = opt.step(cppn.params, grads)
        if it % 50 == 0 or it == N_ITERS - 1:
            losses.append((it, float(loss)))
    final_loss = losses[-1][1]
    metrics = factoredness_metrics(cppn, X, cppn.params, n_weights=96, delta=1e-2, seed=1)
    img = cppn.render(X).reshape(RES, RES, 3)
    return dict(label=label, kind=kind, p=p, final_loss=final_loss,
                losses=losses, **metrics), img


def load_target():
    im = Image.open(os.path.join(ROOT, "data", "target_64.png")).convert("RGB")
    if im.size != (RES, RES):
        im = im.resize((RES, RES), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64).reshape(-1, 3) / 255.0


RESULTS_JSON = os.path.join(ROOT, "results", "results.json")
IMGS_NPZ = os.path.join(ROOT, "results", "imgs.npz")


def run_one(index):
    """Train a single ladder entry and append its result (resumable across calls)."""
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    X = make_coordinate_grid(RES)
    target = load_target()
    label, kind, p = build_ladder()[index]
    t0 = time.time()
    r, img = train(label, kind, p, X, target)
    r["seconds"] = round(time.time() - t0, 1)
    results = []
    if os.path.exists(RESULTS_JSON):
        results = json.load(open(RESULTS_JSON))
    results = [x for x in results if x["label"] != label] + [r]
    json.dump(results, open(RESULTS_JSON, "w"), indent=2)
    imgs = dict(np.load(IMGS_NPZ)) if os.path.exists(IMGS_NPZ) else {}
    imgs[label] = img
    np.savez(IMGS_NPZ, **imgs)
    print(f"[{index}] {label:26s} loss={r['final_loss']:.4f} sens={r['sensitivity']:.3f} "
          f"loc={r['locality']:.3f} redund={r['redundancy']:.3f} "
          f"featER={r['feat_effrank_frac']:.3f}  ({r['seconds']}s)")


def make_plots():
    ladder = build_ladder()
    order = [l for l, _, _ in ladder]
    results = json.load(open(RESULTS_JSON))
    results = sorted(results, key=lambda r: order.index(r["label"]))
    imgs_npz = np.load(IMGS_NPZ)
    target = load_target()

    # ---- figure 1: fitted images ----
    imgs = [(r["label"], imgs_npz[r["label"]]) for r in results]
    n = len(imgs) + 1
    fig, ax = plt.subplots(1, n, figsize=(2.1 * n, 2.4))
    tgt = target.reshape(RES, RES, 3)
    ax[0].imshow(np.clip(tgt, 0, 1)); ax[0].set_title("target (PB skull)", fontsize=8)
    ax[0].axis("off")
    for i, (label, img) in enumerate(imgs):
        ax[i + 1].imshow(np.clip(img, 0, 1))
        ax[i + 1].set_title(label, fontsize=7); ax[i + 1].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "results", "fits.png"), dpi=130)
    plt.close(fig)

    # ---- figure 2: dose-response over the spectral knob ----
    spec = [r for r in results if r["kind"] == "spectral"]
    spec = sorted(spec, key=lambda r: -r["p"])   # p: 1 -> 0
    ps = [r["p"] for r in spec]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    for a, key, title in zip(
        ax,
        ["locality", "feat_effrank_frac", "sensitivity", "redundancy"],
        ["locality  (up = factored)", "feature eff-rank frac (up = factored)",
         "sweep sensitivity (down = factored)", "redundancy (down = factored)"],
    ):
        a.plot(ps, [r[key] for r in spec], "o-", lw=2)
        a.set_xlabel("spectral power p   (1=GD  ->  0=Muon)")
        a.set_title(title, fontsize=9)
        a.invert_xaxis(); a.grid(alpha=0.3)
    fig.suptitle("Dose-response: update geometry vs factoredness (fit fixed PB-skull image)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "results", "dose_response.png"), dpi=130)
    plt.close(fig)

    print("Wrote results/fits.png and results/dose_response.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None,
                    help="ladder index to train (0..6). Omit to train all in one process.")
    ap.add_argument("--plots", action="store_true", help="build figures from results.json")
    args = ap.parse_args()
    if args.plots:
        make_plots()
    elif args.index is not None:
        run_one(args.index)
    else:
        for i in range(len(build_ladder())):
            run_one(i)
        make_plots()
