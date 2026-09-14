"""
Does ANY design axis move NEURON-STRUCTURE factoredness?

The neuron-structure metric (smoothness / high-frequency energy of each live
hidden neuron) is the one that survives the UFR capacity gate. On the optimizer
ladder it looked FLAT. Here we push on axes beyond Adam-vs-spectral to find out
whether neuron structure is movable at all by inner-loop / architecture choices,
or whether it is essentially fixed by the target + backprop:

  axis 1: optimizer geometry     (adam / spectral p=1 / muon p=0)
  axis 2: width                  (16 / 32 / 64 / 128)
  axis 3: depth                  (3 / 6 / 12 layers)
  axis 4: init scale             (1.0 / 2.5 / 4.0)  -- more init variance
  axis 5: weight decay           (0 / 0.02 / 0.1)   -- shrink weights
  axis 6: heavier orthogonalization via repeated Newton-Schulz is already p=0;
          we add an "over-orthogonalized" proxy by training longer at p=0.

Reports neuron_tv + neuron_hf_energy (LOWER = smoother = more factored) as each
axis is varied with everything else held at a baseline. Fit target = skull.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cppn_numpy import CPPN, make_coordinate_grid       # noqa
from optimizers import SteepestDescent                   # noqa
from metrics import neuron_structure_metrics             # noqa
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = 48
SEEDS = [0, 1, 2]
LR = {"adam": 0.02, "sign": 0.02, "spectral": 0.05}


def load_target():
    im = Image.open(os.path.join(ROOT, "data", "target_64.png")).convert("RGB")
    im = im.resize((RES, RES), Image.LANCZOS)
    return np.asarray(im, float).reshape(-1, 3) / 255.0


def fit_and_score(kind="spectral", p=1.0, width=32, depth=6, init_scale=2.5,
                  wd=0.0, iters=1200, seed=0, X=None, target=None):
    c = CPPN(n_hidden_layers=depth, width=width, init_scale=init_scale, seed=seed)
    base = LR[kind]
    opt = SteepestDescent(c.params, kind=kind, p=p, lr=base, wd=wd)
    loss = None
    for it in range(iters):
        opt.lr = base * 0.5 * (1 + np.cos(np.pi * it / iters))
        loss, g = c.loss_and_grad(X, target)
        c.params = opt.step(c.params, g)
    m = neuron_structure_metrics(c, X, RES)
    m["final_loss"] = float(loss)
    return m


def sweep(name, variants, X, target):
    print(f"\n=== axis: {name} ===   (neuron_tv / hf LOWER = more factored)")
    for label, kw in variants:
        tvs, hfs, ls = [], [], []
        for s in SEEDS:
            m = fit_and_score(seed=s, X=X, target=target, **kw)
            tvs.append(m["neuron_tv"]); hfs.append(m["neuron_hf_energy"])
            ls.append(m["final_loss"])
        print(f"  {label:22s} TV={np.mean(tvs):.4f}±{np.std(tvs):.4f}  "
              f"HF={np.mean(hfs):.4f}±{np.std(hfs):.4f}  loss={np.mean(ls):.4f}")


def main():
    X = make_coordinate_grid(RES)
    target = load_target()
    t0 = time.time()

    sweep("optimizer geometry", [
        ("adam", dict(kind="adam", p=None)),
        ("spectral p=1 (GD)", dict(kind="spectral", p=1.0)),
        ("spectral p=0 (Muon)", dict(kind="spectral", p=0.0)),
    ], X, target)

    sweep("width", [
        (f"width={w}", dict(kind="spectral", p=1.0, width=w)) for w in [16, 32, 64, 128]
    ], X, target)

    sweep("depth", [
        (f"depth={d}", dict(kind="spectral", p=1.0, depth=d)) for d in [3, 6, 12]
    ], X, target)

    sweep("init scale", [
        (f"init={i}", dict(kind="spectral", p=1.0, init_scale=i)) for i in [1.0, 2.5, 4.0]
    ], X, target)

    sweep("weight decay", [
        (f"wd={w}", dict(kind="spectral", p=1.0, wd=w)) for w in [0.0, 0.02, 0.1]
    ], X, target)

    sweep("orthogonalization x train-length", [
        ("p=0 iters=1200", dict(kind="spectral", p=0.0, iters=1200)),
        ("p=0 iters=3000", dict(kind="spectral", p=0.0, iters=3000)),
    ], X, target)

    print(f"\n(total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
