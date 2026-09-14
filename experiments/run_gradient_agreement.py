"""
Sanity check for the Nexus analogy: does CROSS-REGION GRADIENT AGREEMENT separate
the evolved Picbreeder genome (UFR) from SGD-trained / our-optimizer nets (FER)?

Nexus (arXiv:2604.09258) selects among equal-loss minima for one where per-SOURCE
gradients agree (high cosine). The hypothesis we test here: gradient agreement is
a proxy for factoredness -- a unified/factored representation reuses the same
internal machinery for every part of the output, so the gradients that different
parts of the target induce should point the same way in weight space.

We have no natural "sources" in single-image fitting, so we MANUFACTURE them by
splitting the pixel grid into spatial regions. For each region r we compute
g_r = d(MSE over region r)/d(weights). Then we measure the mean pairwise cosine
similarity between region-gradients. Prediction (if the analogy holds):
    evolved (UFR)  >  SGD / our optimizers (FER)   in cross-region gradient cosine.

If instead everything clusters, or the evolved genome is NOT higher, the
"gradient agreement = factoredness" mapping does not hold in this setup and a
Nexus-style objective is unlikely to move our neuron metric.

Reports cosine over (a) all weights and (b) live weights only (evolved is sparse).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fer_cppn import load_genome, ARCH, FERCppn, parse_arch   # noqa
from optimizers import SteepestDescent                         # noqa

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = 48
N_REGIONS = 4          # split into a 4x4 = 16 region grid by default below
GRID = 4


def region_masks(res, grid):
    """Return list of boolean masks over flattened pixels, one per grid cell."""
    masks = []
    idx = np.arange(res * res).reshape(res, res)
    bs = res // grid
    for gy in range(grid):
        for gx in range(grid):
            block = idx[gy*bs:(gy+1)*bs if gy < grid-1 else res,
                        gx*bs:(gx+1)*bs if gx < grid-1 else res].ravel()
            m = np.zeros(res*res, dtype=bool); m[block] = True
            masks.append(m)
    return masks


def region_gradients(cppn, X, target, masks):
    """d(region MSE)/d(weights) for each region, flattened to one vector each."""
    grads = []
    for m in masks:
        # masked target: gradient only from region pixels. We compute full grad
        # of a per-region loss by zeroing the residual outside the region.
        g = _masked_grad(cppn, X, target, m)
        grads.append(np.concatenate([gi.ravel() for gi in g]))
    return np.stack(grads)   # (n_regions, n_params)


def _masked_grad(cppn, X, target, mask):
    """d/d(weights) of the MEAN OUTPUT over a region -- an output-Jacobian, NOT a
    loss gradient. Why: Nexus's per-source LOSS gradients all vanish at a zero-loss
    solution, and the evolved genome is a perfect fit (loss=0 => grad=0 for every
    region), so a loss-gradient design is degenerate there. The output-Jacobian
    'which weights does this region recruit, and in what direction' is well-defined
    everywhere and captures the same reuse notion: if two regions have aligned
    output-Jacobians, they lean on the same weights the same way (shared machinery).
    We sum over the 3 channels of the mean region output as a scalar summary."""
    P = cppn.params
    hsv, (Zs, As, _) = cppn.forward(X, params=P, cache=True)
    Jc = cppn._color_jacobian(hsv)                   # d rgb / d hsv, per pixel
    # scalar per net = mean over region pixels of (r+g+b); backprop that.
    npix = max(int(mask.sum()), 1)
    # d(scalar)/d(rgb) = 1/npix for region pixels, summed over channels
    drgb = (mask[:, None].astype(float) / npix) * np.ones((1, 3))
    dhsv = np.einsum("nij,ni->nj", Jc, drgb)         # chain through color
    grads = [None] * len(P)
    grads[-1] = As[-1].T @ dhsv
    dA = dhsv @ P[-1].T
    for l in reversed(range(cppn.n_layers)):
        dZ = dA * cppn._dact(Zs[l])
        grads[l] = As[l].T @ dZ
        dA = dZ @ P[l].T
    return grads


def mean_pairwise_cosine(G, live_mask=None):
    if live_mask is not None:
        G = G[:, live_mask]
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    keep = norms.ravel() > 1e-12
    G = G[keep] / (norms[keep] + 1e-12)
    if G.shape[0] < 2:
        return float("nan")
    C = G @ G.T
    n = C.shape[0]
    return float((C.sum() - np.trace(C)) / (n * (n - 1)))


def fresh_trained(target_name, kind, p, X, target, seed=0, iters=2500, lr=0.02):
    c = FERCppn(ARCH[target_name], np.zeros(_nparams(target_name)))
    rng = np.random.default_rng(seed)
    c.params = [rng.standard_normal(pp.shape) * (1.0/np.sqrt(pp.shape[0]))
                for pp in c.params]
    opt = SteepestDescent(c.params, kind=kind, p=p, lr=lr)
    for it in range(iters):
        opt.lr = lr * 0.5 * (1 + np.cos(np.pi*it/iters))
        _, g = c.loss_and_grad(X, target); c.params = opt.step(c.params, g)
    return c


def _nparams(name):
    nl, blocks, width = parse_arch(ARCH[name])
    fan, tot = 4, 0
    for _ in range(nl):
        tot += fan*width; fan = width
    return tot + fan*3


def live_mask_of(cppn):
    return np.concatenate([(np.abs(p) > 1e-8).ravel() for p in cppn.params])


def main():
    target_name = "skull"
    evolved = load_genome(target_name)
    X = evolved.coordinate_grid(RES)
    target = evolved.render(X)
    masks = region_masks(RES, GRID)
    print(f"### cross-region gradient agreement (skull, {len(masks)} regions) ###")
    print("higher cosine = region-gradients agree = shared machinery (Nexus criterion)\n")

    results = {}

    def report(label, cppn):
        G = region_gradients(cppn, X, target, masks)
        lm = live_mask_of(cppn)
        all_c = mean_pairwise_cosine(G)
        live_c = mean_pairwise_cosine(G, lm)
        results[label] = dict(cos_all=all_c, cos_live=live_c,
                              nnz_frac=float(lm.mean()))
        print(f"  {label:26s} cos(all)={all_c:+.4f}  cos(live)={live_c:+.4f}  "
              f"nnz={lm.mean():.3f}")

    report("EVOLVED (UFR)", evolved)
    report("SGD (FER)", load_genome(f"sgd_{target_name}"))
    for kind, p, lab in [("adam", None, "our Adam"),
                         ("spectral", 1.0, "our Spectral p=1"),
                         ("spectral", 0.0, "our Muon p=0")]:
        c = fresh_trained(target_name, kind, p, X, target)
        report(lab, c)

    out = os.path.join(ROOT, "results", "gradient_agreement.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nWrote {os.path.relpath(out, ROOT)}")
    ev = results["EVOLVED (UFR)"]["cos_live"]
    others = [v["cos_live"] for k, v in results.items() if k != "EVOLVED (UFR)"]
    print(f"\nVERDICT: evolved cos(live)={ev:+.4f} vs trained max={max(others):+.4f} -> "
          + ("SEPARATES (supports Nexus analogy)" if ev > max(others)
             else "does NOT separate (analogy weak in this setup)"))


if __name__ == "__main__":
    main()
