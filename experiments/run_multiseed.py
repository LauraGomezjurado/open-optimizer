"""
Multi-seed, confound-audited dose-response.

Design choices for statistical robustness:
  - PAIRED seeds: every geometry is trained from the SAME set of init seeds, so
    the coordinatewise-vs-spectral comparison is within-init (controls the basin).
  - fixed iters, and we report final loss per run so loss-matching can be checked
    rather than assumed.
  - each run also stores confound diagnostics (weight scale, weight eff-rank,
    saturation) and a scale-invariant elastic sensitivity.
  - --stats aggregates mean +/- std and runs a per-seed Spearman correlation
    between the spectral power p and each metric, so "monotone dose-response" is
    a measured claim (sign + significance across seeds), not eyeballing.

Runs are appended to results/multiseed.json and are resumable across processes:
  python experiments/run_multiseed.py --index I --seed S
  python experiments/run_multiseed.py --stats
"""
import os, sys, json, time, argparse, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cppn_numpy import CPPN, make_coordinate_grid    # noqa
from optimizers import SteepestDescent, build_ladder  # noqa
from metrics import factoredness_metrics              # noqa

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = 40
N_ITERS = 1200
N_HIDDEN = 6
WIDTH = 32
SEEDS = [0, 1, 2, 3, 4]
LR = {"adam": 0.02, "sign": 0.02, "spectral": 0.05, "scalar_adam": 0.02}
# init_scale of the CPPN weights. 2.5 is the original; on the butterfly target
# large init drops normalized-spectral optimizers into a flat-image basin from
# some seeds (confound C7), so butterfly is re-run at 1.5 where ALL geometries
# converge on every seed (see --init_scale).
INIT_SCALE = 2.5

# --- target selection (generality test): each target writes its own JSON so
#     the per-target sweeps do not collide. "skull" keeps the original file name
#     for backward compatibility with the existing 65-run multiseed.json. ---
TARGETS = {
    "skull": "target_64.png",
    "apple": "target_apple_64.png",
    "butterfly": "target_butterfly_64.png",
}
TARGET = "skull"


def _json_path():
    # init_scale 2.5 keeps the original file names (backward compat); any other
    # init_scale gets its own suffix so runs never overwrite across settings.
    isuf = "" if INIT_SCALE == 2.5 else f"_is{INIT_SCALE:g}"
    if TARGET == "skull" and not isuf:
        return os.path.join(ROOT, "results", "multiseed.json")
    return os.path.join(ROOT, "results", f"multiseed_{TARGET}{isuf}.json")


def cosine_lr(base, it, total):
    return base * 0.5 * (1 + np.cos(np.pi * it / total))


def load_target():
    im = Image.open(os.path.join(ROOT, "data", TARGETS[TARGET])).convert("RGB")
    if im.size != (RES, RES):
        im = im.resize((RES, RES), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64).reshape(-1, 3) / 255.0


def train_one(label, kind, p, seed, X, target, opts=None):
    opts = opts or {}
    cppn = CPPN(n_hidden_layers=N_HIDDEN, width=WIDTH, init_scale=INIT_SCALE, seed=seed)
    base_lr = LR.get(kind, 0.02)
    opt = SteepestDescent(cppn.params, kind=kind, p=p, lr=base_lr,
                          wd=opts.get("wd", 0.0))
    loss = None
    for it in range(N_ITERS):
        opt.lr = cosine_lr(base_lr, it, N_ITERS)
        loss, grads = cppn.loss_and_grad(X, target)
        cppn.params = opt.step(cppn.params, grads)
    m = factoredness_metrics(cppn, X, cppn.params, n_weights=80, delta=1e-2, seed=99)
    return dict(label=label, kind=kind, p=p, seed=seed,
                final_loss=float(loss), **m)


def _ladder_entry(index):
    e = build_ladder()[index]
    label, kind, p = e[0], e[1], e[2]
    opts = e[3] if len(e) > 3 else {}
    return label, kind, p, opts


def run_one(index, seed):
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    X = make_coordinate_grid(RES)
    target = load_target()
    label, kind, p, opts = _ladder_entry(index)
    t0 = time.time()
    r = train_one(label, kind, p, seed, X, target, opts=opts)
    r["seconds"] = round(time.time() - t0, 1)
    jp = _json_path()
    data = json.load(open(jp)) if os.path.exists(jp) else []
    data = [x for x in data if not (x["label"] == label and x["seed"] == seed)] + [r]
    json.dump(data, open(jp, "w"), indent=2)
    print(f"idx{index} seed{seed} {label:24s} loss={r['final_loss']:.4f} "
          f"sens={r['sensitivity']:.2f} sensE={r['sensitivity_elastic']:.3f} "
          f"featER={r['feat_effrank_frac']:.3f} wER={r['weight_effrank_frac']:.3f} "
          f"sat={r['sat_frac']:.3f} ({r['seconds']}s)")


def _spearman(x, y):
    # rank correlation without scipy
    def rank(a):
        order = np.argsort(a)
        r = np.empty(len(a)); r[order] = np.arange(len(a))
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def _partial_sat_regression(data):
    """S2 (confound C2): how much factoredness signal survives partialling out
    saturation? For each metric we compare the raw coordinatewise-vs-spectral gap
    to the gap AFTER regressing sat_frac out of both the metric and the group
    indicator. If the partial gap collapses toward 0, the effect was saturation;
    if it survives, factoredness is a structural property beyond activation clipping.

    Method (no statsmodels): partial correlation & partialled group means.
      group g = 1 for coordinatewise-adaptive (adam/sign, no wd), 0 for spectral.
      For metric y: residualize y and g on sat_frac (linear), then the residual
      group-mean difference is the saturation-controlled effect.
    """
    def resid_on(x, z):
        x = np.asarray(x, float); z = np.asarray(z, float)
        A = np.vstack([z, np.ones_like(z)]).T
        beta, *_ = np.linalg.lstsq(A, x, rcond=None)
        return x - A @ beta

    # classify runs
    coord_labels = {"Adam (adaptive coord.)", "SignSGD (L-inf)"}
    spec_labels = {"Spectral p=1.0 (L2/GD)", "Spectral p=0.75", "Spectral p=0.5",
                   "Spectral p=0.25", "Spectral p=0.0 (Muon)"}
    rows = [d for d in data if d["label"] in coord_labels | spec_labels
            and "sat_frac" in d]
    if len(rows) < 6:
        print("\n[S2] not enough runs with sat_frac for partial regression."); return
    g = np.array([1.0 if d["label"] in coord_labels else 0.0 for d in rows])
    sat = np.array([d["sat_frac"] for d in rows])

    print("\n=== [S2] saturation-controlled effect (coordwise=1 vs spectral=0) ===")
    print("raw gap = mean(coord) - mean(spec); partial = same after regressing out sat_frac.")
    print(f"  {'metric':24s} {'raw gap':>9s} {'partial gap':>12s} {'corr(y,sat)':>12s} {'partialR':>9s}")
    for m in ["sensitivity", "sensitivity_elastic", "redundancy",
              "feat_effrank_frac", "feat_effrank_corr_frac",
              "influence_orthogonality", "locality"]:
        vals = [(d.get(m), d) for d in rows]
        if any(v is None for v, _ in vals):
            continue
        y = np.array([v for v, _ in vals])
        raw_gap = y[g == 1].mean() - y[g == 0].mean()
        yr = resid_on(y, sat); gr = resid_on(g, sat)
        # partial group effect: regress residual y on residual g
        denom = (gr * gr).sum()
        partial_gap = float((gr * yr).sum() / denom) if denom > 0 else float("nan")
        c_ys = _spearman(y, sat)
        # partial corr of y with g controlling sat
        pr = float((gr * yr).sum() / (np.sqrt((gr**2).sum() * (yr**2).sum()) + 1e-12))
        print(f"  {m:24s} {raw_gap:9.3f} {partial_gap:12.3f} {c_ys:12.3f} {pr:9.3f}")
    print("  (partial gap ~ raw gap => effect survives saturation control;")
    print("   partial gap -> 0 => the metric was mostly re-measuring saturation.)")


LOSS_GATE = 5e-3   # runs above this did not reach the shared behavior -> excluded


def stats():
    data_all = json.load(open(_json_path()))
    print(f"\n########## TARGET = {TARGET} ##########")
    # loss-matching gate: metric comparisons are only valid among runs that
    # actually reached the fixed target (some geometries fail to converge on
    # harder targets from some seeds -> not behaviorally equivalent, so excluded).
    data = [d for d in data_all if d.get("final_loss", 1.0) < LOSS_GATE]
    n_drop = len(data_all) - len(data)
    if n_drop:
        dropped = {}
        for d in data_all:
            if d.get("final_loss", 1.0) >= LOSS_GATE:
                dropped[d["label"]] = dropped.get(d["label"], 0) + 1
        print(f"[loss-gate {LOSS_GATE:g}] excluded {n_drop} non-converged run(s): "
              + ", ".join(f"{k}×{v}" for k, v in dropped.items()))
    ladder = build_ladder()
    order = [e[0] for e in ladder]
    metrics = ["final_loss", "sensitivity", "sensitivity_elastic", "locality",
               "redundancy", "feat_effrank_frac", "feat_effrank_corr_frac",
               "influence_orthogonality", "weight_effrank_frac",
               "weight_frob", "sat_frac"]

    def _vals(rows, m):
        return [r[m] for r in rows if m in r and r[m] is not None]

    print("\n=== mean +/- std over seeds ===")
    print("(feat_effrank_corr_frac & influence_orthogonality are the "
          "saturation-invariant readouts; up=factored)")
    header = "geometry".ljust(26) + "  " + "  ".join(m[:9].rjust(9) for m in metrics)
    print(header)
    agg = {}
    for label in order:
        rows = [d for d in data if d["label"] == label]
        if not rows:
            continue
        agg[label] = {}
        for m in metrics:
            v = _vals(rows, m)
            agg[label][m] = (np.mean(v), np.std(v)) if v else (float("nan"), 0.0)
        line = label.ljust(26) + "  " + "  ".join(
            f"{agg[label][m][0]:9.3f}" for m in metrics)
        print(line)

    mono_metrics = ["feat_effrank_frac", "feat_effrank_corr_frac",
                    "influence_orthogonality", "sensitivity_elastic",
                    "sensitivity", "redundancy", "locality", "weight_effrank_frac"]

    def _monotonicity(sub_labels, knob_of, header, note):
        """Per-seed Spearman(knob, metric) across a family of rungs."""
        print(f"\n=== {header} ===")
        print(note)
        for m in mono_metrics:
            corrs = []
            for s in SEEDS:
                pts = [(knob_of[l],
                        next((d[m] for d in data
                              if d["label"] == l and d["seed"] == s and m in d), None))
                       for l in sub_labels]
                pts = [(a, b) for a, b in pts if b is not None]
                if len(pts) >= 3:
                    corrs.append(_spearman([a for a, _ in pts], [b for _, b in pts]))
            if corrs:
                arr = np.array(corrs)
                print(f"  {m:24s} mean rho={arr.mean():+.3f} +/- {arr.std():.3f}   "
                      f"frac seeds rho>0 = {float(np.mean(arr > 0)):.2f}   (n={len(corrs)})")

    # spectral family: knob = (1-p), + = toward Muon (orthogonalized)
    spec = [(e[0], e[2]) for e in ladder if e[1] == "spectral"]
    p_of = {l: p for l, p in spec}
    _monotonicity([l for l, _ in spec], {l: 1 - p for l, p in spec},
                  "monotonicity across SPECTRAL knob p (per-seed Spearman)",
                  "(corr with (1-p); + = improves toward Muon / pure orthogonalization)")

    # coordinatewise family (confound C1): knob = q = per-entry-rescaling strength.
    # + = MORE per-coordinate preconditioning. Prediction: fracture rises with q.
    coord = [(e[0], (1.0 if e[2] is None else e[2]))
             for e in ladder if e[1] == "adam" and (len(e) < 4 or not e[3])]
    if len({q for _, q in coord}) >= 3:
        _monotonicity([l for l, _ in coord], {l: q for l, q in coord},
                      "monotonicity across COORDINATEWISE knob q (per-seed Spearman)",
                      "(q = per-entry rescaling strength; + = metric rises with MORE "
                      "per-coordinate preconditioning -> localizes the fracture lever)")

    # confound check: correlation of feat_effrank vs weight_effrank across geometries (seed-avg)
    labs = [l for l in order if l in agg and not np.isnan(agg[l]["feat_effrank_frac"][0])]
    fe = [agg[l]["feat_effrank_frac"][0] for l in labs]
    we = [agg[l]["weight_effrank_frac"][0] for l in labs]
    print(f"\n[confound #2] corr(feature eff-rank, weight eff-rank) across geometries "
          f"= {_spearman(fe, we):+.3f}  (near +1 => feature-rank may just echo weight-rank)")

    _partial_sat_regression(data)
    _plot(agg, order, p_of)


def _plot(agg, order, p_of):
    spec = [l for l in order if l in p_of and l in agg]
    spec = sorted(spec, key=lambda l: -p_of[l])
    ps = [p_of[l] for l in spec]
    panels = [("feat_effrank_frac", "feature eff-rank frac  (up=factored)"),
              ("sensitivity_elastic", "elastic sensitivity [scale-inv] (down=factored)"),
              ("sensitivity", "raw sensitivity (down=factored)"),
              ("weight_effrank_frac", "weight eff-rank frac [confound]")]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for a, (key, title) in zip(ax, panels):
        mu = [agg[l][key][0] for l in spec]
        sd = [agg[l][key][1] for l in spec]
        a.errorbar(ps, mu, yerr=sd, fmt="o-", lw=2, capsize=3)
        a.set_xlabel("spectral power p  (1=GD -> 0=Muon)")
        a.set_title(title, fontsize=9); a.invert_xaxis(); a.grid(alpha=0.3)
    fig.suptitle(f"Multi-seed dose-response (n={len(SEEDS)} seeds, mean +/- std)", fontsize=12)
    fig.tight_layout()
    suffix = "" if TARGET == "skull" else f"_{TARGET}"
    outp = os.path.join(ROOT, "results", f"dose_response_multiseed{suffix}.png")
    fig.savefig(outp, dpi=130)
    plt.close(fig)
    print(f"\nWrote {os.path.relpath(outp, ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="run all index x seed here")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--target", choices=list(TARGETS), default="skull",
                    help="which fixed target image to fit (generality test)")
    ap.add_argument("--init_scale", type=float, default=2.5,
                    help="CPPN weight init scale (1.5 makes all geometries "
                         "converge on butterfly; see confound C7)")
    args = ap.parse_args()
    TARGET = args.target
    INIT_SCALE = args.init_scale
    if args.stats:
        stats()
    elif args.all:
        for i in range(len(build_ladder())):
            for s in SEEDS:
                run_one(i, s)
        stats()
    elif args.index is not None and args.seed is not None:
        run_one(args.index, args.seed)
    else:
        raise SystemExit("need --index & --seed, or --all, or --stats")
