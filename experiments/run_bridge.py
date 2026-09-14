"""
The bridge experiment: put the evolved genome (UFR), the FER paper's SGD genome
(FER), and OUR optimizer ladder ALL in the same architecture, same target, same
metric -- so for the first time they are directly comparable on one ruler.

Setup:
  - architecture: the FER paper's own CPPN for the chosen target (e.g. skull:
    12 layers, their activation mix, HSV readout) -- via src/fer_cppn.py.
  - target image: the exact RGB the EVOLVED genome renders (the literal UFR
    output), so our optimizers are fitting the true UFR image pixel-for-pixel.
  - contestants:
      EVOLVED   : the evolved genome as-is (UFR reference)
      SGD(FER)  : the FER paper's SGD-trained genome as-is (FER reference)
      our ladder: Adam / Spectral(p) / Muon / ... trained in THIS architecture
                  to fit the target, from paired inits.
  - metric: the same factoredness_metrics used everywhere else.

This answers the question Experiment B could not: where do OUR optimizers sit
relative to true UFR and true FER, on identical footing?

Run:  python experiments/run_bridge.py --target skull
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fer_cppn import FERCppn, ARCH, load_genome     # noqa
from optimizers import SteepestDescent, build_ladder  # noqa
from metrics import factoredness_metrics, neuron_structure_metrics  # noqa

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = 48
N_ITERS = 2500
INIT_SCALE = 1.0            # fan-in-scaled; scale=1 converges in their arch
SEEDS = [0, 1, 2]
N_WEIGHTS = 200
SWEEP_SEED = 99
LR = {"adam": 0.02, "sign": 0.02, "spectral": 0.02, "scalar_adam": 0.02}


def cosine_lr(base, it, total):
    return base * 0.5 * (1 + np.cos(np.pi * it / total))


def fresh_net(target_name, seed):
    c = FERCppn(ARCH[target_name], np.zeros(_nparams(target_name)))
    rng = np.random.default_rng(seed)
    c.params = [rng.standard_normal(p.shape) * (INIT_SCALE / np.sqrt(p.shape[0]))
                for p in c.params]
    return c


def _nparams(name):
    from fer_cppn import parse_arch
    nl, blocks, width = parse_arch(ARCH[name])
    fan, tot = 4, 0
    for _ in range(nl):
        tot += fan * width; fan = width
    return tot + fan * 3


def score(cppn, X):
    m = factoredness_metrics(cppn, X, cppn.params, n_weights=N_WEIGHTS,
                             delta=1e-2, seed=SWEEP_SEED, n_boot=100)
    m.update(neuron_structure_metrics(cppn, X, RES))   # capacity-invariant
    nnz = int(sum(np.count_nonzero(p) for p in cppn.params))
    tot = int(sum(p.size for p in cppn.params))
    m["nnz_frac"] = nnz / tot
    return m


def train_ladder(target_name, X, target):
    rows = []
    for label, kind, p, *rest in build_ladder():
        opts = rest[0] if rest else {}
        for seed in SEEDS:
            c = fresh_net(target_name, seed)
            base = LR.get(kind, 0.02)
            opt = SteepestDescent(c.params, kind=kind, p=p, lr=base,
                                  wd=opts.get("wd", 0.0))
            loss = None
            for it in range(N_ITERS):
                opt.lr = cosine_lr(base, it, N_ITERS)
                loss, g = c.loss_and_grad(X, target)
                c.params = opt.step(c.params, g)
            m = score(c, X)
            rows.append(dict(group="ladder", label=label, kind=kind, p=p,
                             seed=seed, final_loss=float(loss), **m))
            print(f"  {label:26s} seed{seed} loss={loss:.4f} "
                  f"sens={m['sensitivity']:.2f} orth={m['influence_orthogonality']:.3f} "
                  f"featERc={m['feat_effrank_corr_frac']:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["skull", "butterfly", "apple"],
                    default="skull")
    args = ap.parse_args()
    tname = args.target

    evolved = load_genome(tname)
    X = evolved.coordinate_grid(RES)
    target = evolved.render(X)          # the literal UFR image
    sgd = load_genome(f"sgd_{tname}")

    print(f"\n### BRIDGE on FER architecture, target = evolved {tname} image ###")
    print("reference networks (as-is):")
    ref = []
    for lab, net in [("EVOLVED(UFR)", evolved), ("SGD(FER)", sgd)]:
        m = score(net, X)
        ref.append(dict(group="reference", label=lab, seed=None,
                        final_loss=None, **m))
        print(f"  {lab:14s} sens={m['sensitivity']:.2f} "
              f"orth={m['influence_orthogonality']:.3f} "
              f"featERc={m['feat_effrank_corr_frac']:.3f} "
              f"redund={m['redundancy']:.3f} nnz={m['nnz_frac']:.3f}")

    print("our optimizer ladder (trained in this arch):")
    t0 = time.time()
    ladder = train_ladder(tname, X, target)
    print(f"  (ladder trained in {time.time()-t0:.0f}s)")

    out = os.path.join(ROOT, "results", f"bridge_{tname}.json")
    json.dump(ref + ladder, open(out, "w"), indent=2)
    print(f"\nWrote {os.path.relpath(out, ROOT)}")
    _summary(ref, ladder)


def _summary(ref, ladder):
    keys = ["final_loss", "sensitivity", "redundancy", "influence_orthogonality",
            "neuron_tv", "neuron_hf_energy", "n_live_neurons", "nnz_frac"]
    print("\n=== SUMMARY (all on the same arch/target/metric) ===")
    print("group      label                    " +
          "".join(k[:9].rjust(11) for k in keys))
    # references
    for r in ref:
        print("reference  " + r["label"].ljust(24) +
              "".join((f"{r[k]:11.3f}" if r.get(k) is not None else "    n/a".rjust(11))
                      for k in keys))
    # ladder means over seeds
    labels = []
    for r in ladder:
        if r["label"] not in labels:
            labels.append(r["label"])
    for lab in labels:
        rs = [r for r in ladder if r["label"] == lab]
        print("ladder     " + lab.ljust(24) +
              "".join(f"{np.mean([r[k] for r in rs]):11.3f}" for k in keys))


if __name__ == "__main__":
    main()
