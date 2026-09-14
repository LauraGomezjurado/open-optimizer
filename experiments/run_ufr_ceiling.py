"""
UFR ceiling: score the ACTUAL evolved Picbreeder genome (the UFR reference from
the FER paper) on the SAME factoredness metrics we run on our optimizer ladder,
alongside the FER paper's own SGD-trained genome (same architecture, same target).

This is the missing absolute anchor. Until now every comparison was optimizer-vs-
optimizer; here we finally measure where TRUE UFR (evolution) and TRUE FER (their
SGD net) sit on our axes. Two questions it answers:

  1. VALIDATION: do our metrics rank evolved (UFR) as more factored than SGD (FER)?
     If not, the metrics don't measure what we claim. This is the external check
     the metrics never had (the FER paper reports no numbers).
  2. CEILING: how far is the evolved genome from our best optimizer? That tells us
     whether inner-loop geometry is a rounding error vs a real ingredient.

Note: evolved and SGD genomes share architecture, target image, and renderer, so
the metric comparison between them is clean. Comparison to the optimizer ladder
is cross-architecture (our probe net is 6x32 sigmoid-RGB; theirs is deep/narrow
HSV), so ladder numbers are indicative, not identical-footing -- we report the
evolved-vs-SGD gap as the primary, footing-matched result.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fer_cppn import load_genome, ARCH          # noqa
from metrics import factoredness_metrics         # noqa

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = 48
N_WEIGHTS = 200      # these nets are sparse + deep; sample more weights
SWEEP_SEED = 99


def score(name):
    cppn = load_genome(name)
    X = cppn.coordinate_grid(RES)
    m = factoredness_metrics(cppn, X, cppn.params, n_weights=N_WEIGHTS,
                             delta=1e-2, seed=SWEEP_SEED, n_boot=200)
    # count live (nonzero) weights -- evolved genomes are very sparse
    nnz = int(sum(np.count_nonzero(p) for p in cppn.params))
    tot = int(sum(p.size for p in cppn.params))
    m["nnz_frac"] = nnz / tot
    return m


def main():
    rows = []
    for genome in ["skull", "butterfly", "apple"]:
        evolved = score(genome)          # UFR
        sgd = score(f"sgd_{genome}")     # FER
        rows.append((genome, evolved, sgd))

    keys = ["sensitivity", "sensitivity_elastic", "redundancy",
            "feat_effrank_frac", "feat_effrank_corr_frac",
            "influence_orthogonality", "locality", "sat_frac", "nnz_frac"]
    print(f"\n=== UFR ceiling: evolved (UFR) vs SGD (FER), FER paper's own genomes ===")
    print(f"(same arch/target/renderer within each row; n_weights={N_WEIGHTS})")
    print("arrows: sens/redundancy/sat DOWN = factored; effrank/orth/locality UP = factored\n")
    hdr = "target    net      " + "".join(k[:10].rjust(11) for k in keys)
    print(hdr)
    for genome, ev, sg in rows:
        for lab, m in [("EVOLVED", ev), ("SGD", sg)]:
            line = f"{genome:9s} {lab:8s}" + "".join(f"{m[k]:11.3f}" for k in keys)
            print(line)
        # direction check on the footing-matched pair
        checks = {
            "sensitivity": ev["sensitivity"] < sg["sensitivity"],
            "redundancy": ev["redundancy"] < sg["redundancy"],
            "feat_effrank_corr_frac": ev["feat_effrank_corr_frac"] > sg["feat_effrank_corr_frac"],
            "influence_orthogonality": ev["influence_orthogonality"] > sg["influence_orthogonality"],
        }
        ok = sum(checks.values())
        print(f"          -> metrics ranking EVOLVED as more factored: {ok}/4  "
              + ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in checks.items()))
        print()

    out = os.path.join(ROOT, "results", "ufr_ceiling.json")
    json.dump([{"target": g, "evolved": ev, "sgd": sg} for g, ev, sg in rows],
              open(out, "w"), indent=2)
    print(f"Wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
