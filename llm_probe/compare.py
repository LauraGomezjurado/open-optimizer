"""
Aggregate the arms and print the headline comparison.

The question: at matched capability (ar_ce), does the OBJECTIVE move the
factoredness proxy? Higher effrank_frac = more independent feature directions =
more factored / less superposed.

Reads results/<arm>.json for each arm+seed, prints mean±std and flags whether
staged / mdlm separate from flat_ar on the proxy AT COMPARABLE ar_ce.
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
ARMS = ["flat_ar", "staged", "mdlm"]
KEYS = ["ar_ce", "mlp_effrank", "resid_effrank", "mlp_pr"]


def load():
    rows = {}
    for f in glob.glob(os.path.join(RES, "*.json")):
        d = json.load(open(f))
        arm = d["arm"]; seed = d["cfg"].get("seed", 0)
        rows.setdefault(arm, []).append(d["final"])
    return rows


def main():
    rows = load()
    if not rows:
        print("no results yet in", RES); return
    print("\n=== objective-lever @ transformer scale (final checkpoint) ===")
    print("higher *_effrank = more factored; compare AT similar ar_ce\n")
    print("arm        n   " + "".join(k.rjust(13) for k in KEYS))
    agg = {}
    for arm in ARMS:
        rs = rows.get(arm, [])
        if not rs:
            continue
        agg[arm] = {k: (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs]))
                    for k in KEYS}
        n = len(rs)
        print(f"{arm:10s} {n:<3d} " +
              "".join(f"{agg[arm][k][0]:13.3f}" for k in KEYS))
    if "flat_ar" in agg:
        base = agg["flat_ar"]
        print("\ndeltas vs flat_ar (mlp_effrank; + = more factored):")
        for arm in ARMS:
            if arm == "flat_ar" or arm not in agg:
                continue
            d = agg[arm]["mlp_effrank"][0] - base["mlp_effrank"][0]
            dce = agg[arm]["ar_ce"][0] - base["ar_ce"][0]
            print(f"  {arm:10s} Δmlp_effrank={d:+.3f}  Δar_ce={dce:+.3f} "
                  f"({'cleaner' if d>0 else 'not cleaner'}; "
                  f"{'similar loss' if abs(dce)<0.2 else 'LOSS GAP - caution'})")
    print("\nreminder: this is the proxy (cheap gate). Run the SAE only on arms "
          "that show a real Δ here at comparable ar_ce.")


if __name__ == "__main__":
    main()
