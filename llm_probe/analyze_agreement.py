"""
Metric-agreement analysis (the core of the metric-first plan).

Reads the exp2 outputs and answers: do the cheap proxy (effrank_frac), the SAE
monosemanticity proxy, and the decoder-interference tiebreaker AGREE on how the
objective arms rank? Agreement => metric trustworthy; disagreement => the metric
was the bottleneck and the disagreement is itself the finding.

Ranks arms (flat_ar, staged, staged_strong) on each metric and reports whether
they induce the same ordering. Higher = more factored for effrank & mono; LOWER
= more factored for decoder_interference (so we invert it for ranking).

Run locally after pulling results:  python llm_probe/analyze_agreement.py
"""
import os, json, glob
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
CAUSAL_ARMS = ["flat_ar", "staged", "staged_strong"]   # matched-task arms


def load_proxy():
    """final effrank per arm (mean over seeds)."""
    by = {}
    for f in glob.glob(os.path.join(RES, "*_seed*.json")):
        d = json.load(open(f))
        if "final" not in d:
            continue
        by.setdefault(d["arm"], []).append(d["final"])
    return {a: {"mlp_effrank": np.mean([x["mlp_effrank"] for x in v]),
                "ar_ce": np.mean([x["ar_ce"] for x in v])}
            for a, v in by.items()}


def load_sae():
    """SAE metrics per arm (seed 0 checkpoints)."""
    by = {}
    for f in glob.glob(os.path.join(RES, "ckpt", "*_sae_L*.json")):
        d = json.load(open(f))
        # ckpt name like flat_ar_seed0
        arm = os.path.basename(f).split("_seed")[0]
        by[arm] = d
    return by


def rank(d, higher_better=True):
    """return arms sorted best->worst on value dict."""
    items = [(a, v) for a, v in d.items() if a in CAUSAL_ARMS]
    return [a for a, _ in sorted(items, key=lambda kv: kv[1],
                                 reverse=higher_better)]


def main():
    proxy = load_proxy()
    sae = load_sae()
    if not proxy:
        print("no arm results yet."); return

    print("=== arm metrics (matched-task causal arms) ===")
    print(f"{'arm':14s} {'ar_ce':>7s} {'effrank':>8s} {'mono':>7s} "
          f"{'interf':>8s} {'fvu':>6s} {'dead':>6s}")
    for a in CAUSAL_ARMS:
        p = proxy.get(a, {}); s = sae.get(a, {})
        print(f"{a:14s} {p.get('ar_ce',float('nan')):7.3f} "
              f"{p.get('mlp_effrank',float('nan')):8.3f} "
              f"{s.get('mono_proxy',float('nan')):7.3f} "
              f"{s.get('decoder_interference',float('nan')):8.3f} "
              f"{s.get('recon_fvu',float('nan')):6.3f} "
              f"{s.get('dead_frac',float('nan')):6.3f}")

    print("\n=== rankings (best->worst = most->least factored) ===")
    r_effrank = rank({a: proxy[a]["mlp_effrank"] for a in proxy if a in CAUSAL_ARMS})
    print(f"  effrank_frac (proxy): {r_effrank}")
    if sae:
        r_mono = rank({a: sae[a]["mono_proxy"] for a in sae if a in CAUSAL_ARMS})
        r_intf = rank({a: sae[a]["decoder_interference"] for a in sae if a in CAUSAL_ARMS},
                      higher_better=False)   # low interference = factored
        print(f"  SAE mono_proxy:       {r_mono}")
        print(f"  decoder_interference: {r_intf}")
        agree = (r_effrank == r_mono == r_intf)
        print(f"\nVERDICT: metrics {'AGREE' if agree else 'DISAGREE'} on arm ordering.")
        if agree:
            print("  => metric trusted. The objective-lever answer under this "
                  "ordering is the real one.")
        else:
            print("  => metric was (part of) the bottleneck. The disagreement is "
                  "the finding; use decoder_interference (most direct superposition "
                  "measure) as tiebreaker.")
    else:
        print("  (SAE results not present yet — rerun after SAE stage.)")


if __name__ == "__main__":
    main()
