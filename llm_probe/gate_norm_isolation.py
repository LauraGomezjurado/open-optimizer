"""
PATH 2 (no retraining): isolate whether Muon's effrank advantage is a WEIGHT-SCALE
effect or survives scale-normalization. Reuses existing checkpoints.

Key idea: our headline effrank_frac already uses the CORRELATION matrix (per-feature
standardized) -> scale-invariant per feature. If Muon's advantage lives purely in
weight scale, then:
  - COVARIANCE effrank (scale-sensitive) should show a big Muon>AdamW gap, AND
  - CORRELATION effrank (scale-invariant) should show a SMALLER/absent gap.
If instead the gap is similar under both, the advantage is structural (how many
independent directions), not a scale artifact.

We also L2-normalize each token's activation vector (removes overall magnitude) and
recompute, as a second scale control. Reports all three effrank variants per
checkpoint so we can see where the Muon-AdamW gap actually lives.
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import TinyGPT       # noqa
from data import get_data       # noqa


def _effrank(sv, eps=1e-12):
    sv = sv[sv > eps]
    if sv.size == 0: return 0.0
    p = sv / sv.sum()
    return float(np.exp(-(p*np.log(p)).sum()))


def variants(A):
    """A: (N, d) activations. Return effrank_frac under 3 normalizations."""
    A = A - A.mean(0, keepdims=True)
    d = A.shape[1]
    # 1. covariance (scale-SENSITIVE)
    cov = (A.T @ A) / A.shape[0]
    er_cov = _effrank(np.linalg.svdvals(cov)) / d
    # 2. correlation (per-feature scale-INVARIANT) -- our headline metric
    sd = A.std(0, keepdims=True) + 1e-8
    Ac = A / sd
    corr = (Ac.T @ Ac) / Ac.shape[0]
    er_corr = _effrank(np.linalg.svdvals(corr)) / d
    # 3. per-token L2-normalized (removes overall magnitude per token)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    covn = (An.T @ An) / An.shape[0]
    er_l2 = _effrank(np.linalg.svdvals(covn)) / d
    return er_cov, er_corr, er_l2


@torch.no_grad()
def gather(ckpt, device, layer, n_batches=6):
    ck = torch.load(ckpt, map_location=device); cfg = ck["cfg"]
    m = TinyGPT(vocab_size=ck["vocab_size"], d_model=cfg["d_model"],
                n_layer=cfg["n_layer"], n_head=cfg["n_head"], max_len=cfg["seq_len"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    data = get_data(cfg["corpus"], cfg["seq_len"], cfg["batch_size"], device)
    acts = []
    for _ in range(n_batches):
        b = data.train_batch()
        m(b, causal=True, capture=True)
        a = m.captured()["mlp_hidden"][layer]
        acts.append(a.reshape(-1, a.shape[-1]).cpu().numpy())
    return np.concatenate(acts), cfg.get("optimizer", "adamw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    print(f"{'ckpt':34s} {'opt':7s} {'er_cov':>8s} {'er_corr':>8s} {'er_l2':>8s}")
    for ck in args.ckpts:
        A, opt = gather(ck, device, args.layer)
        cov, corr, l2 = variants(A)
        rows.append(dict(ckpt=os.path.basename(ck), optimizer=opt,
                         er_cov=cov, er_corr=corr, er_l2=l2))
        print(f"{os.path.basename(ck):34s} {opt:7s} {cov:>8.3f} {corr:>8.3f} {l2:>8.3f}")
    out = args.out or os.path.join(os.path.dirname(args.ckpts[0]), "..", "gate_norm_isolation.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nWrote {out}")
    print("If Muon>AdamW gap is similar in er_cov AND er_corr/er_l2 -> STRUCTURAL (not scale).")
    print("If gap is big in er_cov but vanishes in er_corr/er_l2 -> SCALE artifact.")


if __name__ == "__main__":
    main()
