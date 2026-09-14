"""
PATH B: measure the DEPTH-RESOLVED activation-rank profile to test the mechanism.

DERIVATION_v2 predicts Muon's spectral update keeps W's singular values flat, so
Σ_z = W Σ_x Wᵀ stays isotropic, and this conditioning COMPOUNDS over depth
(x^{(l+1)}=a^{(l)}), so the Muon−AdamW activation-effrank gap should GROW with layer
index. This script measures per-layer activation effrank (correlation-matrix,
scale-invariant) for each checkpoint and reports the profile + the gap-vs-depth
slope — the direct test of the compounding mechanism.

Also reports, per layer, the effective rank of the WEIGHT matrix W itself (its
singular-value-spectrum flatness) — the derivation's premise that Muon keeps W's
spectrum flatter than AdamW. If both (flatter W spectrum) and (growing activation
gap) hold, the mechanism chain is empirically supported end to end.

No training. Uses saved checkpoints.
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


def corr_effrank_frac(A):
    A = A - A.mean(0, keepdims=True)
    A = A / (A.std(0, keepdims=True) + 1e-8)
    C = (A.T @ A) / A.shape[0]
    return _effrank(np.linalg.svdvals(C)) / A.shape[1]


@torch.no_grad()
def profile(ckpt, device, n_batches=6):
    ck = torch.load(ckpt, map_location=device); cfg = ck["cfg"]
    m = TinyGPT(vocab_size=ck["vocab_size"], d_model=cfg["d_model"],
                n_layer=cfg["n_layer"], n_head=cfg["n_head"], max_len=cfg["seq_len"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    data = get_data(cfg["corpus"], cfg["seq_len"], cfg["batch_size"], device)
    L = cfg["n_layer"]
    acts = [[] for _ in range(L)]
    for _ in range(n_batches):
        b = data.train_batch()
        m(b, causal=True, capture=True)
        for l, a in enumerate(m.captured()["mlp_hidden"]):
            acts[l].append(a.reshape(-1, a.shape[-1]).cpu().numpy())
    act_er = [corr_effrank_frac(np.concatenate(acts[l])) for l in range(L)]
    # weight-spectrum flatness: effrank of the up-proj (mlp[0]) singular values / min-dim
    w_er = []
    for blk in m.blocks:
        W = blk.mlp[0].weight.detach().cpu().numpy()   # (d_ff, d_model) up-proj
        sv = np.linalg.svd(W, compute_uv=False)
        w_er.append(_effrank(sv) / min(W.shape))
    return act_er, w_er, cfg.get("optimizer", "adamw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ck in args.ckpts:
        act_er, w_er, opt = profile(ck, device)
        rows.append(dict(ckpt=os.path.basename(ck), optimizer=opt,
                         act_effrank_by_layer=act_er, weight_effrank_by_layer=w_er))
        print(f"\n{os.path.basename(ck)} (opt={opt})")
        print("  layer:        " + " ".join(f"{i:5d}" for i in range(len(act_er))))
        print("  act_effrank:  " + " ".join(f"{v:5.3f}" for v in act_er))
        print("  weight_effrk: " + " ".join(f"{v:5.3f}" for v in w_er))
    out = args.out or os.path.join(os.path.dirname(args.ckpts[0]), "..", "mechanism_depth.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nWrote {out}")
    # gap-vs-depth slope: muon act_effrank − adamw act_effrank per layer
    byopt = {r["optimizer"]: r["act_effrank_by_layer"] for r in rows}
    if "muon" in byopt and "adamw" in byopt:
        gap = [mu - ad for mu, ad in zip(byopt["muon"], byopt["adamw"])]
        print("\nMuon−AdamW activation-effrank GAP by layer:", [f"{g:+.3f}" for g in gap])
        print("PREDICTION: gap grows with layer (compounding). Check monotonic trend.")


if __name__ == "__main__":
    main()
