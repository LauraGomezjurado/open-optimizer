"""
GATE 2: does the quantity the Recombinator would optimize (isotropy of the
output-leverage Gram L = JᵀJ, J = ∂logits/∂hidden) actually PREDICT the measured
recombinability frontier? If L-isotropy does not track recombinability across the
existing matched checkpoints, the derivation targets the wrong thing.

For each checkpoint we compute L on a mid MLP layer via Hutchinson random probes
(L̂ = (1/k) Σ (Jᵀgᵢ)(Jᵀgᵢ)ᵀ, gᵢ ~ Rademacher over the vocab-logit output), then
report isotropy summaries that are directly comparable across models:
  - L_effrank_frac : effective rank of L̂ / m  (higher = more isotropic = target)
  - L_offdiag      : mean |off-diagonal corr| of L̂ (lower = more decorrelated)
  - L_cond         : condition-ish (top eig / mean eig) of the estimable spectrum
We then place these next to the measured recombinability (novelty@frac2, ppl@frac2)
for adamw/muon/aurora and eyeball/rank-correlate whether L-isotropy ordering ==
recombinability ordering.

No training. Uses saved checkpoints. k probes (default 32) — low-rank estimate,
so L_effrank is a floor; we compare RELATIVELY across checkpoints (same k, layer).
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import TinyGPT          # noqa
from data import get_data          # noqa


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device); cfg = ck["cfg"]
    m = TinyGPT(vocab_size=ck["vocab_size"], d_model=cfg["d_model"],
                n_layer=cfg["n_layer"], n_head=cfg["n_head"], max_len=cfg["seq_len"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, cfg


def leverage_gram(model, cfg, device, layer, k=32, n_batches=4):
    """Hutchinson estimate of L=JᵀJ where J=∂logits/∂h at `layer` MLP hidden.
    For each probe g (random over logits), Jᵀg via autograd = ∂(g·logits)/∂h."""
    data = get_data(cfg["corpus"], cfg["seq_len"], cfg["batch_size"], device)
    m_dim = None
    Lsum = None
    count = 0
    for _ in range(n_batches):
        batch = data.train_batch()
        # forward with grad, capture mlp hidden at `layer`
        model.zero_grad(set_to_none=True)
        logits = model(batch, causal=True, capture_grad=True)
        h = model.blocks[layer].mlp_hidden          # (B,T,d_ff), requires grad
        # ARTIFACT FIX: probe over log_softmax (shape) not raw logits (confidence),
        # so the leverage Gram isn't confounded by per-model logit scale/weight-norm.
        y = torch.log_softmax(logits, dim=-1)
        if m_dim is None:
            m_dim = h.shape[-1]; Lsum = torch.zeros(m_dim, m_dim, device=device)
        B, T, V = logits.shape
        for _ in range(k):
            g = (torch.randint(0, 2, (B, T, V), device=device).float() * 2 - 1)
            jtg, = torch.autograd.grad((y * g).sum(), h, retain_graph=True)
            v = jtg.reshape(-1, m_dim)               # (B*T, d_ff) = Jᵀg per token
            Lsum += v.T @ v / v.shape[0]
        count += k
    L = (Lsum / count).cpu().numpy()
    return L


def isotropy(L, eps=1e-9):
    ev = np.linalg.eigvalsh(L); ev = np.clip(ev, 0, None)
    s = ev.sum()
    p = ev / (s + eps); p = p[p > eps]
    effrank = float(np.exp(-(p * np.log(p)).sum()))
    # correlation off-diagonal
    d = np.sqrt(np.clip(np.diag(L), eps, None))
    C = L / np.outer(d, d)
    m = C.shape[0]
    offdiag = float((np.abs(C).sum() - np.abs(np.diag(C)).sum()) / (m * (m - 1)))
    return dict(L_effrank_frac=effrank / L.shape[0], L_offdiag=offdiag,
                L_effrank=effrank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ck in args.ckpts:
        model, cfg = load(ck, device)
        L = leverage_gram(model, cfg, device, args.layer, k=args.k)
        iso = isotropy(L)
        iso.update(ckpt=os.path.basename(ck), optimizer=cfg.get("optimizer", "adamw"))
        rows.append(iso)
        print(f"{iso['ckpt']:32s} opt={iso['optimizer']:7s} "
              f"L_effrank_frac={iso['L_effrank_frac']:.4f} L_offdiag={iso['L_offdiag']:.4f}")
    out = args.out or os.path.join(os.path.dirname(args.ckpts[0]), "..", "gate2_leverage.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nWrote {out}")
    print("\nGATE 2 check: does L_effrank_frac / L_offdiag ordering match the")
    print("recombinability ordering (aurora >= muon > adamw)? If yes, the target predicts")
    print("the measured property. If no, the derivation targets the wrong quantity.")


if __name__ == "__main__":
    main()
