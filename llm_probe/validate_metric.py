"""
Transformer-native metric validation gate (replaces the confounded genome gate).

The evolved-vs-SGD genome gate conflates sparsity (44-63% dead units) with
factoredness, so it can't fairly validate a superposition metric. Instead we
build a ground-truth pair with NO density confound, in the transformer domain:

  WIDE  (d_model large): plenty of room -> features can spread -> LESS superposed.
  NARROW(d_model small): forced to cram > d features into few dims -> MORE superposed.

Same data, same objective (flat_ar), same tokens. Superposition is *known* to be
higher in NARROW by construction (Elhage et al. toy-models-of-superposition logic).
A valid factoredness metric must rank WIDE as more factored than NARROW.

We score both checkpoints on every candidate:
  - effrank_frac        (expect: can't tell, or ambiguous)
  - activation_kurtosis (expect: NARROW higher if it truly tracks superposition)
  - SAE fvu / interference (from sae.py, run separately)

Usage on pod:
  PY train.py --arm flat_ar --d_model 128 --tag narrow --save_ckpt --seed 0
  PY train.py --arm flat_ar --d_model 512 --tag wide   --save_ckpt --seed 0
  PY validate_metric.py --narrow results/ckpt/flat_ar_seed0_narrow.pt \
                        --wide   results/ckpt/flat_ar_seed0_wide.pt
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import TinyGPT                       # noqa
from data import get_data                        # noqa


def activation_kurtosis(A, eps=1e-9):
    """mean per-feature excess kurtosis; HIGH = superposed. (inlined from
    src/metrics.py so this runs on the pod without the src tree)."""
    A = np.asarray(A, np.float64)
    keep = A.std(0) > eps
    A = A[:, keep]
    if A.shape[1] == 0:
        return 0.0
    z = (A - A.mean(0)) / (A.std(0) + eps)
    return float((z ** 4).mean() - 3.0)


def effrank_frac_np(A, eps=1e-9):
    A = A - A.mean(0); sd = A.std(0); sd[sd < eps] = 1; A = A / sd
    c = (A.T @ A) / A.shape[0]
    sv = np.linalg.svd(c, compute_uv=False); s = sv[sv > 1e-12]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()) / A.shape[1])


@torch.no_grad()
def gather(ckpt_path, device, layer=None, n_batches=30):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["cfg"]
    model = TinyGPT(vocab_size=ck["vocab_size"], d_model=cfg["d_model"],
                    n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                    max_len=cfg["seq_len"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    data = get_data(cfg["corpus"], cfg["seq_len"], cfg["batch_size"], device)
    L = layer if layer is not None else cfg["n_layer"] // 2
    resid, mlp = [], []
    for _ in range(n_batches):
        b = data.train_batch()
        model(b, causal=True, capture=True)
        cap = model.captured()
        resid.append(cap["resid"][L].reshape(-1, cap["resid"][L].shape[-1]).cpu().numpy())
        mlp.append(cap["mlp_hidden"][L].reshape(-1, cap["mlp_hidden"][L].shape[-1]).cpu().numpy())
    return np.concatenate(resid), np.concatenate(mlp), cfg["d_model"]


def score(name, resid, mlp):
    return dict(name=name,
                resid_effrank=effrank_frac_np(resid),
                mlp_effrank=effrank_frac_np(mlp),
                resid_kurtosis=activation_kurtosis(resid),
                mlp_kurtosis=activation_kurtosis(mlp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--narrow", required=True)
    ap.add_argument("--wide", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rn, mn, dn = gather(args.narrow, device)
    rw, mw, dw = gather(args.wide, device)
    sn = score(f"NARROW(d={dn})", rn, mn)
    sw = score(f"WIDE(d={dw})", rw, mw)
    print("\n=== transformer-native superposition gate (NARROW=more superposed) ===")
    for k in ["resid_effrank", "mlp_effrank", "resid_kurtosis", "mlp_kurtosis"]:
        print(f"  {k:16s} narrow={sn[k]:+.3f}  wide={sw[k]:+.3f}")
    print("\nA valid superposition metric should read NARROW as MORE superposed:")
    print(f"  kurtosis: narrow>{'>' if sn['mlp_kurtosis']>sw['mlp_kurtosis'] else '<'} wide "
          f"-> kurtosis {'PASSES' if sn['mlp_kurtosis']>sw['mlp_kurtosis'] else 'FAILS'}")
    json.dump(dict(narrow=sn, wide=sw),
              open(os.path.join(os.path.dirname(args.narrow), "..", "metric_gate.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
