"""
Causal spectral surgery — the do()-operation on the hypothesized mediator.

Motivation: three activation-statistics levers failed to beat Muon's
recombinability ceiling (see DERIVATION_v2.md). The remaining correlate is
Muon/Aurora's flat WEIGHT singular spectrum. Correlation is not cause. Here we
INTERVENE directly on a trained checkpoint's spectrum, WITHOUT retraining, and
measure whether recombinability moves as the flat-spectrum hypothesis predicts.

Intervention (per 2D hidden weight matrix W = U S Vᵀ):
  S' = S**p                         # p<1 flattens (toward Muon), p>1 sharpens
  W' = U diag(S') Vᵀ, then rescale to preserve ||W||_F (isolate SHAPE from SCALE).
U and V are left untouched, so we edit ONLY the spectrum shape.

Controls that make this a valid causal test:
  - p=1.0 is an EXACT identity (must reproduce the original numbers; sanity gate).
  - Frobenius norm preserved => any effect is spectral shape, not gross scaling.
  - We report ar_ce (capability) alongside recombinability: a flatten that
    destroys the model makes any recombinability delta meaningless.

Prediction of the flat-spectrum-is-causal hypothesis:
  flatten (p<1) on an AdamW ckpt  -> effrank UP, recombinability UP (toward Muon)
  sharpen (p>1) on a Muon ckpt    -> effrank DOWN, recombinability DOWN (toward Adam)
If the spectrum edit does NOT move recombinability, the spectrum is not the lever.
"""
import os, sys, json, argparse, copy
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import TinyGPT              # noqa
from data import get_data              # noqa
import recombinability as recomb       # noqa
from mechanism_depth import corr_effrank_frac, _effrank   # noqa


# ---- which matrices to operate on -------------------------------------------
def surgery_targets(model):
    """2D weight matrices whose spectrum we edit: MLP up/down proj per block.
    We deliberately EXCLUDE the tied embedding/head (vocab table, not a mixing
    map) and LayerNorm/attn-packed params (attn is nn.MultiheadAttention with a
    fused in_proj; editing its spectrum is ambiguous). MLP projections are the
    clean, unambiguous mixing maps and are where effrank compounds with depth."""
    ts = []
    for li, blk in enumerate(model.blocks):
        ts.append((f"b{li}.mlp.up", blk.mlp[0].weight))     # (d_ff, d_model)
        ts.append((f"b{li}.mlp.down", blk.mlp[2].weight))   # (d_model, d_ff)
    return ts


@torch.no_grad()
def reshape_spectrum(W, p):
    """Return W' with singular values raised to power p, ||W'||_F == ||W||_F."""
    W32 = W.float()
    U, S, Vh = torch.linalg.svd(W32, full_matrices=False)
    S2 = S.clamp_min(0).pow(p)
    # preserve Frobenius norm (== sqrt(sum S^2)) so we edit SHAPE not SCALE
    scale = (S.pow(2).sum().sqrt()) / (S2.pow(2).sum().sqrt() + 1e-12)
    S2 = S2 * scale
    Wp = (U * S2) @ Vh
    return Wp.to(W.dtype), S.cpu().numpy(), S2.cpu().numpy()


@torch.no_grad()
def apply_surgery(model, p):
    """Edit every target matrix in place; return list of (name, sv_before, sv_after)."""
    log = []
    for name, W in surgery_targets(model):
        Wp, sv0, sv1 = reshape_spectrum(W, p)
        W.copy_(Wp)
        log.append((name, sv0, sv1))
    return log


@torch.no_grad()
def eval_ce(model, data, n_batches=8):
    """Autoregressive cross-entropy on held-out probe blocks (capability guard)."""
    losses = []
    probe = data.probe_blocks
    B = min(data.batch_size, probe.shape[0])
    for k in range(n_batches):
        i = (k * B) % max(1, probe.shape[0] - B)
        b = probe[i:i+B].to(data.device)
        if b.shape[0] < 2:
            b = probe[:B].to(data.device)
        logits = model(b[:, :-1], causal=True)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               b[:, 1:].reshape(-1))
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def act_effrank_by_layer(model, data, n_batches=6):
    """Correlation effrank of MLP hidden per layer (the FER axis)."""
    L = len(model.blocks)
    acts = [[] for _ in range(L)]
    for _ in range(n_batches):
        b = data.train_batch()
        model(b, causal=True, capture=True)
        for l, a in enumerate(model.captured()["mlp_hidden"]):
            acts[l].append(a.reshape(-1, a.shape[-1]).cpu().numpy())
    return [corr_effrank_frac(np.concatenate(acts[l])) for l in range(L)]


def weight_effrank(model):
    """Mean spectrum-flatness (effrank/min-dim) across target matrices."""
    vals = []
    for _, W in surgery_targets(model):
        sv = np.linalg.svd(W.detach().cpu().numpy(), compute_uv=False)
        vals.append(_effrank(sv) / min(W.shape))
    return float(np.mean(vals))


# ---- recombinability on an already-loaded (edited) model --------------------
@torch.no_grad()
def recomb_on_model(model, cfg, data, device, seed=0, n_prompts=40, n_samples=8,
                    fracs=(0.0, 0.5, 1.0, 2.0), n_new=24, layer=None):
    """Same measurement as recombinability.run_ckpt but on an in-memory model
    (so we can run it on the surgically-edited weights without a round-trip to
    disk). Returns {frac: {novelty, perplexity, n}}."""
    L = layer if layer is not None else cfg["n_layer"] // 2
    g = torch.Generator(device="cpu").manual_seed(seed)
    probe = data.probe_batch()[:n_prompts]
    plen = min(16, probe.shape[1] // 2)
    out = {}
    for frac in fracs:
        gens, ppls = [], []
        for pi in range(min(n_prompts, probe.shape[0])):
            pin = probe[pi:pi+1, :plen].to(device)
            rep = recomb.clean_rep(model, pin, L)
            norm = rep.norm().item()
            for si in range(n_samples):
                if frac == 0.0 and si > 0:
                    break
                dirn = torch.randn(rep.shape, generator=g).to(device)
                dirn = dirn / (dirn.norm() + 1e-8)
                delta = dirn * (frac * norm)
                gen, ppl = recomb.generate_with_patch(model, pin, L, delta[0],
                                                      n_new, device)
                gens.append(gen); ppls.append(ppl)
        nov = [recomb.token_distance(gens[i], gens[j])
               for i in range(len(gens)) for j in range(i+1, len(gens))]
        out[frac] = dict(novelty=float(np.mean(nov)) if nov else 0.0,
                         perplexity=float(np.exp(np.mean(ppls))) if ppls else 0.0,
                         n=len(gens))
    return out


def _fresh_model(ck, cfg, device):
    m = TinyGPT(vocab_size=ck["vocab_size"], d_model=cfg["d_model"],
                n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                max_len=cfg["seq_len"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m


def run_surgery(ckpt_path, device, powers, seed=0):
    """For each p in `powers`: reload the ORIGINAL weights, apply spectral
    surgery, measure {weight_effrank, act_effrank (mean+by_layer), ar_ce,
    recombinability}. p=1.0 must reproduce the untouched model (identity gate)."""
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["cfg"]
    data = get_data(cfg["corpus"], cfg["seq_len"], cfg["batch_size"], device)
    rows = []
    for p in powers:
        model = _fresh_model(ck, cfg, device)   # always start from the original
        apply_surgery(model, p)
        act_er = act_effrank_by_layer(model, data)
        row = dict(
            p=float(p),
            w_effrank=weight_effrank(model),
            act_effrank_mean=float(np.mean(act_er)),
            act_effrank_by_layer=[float(x) for x in act_er],
            ar_ce=eval_ce(model, data),
            recomb=recomb_on_model(model, cfg, data, device, seed=seed),
        )
        rows.append(row)
    return dict(ckpt=os.path.basename(ckpt_path),
                optimizer=cfg.get("optimizer", "adamw"),
                layer=cfg["n_layer"] // 2, rows=rows)


def _fmt(r):
    lines = [f"\n=== {r['ckpt']} (opt={r['optimizer']}, patch L={r['layer']}) ==="]
    lines.append(f"  {'p':>5} {'w_er':>6} {'act_er':>7} {'ar_ce':>7} "
                 f"{'nov@f1':>7} {'ppl@f1':>8}")
    for row in r["rows"]:
        f1 = row["recomb"].get(1.0) or row["recomb"].get("1.0") or {}
        lines.append(f"  {row['p']:>5.2f} {row['w_effrank']:>6.3f} "
                     f"{row['act_effrank_mean']:>7.3f} {row['ar_ce']:>7.3f} "
                     f"{f1.get('novelty', 0):>7.3f} {f1.get('perplexity', 0):>8.1f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--powers", nargs="+", type=float,
                    default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    allr = []
    for ck in args.ckpts:
        r = run_surgery(ck, device, args.powers, seed=args.seed)
        allr.append(r)
        print(_fmt(r))
    out = args.out or os.path.join(os.path.dirname(args.ckpts[0]), "..",
                                   "spectrum_surgery.json")
    json.dump(allr, open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
