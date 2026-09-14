"""
Recombinability: does Muon's higher effrank buy more distinct-yet-coherent
variations under representation-space perturbation? (open-endedness stepping-stone
test). See RECOMBINABILITY_PLAN.md.

Method per checkpoint:
  - take probe prompts, capture residual at layer L for the last position
  - perturb that vector (random direction, magnitude = frac * ||act||), patch it
    back at layer L, free-run to generate a continuation
  - novelty  = mean pairwise token-distance across the perturbed generations
  - coherence= negative mean per-token NLL of the generation UNDER THE SAME MODEL
               (on-manifold => low perplexity => high coherence). We report
               perplexity; lower = more coherent.
  - the open-endedness quantity is novelty at matched coherence (Pareto frontier).

Matched-loss checkpoints only. Perturbation magnitude normalized to activation
norm so Muon/Adam scale differences don't confound.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import TinyGPT                       # noqa
from data import get_data                        # noqa


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["cfg"]
    m = TinyGPT(vocab_size=ck["vocab_size"], d_model=cfg["d_model"],
                n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                max_len=cfg["seq_len"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, cfg


@torch.no_grad()
def clean_rep(model, prompt_ids, layer):
    """residual at `layer` for the last position: (d,)."""
    model(prompt_ids, causal=True, capture=True)
    return model._resid[layer][:, -1, :]          # (B,d)


@torch.no_grad()
def generate_with_patch(model, prompt_ids, layer, delta_vec, n_new, device):
    """Free-run generation while adding delta_vec to the residual at `layer`,
    last position, at every step. Returns generated token ids and per-token NLL."""
    ids = prompt_ids.clone()
    nlls = []
    for _ in range(n_new):
        def patch(x, d=delta_vec):
            x = x.clone(); x[:, -1, :] = x[:, -1, :] + d
            return x
        logits = model(ids, causal=True, patch=(layer, patch))
        logp = F.log_softmax(logits[:, -1, :], dim=-1)
        nxt = logp.argmax(-1, keepdim=True)         # greedy (deterministic given delta)
        nlls.append(-logp.gather(-1, nxt).item())
        ids = torch.cat([ids, nxt], dim=1)
    gen = ids[0, prompt_ids.shape[1]:].tolist()
    return gen, float(np.mean(nlls))


def token_distance(a, b):
    """1 - (shared unique tokens / union) = Jaccard distance on token sets."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return 1.0 - len(sa & sb) / max(1, len(sa | sb))


def run_ckpt(ckpt_path, device, layer=None, n_prompts=40, n_samples=8,
             fracs=(0.0, 0.5, 1.0, 2.0), n_new=24, seed=0):
    model, cfg = load_model(ckpt_path, device)
    L = layer if layer is not None else cfg["n_layer"] // 2
    data = get_data(cfg["corpus"], cfg["seq_len"], cfg["batch_size"], device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    probe = data.probe_batch()[:n_prompts]        # (P,T)
    plen = min(16, probe.shape[1] // 2)
    results = {}
    for frac in fracs:
        gens, ppls = [], []
        for pi in range(min(n_prompts, probe.shape[0])):
            pin = probe[pi:pi+1, :plen].to(device)
            rep = clean_rep(model, pin, L)          # (1,d)
            norm = rep.norm().item()
            for si in range(n_samples):
                if frac == 0.0 and si > 0:
                    break                           # unperturbed is deterministic
                dirn = torch.randn(rep.shape, generator=g).to(device)
                dirn = dirn / (dirn.norm() + 1e-8)
                delta = dirn * (frac * norm)
                gen, ppl = generate_with_patch(model, pin, L, delta[0], n_new, device)
                gens.append(gen); ppls.append(ppl)
        # novelty = mean pairwise token-distance; coherence = mean perplexity
        nov = []
        for i in range(len(gens)):
            for j in range(i+1, len(gens)):
                nov.append(token_distance(gens[i], gens[j]))
        results[frac] = dict(
            novelty=float(np.mean(nov)) if nov else 0.0,
            perplexity=float(np.exp(np.mean(ppls))) if ppls else 0.0,
            n=len(gens))
    return dict(ckpt=os.path.basename(ckpt_path), layer=L,
                optimizer=cfg.get("optimizer", "adamw"), by_frac=results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    allr = []
    for ck in args.ckpts:
        r = run_ckpt(ck, device, layer=args.layer)
        allr.append(r)
        print(f"\n{r['ckpt']} (opt={r['optimizer']}, L={r['layer']})")
        print(f"  {'frac':>5} {'novelty':>8} {'perplexity':>11}")
        for frac, m in r["by_frac"].items():
            print(f"  {frac:>5} {m['novelty']:>8.3f} {m['perplexity']:>11.1f}")
    out = args.out or os.path.join(os.path.dirname(args.ckpts[0]), "..",
                                   "recombinability.json")
    json.dump(allr, open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
