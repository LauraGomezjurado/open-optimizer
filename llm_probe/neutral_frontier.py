"""
Neutral-judge recombinability frontier — kills the two circularities in
recombinability.py that inflated the "Muon +40%" headline.

Problem: the old frontier used (a) SELF-perplexity for coherence (a model that is
confidently wrong scores itself coherent) and (b) raw token-Jaccard novelty (a
higher-entropy model looks "more novel" for free). Under those, novelty and
coherence rode the same entropy axis, so "Muon is more novel" was largely "Muon's
generations are higher-entropy / less coherent."

Fixes here:
  - COHERENCE = negative mean per-token NLL under a NEUTRAL external judge
    (pretrained GPT-2), which shares our exact GPT-2 BPE vocab. On-manifold English
    => low judge-perplexity, independent of the model being probed.
  - NOVELTY = mean pairwise token-Jaccard distance, reported ALONGSIDE the mean
    per-generation token entropy so we can regress novelty on entropy and read the
    entropy-MATCHED novelty (novelty at a common entropy), not raw novelty.

Everything else (perturb residual at layer L by frac*||act||, greedy free-run) is
identical to recombinability.py so the comparison is apples-to-apples.

Our vocab = gpt2.vocab_size(50257)+1 (last id = MASK). GPT-2 judge accepts ids
0..50256; any MASK/OOV id is dropped before scoring.
"""
import os, sys, json, argparse, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import recombinability as recomb        # reuse load_model, clean_rep, generate_with_patch
from data import get_data                # noqa

_GPT2_VOCAB = 50257                       # ids >= this are our MASK / OOV -> drop


def load_judge(device):
    from transformers import GPT2LMHeadModel
    j = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    return j


@torch.no_grad()
def judge_nll(judge, gen_ids, device):
    """Mean per-token NLL of a generated id sequence under pretrained GPT-2.
    Drops any id >= 50257 (MASK/OOV). Returns None if <2 valid tokens."""
    ids = [t for t in gen_ids if 0 <= t < _GPT2_VOCAB]
    if len(ids) < 2:
        return None
    x = torch.tensor([ids], device=device)
    out = judge(x, labels=x)
    return float(out.loss)                # HF returns mean CE over the sequence


def token_entropy(gen_ids):
    """Shannon entropy (nats) of the token-id empirical distribution in one gen."""
    if not gen_ids:
        return 0.0
    _, counts = np.unique(np.asarray(gen_ids), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


@torch.no_grad()
def run_ckpt(ckpt_path, device, judge, layer=None, n_prompts=40, n_samples=8,
             fracs=(0.0, 0.25, 0.5, 1.0), n_new=24, seed=0):
    """Same perturb-and-generate loop as recombinability.run_ckpt, but scores
    coherence with the NEUTRAL judge and records per-gen entropy for the
    entropy-matched novelty regression."""
    model, cfg = recomb.load_model(ckpt_path, device)
    L = layer if layer is not None else cfg["n_layer"] // 2
    data = get_data(cfg["corpus"], cfg["seq_len"], cfg["batch_size"], device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    probe = data.probe_batch()[:n_prompts]
    plen = min(16, probe.shape[1] // 2)
    by_frac = {}
    for frac in fracs:
        gens, jnlls, ents = [], [], []
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
                gen, _self_ppl = recomb.generate_with_patch(model, pin, L, delta[0],
                                                            n_new, device)
                jn = judge_nll(judge, gen, device)
                if jn is None:
                    continue
                gens.append(gen); jnlls.append(jn); ents.append(token_entropy(gen))
        nov = [recomb.token_distance(gens[i], gens[j])
               for i in range(len(gens)) for j in range(i+1, len(gens))]
        by_frac[frac] = dict(
            novelty=float(np.mean(nov)) if nov else 0.0,
            judge_ppl=float(np.exp(np.mean(jnlls))) if jnlls else 0.0,
            judge_nll=float(np.mean(jnlls)) if jnlls else 0.0,
            mean_entropy=float(np.mean(ents)) if ents else 0.0,
            n=len(gens))
    return dict(ckpt=os.path.basename(ckpt_path),
                optimizer=cfg.get("optimizer", "adamw"),
                lr=cfg.get("lr"), seed=cfg.get("seed"), layer=L, by_frac=by_frac)


def _fmt(r):
    lines = [f"\n=== {r['ckpt']} opt={r['optimizer']} lr={r['lr']} seed={r['seed']} L={r['layer']} ==="]
    lines.append(f"  {'frac':>5} {'novelty':>8} {'judgePPL':>9} {'entropy':>8} {'n':>5}")
    for frac, m in r["by_frac"].items():
        lines.append(f"  {frac:>5} {m['novelty']:>8.3f} {m['judge_ppl']:>9.1f} "
                     f"{m['mean_entropy']:>8.3f} {m['n']:>5d}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    judge = load_judge(device)
    allr = []
    for ck in args.ckpts:
        r = run_ckpt(ck, device, judge, layer=args.layer)
        allr.append(r)
        print(_fmt(r))
    out = args.out or os.path.join(os.path.dirname(args.ckpts[0]), "..",
                                   "neutral_frontier.json")
    json.dump(allr, open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
