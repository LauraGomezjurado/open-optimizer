"""
Capacity-invariant factoredness proxy for transformer activations.

Ported from the toy's lesson: the metric that survived the UFR gate was NOT
weight-perturbation magnitude (capacity-confounded) but a *representation-
structure* quantity normalized so width/capacity can't drive it. Here we compute,
per layer, over a batch of real tokens:

  effrank_frac : effective rank of the activation CORRELATION matrix / width.
                 (correlation, not covariance -> standardized, so a few
                 high-variance units can't dominate; /width -> width-invariant.)
                 LOW  = features collapsed / superposed (fractured).
                 HIGH = many independent directions used (factored).

  token_pr_frac: mean per-token participation ratio of |activations| / width.
                 How many features an average token lights up, normalized.
                 (diagnostic; superposition packs many features per token.)

Both are in (0,1], width-invariant by construction, so they are comparable
across model sizes and across arms. Computed on MLP hidden ("neurons") and on
the residual stream. No SAE required -- this is the cheap gate; SAE only runs
where this shows a gap.

PREP ONLY (needs torch); mirror of src/metrics.py:_feat_effrank_corr_frac.
"""
import torch


@torch.no_grad()
def _effective_rank(sv, eps=1e-12):
    sv = sv[sv > eps]
    if sv.numel() == 0:
        return torch.tensor(0.0)
    p = sv / sv.sum()
    ent = -(p * torch.log(p)).sum()
    return torch.exp(ent)


@torch.no_grad()
def effrank_frac(acts):
    """acts: (N, d) activations (tokens x features). Returns effrank/d in (0,1].
    Uses the CORRELATION matrix (standardized cols) -- our width-invariant variant."""
    A = acts.float()
    A = A - A.mean(0, keepdim=True)
    sd = A.std(0, keepdim=True)
    A = A / (sd + 1e-8)                    # correlation matrix path
    corr = (A.T @ A) / A.shape[0]
    sv = torch.linalg.svdvals(corr)
    return float(_effective_rank(sv) / A.shape[1])


@torch.no_grad()
def raw_erank(acts):
    """RAW effective rank exactly as arXiv:2606.09658 defines it: exp-entropy of
    the squared singular values of the CENTERED hidden-state matrix (NOT
    standardized, NOT normalized by width). Lets us put our number next to their
    Table 1 (GPT-2: Adam 11.1, Muon 16.0)."""
    A = acts.float()
    A = A - A.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(A)           # singular values of Z directly
    return float(_effective_rank(sv))


@torch.no_grad()
def token_pr_frac(acts):
    """Mean per-token participation ratio of |acts|, normalized by width."""
    A = acts.abs().float()
    s1 = A.sum(1)
    s2 = (A * A).sum(1)
    pr = (s1 * s1) / (s2 * A.shape[1] + 1e-12)   # in (0,1]
    return float(pr.mean())


@torch.no_grad()
def factoredness_proxy(captured, max_tokens=20000):
    """captured: dict from TinyGPT.captured(). Returns per-layer + mean metrics.
    Flattens (B,T,d)->(B*T,d), subsamples to max_tokens for the SVD."""
    out = {}
    for name in ["resid", "mlp_hidden"]:
        layers = captured[name]
        er, pr, rawer = [], [], []
        for A in layers:
            A = A.reshape(-1, A.shape[-1])
            if A.shape[0] > max_tokens:
                idx = torch.randperm(A.shape[0], device=A.device)[:max_tokens]
                A = A[idx]
            er.append(effrank_frac(A))
            pr.append(token_pr_frac(A))
            rawer.append(raw_erank(A))
        out[f"{name}_effrank_frac"] = er
        out[f"{name}_effrank_frac_mean"] = float(sum(er) / len(er))
        out[f"{name}_token_pr_frac_mean"] = float(sum(pr) / len(pr))
        out[f"{name}_raw_erank_mean"] = float(sum(rawer) / len(rawer))  # cf. their Table 1
    return out
