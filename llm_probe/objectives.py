"""
The three training objectives — the ONLY thing that differs across arms.

Everything else (model, data, optimizer, total tokens) is held fixed, so any
difference in the factoredness proxy is attributable to the objective, mirroring
the toy's controlled design.

  arm "flat_ar"   : autoregressive next-token. causal mask; loss on every
                    position predicting the next token. (FER-regime baseline.)

  arm "staged"    : Stanley-lineage analog = a SEQUENCE of objectives that build
                    on each other. Default schedule: masked-denoising warmup
                    (bidirectional, reconstruct masked spans) -> then switch to
                    autoregressive, warm-started. The objective itself changes
                    mid-training, like moving through coherent stepping stones.

  arm "mdlm"      : masked-diffusion / any-order = multi-scale objective. Each
                    step samples a mask ratio t~U(0,1), masks that fraction,
                    predicts the masked tokens bidirectionally. "Predict at every
                    corruption level" is the fair, distribution-wide version of
                    the toy's coarse->fine blur.

A "step spec" tells the loop which objective to apply at the current step, so
`staged` is just a schedule over these primitives.
"""
import torch
import torch.nn.functional as F


def ar_loss(model, batch):
    """batch: (B,T) token ids. Standard next-token CE."""
    x = batch[:, :-1]
    y = batch[:, 1:]
    logits = model(x, causal=True)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def masked_loss(model, batch, mask_ratio, mask_id):
    """Bidirectional masked denoising. Mask `mask_ratio` of tokens, predict them.
    Loss only on masked positions (BERT/MDLM-style)."""
    B, T = batch.shape
    inp = batch.clone()
    rand = torch.rand(B, T, device=batch.device)
    mask = rand < mask_ratio
    # ensure at least one masked token per row
    mask[torch.arange(B), rand.argmin(dim=1)] = True
    inp[mask] = mask_id
    logits = model(inp, causal=False)
    loss = F.cross_entropy(
        logits[mask], batch[mask]) if mask.any() else logits.sum() * 0.0
    return loss


def mdlm_loss(model, batch, mask_id, rng=None):
    """Masked-diffusion: sample a random mask ratio each step (multi-scale)."""
    t = float(torch.rand(1).item()) if rng is None else float(rng.random())
    t = 0.15 + 0.7 * t          # keep in [0.15, 0.85] to avoid degenerate ends
    return masked_loss(model, batch, t, mask_id)


def ar_loss_ctx(model, batch, ctx):
    """AR loss restricted to the first `ctx` tokens (short->long context curriculum).
    The 'coherent intermediate' grows: model first learns to model short spans,
    then progressively longer ones -- a lineage of increasingly-complete tasks."""
    b = batch[:, :ctx]
    x, y = b[:, :-1], b[:, 1:]
    logits = model(x, causal=True)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def step_loss(arm, model, batch, step, total_steps, mask_id):
    """Dispatch the loss for the current step given the arm."""
    if arm == "flat_ar":
        return ar_loss(model, batch), "ar"
    if arm == "mdlm":
        return mdlm_loss(model, batch, mask_id), "mdlm"
    if arm == "staged":
        # mild: one objective switch, masked-denoising warmup -> autoregressive.
        if step < total_steps // 2:
            return masked_loss(model, batch, mask_ratio=0.30, mask_id=mask_id), "mask"
        return ar_loss(model, batch), "ar"
    if arm == "staged_strong":
        # STRONG lineage: MANY progressive stages, each a fuller coherent task,
        # warm-started. denoise(coarse->fine mask) -> AR(short->long context).
        # 6 stages over training; the objective changes 5 times.
        frac = step / max(total_steps, 1)
        T = batch.shape[1]
        if frac < 1/6:   return masked_loss(model, batch, 0.50, mask_id), "mask.50"
        if frac < 2/6:   return masked_loss(model, batch, 0.30, mask_id), "mask.30"
        if frac < 3/6:   return masked_loss(model, batch, 0.15, mask_id), "mask.15"
        if frac < 4/6:   return ar_loss_ctx(model, batch, max(8, T//4)), "ar.q"
        if frac < 5/6:   return ar_loss_ctx(model, batch, max(16, T//2)), "ar.h"
        return ar_loss(model, batch), "ar.full"
    raise ValueError(arm)


ARMS = ["flat_ar", "staged", "mdlm", "staged_strong"]
