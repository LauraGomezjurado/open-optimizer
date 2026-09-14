"""
Compact decoder-only transformer for the objective-lever experiment.

Deliberately small (~10-50M params) and dependency-light (torch only). Supports
BOTH training objectives we need from ONE architecture, chosen at forward time:
  - causal (autoregressive): standard causal mask
  - bidirectional (masked-denoising / any-order): no causal mask, predict
    masked positions -- lets the SAME model train arm-1 (AR), arm-2 (staged
    mask->AR), and arm-3 (masked-diffusion) with only the mask/loss changing.

Exposes activation capture (residual stream per block + MLP hidden) so the
factoredness proxy reads internal structure without re-running the model.

PREP ONLY: authored for a GPU pod; not run locally (no torch here).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.mlp_hidden = None          # captured for the metric

    def forward(self, x, attn_mask=None, key_padding_mask=None, capture=False,
                capture_grad=False):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        # split the mlp to grab the post-GELU hidden (the "neurons")
        pre = self.mlp[0](h)
        hid = self.mlp[1](pre)
        # capture_grad keeps the graph (for the anti-collapse regularizer);
        # capture (metric) detaches for cheap read-only measurement.
        self.mlp_hidden = hid if capture_grad else (hid.detach() if capture else None)
        x = x + self.mlp[2](hid)
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layer=6, n_head=6,
                 d_ff=None, max_len=512, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight          # weight tying
        self._resid = []                             # captured residual streams
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _causal_mask(self, T, device):
        return torch.triu(torch.full((T, T), float("-inf"), device=device), 1)

    def forward(self, idx, causal=True, capture=False, patch=None, capture_grad=False):
        """patch: optional (layer_idx, fn) -- after block `layer_idx` writes the
        residual stream x (B,T,d), replace it with fn(x). Used for recombinability
        (activation patching): perturb/interpolate the representation and free-run."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None])
        attn_mask = self._causal_mask(T, idx.device) if causal else None
        self._resid = []
        for li, blk in enumerate(self.blocks):
            x = blk(x, attn_mask=attn_mask, capture=capture, capture_grad=capture_grad)
            if patch is not None and patch[0] == li:
                x = patch[1](x)
            if capture:
                self._resid.append(x.detach())
        x = self.ln_f(x)
        return self.head(x)

    def captured(self):
        """Return dict of activations for the factoredness proxy:
        residual stream per block (B,T,d) and MLP hidden per block (B,T,d_ff)."""
        return {
            "resid": self._resid,
            "mlp_hidden": [b.mlp_hidden for b in self.blocks],
        }


def count_params(m):
    return sum(p.numel() for p in m.parameters())
