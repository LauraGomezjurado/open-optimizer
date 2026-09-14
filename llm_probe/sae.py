"""
Sparse autoencoder (SAE) monosemanticity gate for the objective-lever arms.

The cheap proxy (effrank) said staged/mdlm did NOT beat flat_ar. This is the
real check the whole project keeps deferring to: train an SAE on the residual
stream and measure how MONOSEMANTIC / disentangled the learned features are —
the field's actual operationalization of UFR vs FER at LLM scale.

Given a trained model checkpoint, we:
  1. gather residual-stream activations at a chosen layer over eval tokens
  2. train a top-k sparse autoencoder (dict size = expansion * d_model)
  3. report, at matched sparsity:
       - recon_fvu       : fraction of variance unexplained (lower = SAE captures
                           the representation well)
       - dead_frac       : fraction of dictionary features that never fire
                           (high = wasted dictionary; healthy SAEs keep it low)
       - feat_activ_freq : mean firing rate of live features
       - mono_proxy      : a monosemanticity proxy = mean max-activation
                           concentration per feature (how peaked each feature's
                           activations are across tokens; higher = more selective
                           = more monosemantic). This is a cheap stand-in for
                           human-labeled monosemanticity, comparable across arms.

Comparison across arms at the SAME SAE hyperparameters (dict size, k, steps) is
what matters: does the staged/lineage arm yield a cleaner dictionary?

PREP ONLY (needs torch); runs on the pod after checkpoints exist.
"""
import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import TinyGPT                      # noqa
from data import get_data                       # noqa


class TopKSAE(nn.Module):
    def __init__(self, d_in, d_dict, k):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d_in, d_dict)
        self.dec = nn.Linear(d_dict, d_in, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        with torch.no_grad():                    # unit-norm decoder columns
            self.dec.weight.data = F.normalize(self.dec.weight.data, dim=0)

    def encode(self, x):
        z = self.enc(x - self.pre_bias)
        topv, topi = z.topk(self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topi, F.relu(topv))
        return z_sparse

    def forward(self, x):
        z = self.encode(x)
        return self.dec(z) + self.pre_bias, z


@torch.no_grad()
def gather_acts(model, data, device, layer, n_batches=40):
    acts = []
    for _ in range(n_batches):
        b = data.train_batch()
        model(b, causal=True, capture=True)
        r = model.captured()["resid"][layer]          # (B,T,d_model)
        acts.append(r.reshape(-1, r.shape[-1]).cpu())
    return torch.cat(acts, 0)


def train_sae(acts, d_dict, k, steps, device, lr=1e-3, bs=4096):
    d_in = acts.shape[1]
    sae = TopKSAE(d_in, d_dict, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    acts = acts.to(device)
    fired = torch.zeros(d_dict, device=device)
    for step in range(steps):
        idx = torch.randint(0, acts.shape[0], (bs,), device=device)
        x = acts[idx]
        xh, z = sae(x)
        loss = F.mse_loss(xh, x)
        opt.zero_grad(); loss.backward()
        with torch.no_grad():                    # keep decoder unit-norm
            sae.dec.weight.data = F.normalize(sae.dec.weight.data, dim=0)
        opt.step()
        fired += (z > 0).float().sum(0)
    return sae, fired


@torch.no_grad()
def metrics(sae, acts, fired, device):
    acts = acts.to(device)
    xh, z = sae(acts)
    var = acts.var(0).sum()
    fvu = float(((acts - xh) ** 2).mean(0).sum() / (var + 1e-9))
    dead = float((fired == 0).float().mean())
    live = fired > 0
    activ_freq = float((fired[live] / acts.shape[0]).mean()) if live.any() else 0.0
    # monosemanticity proxy: for each live feature, how concentrated are its
    # activations across tokens (participation ratio, inverted). Peaked = selective.
    zc = z.clone(); zc[zc < 0] = 0
    col = zc[:, live]
    s1 = col.sum(0); s2 = (col * col).sum(0)
    pr = (s1 * s1) / (s2 * col.shape[0] + 1e-9)          # in (0,1], low = peaked
    mono = float((1.0 - pr).mean()) if live.any() else 0.0
    # metric D (tiebreaker): decoder feature INTERFERENCE. Superposition packs
    # more features than dims into a non-orthogonal (interfering) dictionary, so
    # the mean |off-diagonal cosine| of live decoder columns is HIGH for
    # fractured/superposed, LOW for clean/factored. Directly operationalizes
    # superposition, independent of the activation-concentration mono_proxy.
    W = F.normalize(sae.dec.weight[:, live], dim=0)      # (d_in, n_live)
    G = (W.T @ W).abs()
    n = G.shape[0]
    interference = float((G.sum() - torch.diagonal(G).sum()) / (n * (n - 1) + 1e-9))
    return dict(recon_fvu=fvu, dead_frac=dead, feat_activ_freq=activ_freq,
                mono_proxy=mono, decoder_interference=interference,
                n_live=int(live.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["cfg"]
    data = get_data(cfg["corpus"], cfg["seq_len"], cfg["batch_size"], device)
    model = TinyGPT(vocab_size=ck["vocab_size"], d_model=cfg["d_model"],
                    n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                    max_len=cfg["seq_len"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    acts = gather_acts(model, data, device, args.layer)
    d_dict = args.expansion * cfg["d_model"]
    sae, fired = train_sae(acts, d_dict, args.k, args.steps, device)
    m = metrics(sae, acts, fired, device)
    m.update(ckpt=os.path.basename(args.ckpt), layer=args.layer,
             d_dict=d_dict, k=args.k)
    out = args.ckpt.replace(".pt", f"_sae_L{args.layer}.json")
    json.dump(m, open(out, "w"), indent=2)
    print(f"{os.path.basename(args.ckpt)} L{args.layer}: "
          f"fvu={m['recon_fvu']:.3f} dead={m['dead_frac']:.3f} "
          f"mono={m['mono_proxy']:.3f} n_live={m['n_live']} -> {out}")


if __name__ == "__main__":
    main()
