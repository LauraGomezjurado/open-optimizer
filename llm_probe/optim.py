"""
Optimizers for the transformer optimizer-vs-factoredness test.

  - adamw : torch.optim.AdamW (baseline)
  - sgd   : torch.optim.SGD w/ momentum
  - muon  : Muon recipe -- orthogonalized (Newton-Schulz) momentum updates on 2D
            weight matrices, AdamW on everything else (embeddings, LayerNorm,
            biases, the tied head). This is the standard hybrid; Muon only makes
            sense on 2D hidden matrices.

Muon reference: Jordan et al. 2024 (the Newton-Schulz zeropower iteration that
orthogonalizes the momentum matrix so the update is ~semi-orthogonal).
"""
import torch
from torch.optim import AdamW, SGD


@torch.no_grad()
def _zeropower_via_newtonschulz(G, steps=5, eps=1e-7):
    """Orthogonalize G (m x n) via quintic Newton-Schulz. Returns ~UV^T of G."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    X = X / (X.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


# CANS-12 polar (from tilde-research/aurora-release/src/polar.py): 9 Chebyshev-
# optimized cubic Newton-Schulz iterations + 3 classic. Approximates polar(G)=UVᵀ.
_CANS12 = (
    (5.182503604966906, -5.178098480082684), (2.586120737395915, -0.6479542005271643),
    (2.567364126726186, -0.6454968804392178), (2.520560084348265, -0.6393528082067044),
    (2.410759275435182, -0.6248683598710716), (2.1883348130094173, -0.5952022073798908),
    (1.8595760874873613, -0.5504490972723968), (1.589020160467417, -0.5126569802066718),
    (1.5051653981684994, -0.5007377068751799), (1.5, -0.5), (1.5, -0.5), (1.5, -0.5),
)


@torch.no_grad()
def polar_cans12(G, eps=1e-7):
    """Polar factor UVᵀ via CANS-12 Newton-Schulz (fp32). Shape-preserving."""
    X = G.to(torch.float32)
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for a, b in _CANS12:
        A = X @ X.mT
        X = a * X + b * A @ X
    if tall:
        X = X.mT
    return X


@torch.no_grad()
def aurora_update(M, pp_iterations=2, pp_beta=0.5, eps=1e-7):
    """Aurora leverage-uniform polar (faithful port of aurora-release/src/aurora.py).
    M: momentum (m x n). For tall m>n: alternate polar() with a damped diagonal
    row-norm preconditioner so the update lands on Stiefel ∩ oblique (orthogonal
    AND equal row norms = n/m), fixing Muon's neuron death. Square/wide: plain polar.
    Returns the (unscaled) update direction; caller applies Muon aspect-ratio scaling."""
    m, n = M.size(-2), M.size(-1)
    if m <= n:
        return polar_cans12(M)
    G32 = M.to(torch.float32)
    target_row_sq = n / m                             # derived: ‖row‖²=n/m for column-orthonormal tall U
    row_norm = G32.norm(dim=-1, keepdim=True).clamp_(min=eps)
    D = 1.0 / row_norm                                # (A) warm-start preconditioner
    U = None
    for k in range(pp_iterations):
        U = polar_cans12(D * G32)                     # (B) orthogonalize the PRE-scaled matrix
        if k < pp_iterations - 1:
            row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp_(min=eps * eps)
            D = D * (target_row_sq / row_sq).pow(pp_beta)   # (C) damped multiplicative correction
    return U


class Muon(torch.optim.Optimizer):
    """Muon on 2D params; caller must put ONLY 2D hidden matrices in this group.
    1D / embedding / head params should go to a separate AdamW optimizer.
    Supports DECOUPLED weight decay (needed for Gate 1b weight-norm matching)."""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, wd=0.0):
        super().__init__(list(params),
                         dict(lr=lr, momentum=momentum, nesterov=nesterov,
                              ns_steps=ns_steps, wd=wd))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(g)
                buf = st["m"]
                buf.mul_(mom).add_(g)
                gg = g.add(buf, alpha=mom) if group["nesterov"] else buf
                if group.get("wd", 0.0):                    # decoupled weight decay
                    p.mul_(1 - group["lr"] * group["wd"])
                if gg.ndim == 2:
                    o = _zeropower_via_newtonschulz(gg, steps=group["ns_steps"])
                    # scale like the reference: sqrt(max/min dim) keeps update RMS sane
                    scale = (max(gg.shape) / min(gg.shape)) ** 0.5
                    p.add_(o, alpha=-group["lr"] * scale)
                else:
                    p.add_(gg, alpha=-group["lr"])


class Aurora(torch.optim.Optimizer):
    """Aurora on 2D params: identical to Muon EXCEPT the orthogonalization step is
    the leverage-uniform polar (aurora_update) -- tall matrices get the damped
    alternating Stiefel∩oblique projection, square/wide fall back to plain polar.
    Same aspect-ratio LR scaling as Muon. See AURORA_NOTES.md."""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 pp_iterations=2, pp_beta=0.5):
        super().__init__(list(params),
                         dict(lr=lr, momentum=momentum, nesterov=nesterov,
                              pp_iterations=pp_iterations, pp_beta=pp_beta))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(g)
                buf = st["m"]
                buf.mul_(mom).add_(g)
                gg = g.add(buf, alpha=mom) if group["nesterov"] else buf
                if gg.ndim == 2:
                    o = aurora_update(gg, pp_iterations=group["pp_iterations"],
                                      pp_beta=group["pp_beta"]).to(gg.dtype)
                    scale = (max(gg.shape) / min(gg.shape)) ** 0.5
                    p.add_(o, alpha=-group["lr"] * scale)
                else:
                    p.add_(gg, alpha=-group["lr"])


class SigmaXTracker:
    """Maintains an EMA of the INPUT covariance Σ_x = E[xxᵀ] for each nn.Linear
    whose weight is optimized by WhitenedPolar. Forward hooks capture the layer
    input x (…,in); we accumulate xᵀx. Provides Σ_x^{−1/2} (damped) per weight.
    This is the cheap, well-posed lever from DERIVATION_v2 (n×n, no vjps)."""
    def __init__(self, model, beta=0.95, eps=0.05):
        self.beta, self.eps = beta, eps
        self.cov = {}          # weight tensor id -> running (in,in) cov
        self.handles = []
        self._wid = {}         # module -> its weight tensor
        import torch.nn as nn
        for m in model.modules():
            if isinstance(m, nn.Linear):
                self._wid[m] = m.weight
                self.handles.append(m.register_forward_hook(self._hook))

    def _hook(self, module, inp, out):
        x = inp[0].detach()
        x = x.reshape(-1, x.shape[-1])                 # (N, in)
        c = (x.T @ x) / x.shape[0]
        w = self._wid[module]
        key = id(w)
        if key not in self.cov:
            self.cov[key] = c
        else:
            self.cov[key].mul_(self.beta).add_(c, alpha=1 - self.beta)
        # also track OUTPUT (post-linear) second moment per feature, for the
        # phi-gap diagonal post-correction: E[out_i^2] over tokens. (out is the
        # pre-activation; a=phi(out), but per-feature variance of out is the cheap
        # proxy the debate agent asked for to equalize activation magnitudes.)
        o = out.detach().reshape(-1, out.shape[-1])
        d2 = (o * o).mean(0)                            # (out,)
        k2 = ("out", key)
        if k2 not in self.cov:
            self.cov[k2] = d2
        else:
            self.cov[k2].mul_(self.beta).add_(d2, alpha=1 - self.beta)

    @torch.no_grad()
    def out_rms(self, weight, eps=1e-8):
        """Per-output-feature RMS sqrt(E[out_i^2]); None if untracked."""
        k2 = ("out", id(weight))
        if k2 not in self.cov:
            return None
        return self.cov[k2].clamp_min(eps).sqrt()

    @torch.no_grad()
    def isqrt(self, weight, alpha=0.5):
        """Σ_x^{−alpha} for this weight (alpha=0 -> identity=Muon)."""
        key = id(weight)
        if key not in self.cov or alpha == 0.0:
            return None
        C = self.cov[key].float()
        d = C.shape[0]
        C = C + self.eps * (torch.trace(C) / d) * torch.eye(d, device=C.device)
        evals, evecs = torch.linalg.eigh(C)
        evals = evals.clamp_min(1e-8)
        return (evecs * evals.pow(-alpha)) @ evecs.T     # Σ_x^{−alpha}


class WhitenedPolar(torch.optim.Optimizer):
    """Input-whitened polar (DERIVATION_v2 corrected optimizer):
        O* = polar(M · Σ_x^{−α}) · Σ_x^{−α},  α∈[0,1]  (α=0 ⇒ Muon)
    Targets high ACTIVATION-covariance rank via the input covariance Σ_x (fed by a
    SigmaXTracker). Weight W is (out,in); Σ_x is (in,in) so both mults are on the
    input index. Same aspect-ratio LR scaling as Muon; decoupled weight decay."""
    def __init__(self, params, tracker, lr=0.02, momentum=0.95, nesterov=True,
                 alpha=0.5, wd=0.0, place="up", renorm=True, postcorr=0.0,
                 base="muon", pp_iterations=2, pp_beta=0.5):
        """place: which matrices get input-whitened -- 'up' (tall m>n only),
                  'both' (all 2D), 'down' (wide m<=n only), 'none'.
        postcorr: strength in [0,1] of the phi-gap DIAGONAL post-correction --
                  divide update rows by per-output-feature RMS^postcorr so the
                  update equalizes activation magnitudes (Aurora-style, but derived
                  against measured E[out^2] instead of n/m). 0 = off.
        base: 'muon' (polar) or 'aurora' (leverage-uniform polar) as the
              orthogonalization core -- lets us stack whitening on Aurora."""
        super().__init__(list(params), dict(lr=lr, momentum=momentum,
                         nesterov=nesterov, alpha=alpha, wd=wd, place=place,
                         renorm=renorm, postcorr=postcorr, base=base,
                         pp_iterations=pp_iterations, pp_beta=pp_beta))
        self.tracker = tracker

    def _whiten_this(self, place, m, n):
        if place == "none": return False
        if place == "both": return True
        if place == "up":   return m > n
        if place == "down": return m <= n
        return False

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mom = group["momentum"]; alpha = group["alpha"]
            base = group["base"]; postcorr = group["postcorr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(g)
                buf = st["m"]; buf.mul_(mom).add_(g)
                gg = g.add(buf, alpha=mom) if group["nesterov"] else buf
                if group.get("wd", 0.0):
                    p.mul_(1 - group["lr"] * group["wd"])
                if gg.ndim == 2:
                    m, n = gg.shape
                    def _orth(X):
                        return (aurora_update(X, pp_iterations=group["pp_iterations"],
                                              pp_beta=group["pp_beta"])
                                if base == "aurora" else polar_cans12(X))
                    Sx = self.tracker.isqrt(p, alpha=alpha) if self._whiten_this(group["place"], m, n) else None
                    if Sx is not None:
                        M = gg.float() @ Sx
                        O = _orth(M) @ Sx
                        if group["renorm"]:               # restore unit-RMS step
                            O = O * (O.numel() ** 0.5 / (O.norm() + 1e-8))
                    else:
                        O = _orth(gg.float())
                    if postcorr > 0:                      # phi-gap diagonal fix
                        rms = self.tracker.out_rms(p)     # (out,) = (m,)
                        if rms is not None and rms.numel() == O.shape[0]:
                            w = (rms / rms.mean()).clamp_min(1e-6).pow(-postcorr)
                            O = O * w[:, None]
                            if group["renorm"]:
                                O = O * (O.numel() ** 0.5 / (O.norm() + 1e-8))
                    scale = (max(gg.shape) / min(gg.shape)) ** 0.5
                    p.add_(O.to(gg.dtype), alpha=-group["lr"] * scale)
                else:
                    p.add_(gg, alpha=-group["lr"])


def _split_2d(model):
    """2D hidden matrices (Muon/Aurora act on these) vs the rest (AdamW)."""
    twod, rest = [], []
    for n, p in model.named_parameters():
        if p.ndim == 2 and "tok" not in n and "pos" not in n and "head" not in n:
            twod.append(p)
        else:
            rest.append(p)
    return twod, rest


def build_optimizer(kind, model, lr, wd, wcfg=None):
    """Return optimizer_list. muon/aurora: [<2D optimizer>, AdamW(rest)]."""
    if kind == "adamw":
        return [AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))]
    if kind == "sgd":
        return [SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)]
    if kind == "muon":
        twod, rest = _split_2d(model)
        return [Muon(twod, lr=lr, momentum=0.95, wd=wd),
                AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, 0.95))]
    if kind == "aurora":
        twod, rest = _split_2d(model)
        return [Aurora(twod, lr=lr, momentum=0.95),
                AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, 0.95))]
    if kind == "whitened":
        w = dict(alpha=0.5, place="up", postcorr=0.0, base="muon")
        w.update(wcfg or {})
        twod, rest = _split_2d(model)
        tracker = SigmaXTracker(model)
        opts = [WhitenedPolar(twod, tracker, lr=lr, momentum=0.95, wd=wd,
                              alpha=w["alpha"], place=w["place"],
                              postcorr=w["postcorr"], base=w["base"]),
                AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, 0.95))]
        return opts
    raise ValueError(kind)
