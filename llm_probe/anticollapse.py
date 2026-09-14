"""
Anti-collapse richness regularizer R (design principle #2).

Adds an EXPLICIT term that fights representational collapse, so the optimizer
prizes keeping the representation high-rank/decorrelated (navigable) rather than
taking the cheapest collapsed path to low loss. Objective: L = CE - lambda * R.

Two surrogates (both differentiable, no entropy-of-SVD needed):
  offdiag : R = - mean(offdiagonal(Corr)^2). Maximizing R decorrelates features
            => pushes toward higher effective rank. Cheapest, most stable.
  logdet  : R = mean log(eigval(C + eps I)). Maximizing log-det spreads variance
            across ALL directions (isotropic covariance = max effective rank).
            The principled "volume/diversity" term.

Computed on a hidden activation matrix A (tokens x features), subsampled for cost.
IMPORTANT (anti-circularity): validate the trained model on a DIFFERENT property
(recombinability) and a DIFFERENT layer than the one R is applied to.
"""
import torch


def richness_reg(A, kind="offdiag", eps=1e-3, max_tokens=4096):
    """A: (N, d) activations (differentiable). Returns scalar R to be MAXIMIZED
    (added as -lambda*R... i.e. loss = CE - lambda*R)."""
    if A.dim() == 3:
        A = A.reshape(-1, A.shape[-1])
    if A.shape[0] > max_tokens:
        idx = torch.randperm(A.shape[0], device=A.device)[:max_tokens]
        A = A[idx]
    A = A - A.mean(0, keepdim=True)
    if kind == "offdiag":
        sd = A.std(0, keepdim=True) + 1e-6
        Z = A / sd
        C = (Z.T @ Z) / Z.shape[0]                 # correlation matrix
        d = C.shape[0]
        offsq = (C ** 2).sum() - torch.diagonal(C).pow(2).sum()
        # R = -mean offdiag^2  (higher = more decorrelated = higher rank)
        return -offsq / (d * (d - 1) + 1e-9)
    if kind == "logdet":
        C = (A.T @ A) / A.shape[0]
        d = C.shape[0]
        C = C + eps * torch.eye(d, device=A.device, dtype=A.dtype)
        # normalize scale so it doesn't just reward big activations
        C = C / (torch.diagonal(C).mean() + 1e-9)
        sign, logabsdet = torch.linalg.slogdet(C)
        return logabsdet / d                        # per-dim log-volume
    if kind == "negentropy":
        # HIGHER-ORDER independence (ICA / FastICA log-cosh contrast).
        # Second-order levers (offdiag/logdet, and the whitened-polar optimizer)
        # only reach DEcorrelation; negentropy reaches statistical INDEPENDENCE.
        # J(u) ~ (E[G(u)] - E[G(nu)])^2 with G(u)=log cosh(u), u a whitened feature,
        # nu~N(0,1). Maximizing summed per-feature negentropy pushes marginals AWAY
        # from Gaussian => the ZCA-whitened axes become an independent basis
        # (breaks the O(m) rotation-invariance that covariance targets cannot).
        return _negentropy(A, eps=eps)
    if kind == "cumulant":
        # Direct 4th-order cross-decorrelation the covariance can't see: after
        # whitening, penalize off-diagonal fourth-order cross-cumulants
        # E[u_i^2 u_j^2]-δ_ij-related terms. R = -mean(offdiag(K4)^2), maximized.
        return _cross_cumulant(A, eps=eps)
    raise ValueError(kind)


def _zca_whiten(A, eps=1e-3):
    """ZCA-whiten columns of A (N,d): output U has E[UᵀU]/N ≈ I. Differentiable.
    Robust to ill-conditioned activation covariances (dead/low-variance units):
    computes in float32, jitters by a TRACE-relative ridge so repeated ~0
    eigenvalues don't stall eigh, and standardizes per-feature first so scale
    imbalance across units doesn't dominate."""
    A32 = A.float()
    A32 = A32 - A32.mean(0, keepdim=True)
    A32 = A32 / (A32.std(0, keepdim=True) + 1e-6)      # per-feature standardize
    d = A32.shape[1]
    C = (A32.T @ A32) / A32.shape[0]
    ridge = eps * (torch.diagonal(C).mean() + 1e-8)    # trace-relative, not absolute
    C = C + ridge * torch.eye(d, device=A32.device, dtype=C.dtype)
    # DETACH the whitening transform: ZCA is a change-of-basis PRECONDITIONER, not
    # part of the objective. Backprop through eigh's eigenvectors is numerically
    # explosive when eigenvalues are near-degenerate (dead units) -> NaN loss.
    # Computing W under no_grad and applying it to the (grad-carrying) A32 keeps
    # gradients flowing through the log-cosh / cumulant contrast ONLY, which is
    # exactly the FastICA formulation (whiten, then optimize non-Gaussianity).
    with torch.no_grad():
        try:
            evals, evecs = torch.linalg.eigh(C)
            evals = evals.clamp_min(ridge)
            W = (evecs * evals.rsqrt()) @ evecs.T      # C^{-1/2}  (detached)
        except torch._C._LinAlgError:
            W = None
    if W is None:
        return A32.to(A.dtype)                         # already standardized ≈ identity
    return (A32 @ W).to(A.dtype)


def _negentropy(A, eps=1e-3):
    """Summed per-feature negentropy on ZCA-whitened features (to be MAXIMIZED).
    G(u)=log cosh(u); k=E_gauss[G]≈0.3746 for N(0,1). Higher = more non-Gaussian
    (more independent structure) along the whitened axes."""
    U = _zca_whiten(A, eps=eps)                    # unit-variance, decorrelated
    G = torch.log(torch.cosh(U) + 1e-12)           # (N,d)
    k_gauss = 0.3746               # E_{nu~N(0,1)}[log cosh nu], precomputed
    J = (G.mean(0) - k_gauss).pow(2)               # (d,) per-feature negentropy proxy
    return J.mean()


def _cross_cumulant(A, eps=1e-3):
    """Fourth-order cross-cumulant off-diagonal penalty on whitened features
    (to be MAXIMIZED as R = -mean offdiag^2). For whitened unit-variance u,
    E[u_i^2 u_j^2] = 1 under independence (i≠j); deviation = higher-order coupling
    invisible to the covariance. Drives toward 4th-order independence."""
    U = _zca_whiten(A, eps=eps)
    N, d = U.shape
    U2 = U * U                                      # (N,d)
    M = (U2.T @ U2) / N                             # E[u_i^2 u_j^2], (d,d)
    off = M - torch.diag(torch.diagonal(M))
    # under independence off-diagonals -> 1; center on 1 and penalize deviation
    dev = (off - 1.0) * (1.0 - torch.eye(d, device=A.device, dtype=A.dtype))
    return -(dev.pow(2).sum()) / (d * (d - 1) + 1e-9)
