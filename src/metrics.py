"""
Factoredness readouts.

The FER paper diagnoses fractured-vs-factored representations qualitatively, by
*weight sweeps*: perturb a single weight and watch the output image. In a
factored (UFR) network the change is orderly, localized and semantically
meaningful; in a fractured (FER) network a single weight causes global,
"skull-crushing" distortion, and many weights redundantly do overlapping things.

We turn that into three scalars by building a *sampled parameter->image
Jacobian* via central finite differences on K randomly chosen weights:

    S[k, :] = ( I(w_k + delta) - I(w_k - delta) ) / (2 delta)      # flattened image

From S we compute:

  1. sensitivity   = mean_k || S[k] ||_2
        How violently the image reacts to a unit weight change. Fractured nets
        are brittle -> high sensitivity.

  2. locality      = mean_k ( 1 - PR_spatial(S[k]) )
        PR_spatial is the normalized participation ratio of the per-pixel change
        magnitude (in (0,1]; ~1 = change spread over whole image, ~1/Npix =
        change on a single pixel). locality high => change is spatially
        concentrated => more factored / modular.

  3. redundancy    = 1 - effrank(S) / K
        effrank = exp(entropy of normalized singular values of S). If the K
        weights produce near-duplicate influence maps (entanglement/redundancy,
        the FER prediction), S is low-rank -> low effrank -> high redundancy.
        Factored networks spend weights on distinct effects -> high effrank ->
        low redundancy.

We also report a feature-level readout that needs no sweeps:

  4. feat_effrank_frac = effrank(last-hidden activations covariance) / width
        Redundant (fractured) features are correlated -> low effective rank.

Higher `locality` and `feat_effrank_frac`, and lower `sensitivity` and
`redundancy`, all point toward the *factored* (UFR) end.

--- Saturation-invariant additions (confound C2) ---

The proxies above co-move with unit saturation: coordinatewise-adaptive
optimizers clip more units past |a|>0.95, and a saturated activation has both a
degenerate covariance (low feat-rank) and a cliff-like weight response (high
sensitivity). To measure *structure* rather than *how many units got clipped* we
add two saturation-robust readouts:

  5. influence_orthogonality = mean off-diagonal |cos| of the sweep Jacobian rows
        Build S (K x Npix*3), L2-normalize each row, form the Gram G = S_hat S_hat^T.
        The mean |off-diagonal| entry is "how aligned are the image-effects of
        different weights". It is invariant to each row's *magnitude* (so a
        saturated-but-orthogonal net is not penalized), unlike `redundancy`
        (effective rank) which is depressed by any magnitude imbalance. High
        overlap => entangled/fractured; low overlap => factored. We report
        influence_orthogonality = 1 - mean|offdiag cos| (up = factored) so it
        aligns with locality/feat_effrank direction.

  6. feat_effrank_corr_frac = effrank of the CORRELATION matrix of last-hidden
        activations (columns standardized to unit variance first). This removes
        confound C3 -- raw covariance eff-rank is dominated by whichever
        activation family (identity is unbounded; sin/tanh/gauss are O(1)) has
        largest variance, which is itself geometry-dependent. Standardizing asks
        "how many independent directions" rather than "which activations got big".
"""
import numpy as np


def _participation_ratio(v):
    v = np.asarray(v, dtype=np.float64)
    s1 = v.sum()
    s2 = (v * v).sum()
    if s2 <= 0:
        return 1.0
    return (s1 * s1) / (s2 * len(v))   # in (0,1]


def _effective_rank(singular_values, eps=1e-12):
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    ent = -(p * np.log(p)).sum()
    return float(np.exp(ent))


def weight_sweep_jacobian(cppn, X, params, n_weights=96, delta=1e-2, seed=0,
                          relative=False):
    """Central-difference param->image Jacobian on a random sample of weights.

    relative=False : additive perturbation w +/- delta          -> dI/dw
    relative=True  : multiplicative perturbation w*(1 +/- delta) -> ~ w * dI/dw
                     = dI/dlog|w|, a SCALE-INVARIANT elasticity. This controls the
                     confound where different geometries land at different weight
                     magnitudes, so a fixed additive delta means different relative
                     perturbations.
    """
    rng = np.random.default_rng(seed)
    # flatten weight index space
    shapes = [w.shape for w in params]
    sizes = [w.size for w in params]
    total = sum(sizes)
    idx = rng.choice(total, size=min(n_weights, total), replace=False)
    Npix = X.shape[0]
    rows = []
    base = [w.copy() for w in params]
    for flat_i in idx:
        # locate (layer, row, col)
        li, off = 0, flat_i
        while off >= sizes[li]:
            off -= sizes[li]; li += 1
        r, c = np.unravel_index(off, shapes[li])
        w0 = base[li][r, c]
        if relative:
            h = delta * (abs(w0) + 1e-6)
        else:
            h = delta
        base[li][r, c] = w0 + h
        ip = cppn.render(X, params=base)
        base[li][r, c] = w0 - h
        im = cppn.render(X, params=base)
        base[li][r, c] = w0
        if relative:
            rows.append((((ip - im) / (2 * h)) * (abs(w0) + 1e-6)).ravel())
        else:
            rows.append(((ip - im) / (2 * h)).ravel())
    S = np.stack(rows, axis=0)   # (K, Npix*3)
    return S, Npix


def confound_diagnostics(cppn, X, params):
    """Diagnostics that expose confounds in the factoredness readouts."""
    frob, ers = 0.0, []
    for w in params:
        frob += float(np.linalg.norm(w))
        sv = np.linalg.svd(w, compute_uv=False)
        ers.append(_effective_rank(sv) / min(w.shape))
    _, (_, As, _) = cppn.forward(X, params=params, cache=True)
    A = As[-1]
    sat_frac = float(np.mean(np.abs(A) > 0.95))
    return dict(
        weight_frob=frob,
        weight_effrank_frac=float(np.mean(ers)),   # confound #2: does feat-rank just echo this?
        sat_frac=sat_frac,                         # confound #3: saturation inflating/deflating rank
    )


def _influence_orthogonality(S, eps=1e-12, remove_common_mode=True):
    """1 - mean|off-diagonal cosine| of the sweep-Jacobian rows.

    Row-normalize S so each weight's influence map is a unit vector, then the
    Gram matrix is the cosine-similarity matrix. The mean absolute off-diagonal
    entry is how aligned different weights' effects are -- SCALE-INVARIANT (a
    saturated-but-orthogonal net is not penalized), unlike effective-rank
    redundancy which is depressed by magnitude imbalance. We return
    1 - overlap so up = factored (distinct effects).

    remove_common_mode: subtract the mean influence map across weights first.
    Every weight sweep tends to induce a similar *global* image change (overall
    brightness/tint), a common-mode term that aligns all rows regardless of
    factoredness and saturates the metric near 1. Removing it isolates the
    STRUCTURED, weight-specific part of each influence map -- which is what
    'do different weights do different things' actually means."""
    K = S.shape[0]
    if K < 2:
        return 0.0
    if remove_common_mode:
        S = S - S.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    keep = (norms.ravel() > eps)
    Sn = S[keep] / (norms[keep] + eps)
    if Sn.shape[0] < 2:
        return 0.0
    G = np.abs(Sn @ Sn.T)
    m = G.shape[0]
    off = (G.sum() - np.trace(G)) / (m * (m - 1))   # mean |off-diagonal cosine|
    return float(1.0 - off)


def _feat_effrank_corr_frac(cppn, X, params):
    """Effective rank of the CORRELATION (not covariance) matrix of last-hidden
    activations. Standardizing columns to unit variance removes confound C3:
    raw covariance eff-rank is dominated by whichever activation family got the
    largest variance under a given geometry."""
    _, (_, As, _) = cppn.forward(X, params=params, cache=True)
    A = As[-1]
    A = A - A.mean(0, keepdims=True)
    sd = A.std(0, keepdims=True)
    A = A / (sd + 1e-8)                 # unit-variance columns => correlation matrix
    corr = (A.T @ A) / A.shape[0]
    sv = np.linalg.svd(corr, compute_uv=False)
    return float(_effective_rank(sv) / A.shape[1])


def activation_kurtosis(A, eps=1e-9):
    """Mean per-feature excess kurtosis of an activation matrix A (N samples x
    F features). This is the SUPERPOSITION discriminator that effective rank
    cannot provide: economical/factored bases and collapsed/superposed ones can
    have the SAME (low) effective rank, but superposition packs many sparse
    features into few dims, giving heavy-tailed (high-kurtosis) per-feature
    activations, whereas a clean independent factor basis is ~Gaussian (kurtosis
    ~0). Validated on a synthetic economical-vs-superposed pair at matched
    effective rank (econ ~0, superposed +0.7..+1.5).

    HIGH kurtosis = superposed/fractured ; LOW (~0) = economical/factored.
    Invariant to effective rank by construction (it's a per-feature shape stat)."""
    A = np.asarray(A, np.float64)
    keep = A.std(0) > eps
    A = A[:, keep]
    if A.shape[1] == 0:
        return 0.0
    z = (A - A.mean(0)) / (A.std(0) + eps)
    return float((z ** 4).mean() - 3.0)


def neuron_structure_metrics(cppn, X, res, params=None):
    """CAPACITY-INVARIANT factoredness readout, in NEURON space (not weight
    space). Renders every LIVE hidden neuron as an image over the res x res grid
    and measures how smooth / low-frequency each neuron image is.

    Motivation: the FER phenomenon is 'is each neuron a clean coherent primitive
    (factored) or a noisy high-frequency redundant fragment (fractured)?'. That
    is a property of the neuron IMAGE, independent of how many weights feed it --
    so unlike weight-perturbation `sensitivity`/`redundancy` it does not collapse
    when a network is sparse / low-capacity. Validated to rank the evolved
    Picbreeder genome (UFR) as more factored than the FER paper's SGD genome on
    all three targets, the direction the weight-space metrics get backwards.

    Returns (all per-neuron, aggregated over live neurons):
      neuron_tv        : median total-variation of the normalized neuron image
                         (LOW = smooth = factored)
      neuron_hf_energy : median fraction of spectral energy above a low-frequency
                         band (LOW = low-frequency = factored)
      neuron_smoothness: 1 - mean_tv  (HIGH = factored), for a single up=factored scalar
      n_live_neurons   : count of non-constant hidden neurons
    """
    _, (_, As, _) = cppn.forward(X, params=params, cache=True)
    hidden = As[1:]                       # drop the input layer
    tvs, hfs = [], []
    yy, xx = np.ogrid[:res, :res]
    r = np.sqrt((yy - res / 2) ** 2 + (xx - res / 2) ** 2)
    hf_mask = r > res * 0.15              # "high frequency" = outside central band
    for A in hidden:
        w = A.shape[1]
        imgs = A.reshape(res, res, w)
        for j in range(w):
            m = imgs[:, :, j]
            rng = m.max() - m.min()
            if rng < 1e-6:                # dead / constant neuron -> not live
                continue
            mn = (m - m.min()) / rng
            tv = (np.abs(np.diff(mn, axis=0)).mean() +
                  np.abs(np.diff(mn, axis=1)).mean()) / 2.0
            tvs.append(tv)
            F = np.abs(np.fft.fftshift(np.fft.fft2(mn - mn.mean()))) ** 2
            hfs.append(F[hf_mask].sum() / (F.sum() + 1e-12))
    if not tvs:
        return dict(neuron_tv=0.0, neuron_hf_energy=0.0, neuron_smoothness=1.0,
                    n_live_neurons=0)
    tvs = np.array(tvs)
    return dict(
        neuron_tv=float(np.median(tvs)),
        neuron_hf_energy=float(np.median(hfs)),
        neuron_smoothness=float(1.0 - tvs.mean()),
        n_live_neurons=int(len(tvs)),
    )


def factoredness_metrics(cppn, X, params, n_weights=96, delta=1e-2, seed=0,
                         n_boot=200):
    S, Npix = weight_sweep_jacobian(cppn, X, params, n_weights, delta, seed)
    K = S.shape[0]
    # 1. sensitivity
    row_norms = np.linalg.norm(S, axis=1)
    sensitivity = float(row_norms.mean())
    # 2. locality (per-pixel magnitude across 3 channels)
    Simg = S.reshape(K, Npix, 3)
    pr = []
    for k in range(K):
        mag = np.linalg.norm(Simg[k], axis=1)   # (Npix,)
        if mag.sum() <= 0:
            continue
        pr.append(_participation_ratio(mag))
    locality = float(1.0 - np.mean(pr)) if pr else 0.0
    # 3. redundancy via effective rank of S
    sv = np.linalg.svd(S, compute_uv=False)
    er = _effective_rank(sv)
    redundancy = float(1.0 - er / K)
    # 4. feature effective rank of last hidden layer (raw covariance -- has C3)
    _, (_, As, _) = cppn.forward(X, params=params, cache=True)
    A_last = As[-1]                     # (N, width)
    A_last = A_last - A_last.mean(0, keepdims=True)
    cov = (A_last.T @ A_last) / A_last.shape[0]
    fsv = np.linalg.svd(cov, compute_uv=False)
    feat_effrank_frac = float(_effective_rank(fsv) / A_last.shape[1])

    # 5. SCALE-INVARIANT sensitivity via elastic (multiplicative) sweep
    S_rel, _ = weight_sweep_jacobian(cppn, X, params, n_weights, delta=0.05,
                                     seed=seed, relative=True)
    sensitivity_elastic = float(np.linalg.norm(S_rel, axis=1).mean())

    # 6. SATURATION-INVARIANT structure readouts (confounds C2/C3)
    influence_orthogonality = _influence_orthogonality(S)
    feat_effrank_corr_frac = _feat_effrank_corr_frac(cppn, X, params)

    # 7. bootstrap CIs over the sampled weights (confound C6): resample rows of S
    rng = np.random.default_rng(seed + 12345)
    boot_sens, boot_orth = [], []
    for _ in range(n_boot):
        bi = rng.integers(0, K, size=K)
        boot_sens.append(float(row_norms[bi].mean()))
        boot_orth.append(_influence_orthogonality(S[bi]))
    def _ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    out = dict(
        sensitivity=sensitivity,
        sensitivity_elastic=sensitivity_elastic,
        locality=locality,
        redundancy=redundancy,
        effrank_jacobian=float(er),
        feat_effrank_frac=feat_effrank_frac,
        influence_orthogonality=influence_orthogonality,
        feat_effrank_corr_frac=feat_effrank_corr_frac,
        sensitivity_ci=_ci(boot_sens),
        influence_orthogonality_ci=_ci(boot_orth),
    )
    out.update(confound_diagnostics(cppn, X, params))
    return out
