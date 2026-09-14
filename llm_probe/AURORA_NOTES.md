# Aurora: implementation notes + the design template for our optimizer

Implemented from the reference (github.com/tilde-research/aurora-release, src/aurora.py
+ polar.py) so we understand the mechanism concretely before deriving our own.

## The mechanism, in one paragraph

Muon's update = polar(momentum) = UVᵀ (nearest semi-orthogonal matrix to the
momentum), which equalizes the update's singular values → spectral-norm steepest
descent. PROBLEM on tall matrices (m>n, e.g. MLP up/gate, m rows = neurons, n cols
= d_model): orthogonalization only constrains UᵀU=I (columns orthonormal); it says
nothing about ROW norms. Rows (neurons) with small gradient get small orthogonal
rows and STAY small → their preactivation contribution vanishes → the neuron
effectively dies. Aurora fixes this by ALSO forcing every row norm to the value a
truly column-orthogonal tall matrix would have on average: ‖row_i‖² = n/m.

## The algorithm (what each line is doing, and why it matters for us)

Input: momentum M (m×n), iters K=2, damping β=0.5.
  if m<=n:  update = polar(M)                       # square/wide → plain Muon
  else:                                             # tall → leverage-uniform polar
    D = 1/rownorm(M)                                # (A) warm-start preconditioner
    for k in 0..K-1:
      U = polar(D * M)                              # (B) orthogonalize the PRE-scaled matrix
      if k < K-1:
        row_sq = rowsumsq(U)                        # how far are U's rows from target?
        D = D * (target_row_sq / row_sq)^β          # (C) damped multiplicative correction
    update = U
  update *= max(1, m/n)^0.5                          # (D) Muon aspect-ratio LR scaling

Design choices that ARE the template (each is a lever for our optimizer):
- **(KEY) It's ALTERNATING PROJECTION onto an intersection of two manifolds.**
  polar(·) projects onto the Stiefel manifold (UᵀU=I). The diagonal-D rescaling
  nudges toward the oblique manifold (fixed row norms). Iterating between them
  approximates projection onto Stiefel ∩ oblique. THIS is the reusable pattern:
  to add a constraint to Muon, express it as a second manifold and alternate its
  projection with the polar step.
- **The constraint is applied by PRE-conditioning the input to polar, not by
  post-scaling the output.** Post-scaling (NorMuon) would break orthogonality;
  pre-scaling + re-orthogonalizing preserves it (to the iteration's accuracy).
  For OUR constraint we likewise need to fold it into what polar sees, not patch
  the result.
- **Damping β on the multiplicative correction** (not an additive/EMA step):
  D *= ratio^β with β<1 under-corrects each step to avoid oscillation between the
  two projections (classic alternating-projection stability trick). Any second
  constraint we add needs the analogous damping.
- **Target = n/m is DERIVED, not tuned**: a column-orthonormal tall matrix has
  Σᵢ‖rowᵢ‖² = ‖U‖_F² = n (trace of UᵀU = n), spread over m rows → mean n/m. So
  "equal row norms consistent with orthogonality" = n/m. Our constraint must
  similarly be derived from "what value is consistent with the base geometry."
- **Only tall matrices**: for square/wide, orthogonality already ⇒ uniform rows,
  so the extra constraint is vacuous/harmful. Our constraint's domain of action
  must be identified the same way.
- **CANS-12 polar** (9 Chebyshev-optimized cubic NS iters + 3 classic) — a fast
  accurate polar; we reuse it verbatim (polar.py).

## Why this is the right template for a "curiosity-able" optimizer

Aurora shows the RECIPE: (1) pick a representational pathology (neuron death =
dead, unrecombinable directions), (2) express its fix as a constraint MANIFOLD,
(3) derive the target value from consistency with the base (orthogonal) geometry,
(4) realize it as damped alternating projection folded into the polar input.
Our target is recombinability. Neuron death is ONE failure of recombinability
(dead directions); redundancy is another (duplicated directions). So our optimizer
would add the constraint that equalizes each direction's LEVERAGE ON THE OUTPUT —
same recipe, different (output-space, not weight-row) constraint. Aurora fixes
input-side row death; we want output-side direction viability. The math (next) is
about writing that constraint as a manifold and finding its projection.

## Our re-implementation

`aurora_opt.py` — faithful port (their aurora + CANS-12 polar), wired into our
train.py as --optimizer aurora, applied to tall 2D matrices (MLP up/gate),
plain-Muon on other 2D, AdamW on 1D/embeddings/head (our existing Muon split).
Validated on the SAME recombinability frontier as Muon/AdamW.
