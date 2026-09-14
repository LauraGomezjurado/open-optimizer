# Deriving a recombinability-preserving optimizer (Muon → Aurora → "Recombinator")

Following the Aurora template (AURORA_NOTES.md): pick a representational pathology,
express its fix as a constraint manifold, derive the target from consistency with
the base geometry, realize it as damped alternating projection folded into polar.

## Notation

Weight matrix W ∈ ℝ^{m×n} for an MLP up/gate projection: m rows = hidden neurons
(post-activation features), n cols = d_model inputs. Forward: pre-activation
h = W x, x ∈ ℝ^n. Update direction O = the (semi-)orthogonalized momentum.
Tall case m > n (our arch: d_ff=4d rows, d cols).

## Step 1 — the base geometry (Muon)

Muon: O = polar(M) = argmin_{OᵀO=I} ‖O − M‖_F = UVᵀ (M = UΣVᵀ). Constraint set =
**Stiefel manifold** St(m,n) = {O ∈ ℝ^{m×n} : OᵀO = I_n}. Equalizes update singular
values → spectral-norm steepest descent. Pathology (tall): OᵀO=I constrains
*columns*; *rows* (neurons) are unconstrained → small-gradient rows stay small →
neuron death.

## Step 2 — Aurora's added constraint (the template instance)

Aurora adds **equal row norms**: ‖O_{i,:}‖² = c ∀i. Target c derived from
consistency with Stiefel: for O ∈ St(m,n), Σᵢ‖O_{i,:}‖² = ‖O‖_F² = tr(OᵀO) =
tr(I_n) = n, spread over m rows → c = n/m. Constraint set = **oblique manifold**
Ob = {O : diag(OOᵀ) = (n/m) I_m}. Aurora targets **St(m,n) ∩ Ob** via damped
alternating projection (polar ↔ diagonal row-rescale). Fixes input-side death.

## Step 3 — our pathology: recombinability is OUTPUT-side, not input-side

Aurora keeps every *neuron* (row) alive. But recombinability (our measured target)
is about the **output** being perturbable into distinct-yet-coherent variations.
The relevant object is not the weight rows but each direction's **leverage on the
output distribution**. Two failure modes of recombinability:
  (a) DEAD output directions — a hidden direction that no longer influences logits
      (Aurora's row-death is a special case, but death can also happen through the
      down-projection / readout, which Aurora doesn't touch);
  (b) REDUNDANT output directions — two hidden directions that move the output the
      same way ⇒ perturbing along them is not independent ⇒ no new stepping stone.

So our constraint should equalize, and de-correlate, each hidden unit's *effect on
the output*, not its weight-row norm. This is the output-space analog of Aurora.

## Step 4 — formalize "output leverage"

For up-proj W (m×n) feeding activation φ then down-proj/readout chain, the
first-order effect of hidden unit i on the output y is captured by the Jacobian
column J_{:,i} = ∂y/∂h_i (averaged over a data batch). Define the **output-leverage
Gram** L = JᵀJ ∈ ℝ^{m×m}: L_{ii} = ‖∂y/∂h_i‖² (unit i's influence magnitude),
L_{ij} = ⟨∂y/∂h_i, ∂y/∂h_j⟩ (how redundant units i,j are).

Recombinability wants:
  - no dead unit: L_{ii} bounded away from 0 (every direction matters) — like Aurora
    but measured through the FULL output path, not just the weight row;
  - no redundancy: L_{ij}≈0 for i≠j (directions independent) — this is the NEW part,
    the output-space decorrelation Aurora lacks.
Ideal: **L ∝ I_m** (isotropic output leverage). This is precisely "every direction
is an independent, live stepping stone."

## Step 5 — the constraint as a manifold + the derived target

We cannot directly constrain L via the up-proj update alone (L depends on the whole
downstream path). Two tractable routes:

**Route A (weight-space surrogate, Aurora-compatible, cheap).** Approximate output
leverage by the update's own row structure PLUS a decorrelation term. Constraint:
O ∈ St(m,n) AND OOᵀ is as diagonal as possible (rows orthogonal *to each other*,
not just unit-norm). But for tall O ∈ St(m,n), OOᵀ is an m×m rank-n projector — it
CANNOT be diagonal (rank n < m). So pure row-orthogonality is infeasible tall; the
best is Aurora's equal-diagonal (n/m) — which is why Aurora stops there. **Insight:
in weight space alone you cannot get output-decorrelation for tall matrices; the
extra degrees of freedom aren't there. Recombinability's decorrelation MUST use
downstream (output-path) information.** This is the precise sense in which our
optimizer must go beyond Aurora.

**Route B (output-informed preconditioner).** Fold a cheap estimate of L into the
polar input, exactly as Aurora folds the row-norm preconditioner D. Let P be a
diagonal (or low-rank) preconditioner chosen so that after the update, the
output-leverage Gram moves toward isotropy: precondition M̃ = f(L) · M before
polar, then re-orthogonalize (damped alternating projection, same as Aurora). The
target for f(L) is derived from consistency: at the fixed point we want
diag(L)=const and off-diag(L)→0, i.e. f drives L toward (tr L / m) I_m.

## Step 6 — the projection: is there a Newton-Schulz-style iteration?

Aurora's iteration works because its two projections each have closed/cheap forms
(polar via NS; row-rescale via diagonal division) and their alternation converges.
For ours:
  - Stiefel projection: polar (same CANS-12 NS). ✓ cheap.
  - Output-isotropy projection: we need to move O so JᵀJ → isotropic. J depends on
    the downstream weights, so exact projection is not closed-form. BUT a **damped
    diagonal preconditioner on the hidden index**, updated from a running estimate
    of L's diagonal and a cheap off-diagonal proxy, mirrors Aurora's D-update:
        D ← D · (target_leverage / est_leverage_i)^β    (kills dead directions)
        plus an off-diagonal decorrelation nudge (whitening step on the hidden index)
    then O = polar(D̃ · M). This IS a Newton-Schulz-style alternating iteration —
    Aurora's, generalized from "row norm" to "output leverage." The off-diagonal
    (whitening) piece is the genuinely new, more expensive term (needs L, not just
    row norms) — the cost/accuracy tradeoff is the open design question.

## Step 7 — what to derive next (the math still owed)

1. A cheap estimator of the output-leverage Gram L per step (subsample tokens;
   use the captured Jacobian of logits w.r.t. hidden — we already patch/capture
   activations, so ∂y/∂h is one vjp).
2. Convergence of the damped alternating projection (Stiefel ↔ output-isotropy):
   does it converge like Aurora's? Aurora's converges because oblique∩Stiefel is
   benign; output-isotropy is data-dependent, so this needs either a fixed-per-step
   L estimate (freeze L within the inner iterations) or a proof under slow-L drift.
3. The derived target: show that O with JᵀJ ∝ I is the max-recombinability update
   at fixed loss-descent (Lagrangian: minimize ⟨O,−M⟩ s.t. OᵀO=I and JᵀJ∝I).

## The one-line thesis

Muon constrains update *singular values* (Stiefel). Aurora adds *equal weight-row
leverage* (∩ oblique), fixing input-side neuron death. **Recombinability requires
equal + decorrelated OUTPUT leverage (JᵀJ ∝ I)** — which is infeasible in weight
space alone for tall matrices (Route A impossibility) and therefore demands an
output-informed damped-alternating-projection step (Route B) — Aurora's iteration
generalized from row-norms to the output-leverage Gram. That is the "Recombinator"
optimizer, and the empirical target it must beat is the recombinability frontier.
